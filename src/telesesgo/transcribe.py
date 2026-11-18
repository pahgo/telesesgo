"""Transcripción local con faster-whisper y diarización opcional con pyannote."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from dataclasses import asdict, dataclass

import numpy as np
from pathlib import Path

from . import db
from .config import Config, SttConfig

log = logging.getLogger(__name__)


@dataclass
class Line:
    idx: int
    start: float
    end: float
    text: str
    speaker: str | None = None


_model_cache: dict[tuple, object] = {}


def release_whisper() -> None:
    """Libera Whisper de la GPU (4 GB no dan para Whisper y la diarización a la vez: si no caben, Windows pasa parte
    a la memoria del sistema y todo va varias veces más lento). Se vuelve a cargar en ~2 s cuando haga falta."""
    if _model_cache:
        _model_cache.clear()
        import gc
        gc.collect()


def _preload_cuda_libs() -> None:
    """Carga cuBLAS/cuDNN de los wheels nvidia-* (extra `cuda`) para que ctranslate2 los encuentre
    sin tener que exportar LD_LIBRARY_PATH."""
    import ctypes
    import glob

    try:
        import nvidia
    except ImportError:
        return
    for base in nvidia.__path__:
        for pattern in ("cublas/lib/libcublasLt.so.*", "cublas/lib/libcublas.so.*", "cudnn/lib/libcudnn*.so.*"):
            for lib in sorted(glob.glob(os.path.join(base, pattern))):
                try:
                    ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
                except OSError as e:
                    log.debug("No se pudo cargar %s: %s", lib, e)


def _load_model(stt: SttConfig):
    _preload_cuda_libs()
    from faster_whisper import WhisperModel

    compute = stt.compute_type
    if compute == "default":
        import ctranslate2

        gpu = stt.device == "cuda" or (stt.device == "auto" and ctranslate2.get_cuda_device_count() > 0)
        compute = "int8_float16" if gpu else "int8"  # 4 GB de VRAM: int8_float16 deja margen
    key = (stt.model, stt.device, compute)
    if key not in _model_cache:
        log.info("Cargando Whisper %s (device=%s, compute=%s)", *key)
        _model_cache[key] = WhisperModel(stt.model, device=stt.device, compute_type=compute)
    return _model_cache[key]


def transcribe_audio(audio_path: Path, stt: SttConfig, on_progress=None) -> tuple[list[Line], str]:
    """`on_progress(segundos_hechos, segundos_totales, velocidad)` se llama cada minuto de audio."""
    model = _load_model(stt)
    common = dict(
        language=stt.language,
        beam_size=stt.beam_size,
        vad_filter=stt.vad_filter,
        vad_parameters={"min_silence_duration_ms": 500},
        initial_prompt=stt.initial_prompt,
        # Evita bucles de alucinación típicos en audios largos con música/cortinillas.
        condition_on_previous_text=False,
    )
    if stt.batch_size > 0:
        from faster_whisper import BatchedInferencePipeline

        # without_timestamps=False: segmentos con marcas de tiempo propias (frases), no un bloque por fragmento VAD.
        segments, info = BatchedInferencePipeline(model=model).transcribe(
            str(audio_path), batch_size=stt.batch_size, without_timestamps=False, **common)
    else:
        segments, info = model.transcribe(str(audio_path), **common)
    total = info.duration or 0
    t0 = time.monotonic()
    next_log = 60.0
    lines: list[Line] = []
    for seg in segments:
        if seg.end >= next_log:
            elapsed = time.monotonic() - t0
            speed = seg.end / elapsed if elapsed else 0
            eta = (total - seg.end) / speed if speed else 0
            log.info("  %s / %s audio  (%.2fx tiempo real, quedan ~%d min)",
                     _mmss(seg.end), _mmss(total), speed, round(eta / 60))
            if on_progress:
                on_progress(seg.end, total, speed)
            next_log = (seg.end // 60 + 1) * 60
        text = seg.text.strip()
        if text:
            lines.append(Line(idx=len(lines), start=round(seg.start, 2), end=round(seg.end, 2), text=text))
    log.info("Transcritos %s de audio en %.1f min%s", _mmss(total), (time.monotonic() - t0) / 60,
             f" (lotes de {stt.batch_size})" if stt.batch_size > 0 else "")
    return lines, info.language


def _mmss(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


_diar_cache: dict[str, object] = {}


def _diarization_pipeline(dc):
    from pyannote.audio import Pipeline
    import torch

    if dc.model not in _diar_cache:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("Falta HF_TOKEN para descargar el modelo de diarización (ver PASOS_PENDIENTES.md)")
        try:
            pipe = Pipeline.from_pretrained(dc.model, token=token)
        except TypeError:  # pyannote < 4
            pipe = Pipeline.from_pretrained(dc.model, use_auth_token=token)
        if pipe is None:
            raise RuntimeError(f"No se pudo cargar {dc.model}: ¿has aceptado sus condiciones en Hugging Face?")
        for attr in ("embedding_batch_size", "segmentation_batch_size"):
            if hasattr(pipe, attr):
                try:
                    setattr(pipe, attr, dc.batch_size)
                except Exception as exc:  # versiones de pyannote sin ese ajuste
                    log.debug("No se pudo fijar %s: %s", attr, exc)
        _diar_cache[dc.model] = pipe
    return _diar_cache[dc.model]


def _diar_device(dc) -> str:
    import torch
    return "cuda" if dc.device == "cuda" or (dc.device == "auto" and torch.cuda.is_available()) else "cpu"


def diarize(audio_path: Path, lines: list[Line], dc) -> dict[str, tuple[float, object]]:
    """Asigna SPEAKER_nn a cada línea por máximo solape y devuelve {etiqueta: (segundos, huella de voz)}."""
    import torch
    from faster_whisper import decode_audio

    pipe = _diarization_pipeline(dc)
    audio = decode_audio(str(audio_path), sampling_rate=16000)
    inp = {"waveform": torch.from_numpy(audio)[None], "sample_rate": 16000}
    device = _diar_device(dc)
    if device == "cuda":
        release_whisper()  # la GPU, solo para la diarización mientras dura
        pipe.to(torch.device("cuda"))
    try:
        try:
            result = pipe(inp)
        except (RuntimeError, torch.OutOfMemoryError) as exc:
            if "out of memory" not in str(exc).lower() or device != "cuda":
                raise
            log.warning("Diarización sin memoria en la GPU; se repite en CPU (más lento)")
            pipe.to(torch.device("cpu"))
            torch.cuda.empty_cache()
            result = pipe(inp)
    finally:
        if device == "cuda":  # devolver la GPU a Whisper
            pipe.to(torch.device("cpu"))
            torch.cuda.empty_cache()
    # pyannote 4: DiarizeOutput con la versión sin solapes y una huella por hablante
    annotation = getattr(result, "exclusive_speaker_diarization", None) or getattr(result, "speaker_diarization", result)
    embeddings = getattr(result, "speaker_embeddings", None)
    order = list(getattr(result, "speaker_diarization", annotation).labels())
    turns = [(t.start, t.end, spk) for t, _, spk in annotation.itertracks(yield_label=True)]
    for ln in lines:
        overlap: dict[str, float] = {}
        for s0, e0, spk in turns:
            o = min(e0, ln.end) - max(s0, ln.start)
            if o > 0:
                overlap[spk] = overlap.get(spk, 0) + o
        ln.speaker = max(overlap, key=overlap.get) if overlap else None
    seconds: dict[str, float] = {}
    for ln in lines:
        if ln.speaker:
            seconds[ln.speaker] = seconds.get(ln.speaker, 0.0) + (ln.end - ln.start)
    out = {}
    for spk, secs in seconds.items():
        emb = None
        if embeddings is not None and spk in order and order.index(spk) < len(embeddings):
            emb = embeddings[order.index(spk)]
            if emb is not None and not np.all(np.isfinite(emb)):
                emb = None
        out[spk] = (secs, emb)
    return out


def diarize_episode(cfg: Config, conn: sqlite3.Connection, ep: sqlite3.Row, lines: list[Line] | None = None) -> int:
    """Diariza un episodio ya transcrito, guarda los hablantes y los compara con la biblioteca de voces."""
    from .voices import match_library, store_episode_speakers

    audio = Path(ep["audio_path"] or "")
    if not audio.exists():
        raise FileNotFoundError(f"No está el audio para diarizar: {audio}")
    lines = lines if lines is not None else load_lines(conn, ep["id"])
    speakers = diarize(audio, lines, cfg.stt.diarization)
    conn.executemany("UPDATE transcript_lines SET speaker = ? WHERE episode_id = ? AND idx = ?",
                     [(l.speaker, ep["id"], l.idx) for l in lines])
    conn.execute("UPDATE transcripts SET diarized = 1 WHERE episode_id = ?", (ep["id"],))
    store_episode_speakers(conn, ep["id"], speakers)
    known = match_library(cfg, conn, ep["id"])
    log.info("Diarizado %s: %d hablantes, %d reconocidos por la voz", ep["id"], len(speakers), known)
    return len(speakers)


def needs_diarization(cfg: Config, conn: sqlite3.Connection, episode_id: int) -> bool:
    if not cfg.stt.diarization.enabled:
        return False
    r = conn.execute("SELECT diarized FROM transcripts WHERE episode_id = ?", (episode_id,)).fetchone()
    return r is not None and not r[0]


def transcribe_episode(cfg: Config, conn: sqlite3.Connection, ep: sqlite3.Row, on_progress=None) -> list[Line]:
    audio = Path(ep["audio_path"])
    log.info("Transcribiendo %s", audio.name)
    lines, lang = transcribe_audio(audio, cfg.stt, on_progress)
    diarized = False  # la diarización es un paso aparte (diarize_episode)

    out = cfg.transcript_dir / ep["source_key"] / f"{ep['air_date']}_{ep['external_id']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"episode_id": ep["id"], "model": cfg.stt.model, "language": lang,
                               "diarized": diarized, "lines": [asdict(l) for l in lines]},
                              ensure_ascii=False, indent=1), encoding="utf-8")

    conn.execute("DELETE FROM transcript_lines WHERE episode_id = ?", (ep["id"],))
    conn.executemany(
        "INSERT INTO transcript_lines (episode_id, idx, start_s, end_s, speaker, text) VALUES (?, ?, ?, ?, ?, ?)",
        [(ep["id"], l.idx, l.start, l.end, l.speaker, l.text) for l in lines],
    )
    conn.execute(
        "INSERT OR REPLACE INTO transcripts (episode_id, model, language, diarized, json_path) VALUES (?, ?, ?, ?, ?)",
        (ep["id"], cfg.stt.model, lang, int(diarized), str(out)),
    )
    db.set_status(conn, ep["id"], "transcribed")
    if cfg.delete_audio_after_transcribe and not cfg.stt.diarization.enabled:
        audio.unlink(missing_ok=True)
    log.info("Transcripción: %d líneas -> %s", len(lines), out.name)
    return lines


def load_lines(conn: sqlite3.Connection, episode_id: int) -> list[Line]:
    rows = conn.execute(
        "SELECT idx, start_s, end_s, text, speaker FROM transcript_lines WHERE episode_id = ? ORDER BY idx", (episode_id,)
    ).fetchall()
    return [Line(idx=r["idx"], start=r["start_s"], end=r["end_s"], text=r["text"], speaker=r["speaker"]) for r in rows]
