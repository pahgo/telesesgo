"""Carga y validación de config.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class SourceConfig(BaseModel):
    key: str
    channel: str
    program: str
    type: Literal["rtve", "atresplayer", "ytdlp"]
    enabled: bool = True
    program_id: int | None = None  # rtve
    format_id: str | None = None  # atresplayer
    url: str | None = None  # ytdlp
    title_regex: str | None = None
    min_duration_s: int = 1200
    # Franja de la edición: "mediodia" (14-15 h) o "noche" (20-21 h). Se usa en `telesesgo run --edicion`.
    edition: Literal["mediodia", "noche"] | None = None


class DownloadConfig(BaseModel):
    format: str = "ba/b"
    transcode_bitrate: str | None = "24k"
    cookies_file: str | None = None
    # Descargas simultáneas. Van en paralelo con la transcripción en `run`.
    workers: int = 4
    # Carpeta temporal para la descarga y la recodificación. Por defecto, la de temporales del sistema
    # (disco de Linux): escribir los fragmentos en C: desde WSL/contenedor (9p) frena la transcripción.
    staging_dir: str | None = None


class DiarizationConfig(BaseModel):
    enabled: bool = False
    model: str = "pyannote/speaker-diarization-community-1"
    device: str = "auto"                # auto|cuda|cpu (si la GPU se queda sin memoria, se pasa a CPU)
    # Reconocimiento de voces conocidas (coseno entre huellas normalizadas)
    match_threshold: float = 0.75       # parecido mínimo (coseno) para dar por reconocida una voz; habitual 0.60-0.80
    match_min_seconds: float = 8.0      # voz mínima de un hablante para intentar reconocerlo
    learn_min_seconds: float = 20.0     # voz mínima para añadirlo o refinarlo en la biblioteca
    learn_min_similarity: float = 0.50  # si la huella nueva se parece menos a la guardada, no se mezcla


class SttConfig(BaseModel):
    model: str = "large-v3-turbo"
    device: str = "auto"
    compute_type: str = "default"
    language: str = "es"
    beam_size: int = 5
    vad_filter: bool = True
    initial_prompt: str | None = None
    # >0: transcripción por lotes (BatchedInferencePipeline), varios fragmentos a la vez en la GPU.
    # 0: modo secuencial clásico. Con 4 GB de VRAM, 8 es un valor prudente.
    batch_size: int = 0
    diarization: DiarizationConfig = Field(default_factory=DiarizationConfig)


class LlmConfig(BaseModel):
    provider: Literal["gemini"] = "gemini"
    model: str = "gemini-3.5-flash"
    # Modelos de reserva, en orden: si el principal está saturado (503) o no existe (404), se pasa al
    # siguiente y se sigue con él el resto de la ejecución. El modelo usado queda en stories.analysis_model.
    fallback_models: list[str] = Field(default_factory=list)
    overload_retries: int = 2     # 503 seguidos con un modelo antes de pasar al de reserva
    # Profundidad de razonamiento: thinking_level en Gemini 3.x, thinking_budget en Gemini 2.x.
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "high"
    max_workers: int = 4          # piezas de un mismo episodio analizadas en paralelo
    max_concurrency: int = 4      # tope global de peticiones simultáneas a la API
    max_retries: int = 6          # reintentos ante 429 (cuota) y errores 5xx
    timeout_s: int = 300           # sin respuesta en este tiempo = reintento (y, si se repite, reserva)
    # False: solo se analizan en detalle las piezas políticas (las demás quedan segmentadas, sin puntuar).
    analyze_non_political: bool = True
    # Observaciones automáticas por franja (data/notes/<fecha>-mediodia.md / -noche.md) al final de `run`.
    write_notes: bool = True
    prompt_version: str = "v1"


class Config(BaseModel):
    data_dir: Path = Path("data")
    db_path: Path = Path("data/telesesgo.db")
    discover_limit: int = 3
    max_age_days: int = 3
    delete_audio_after_transcribe: bool = False
    sources: list[SourceConfig] = Field(default_factory=list)
    download: DownloadConfig = Field(default_factory=DownloadConfig)
    stt: SttConfig = Field(default_factory=SttConfig)
    llm: LlmConfig = Field(default_factory=LlmConfig)
    political_context: str = ""

    @property
    def audio_dir(self) -> Path:
        return self.data_dir / "audio"

    @property
    def transcript_dir(self) -> Path:
        return self.data_dir / "transcripts"

    def source(self, key: str) -> SourceConfig:
        for s in self.sources:
            if s.key == key:
                return s
        raise KeyError(f"Fuente desconocida: {key}")


def load_config(path: str | Path = "config.yaml") -> Config:
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cfg = Config.model_validate(raw)
    base = path.parent
    if not cfg.data_dir.is_absolute():
        cfg.data_dir = base / cfg.data_dir
    if not cfg.db_path.is_absolute():
        cfg.db_path = base / cfg.db_path
    if cfg.download.cookies_file and not Path(cfg.download.cookies_file).is_absolute():
        cfg.download.cookies_file = str(base / cfg.download.cookies_file)
    for d in (cfg.data_dir, cfg.audio_dir, cfg.transcript_dir):
        d.mkdir(parents=True, exist_ok=True)
    return cfg
