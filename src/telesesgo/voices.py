"""Hablantes de cada episodio y biblioteca de voces conocidas.

Flujo por episodio (con diarización activada):
  1. La diarización da etiquetas SPEAKER_nn por línea y una huella de voz (embedding) por hablante.
  2. `match_library`: cada hablante con bastante voz se compara (coseno) con la biblioteca; si se parece lo
     suficiente a una voz conocida, se le asigna su nombre y rol (id_source = 'voz').
  3. La segmentación de Gemini identifica por el contenido a los que faltan: presentador por la entradilla,
     reporteros por su despedida, políticos por cómo se presentan sus totales (id_source = 'llm').
  4. `learn`: las voces con nombre y bastante tiempo se añaden a la biblioteca o refinan su centroide,
     salvo que la huella nueva no se parezca a la guardada (evita contaminar una voz con otra).
"""

from __future__ import annotations

import logging
import sqlite3

import numpy as np

from .config import Config

log = logging.getLogger(__name__)

MEDIA_ROLES = {"presentador", "reportero"}


def to_blob(vec) -> bytes:
    v = np.asarray(vec, dtype=np.float32).ravel()
    n = float(np.linalg.norm(v))
    return (v / n if n else v).astype(np.float32).tobytes()


def from_blob(blob: bytes | None) -> np.ndarray | None:
    return None if blob is None else np.frombuffer(blob, dtype=np.float32)


def short_label(label: str | None) -> str | None:
    """SPEAKER_03 -> H03 (más corto en el texto que se envía al modelo)."""
    if not label:
        return None
    digits = "".join(c for c in label if c.isdigit())
    return f"H{int(digits):02d}" if digits else label


def store_episode_speakers(conn: sqlite3.Connection, episode_id: int, speakers: dict[str, tuple[float, object]]) -> None:
    """speakers: etiqueta -> (segundos de voz, embedding o None). El que más habla queda como presentador probable."""
    conn.execute("DELETE FROM episode_speakers WHERE episode_id = ?", (episode_id,))
    anchor = max(speakers, key=lambda k: speakers[k][0]) if speakers else None
    for label, (secs, emb) in speakers.items():
        is_anchor = label == anchor
        conn.execute(
            "INSERT INTO episode_speakers (episode_id, label, seconds, embedding, role, id_source) VALUES (?, ?, ?, ?, ?, ?)",
            (episode_id, label, round(secs, 1), None if emb is None else to_blob(emb),
             "presentador" if is_anchor else "desconocido", "anchor" if is_anchor else None))
    conn.commit()


def speaker_map(conn: sqlite3.Connection, episode_id: int) -> dict[str, sqlite3.Row]:
    return {r["label"]: r for r in conn.execute("SELECT * FROM episode_speakers WHERE episode_id = ?", (episode_id,))}


def match_library(cfg: Config, conn: sqlite3.Connection, episode_id: int) -> int:
    """Asigna nombre a los hablantes que se parecen a una voz conocida. Devuelve cuántos se reconocieron."""
    lib = [(r["id"], r["name"], r["role"], from_blob(r["embedding"])) for r in conn.execute("SELECT * FROM voices")]
    if not lib:
        return 0
    mat = np.stack([v for *_, v in lib])
    taken: set[int] = set()
    found = 0
    rows = sorted(speaker_map(conn, episode_id).values(), key=lambda r: -r["seconds"])
    for r in rows:
        emb = from_blob(r["embedding"])
        if emb is None or r["seconds"] < cfg.stt.diarization.match_min_seconds:
            continue
        sims = mat @ emb
        best = int(np.argmax(sims))
        sim = float(sims[best])
        if sim < cfg.stt.diarization.match_threshold or best in taken:
            continue
        taken.add(best)  # una voz de la biblioteca solo puede ser un hablante del episodio
        vid, name, role, _ = lib[best]
        conn.execute("UPDATE episode_speakers SET name = ?, role = ?, id_source = 'voz', similarity = ?, voice_id = ?"
                     " WHERE episode_id = ? AND label = ?", (name, role, round(sim, 3), vid, episode_id, r["label"]))
        found += 1
        log.info("Episodio %s: %s reconocido como %s (similitud %.2f)", episode_id, r["label"], name, sim)
    conn.commit()
    return found


def apply_llm_ids(conn: sqlite3.Connection, episode_id: int, ids) -> None:
    """Aplica la identificación por contexto de Gemini (SpeakerId con etiquetas Hnn).
    Si contradice una identificación por voz, se mantiene la de la voz y se registra el conflicto."""
    by_short = {short_label(lbl): row for lbl, row in speaker_map(conn, episode_id).items()}
    for sid in ids:
        row = by_short.get(sid.label) or by_short.get(short_label(sid.label))
        if row is None:
            continue
        name = (sid.name or "").strip() or None
        if row["id_source"] == "voz":
            if name and name != row["name"]:
                log.warning("Episodio %s: %s reconocido por voz como %s, pero Gemini dice %s; se mantiene la voz",
                            episode_id, row["label"], row["name"], name)
            continue
        if sid.role == "desconocido" and not name:
            continue
        conn.execute("UPDATE episode_speakers SET role = ?, name = ?, id_source = 'llm' WHERE episode_id = ? AND label = ?",
                     (sid.role, name, episode_id, row["label"]))
    conn.commit()


def learn(cfg: Config, conn: sqlite3.Connection, episode_id: int, channel: str | None = None) -> int:
    """Añade a la biblioteca (o refina) las voces con nombre identificadas en el episodio."""
    dc = cfg.stt.diarization
    learned = 0
    for r in speaker_map(conn, episode_id).values():
        emb = from_blob(r["embedding"])
        if emb is None or not r["name"] or r["seconds"] < dc.learn_min_seconds or r["id_source"] not in ("llm", "voz"):
            continue
        if len(r["name"].split()) < 2:  # «Vicente» a secas: demasiado ambiguo para la biblioteca
            log.info("Episodio %s: %s sin apellido; no se añade a la biblioteca", episode_id, r["name"])
            continue
        cur = conn.execute("SELECT * FROM voices WHERE name = ?", (r["name"],)).fetchone()
        if cur is None:
            vid = conn.execute("INSERT INTO voices (name, role, channel, embedding, samples, seconds) VALUES (?, ?, ?, ?, 1, ?)",
                               (r["name"], r["role"], channel if r["role"] in MEDIA_ROLES else None, to_blob(emb),
                                r["seconds"])).lastrowid
            log.info("Nueva voz en la biblioteca: %s (%s)", r["name"], r["role"])
        else:
            old = from_blob(cur["embedding"])
            sim = float(old @ emb)
            if sim < dc.learn_min_similarity:
                log.warning("Episodio %s: la voz de %s no se parece a la guardada (%.2f); no se actualiza la biblioteca",
                            episode_id, r["name"], sim)
                continue
            n = cur["samples"]
            vid = cur["id"]
            conn.execute("UPDATE voices SET embedding = ?, samples = ?, seconds = seconds + ?, updated_at = datetime('now')"
                         " WHERE id = ?", (to_blob(old * n + emb), n + 1, r["seconds"], vid))
        conn.execute("UPDATE episode_speakers SET voice_id = ? WHERE episode_id = ? AND label = ?", (vid, episode_id, r["label"]))
        learned += 1
    conn.commit()
    return learned


def legend(conn: sqlite3.Connection, episode_id: int) -> str:
    """Cabecera para el modelo: quién es cada etiqueta Hnn, según lo que ya se sabe."""
    rows = sorted(speaker_map(conn, episode_id).values(), key=lambda r: -r["seconds"])
    if not rows:
        return ""
    out = ["Hablantes (diarización automática; la etiqueta Hnn de cada línea indica la voz):"]
    for r in rows:
        who = r["name"] or "sin identificar"
        how = {"voz": f"reconocido por la voz, similitud {r['similarity'] or 0:.2f}",
               "llm": "identificado por el contexto", "anchor": "presentador probable: es quien más habla"}.get(r["id_source"] or "", "")
        out.append(f"- {short_label(r['label'])}: {who} · {r['role'] or 'desconocido'} · {round(r['seconds'])} s"
                   + (f" ({how})" if how else ""))
    return "\n".join(out)
