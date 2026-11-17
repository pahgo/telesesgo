"""Esquema y acceso a SQLite."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS episodes (
    id            INTEGER PRIMARY KEY,
    source_key    TEXT NOT NULL,
    channel       TEXT NOT NULL,
    program       TEXT NOT NULL,
    external_id   TEXT NOT NULL,
    title         TEXT,
    air_date      TEXT NOT NULL,          -- YYYY-MM-DD
    aired_at      TEXT,                   -- ISO 8601
    url           TEXT NOT NULL,
    duration_s    REAL,
    audio_path    TEXT,
    status        TEXT NOT NULL DEFAULT 'discovered',  -- discovered|downloaded|transcribed|segmented|analyzed|failed
    error         TEXT,
    created_at    TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at    TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (source_key, external_id)
);

CREATE TABLE IF NOT EXISTS transcripts (
    episode_id    INTEGER PRIMARY KEY REFERENCES episodes(id) ON DELETE CASCADE,
    model         TEXT NOT NULL,
    language      TEXT,
    diarized      INTEGER NOT NULL DEFAULT 0,
    json_path     TEXT NOT NULL,
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS transcript_lines (
    episode_id    INTEGER NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
    idx           INTEGER NOT NULL,
    start_s       REAL NOT NULL,
    end_s         REAL NOT NULL,
    speaker       TEXT,
    text          TEXT NOT NULL,
    PRIMARY KEY (episode_id, idx)
);

CREATE TABLE IF NOT EXISTS stories (
    id               INTEGER PRIMARY KEY,
    episode_id       INTEGER NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
    ord              INTEGER NOT NULL,
    start_line       INTEGER NOT NULL,
    end_line         INTEGER NOT NULL,
    start_s          REAL NOT NULL,
    end_s            REAL NOT NULL,
    duration_s       REAL NOT NULL,
    headline         TEXT NOT NULL,
    summary          TEXT,
    section          TEXT,
    story_format     TEXT,
    tags             TEXT,                -- JSON list
    is_political     INTEGER NOT NULL DEFAULT 0,
    -- resultado del análisis (NULL hasta analizar)
    ideological_lean REAL,                -- -2 izquierda .. +2 derecha
    lean_confidence  REAL,                -- 0..1
    tone_government  REAL,                -- -2 .. +2
    tone_opposition  REAL,                -- -2 .. +2
    primary_frame    TEXT,
    source_balance   TEXT,
    editorializing   INTEGER,
    analysis_json    TEXT,
    analysis_model   TEXT,
    prompt_version   TEXT,
    event_id         INTEGER REFERENCES events(id) ON DELETE SET NULL,
    UNIQUE (episode_id, ord)
);

CREATE TABLE IF NOT EXISTS actors (
    id            INTEGER PRIMARY KEY,
    name          TEXT NOT NULL UNIQUE,   -- nombre canónico
    kind          TEXT,
    affiliation   TEXT,
    bloc          TEXT
);

CREATE TABLE IF NOT EXISTS story_actors (
    story_id        INTEGER NOT NULL REFERENCES stories(id) ON DELETE CASCADE,
    actor_id        INTEGER NOT NULL REFERENCES actors(id),
    role            TEXT,
    bloc            TEXT,                 -- bloque en el momento de la emisión
    speaks_on_air   INTEGER NOT NULL DEFAULT 0,
    voice_seconds   REAL NOT NULL DEFAULT 0,
    mention_seconds REAL NOT NULL DEFAULT 0,
    mention_count   INTEGER NOT NULL DEFAULT 0,
    tone            REAL,
    tone_rationale  TEXT,
    PRIMARY KEY (story_id, actor_id)
);

CREATE TABLE IF NOT EXISTS evidence (
    id            INTEGER PRIMARY KEY,
    story_id      INTEGER NOT NULL REFERENCES stories(id) ON DELETE CASCADE,
    dimension     TEXT NOT NULL,          -- lean|tone_government|tone_opposition|frame|loaded_language|editorializing|actor_tone
    quote         TEXT NOT NULL,
    note          TEXT,
    line_idx      INTEGER,
    verified      INTEGER NOT NULL,       -- 1 si la cita aparece literalmente (fuzzy) en la transcripción
    match_score   REAL,
    speaker       TEXT,                   -- quién pronuncia la frase (nombre o rol)
    voice         TEXT,                   -- medio (presentador/redacción) | fuente (total, entrevistado...)
    voice_source  TEXT                    -- diarizacion | actores | llm: de dónde sale la atribución
);

-- Hablantes de cada episodio según la diarización, con su huella de voz y su identificación.
CREATE TABLE IF NOT EXISTS episode_speakers (
    episode_id    INTEGER NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
    label         TEXT NOT NULL,          -- SPEAKER_03
    seconds       REAL NOT NULL,
    embedding     BLOB,                   -- float32 normalizado
    role          TEXT,                   -- presentador | reportero | fuente | desconocido
    name          TEXT,
    id_source     TEXT,                   -- voz (biblioteca) | llm (contexto) | anchor (más tiempo de voz)
    similarity    REAL,                   -- coseno con la voz de la biblioteca, si id_source = voz
    voice_id      INTEGER REFERENCES voices(id) ON DELETE SET NULL,
    PRIMARY KEY (episode_id, label)
);

-- Biblioteca de voces conocidas: centroide de las huellas confirmadas en episodios anteriores.
CREATE TABLE IF NOT EXISTS voices (
    id            INTEGER PRIMARY KEY,
    name          TEXT NOT NULL UNIQUE,
    role          TEXT,                   -- presentador | reportero | fuente
    channel       TEXT,
    embedding     BLOB NOT NULL,
    samples       INTEGER NOT NULL DEFAULT 1,
    seconds       REAL NOT NULL DEFAULT 0,
    updated_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS events (
    id            INTEGER PRIMARY KEY,
    air_date      TEXT NOT NULL,
    label         TEXT NOT NULL,
    section       TEXT,
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Duración de cada paso del pipeline (download, transcribe, segment, analyze, match, html...).
CREATE TABLE IF NOT EXISTS step_timings (
    id          INTEGER PRIMARY KEY,
    episode_id  INTEGER,
    air_date    TEXT,
    step        TEXT NOT NULL,
    seconds     REAL NOT NULL,
    ok          INTEGER NOT NULL,
    detail      TEXT,
    started_at  TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS llm_usage (
    id                 INTEGER PRIMARY KEY,
    episode_id         INTEGER,
    stage              TEXT NOT NULL,
    model              TEXT NOT NULL,
    input_tokens       INTEGER,
    output_tokens      INTEGER,
    cache_read_tokens  INTEGER,
    cache_write_tokens INTEGER,
    stop_reason        TEXT,
    created_at         TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS ix_episodes_status ON episodes(status);
CREATE INDEX IF NOT EXISTS ix_episodes_date ON episodes(air_date);
CREATE INDEX IF NOT EXISTS ix_stories_event ON stories(event_id);

-- Vistas de consulta ------------------------------------------------------

CREATE VIEW IF NOT EXISTS v_stories AS
SELECT e.air_date, e.channel, e.program, s.ord, s.headline, s.section,
       round(s.duration_s) AS duration_s, s.is_political,
       s.ideological_lean, s.lean_confidence, s.tone_government, s.tone_opposition,
       s.primary_frame, s.source_balance, s.editorializing, s.event_id, s.id AS story_id
FROM stories s JOIN episodes e ON e.id = s.episode_id;

-- Tiempo de palabra (voz propia) y de mención por actor y cadena.
CREATE VIEW IF NOT EXISTS v_actor_exposure AS
SELECT e.air_date, e.channel, a.name AS actor, sa.bloc,
       round(sum(sa.voice_seconds)) AS voice_s,
       round(sum(sa.mention_seconds)) AS mention_s,
       sum(sa.mention_count) AS mentions,
       round(avg(sa.tone), 2) AS avg_tone,
       count(DISTINCT s.id) AS stories
FROM story_actors sa
JOIN stories s ON s.id = sa.story_id
JOIN episodes e ON e.id = s.episode_id
JOIN actors a ON a.id = sa.actor_id
GROUP BY e.air_date, e.channel, a.name, sa.bloc;

-- Tiempo de palabra agregado por bloque político.
CREATE VIEW IF NOT EXISTS v_bloc_exposure AS
SELECT air_date, channel, bloc, sum(voice_s) AS voice_s, sum(mention_s) AS mention_s
FROM v_actor_exposure GROUP BY air_date, channel, bloc;

-- Misma noticia en varias cadenas.
CREATE VIEW IF NOT EXISTS v_event_comparison AS
SELECT ev.air_date, ev.id AS event_id, ev.label, e.channel, e.program,
       s.headline, s.ord AS position, round(s.duration_s) AS duration_s,
       s.ideological_lean, s.tone_government, s.tone_opposition,
       s.primary_frame, s.source_balance
FROM events ev
JOIN stories s ON s.event_id = ev.id
JOIN episodes e ON e.id = s.episode_id;
"""


MIGRATIONS = [
    ("evidence", "speaker", "TEXT"),
    ("evidence", "voice", "TEXT"),
    ("evidence", "voice_source", "TEXT"),
]


def connect(path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.executescript(SCHEMA)
    for table, col, kind in MIGRATIONS:  # columnas añadidas después de crear la BD
        cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
        if col not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {kind}")
    conn.commit()
    return conn


def log_timing(conn: sqlite3.Connection, step: str, seconds: float, ok: bool, *, episode_id: int | None = None,
               air_date: str | None = None, started_at: str, detail: str | None = None) -> None:
    conn.execute(
        "INSERT INTO step_timings (episode_id, air_date, step, seconds, ok, detail, started_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (episode_id, air_date, step, round(seconds, 2), int(ok), detail, started_at),
    )
    conn.commit()


def set_status(conn: sqlite3.Connection, episode_id: int, status: str, error: str | None = None, **fields) -> None:
    cols = {"status": status, "error": error, **fields}
    assignments = ", ".join(f"{k} = ?" for k in cols) + ", updated_at = datetime('now')"
    conn.execute(f"UPDATE episodes SET {assignments} WHERE id = ?", (*cols.values(), episode_id))
    conn.commit()


def upsert_actor(conn: sqlite3.Connection, name: str, kind: str | None, affiliation: str | None, bloc: str | None) -> int:
    row = conn.execute("SELECT id FROM actors WHERE name = ?", (name,)).fetchone()
    if row:
        conn.execute(
            "UPDATE actors SET kind = coalesce(?, kind), affiliation = coalesce(?, affiliation), bloc = coalesce(?, bloc) WHERE id = ?",
            (kind, affiliation, bloc, row["id"]),
        )
        return row["id"]
    cur = conn.execute(
        "INSERT INTO actors (name, kind, affiliation, bloc) VALUES (?, ?, ?, ?)", (name, kind, affiliation, bloc)
    )
    return cur.lastrowid


def log_usage(conn: sqlite3.Connection, episode_id: int | None, stage: str, model: str, usage, stop_reason: str | None) -> None:
    conn.execute(
        "INSERT INTO llm_usage (episode_id, stage, model, input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, stop_reason)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            episode_id,
            stage,
            model,
            getattr(usage, "input_tokens", None),
            getattr(usage, "output_tokens", None),
            getattr(usage, "cache_read_input_tokens", None),
            getattr(usage, "cache_creation_input_tokens", None),
            stop_reason,
        ),
    )
    conn.commit()


def dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)
