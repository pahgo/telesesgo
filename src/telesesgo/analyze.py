"""Segmentación en piezas, análisis de sesgo por pieza y cruce entre cadenas."""

from __future__ import annotations

import logging
import re
import sqlite3
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed

from rapidfuzz import fuzz

from . import db, prompts, voices
from .config import Config
from .llm import Llm, LlmError
from .schemas import EventMatching, Quote, Segmentation, Story, StoryAnalysis
from .transcribe import Line, load_lines

log = logging.getLogger(__name__)

SKIP_SECTIONS = {"tiempo", "deportes"}
SKIP_FORMATS = {"sumario_titulares", "promocion_autopromo"}
QUOTE_MATCH_THRESHOLD = 88


# --- Segmentación ------------------------------------------------------------

def normalize_stories(stories: list[Story], n_lines: int) -> list[Story]:
    """Ordena, recorta a rango válido y elimina solapes de la segmentación del LLM."""
    out: list[Story] = []
    last_end = -1
    for st in sorted(stories, key=lambda s: s.start_line):
        start = max(st.start_line, last_end + 1, 0)
        end = min(st.end_line, n_lines - 1)
        if end < start:
            continue
        out.append(st.model_copy(update={"start_line": start, "end_line": end}))
        last_end = end
    return out


def segment_episode(cfg: Config, conn: sqlite3.Connection, llm: Llm, ep: sqlite3.Row) -> int:
    lines = load_lines(conn, ep["id"])
    if not lines:
        raise LlmError("Transcripción vacía")
    system = prompts.SEGMENT_SYSTEM.format(political_context=cfg.political_context)
    legend = voices.legend(conn, ep["id"])
    content = [{"type": "text", "text": f"Informativo: {ep['channel']} — {ep['program']} — {ep['air_date']}\n\n"
                                        + prompts.render_transcript(lines, legend)}]
    seg, msg = llm.structured(system=system, content=content, output=Segmentation)
    db.log_usage(conn, ep["id"], "segment", msg.model, msg.usage, msg.stop_reason)
    if legend and seg.speakers:  # quién es cada voz, y aprender las voces con nombre para próximos episodios
        voices.apply_llm_ids(conn, ep["id"], seg.speakers)
        voices.learn(cfg, conn, ep["id"], ep["channel"])

    stories = normalize_stories(seg.stories, len(lines))
    conn.execute("DELETE FROM stories WHERE episode_id = ?", (ep["id"],))
    for ord_, st in enumerate(stories, start=1):
        start_s, end_s = lines[st.start_line].start, lines[st.end_line].end
        conn.execute(
            "INSERT INTO stories (episode_id, ord, start_line, end_line, start_s, end_s, duration_s, headline, summary,"
            " section, story_format, tags, is_political) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (ep["id"], ord_, st.start_line, st.end_line, start_s, end_s, end_s - start_s, st.headline, st.summary,
             st.section, st.story_format, db.dumps(st.tags), int(st.is_political)),
        )
    db.set_status(conn, ep["id"], "segmented")
    log.info("Segmentado %s: %d piezas", ep["id"], len(stories))
    return len(stories)


# --- Verificación de citas y exposición ------------------------------------

def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^\w\s]", " ", text).split())


def verify_quote(quote: str, lines: list[Line], line_idx: int | None, lo: int, hi: int) -> tuple[bool, float]:
    """Comprueba que la cita aparece (con tolerancia) cerca de la línea indicada o dentro de la pieza."""
    q = _norm(quote)
    if not q:
        return False, 0.0
    windows = []
    if line_idx is not None and 0 <= line_idx < len(lines):
        windows.append(lines[max(0, line_idx - 1): line_idx + 2])
    windows.append(lines[lo: hi + 1])
    best = 0.0
    for w in windows:
        best = max(best, fuzz.partial_ratio(q, _norm(" ".join(l.text for l in w))))
        if best >= QUOTE_MATCH_THRESHOLD:
            break
    return best >= QUOTE_MATCH_THRESHOLD, round(best, 1)


def locate_line(quote: str, lines: list[Line], hint: int | None, lo: int, hi: int) -> int | None:
    """Línea de la pieza donde está la cita (la indicada por el modelo si encaja, si no la que mejor encaja)."""
    q = _norm(quote)
    if not q:
        return None
    best, best_idx = 0.0, None
    order = ([hint] if hint is not None and lo <= hint <= hi else []) + list(range(lo, hi + 1))
    for i in order:
        if not 0 <= i < len(lines):
            continue
        score = fuzz.partial_ratio(q, _norm(lines[i].text))
        if score > best + 0.5:
            best, best_idx = score, i
        if i == hint and score >= QUOTE_MATCH_THRESHOLD:
            return i
    return best_idx if best >= 70 else None


def attribute(quote: str, line: int | None, llm_speaker: str, llm_voice: str, lines: list[Line], lo: int, hi: int,
              spk_map: dict, actors) -> tuple[str, str, str]:
    """Quién dice la cita y con qué voz (medio | fuente), y de dónde sale la atribución.

    1. Diarización: el rol del hablante de esa línea (presentador/reportero = medio; fuente = fuente).
    2. Actores: si la línea está entre las voice_lines de un actor que habla con voz propia, es una fuente.
    3. Si no, lo que dijo el modelo.
    """
    idx = locate_line(quote, lines, line, lo, hi)
    if idx is not None and lines[idx].speaker and lines[idx].speaker in spk_map:
        row = spk_map[lines[idx].speaker]
        if row["role"] in voices.MEDIA_ROLES:
            return row["name"] or row["role"], "medio", "diarizacion"
        if row["role"] == "fuente":
            return row["name"] or llm_speaker or "fuente", "fuente", "diarizacion"
    if idx is not None:
        for act in actors:
            if act.speaks_on_air and idx in act.voice_lines:
                return act.name, "fuente", "actores"
    return llm_speaker, llm_voice, "llm"


def line_seconds(lines: list[Line], idxs: list[int], lo: int, hi: int) -> float:
    return round(sum(lines[i].end - lines[i].start for i in set(idxs) if lo <= i <= hi), 2)


# --- Análisis por pieza ------------------------------------------------------

def _analyze_one(cfg: Config, llm: Llm, system: str, transcript_block: dict, ep: sqlite3.Row, st: sqlite3.Row):
    task = prompts.ANALYSIS_TASK.format(
        ord=st["ord"], start_line=st["start_line"], end_line=st["end_line"], headline=st["headline"],
        section=st["section"], story_format=st["story_format"], channel=ep["channel"], program=ep["program"],
        air_date=ep["air_date"],
    )
    return llm.structured(system=system, content=[transcript_block, {"type": "text", "text": task}],
                          output=StoryAnalysis, max_tokens=32000)


def store_analysis(cfg: Config, conn: sqlite3.Connection, story: sqlite3.Row, lines: list[Line],
                   a: StoryAnalysis, model: str) -> None:
    lo, hi = story["start_line"], story["end_line"]
    sid = story["id"]
    conn.execute("DELETE FROM evidence WHERE story_id = ?", (sid,))
    conn.execute("DELETE FROM story_actors WHERE story_id = ?", (sid,))
    spk_map = voices.speaker_map(conn, story["episode_id"])
    media_editorial = 0

    def insert(dimension: str, quote: str, note: str, line, speaker: str, voice: str):
        ok, score = verify_quote(quote, lines, line, lo, hi)
        who, how, src = attribute(quote, line, speaker, voice, lines, lo, hi, spk_map, a.actors)
        conn.execute(
            "INSERT INTO evidence (story_id, dimension, quote, note, line_idx, verified, match_score, speaker, voice, voice_source)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (sid, dimension, quote, note, line, int(ok), score, who, how, src),
        )
        return how

    for dim, quotes in (("lean", a.ideological_lean.evidence), ("tone_government", a.tone_government.evidence),
                        ("tone_opposition", a.tone_opposition.evidence), ("frame", a.frame_evidence),
                        ("editorializing", a.editorializing_evidence)):
        for q in quotes:
            how = insert(dim, q.quote, q.note, q.line, q.speaker, q.voice)
            if dim == "editorializing" and how == "medio":
                media_editorial += 1
    for t in a.loaded_language:
        insert("loaded_language", t.quote, f"{t.connotation} -> {t.target}", t.line, t.speaker, t.voice)

    # Opinión propia solo si alguna prueba es voz del medio: que opine una fuente no es editorializar.
    editorializing = a.editorializing and (media_editorial > 0 or not a.editorializing_evidence)
    if a.editorializing and not editorializing:
        log.info("Pieza %s: la opinión señalada es de fuentes, no del informativo; no cuenta como editorialización", sid)

    seen: set[int] = set()
    for act in a.actors:
        actor_id = db.upsert_actor(conn, act.name.strip(), act.kind, act.affiliation, act.bloc)
        if actor_id in seen:  # el LLM puede repetir un actor; nos quedamos con la primera aparición
            continue
        seen.add(actor_id)
        conn.execute(
            "INSERT INTO story_actors (story_id, actor_id, role, bloc, speaks_on_air, voice_seconds, mention_seconds,"
            " mention_count, tone, tone_rationale) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (sid, actor_id, act.role, act.bloc, int(act.speaks_on_air), line_seconds(lines, act.voice_lines, lo, hi),
             line_seconds(lines, act.mention_lines, lo, hi),
             len({i for i in act.mention_lines if lo <= i <= hi}), act.tone, act.tone_rationale),
        )

    conf = {"baja": 0.33, "media": 0.66, "alta": 1.0}[a.lean_confidence]
    conn.execute(
        "UPDATE stories SET ideological_lean = ?, lean_confidence = ?, tone_government = ?, tone_opposition = ?,"
        " primary_frame = ?, source_balance = ?, editorializing = ?, analysis_json = ?, analysis_model = ?,"
        " prompt_version = ? WHERE id = ?",
        (a.ideological_lean.score, conf, a.tone_government.score, a.tone_opposition.score, a.primary_frame,
         a.source_balance, int(editorializing), a.model_dump_json(), model, cfg.llm.prompt_version, sid),
    )
    conn.commit()


def needs_analysis(cfg: Config, r: sqlite3.Row) -> bool:
    """Piezas que reciben análisis detallado (sumarios, promociones, tiempo y deportes nunca)."""
    if r["section"] in SKIP_SECTIONS or r["story_format"] in SKIP_FORMATS:
        return False
    return cfg.llm.analyze_non_political or bool(r["is_political"])


def analyze_episode(cfg: Config, conn: sqlite3.Connection, llm: Llm, ep: sqlite3.Row, force: bool = False,
                    on_progress=None) -> int:
    """Analiza cada pieza pendiente. `on_progress(hechas, total)` se llama tras cada pieza."""
    lines = load_lines(conn, ep["id"])
    rows = conn.execute("SELECT * FROM stories WHERE episode_id = ? ORDER BY ord", (ep["id"],)).fetchall()
    todo = [
        r for r in rows
        if needs_analysis(cfg, r)
        and (force or r["analysis_json"] is None or r["prompt_version"] != cfg.llm.prompt_version)
    ]
    if not todo:
        db.set_status(conn, ep["id"], "analyzed")
        return 0

    system = prompts.ANALYSIS_SYSTEM.format(political_context=cfg.political_context)
    # Todas las piezas del episodio comparten este prefijo (transcripción completa): Gemini lo reutiliza
    # con su caché implícita, que abarata y acelera las llamadas siguientes.
    transcript_block = {"type": "text", "text": prompts.render_transcript(lines, voices.legend(conn, ep["id"]))}

    done, failures = 0, 0

    def handle(st, result):
        nonlocal done
        analysis, msg = result
        db.log_usage(conn, ep["id"], "analyze", msg.model, msg.usage, msg.stop_reason)
        store_analysis(cfg, conn, st, lines, analysis, msg.model)
        done += 1
        if on_progress:
            on_progress(done + failures, len(todo))

    def failed(st, exc):
        nonlocal failures
        failures += 1
        log.error("Pieza %s: %s", st["id"], exc)
        if on_progress:
            on_progress(done + failures, len(todo))

    # La primera llamada va sola para que la caché del prefijo esté lista; el resto la reutiliza en paralelo.
    first, rest = todo[0], todo[1:]
    try:
        handle(first, _analyze_one(cfg, llm, system, transcript_block, ep, first))
    except LlmError as exc:
        failed(first, exc)

    with ThreadPoolExecutor(max_workers=cfg.llm.max_workers) as pool:
        futures = {pool.submit(_analyze_one, cfg, llm, system, transcript_block, ep, st): st for st in rest}
        for fut in as_completed(futures):
            st = futures[fut]
            try:
                handle(st, fut.result())
            except LlmError as exc:
                failed(st, exc)

    if failures:
        db.set_status(conn, ep["id"], "segmented", error=f"{failures} piezas sin analizar")
    else:
        db.set_status(conn, ep["id"], "analyzed")
    log.info("Analizadas %d/%d piezas del episodio %s", done, len(todo), ep["id"])
    return done


# --- Cruce entre cadenas -----------------------------------------------------

def match_events(cfg: Config, conn: sqlite3.Connection, llm: Llm, air_date: str) -> int:
    rows = conn.execute(
        "SELECT s.id, e.channel, e.program, s.section, s.headline, s.summary FROM stories s"
        " JOIN episodes e ON e.id = s.episode_id WHERE e.air_date = ? AND s.story_format NOT IN ('sumario_titulares', 'promocion_autopromo')"
        " AND s.section != 'tiempo' ORDER BY e.channel, s.ord",
        (air_date,),
    ).fetchall()
    if len({r["program"] for r in rows}) < 2:
        log.info("%s: menos de dos programas con piezas; no hay nada que cruzar", air_date)
        return 0
    listing = "\n".join(f"{r['id']} | {r['channel']} | {r['program']} | {r['section']} | {r['headline']} | {r['summary']}" for r in rows)
    result, msg = llm.structured(system=prompts.MATCH_SYSTEM, content=[{"type": "text", "text": f"Fecha: {air_date}\n\n{listing}"}],
                                 output=EventMatching)
    db.log_usage(conn, None, "match", msg.model, msg.usage, msg.stop_reason)

    valid = {r["id"] for r in rows}
    old = [r["id"] for r in conn.execute("SELECT id FROM events WHERE air_date = ?", (air_date,))]
    if old:
        marks = ",".join("?" * len(old))
        conn.execute(f"UPDATE stories SET event_id = NULL WHERE event_id IN ({marks})", old)
        conn.execute(f"DELETE FROM events WHERE id IN ({marks})", old)
    assigned: set[int] = set()
    n = 0
    for ev in result.events:
        ids = [i for i in dict.fromkeys(ev.story_ids) if i in valid and i not in assigned]
        if len(ids) < 2:
            continue
        cur = conn.execute("INSERT INTO events (air_date, label, section) VALUES (?, ?, ?)", (air_date, ev.label, ev.section))
        conn.executemany("UPDATE stories SET event_id = ? WHERE id = ?", [(cur.lastrowid, i) for i in ids])
        assigned.update(ids)
        n += 1
    conn.commit()
    log.info("%s: %d acontecimientos comunes", air_date, n)
    return n
