"""Observaciones automáticas por franja (mediodía / noche) a partir de los datos ya medidos.

Gemini recibe un dossier compacto (métricas por programa, noticias comunes y piezas destacadas con citas
verificadas) y devuelve observaciones estructuradas. Las citas «…» del texto se comprueban contra las citas
verificadas y la transcripción; las que no aparecen se marcan como no verificadas.

Se guardan en data/notes/<fecha>-mediodia.md y <fecha>-noche.md. Las notas escritas a mano en
data/notes/<fecha>.md no se tocan.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from pathlib import Path

from rapidfuzz import fuzz

from . import db, prompts
from .analyze import QUOTE_MATCH_THRESHOLD, _norm
from .config import Config
from .htmlreport import BALANCE_LABEL, FRAME_LABEL, _franja, _program_metrics
from .schemas import EditionNotes

log = logging.getLogger(__name__)

FRANJAS = {"mediodia": "Mediodía", "noche": "Noche"}
SLUG = {v: k for k, v in FRANJAS.items()}
MAX_HIGHLIGHTS = 10
MAX_QUOTES = 2


def notes_file(cfg: Config, date: str, franja: str) -> Path:
    return cfg.data_dir / "notes" / f"{date}-{SLUG[franja]}.md"


def _episodes(conn, date: str, franja: str) -> list[sqlite3.Row]:
    rows = conn.execute("SELECT * FROM episodes WHERE air_date = ? AND status = 'analyzed' ORDER BY aired_at", (date,)).fetchall()
    return [r for r in rows if _franja(r["aired_at"]) == franja]


def _fmt(v, digits=2):
    return "–" if v is None else f"{v:+.{digits}f}" if isinstance(v, float) else str(v)


def _quotes(conn, story_id: int) -> list[tuple[str, str]]:
    """(cita, quién): primero las del propio informativo."""
    rows = conn.execute(
        "SELECT quote, speaker, voice FROM evidence WHERE story_id = ? AND verified = 1 ORDER BY voice = 'fuente', CASE dimension"
        " WHEN 'editorializing' THEN 0 WHEN 'loaded_language' THEN 1 WHEN 'tone_government' THEN 2"
        " WHEN 'tone_opposition' THEN 3 ELSE 4 END LIMIT ?", (story_id, MAX_QUOTES)).fetchall()
    out = []
    for r in rows:
        who = (f"fuente: {r['speaker'] or 'sin identificar'}" if r["voice"] == "fuente"
               else f"informativo: {r['speaker'] or 'presentador/redacción'}")
        out.append((r["quote"], who))
    return out


def _selection_notes(row) -> str:
    try:
        return (json.loads(row["analysis_json"] or "{}").get("selection_notes") or "")[:300]
    except json.JSONDecodeError:
        return ""


def build_dossier(cfg: Config, conn: sqlite3.Connection, date: str, franja: str) -> tuple[str, list[str]]:
    """Devuelve (dossier en texto, citas permitidas)."""
    eps = _episodes(conn, date, franja)
    allowed: list[str] = []
    out = [f"FECHA: {date}  ·  FRANJA: {franja}", "", "== MÉTRICAS POR PROGRAMA =="]
    for ep in eps:
        m = _program_metrics(conn, ep)
        voice = ", ".join(f"{k} {round(v or 0)} s" for k, v in m["voice"].items())
        out.append(
            f"- {ep['channel']} — {ep['program']}: {round((m['dur'] or 0) / 60)} min, {m['n']} piezas "
            f"({m['analysed']} puntuadas). Orientación media {_fmt(m['lean'])} (−izq/+der), tono Gobierno "
            f"{_fmt(m['gob'])}, tono oposición {_fmt(m['opo'])}, piezas con opinión propia {m['editorial']}. "
            f"Apertura: «{m['opening']}». Tiempo con voz propia: {voice}.")

    other = "Noche" if franja == "Mediodía" else "Mediodía"
    other_eps = _episodes(conn, date, other)
    if other_eps:
        out += ["", f"== OTRA FRANJA DEL MISMO DÍA ({other}), solo para comparar =="]
        for ep in other_eps:
            m = _program_metrics(conn, ep)
            out.append(f"- {ep['program']}: tono Gobierno {_fmt(m['gob'])}, tono oposición {_fmt(m['opo'])}, "
                       f"opinión propia {m['editorial']}, apertura «{m['opening']}».")

    ids = [e["id"] for e in eps]
    if not ids:
        return "\n".join(out), allowed
    marks = ",".join("?" * len(ids))
    label = {e["id"]: e["program"] for e in eps}

    out += ["", "== NOTICIAS COMUNES ENTRE PROGRAMAS DE ESTA FRANJA =="]
    events = conn.execute(
        f"SELECT ev.id, ev.label FROM events ev JOIN stories s ON s.event_id = ev.id WHERE s.episode_id IN ({marks})"
        " GROUP BY ev.id HAVING count(DISTINCT s.episode_id) >= 2 ORDER BY ev.id", ids).fetchall()
    for ev in events:
        out.append(f"* {ev['label']}")
        for s in conn.execute(f"SELECT * FROM stories WHERE event_id = ? AND episode_id IN ({marks}) ORDER BY episode_id",
                              [ev["id"], *ids]):
            qs = _quotes(conn, s["id"])
            allowed += [q for q, _ in qs]
            out.append(
                f"    - {label[s['episode_id']]}: posición {s['ord']}, {round(s['duration_s'] or 0)} s, tono Gob {_fmt(s['tone_government'], 0)},"
                f" tono Opo {_fmt(s['tone_opposition'], 0)}, orientación {_fmt(s['ideological_lean'], 0)},"
                f" encuadre {FRAME_LABEL.get(s['primary_frame'], s['primary_frame'] or '–')},"
                f" fuentes {BALANCE_LABEL.get(s['source_balance'], s['source_balance'] or '–')}."
                + (f" Nota: {_selection_notes(s)}" if _selection_notes(s) else "")
                + "".join(f" Cita ({who}): «{q}»" for q, who in qs))
    if not events:
        out.append("(sin noticias comunes)")

    out += ["", "== PIEZAS DESTACADAS (opinión propia o tono marcado) =="]
    for ep in eps:
        rows = conn.execute(
            "SELECT * FROM stories WHERE episode_id = ? AND analysis_json IS NOT NULL AND (editorializing = 1"
            " OR abs(coalesce(tone_government, 0)) >= 1 OR abs(coalesce(tone_opposition, 0)) >= 1"
            " OR abs(coalesce(ideological_lean, 0)) >= 1) ORDER BY editorializing DESC,"
            " abs(coalesce(tone_government, 0)) + abs(coalesce(tone_opposition, 0)) DESC LIMIT ?",
            (ep["id"], MAX_HIGHLIGHTS)).fetchall()
        out.append(f"* {ep['program']}")
        for s in rows:
            qs = _quotes(conn, s["id"])
            allowed += [q for q, _ in qs]
            out.append(
                f"    - [{s['ord']}] {s['headline']} ({round(s['duration_s'] or 0)} s): tono Gob {_fmt(s['tone_government'], 0)},"
                f" tono Opo {_fmt(s['tone_opposition'], 0)}, orientación {_fmt(s['ideological_lean'], 0)},"
                f" opinión propia {'sí' if s['editorializing'] else 'no'}."
                + (f" Nota: {_selection_notes(s)}" if _selection_notes(s) else "")
                + "".join(f" Cita ({who}): «{q}»" for q, who in qs))
        if not rows:
            out.append("    (ninguna)")
    return "\n".join(out), allowed


def _verify_quotes(text: str, allowed: list[str], transcripts: list[str]) -> tuple[str, int]:
    """Marca las citas «…» que no aparecen en las citas verificadas ni en la transcripción."""
    pool = [_norm(q) for q in allowed] + transcripts
    bad = 0

    def check(m):
        nonlocal bad
        q = _norm(m.group(1))
        if len(q) < 12 or any(fuzz.partial_ratio(q, p) >= QUOTE_MATCH_THRESHOLD for p in pool if p):
            return m.group(0)
        bad += 1
        return m.group(0) + " *(cita no verificada)*"

    return re.sub(r"«([^»]{3,400})»", check, text), bad


def render(notes: EditionNotes, franja: str, model: str) -> str:
    lines = [f"## {franja}: {notes.title.strip()}", ""]
    lines += [f"- **{o.lead.strip().rstrip('.')}.** {o.text.strip()}" for o in notes.findings]
    if notes.same_story:
        lines += ["", "### Una misma noticia, distintos relatos", ""]
        lines += [f"- **{o.lead.strip().rstrip('.')}.** {o.text.strip()}" for o in notes.same_story]
    if notes.caveats:
        lines += ["", "### Cautelas", ""] + [f"- {c.strip()}" for c in notes.caveats]
    lines += ["", f"*Observaciones generadas automáticamente por {model} a partir de las métricas y citas verificadas del día.*"]
    return "\n".join(lines) + "\n"


def write_edition_notes(cfg: Config, conn: sqlite3.Connection, llm, date: str, franja: str) -> Path | None:
    """Genera las observaciones de una franja. None si no hay ningún programa analizado en ella."""
    eps = _episodes(conn, date, franja)
    if not eps:
        return None
    dossier, allowed = build_dossier(cfg, conn, date, franja)
    system = prompts.NOTES_SYSTEM.format(political_context=cfg.political_context)
    notes, msg = llm.structured(system=system, content=[{"type": "text", "text": dossier}], output=EditionNotes,
                                max_tokens=16000)
    db.log_usage(conn, None, "notes", msg.model, msg.usage, msg.stop_reason)

    transcripts = [_norm(" ".join(r[0] for r in conn.execute(
        "SELECT text FROM transcript_lines WHERE episode_id = ? ORDER BY idx", (e["id"],)))) for e in eps]
    text, bad = _verify_quotes(render(notes, franja, msg.model), allowed, transcripts)
    if bad:
        log.warning("%s %s: %d citas de las observaciones no se encuentran en la transcripción", date, franja, bad)
    path = notes_file(cfg, date, franja)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
