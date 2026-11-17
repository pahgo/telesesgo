"""Informe HTML por día: comparativa por franja, misma noticia en cada cadena, hallazgos y notas.

Se genera a partir de la base de datos en `<data_dir>/reports/<fecha>.html`, más un `index.html`.
Las observaciones escritas a mano se leen de `<data_dir>/notes/<fecha>.md` (Markdown mínimo).
"""

from __future__ import annotations

import html
import json
import logging
import re
import sqlite3
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from .config import Config

log = logging.getLogger(__name__)

# Orden fijo de cadenas en las columnas; las desconocidas van al final por orden alfabético.
CHANNEL_ORDER = ["La 1 (RTVE)", "La 1", "Antena 3", "laSexta", "Telecinco", "Cuatro"]
VOICE_GROUPS = [
    ("Gobierno y su partido", ("gobierno", "partido_gobierno")),
    ("Socios", ("socio_gobierno",)),
    ("Oposición", ("oposicion",)),
    ("Sindicatos", ("sindicato",)),
    ("Patronal y empresas", ("patronal_empresa",)),
]
# Prioridad para elegir la cita que ilustra cada pieza en la comparativa.
QUOTE_PRIORITY = ["editorializing", "loaded_language", "tone_government", "tone_opposition", "lean", "frame"]
FRAME_LABEL = {
    "conflicto": "conflicto", "interes_humano": "interés humano", "consecuencias_economicas": "consecuencias económicas",
    "moralidad": "moralidad", "atribucion_responsabilidad": "atribución de responsabilidad",
    "estrategia_juego_politico": "estrategia / juego político", "seguridad_orden_publico": "seguridad y orden público",
    "gestion_eficacia": "gestión y eficacia", "informativo_neutro": "informativo neutro",
}
BALANCE_LABEL = {"no_aplica": "no aplica", "unilateral": "unilateral", "desequilibrado": "desequilibrado", "equilibrado": "equilibrado"}
SECTION_LABEL = {
    "politica_nacional": "política nacional", "politica_autonomica": "política autonómica",
    "tribunales_corrupcion": "tribunales y corrupción", "economia_trabajo": "economía y trabajo",
    "sucesos_seguridad": "sucesos y seguridad", "medio_ambiente_clima": "medio ambiente y clima",
    "inmigracion": "inmigración", "ciencia_tecnologia": "ciencia y tecnología", "cultura_espectaculos": "cultura y espectáculos",
    "educacion": "educación",
}
DIMENSION_LABEL = {
    "editorializing": "opinión propia", "loaded_language": "lenguaje cargado", "tone_government": "tono hacia el Gobierno",
    "tone_opposition": "tono hacia la oposición", "lean": "orientación", "frame": "encuadre",
}


def reports_dir(cfg: Config) -> Path:
    return cfg.data_dir / "reports"


def notes_path(cfg: Config, date: str) -> Path:
    """Notas escritas a mano para todo el día."""
    return cfg.data_dir / "notes" / f"{date}.md"


def franja_notes_path(cfg: Config, date: str, franja: str) -> Path:
    """Observaciones automáticas de una franja (ver notes.py)."""
    slug = {"Mediodía": "mediodia", "Noche": "noche"}.get(franja, "sin-hora")
    return cfg.data_dir / "notes" / f"{date}-{slug}.md"


# --- Utilidades de formato ---------------------------------------------------

def _e(text) -> str:
    return html.escape("" if text is None else str(text))


def _channel_key(channel: str) -> tuple[int, str]:
    for i, c in enumerate(CHANNEL_ORDER):
        if channel.startswith(c):
            return i, channel
    return len(CHANNEL_ORDER), channel


def _franja(aired_at: str | None) -> str:
    if not aired_at:
        return "Sin hora"
    try:
        return "Mediodía" if datetime.fromisoformat(aired_at).hour < 18 else "Noche"
    except ValueError:
        return "Sin hora"


def _mmss(seconds) -> str:
    if seconds is None:
        return "–"
    m, s = divmod(int(round(seconds)), 60)
    return f"{m}:{s:02d}"


def _num(value, digits=1) -> str:
    if value is None:
        return "–"
    txt = f"{value:+.{digits}f}" if value else f"{0:.{digits}f}"
    return txt.replace(".", ",").replace("-", "−")


def score_cell(value, title: str = "", digits: int = 1) -> str:
    """Celda con la puntuación como texto y el color de la escala divergente (−2 rojo … +2 azul)."""
    if value is None:
        return f'<td class="sc na" title="{_e(title or "no aplica")}">–</td>'
    step = max(-2, min(2, int(round(value))))
    cls = {-2: "n2", -1: "n1", 0: "z", 1: "p1", 2: "p2"}[step]
    return f'<td class="sc {cls}" title="{_e(title)}">{_num(value, digits)}</td>'


def markdown_to_html(md: str) -> str:
    """Markdown mínimo y seguro: títulos ##/###, listas con '- ', párrafos, **negrita** y *cursiva*."""
    out, in_list = [], False

    def inline(t: str) -> str:
        t = _e(t)
        t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
        t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
        return re.sub(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)", r"<em>\1</em>", t)

    for raw in md.splitlines():
        line = raw.rstrip()
        if line.startswith("- "):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{inline(line[2:])}</li>")
            continue
        if in_list:
            out.append("</ul>")
            in_list = False
        if line.startswith("### "):
            out.append(f"<h4>{inline(line[4:])}</h4>")
        elif line.startswith("## "):
            out.append(f"<h3>{inline(line[3:])}</h3>")
        elif line.startswith("# "):
            out.append(f"<h3>{inline(line[2:])}</h3>")
        elif line:
            out.append(f"<p>{inline(line)}</p>")
    if in_list:
        out.append("</ul>")
    return "\n".join(out)


# --- Datos -------------------------------------------------------------------

def _programs(conn: sqlite3.Connection, date: str) -> list[sqlite3.Row]:
    rows = conn.execute(
        "SELECT * FROM episodes e WHERE air_date = ? AND EXISTS (SELECT 1 FROM stories s WHERE s.episode_id = e.id)",
        (date,),
    ).fetchall()
    return sorted(rows, key=lambda r: (_franja(r["aired_at"]), _channel_key(r["channel"]), r["aired_at"] or ""))


def _program_metrics(conn: sqlite3.Connection, ep: sqlite3.Row) -> dict:
    eid = ep["id"]
    m = dict(conn.execute(
        "SELECT sum(duration_s) dur, sum(analysis_json IS NOT NULL) analysed, count(*) n,"
        " avg(ideological_lean) lean, avg(tone_government) gob, avg(tone_opposition) opo,"
        " coalesce(sum(editorializing), 0) editorial,"
        " sum(CASE WHEN event_id IS NOT NULL THEN duration_s ELSE 0 END) shared_dur"
        " FROM stories WHERE episode_id = ?", (eid,)).fetchone())
    opening = conn.execute(
        "SELECT headline FROM stories WHERE episode_id = ? AND story_format NOT IN ('sumario_titulares', 'promocion_autopromo')"
        " AND section NOT IN ('sumario', 'deportes', 'tiempo') ORDER BY ord LIMIT 1", (eid,)).fetchone()
    m["opening"] = opening[0] if opening else "–"
    voice = dict(conn.execute(
        "SELECT sa.bloc, sum(sa.voice_seconds) FROM story_actors sa JOIN stories s ON s.id = sa.story_id"
        " WHERE s.episode_id = ? GROUP BY sa.bloc", (eid,)).fetchall())
    m["voice"] = {label: sum(voice.get(b, 0) or 0 for b in blocs) for label, blocs in VOICE_GROUPS}
    m["verified"], m["quotes"] = conn.execute(
        "SELECT coalesce(sum(verified), 0), count(*) FROM evidence ev JOIN stories s ON s.id = ev.story_id"
        " WHERE s.episode_id = ?", (eid,)).fetchone()
    return m


def _story_quotes(conn: sqlite3.Connection, story_ids: list[int]) -> dict[int, list[sqlite3.Row]]:
    if not story_ids:
        return {}
    marks = ",".join("?" * len(story_ids))
    out: dict[int, list] = defaultdict(list)
    for r in conn.execute(f"SELECT * FROM evidence WHERE verified = 1 AND story_id IN ({marks})", story_ids):
        out[r["story_id"]].append(r)
    for rows in out.values():
        rows.sort(key=lambda r: (r["voice"] == "fuente",
                                 QUOTE_PRIORITY.index(r["dimension"]) if r["dimension"] in QUOTE_PRIORITY else 99))
    return out


def _quote_who(r) -> str:
    """'dice X · ' si la cita es de una fuente; nada si es del propio informativo."""
    if r["voice"] == "fuente":
        return f"lo dice {_e(r['speaker'] or 'una fuente')} · "
    return ""


def _analysis(row: sqlite3.Row) -> dict:
    try:
        return json.loads(row["analysis_json"]) if row["analysis_json"] else {}
    except (TypeError, json.JSONDecodeError):
        return {}


def _spread(values) -> float:
    vals = [v for v in values if v is not None]
    return (max(vals) - min(vals)) if len(vals) >= 2 else 0.0


# --- Secciones ---------------------------------------------------------------

def _franja_table(conn, franja: str, eps: list[sqlite3.Row], metrics: dict[int, dict]) -> str:
    head = "".join(f"<th scope='col'>{_e(ep['channel'])}<br><span class='muted'>{_e(ep['program'])}</span></th>" for ep in eps)
    rows = []

    def row(label, cells, hint=""):
        h = f" <span class='hint' title='{_e(hint)}' aria-label='{_e(hint)}'>i</span>" if hint else ""
        rows.append(f"<tr><th scope='row'>{_e(label)}{h}</th>{''.join(cells)}</tr>")

    row("Duración analizada", [f"<td>{_mmss(metrics[e['id']]['dur'])}</td>" for e in eps])
    row("Piezas analizadas", [f"<td>{metrics[e['id']]['analysed']} de {metrics[e['id']]['n']}</td>" for e in eps],
        "Se omiten sumarios, autopromociones, deportes y tiempo")
    row("Orientación media", [score_cell(metrics[e["id"]]["lean"], "−2 izquierda … +2 derecha") for e in eps],
        "−2 favorece a la izquierda, +2 a la derecha, 0 neutral")
    row("Tono hacia el Gobierno", [score_cell(metrics[e["id"]]["gob"], "−2 muy negativo … +2 muy positivo") for e in eps])
    row("Tono hacia la oposición", [score_cell(metrics[e["id"]]["opo"], "−2 muy negativo … +2 muy positivo") for e in eps])
    row("Piezas con opinión propia", [f"<td>{metrics[e['id']]['editorial']}</td>" for e in eps])
    row("Abre con", [f"<td class='txt'>{_e(metrics[e['id']]['opening'])}</td>" for e in eps])
    for label, _ in VOICE_GROUPS:
        row(f"Tiempo de palabra: {label}", [f"<td>{round(metrics[e['id']]['voice'][label])} s</td>" for e in eps],
            "Segundos con voz propia (totales, declaraciones, entrevistas)")
    row("Citas verificadas", [f"<td>{metrics[e['id']]['verified']} de {metrics[e['id']]['quotes']}</td>" for e in eps],
        "Fragmentos literales contrastados con la transcripción")
    return (f"<h2 id='{_e(franja.lower())}'>{_e(franja)}</h2>\n<div class='scroll'><table class='cmp'>"
            f"<thead><tr><th></th>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>")


def _events_section(conn, date: str, prog_label: dict[int, str]) -> tuple[str, list[dict]]:
    events = conn.execute("SELECT * FROM events WHERE air_date = ? ORDER BY id", (date,)).fetchall()
    data = []
    for ev in events:
        stories = conn.execute(
            "SELECT s.*, e.channel, e.program, e.aired_at FROM stories s JOIN episodes e ON e.id = s.episode_id"
            " WHERE s.event_id = ?", (ev["id"],)).fetchall()
        stories = sorted(stories, key=lambda s: (_channel_key(s["channel"]), _franja(s["aired_at"])))
        spread = max(_spread(s["tone_government"] for s in stories), _spread(s["tone_opposition"] for s in stories),
                     _spread(s["ideological_lean"] for s in stories))
        dur = [s["duration_s"] or 0 for s in stories]
        data.append({"ev": ev, "stories": stories, "spread": spread,
                     "dur_ratio": (max(dur) / min(dur)) if dur and min(dur) > 0 else 0, "total": sum(dur)})
    if not data:
        return "<h2 id='noticias'>La misma noticia en cada cadena</h2><p class='muted'>Sin cruce entre cadenas para este día.</p>", data

    quotes = _story_quotes(conn, [s["id"] for d in data for s in d["stories"]])
    blocks = []
    for d in sorted(data, key=lambda d: (-d["spread"], -d["total"])):
        rows = []
        for s in d["stories"]:
            q = quotes.get(s["id"], [])
            qtxt = (f"«{_e(q[0]['quote'])}» <span class='muted'>({_quote_who(q[0])}"
                    f"{DIMENSION_LABEL.get(q[0]['dimension'], q[0]['dimension'])})</span>"
                    if q else "<span class='muted'>–</span>")
            rows.append(
                f"<tr><td class='txt'>{_e(prog_label[s['episode_id']])}</td><td>{s['ord']}</td><td>{_mmss(s['duration_s'])}</td>"
                f"{score_cell(s['tone_government'], digits=0)}{score_cell(s['tone_opposition'], digits=0)}"
                f"{score_cell(s['ideological_lean'], digits=0)}<td class='txt'>{_e(FRAME_LABEL.get(s['primary_frame'], s['primary_frame'] or '–'))}</td>"
                f"<td class='txt'>{_e(BALANCE_LABEL.get(s['source_balance'], s['source_balance'] or '–'))}</td><td class='txt quote'>{qtxt}</td></tr>")
        flag = " <span class='badge'>gran divergencia</span>" if d["spread"] >= 2 else ""
        blocks.append(
            f"<details{' open' if d['spread'] >= 2 else ''}><summary><strong>{_e(d['ev']['label'])}</strong>{flag}"
            f" <span class='muted'>· {len(d['stories'])} piezas · {_mmss(d['total'])} en total</span></summary>"
            "<div class='scroll'><table class='ev'><thead><tr><th>Programa</th><th>Pos.</th><th>Dur.</th><th>Gob.</th>"
            "<th>Opo.</th><th>Orient.</th><th>Encuadre</th><th>Fuentes</th><th>Cita destacada</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div></details>")
    intro = ("<p class='muted'>Ordenadas de mayor a menor divergencia entre cadenas (diferencia máxima en tono u orientación). "
             "Las que difieren 2 puntos o más aparecen desplegadas.</p>")
    return f"<h2 id='noticias'>La misma noticia en cada cadena</h2>{intro}{''.join(blocks)}", data


def _highlights(conn, date: str, eps: list[sqlite3.Row], prog_label: dict[int, str], events: list[dict]) -> str:
    items = []
    # 1. Divergencias.
    for d in sorted(events, key=lambda d: -d["spread"])[:5]:
        if d["spread"] < 2:
            break
        parts = []
        for field, name in (("tone_government", "Gobierno"), ("tone_opposition", "oposición"), ("ideological_lean", "orientación")):
            vals = [(s[field], prog_label[s["episode_id"]]) for s in d["stories"] if s[field] is not None]
            if len(vals) >= 2 and _spread(v for v, _ in vals) >= 2:
                lo, hi = min(vals), max(vals)
                parts.append(f"{name}: {_e(lo[1])} {_num(lo[0], 0)} frente a {_e(hi[1])} {_num(hi[0], 0)}")
        items.append(f"<li><strong>{_e(d['ev']['label'])}</strong> — {'; '.join(parts)}.</li>")
    for d in sorted(events, key=lambda d: -d["dur_ratio"])[:3]:
        if d["dur_ratio"] < 4:
            break
        longest = max(d["stories"], key=lambda s: s["duration_s"] or 0)
        shortest = min(d["stories"], key=lambda s: s["duration_s"] or 0)
        items.append(f"<li><strong>{_e(d['ev']['label'])}</strong> — {_e(prog_label[longest['episode_id']])} le dedica "
                     f"{_mmss(longest['duration_s'])} y {_e(prog_label[shortest['episode_id']])} {_mmss(shortest['duration_s'])}.</li>")
    sections = [f"<h3>Mayores diferencias entre cadenas</h3><ul>{''.join(items)}</ul>"] if items else []

    ids = [e["id"] for e in eps]
    marks = ",".join("?" * len(ids))
    # 2. Exclusivas políticas.
    excl = conn.execute(
        f"SELECT * FROM stories WHERE episode_id IN ({marks}) AND event_id IS NULL AND is_political = 1"
        " AND analysis_json IS NOT NULL AND duration_s >= 30 ORDER BY duration_s DESC LIMIT 8", ids).fetchall()
    if excl:
        li = "".join(f"<li>{_e(prog_label[s['episode_id']])}: <strong>{_e(s['headline'])}</strong> ({_mmss(s['duration_s'])})</li>"
                     for s in excl)
        sections.append(f"<h3>Temas políticos que solo cubre un programa</h3><ul>{li}</ul>")

    # 3. Opinión propia y lenguaje cargado (solo citas verificadas).
    for dim, title, limit in (("editorializing", "Opinión propia de la cadena", 10), ("loaded_language", "Lenguaje cargado", 12)):
        rows = conn.execute(
            f"SELECT ev.*, s.episode_id, s.headline FROM evidence ev JOIN stories s ON s.id = ev.story_id"
            f" WHERE s.episode_id IN ({marks}) AND ev.dimension = ? AND ev.verified = 1"
            " AND coalesce(ev.voice, 'medio') = 'medio' ORDER BY s.episode_id, s.ord",
            ids + [dim]).fetchall()
        if not rows:
            continue
        by_prog: dict[int, list] = defaultdict(list)
        for r in rows:
            by_prog[r["episode_id"]].append(r)
        li = []
        for eid in ids:
            for r in by_prog.get(eid, [])[: max(2, limit // max(1, len(by_prog)))]:
                note = f" <span class='muted'>→ {_e(r['note'])}</span>" if dim == "loaded_language" and r["note"] else ""
                who = f" <span class='muted'>({_e(r['speaker'])})</span>" if r["speaker"] else ""
                li.append(f"<li>{_e(prog_label[eid])}: «{_e(r['quote'])}»{who}{note} <span class='muted'>— {_e(r['headline'])}</span></li>")
        sections.append(f"<h3>{title}</h3><p class='muted small'>Solo frases dichas por el propio informativo "
                        f"(presentador, redacción); las de sus fuentes no cuentan.</p><ul>{''.join(li)}</ul>")

    # 4. Fuentes desequilibradas.
    unb = conn.execute(
        f"SELECT * FROM stories WHERE episode_id IN ({marks}) AND source_balance IN ('unilateral', 'desequilibrado')"
        " ORDER BY duration_s DESC LIMIT 10", ids).fetchall()
    li = []
    for s in unb:
        missing = _analysis(s).get("missing_perspectives") or []
        if missing:
            li.append(f"<li>{_e(prog_label[s['episode_id']])}: <strong>{_e(s['headline'])}</strong> "
                      f"<span class='muted'>({_e(BALANCE_LABEL.get(s['source_balance'], s['source_balance']))})</span> — falta: {_e('; '.join(missing))}</li>")
    if li:
        sections.append(f"<h3>Piezas con fuentes desequilibradas</h3><ul>{''.join(li)}</ul>")
    if not sections:
        return ""
    return ("<h2 id='hallazgos'>Hallazgos calculados</h2><p class='muted'>Generados automáticamente a partir de las "
            "puntuaciones y de las citas verificadas.</p>" + "".join(sections))


def _program_details(conn, eps: list[sqlite3.Row], prog_label: dict[int, str]) -> str:
    blocks = []
    for ep in eps:
        rows = []
        for s in conn.execute("SELECT * FROM stories WHERE episode_id = ? ORDER BY ord", (ep["id"],)):
            a = _analysis(s)
            notes = a.get("selection_notes") or ""
            skipped = "" if s["analysis_json"] else " skipped"
            rows.append(
                f"<tr class='{skipped.strip()}'><td>{s['ord']}</td><td>{_mmss(s['start_s'])}</td><td>{_mmss(s['duration_s'])}</td>"
                f"<td class='txt'><strong>{_e(s['headline'])}</strong><br><span class='muted'>{_e(s['summary'])}</span></td>"
                f"<td class='txt'>{_e(SECTION_LABEL.get(s['section'], (s['section'] or '').replace('_', ' ')))}</td>"
                f"{score_cell(s['tone_government'], (a.get('tone_government') or {}).get('rationale', ''), 0)}"
                f"{score_cell(s['tone_opposition'], (a.get('tone_opposition') or {}).get('rationale', ''), 0)}"
                f"{score_cell(s['ideological_lean'], (a.get('ideological_lean') or {}).get('rationale', ''), 0)}"
                f"<td>{'sí' if s['editorializing'] else ''}</td><td class='txt small'>{_e(notes)}</td></tr>")
        model = conn.execute("SELECT group_concat(DISTINCT analysis_model) FROM stories WHERE episode_id = ?", (ep["id"],)).fetchone()[0]
        spk = conn.execute("SELECT * FROM episode_speakers WHERE episode_id = ? AND (name IS NOT NULL OR role != 'desconocido')"
                           " ORDER BY seconds DESC LIMIT 15", (ep["id"],)).fetchall()
        how = {"voz": "por la voz", "llm": "por el contexto", "anchor": "quien más habla"}
        voices_html = ("<p class='small'><strong>Voces identificadas:</strong> " + "; ".join(
            f"{_e(r['name'] or 'sin nombre')} <span class='muted'>({_e(r['role'])}, {round(r['seconds'])} s, "
            f"{how.get(r['id_source'], '–')})</span>" for r in spk) + "</p>") if spk else ""
        blocks.append(
            f"<details><summary><strong>{_e(prog_label[ep['id']])}</strong> <span class='muted'>· {_e(ep['title'])} · "
            f"<a href='{_e(ep['url'])}'>emisión</a> · analizado con {_e(model or '–')}</span></summary>{voices_html}"
            "<div class='scroll'><table class='det'><thead><tr><th>#</th><th>Inicio</th><th>Dur.</th><th>Pieza</th><th>Sección</th>"
            "<th>Gob.</th><th>Opo.</th><th>Orient.</th><th>Opinión</th><th>Notas de selección</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div></details>")
    return ("<h2 id='programas'>Detalle por programa</h2><p class='muted'>Pasa el ratón por una puntuación para ver la "
            "justificación. Las filas en gris no se puntúan (sumarios, autopromoción, deportes, tiempo).</p>" + "".join(blocks))


# --- Página ------------------------------------------------------------------

CSS = """
:root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;
--n2:#e34948;--n1:#f19d9a;--z:#f0efec;--p1:#86b6ef;--p2:#2a78d6;--on-strong:#ffffff;--on-soft:#0b0b0b;--accent:#2a78d6}
@media (prefers-color-scheme:dark){:root{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;
--muted:#898781;--grid:#2c2c2a;--n2:#e66767;--n1:#843a39;--z:#383835;--p1:#184f95;--p2:#3987e5;--on-strong:#0b0b0b;
--on-soft:#f0efec;--accent:#3987e5}}
*{box-sizing:border-box}body{margin:0;background:var(--page);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1180px;margin:0 auto;padding:24px 20px 64px}h1{font-size:26px;margin:0 0 4px}
h2{font-size:20px;margin:36px 0 10px;padding-top:8px;border-top:1px solid var(--grid)}h3{font-size:16px;margin:22px 0 6px}
a{color:var(--accent)}.muted{color:var(--muted)}.small{font-size:13px}nav a{margin-right:14px}
.scroll{overflow-x:auto;background:var(--surface);border:1px solid var(--grid);border-radius:8px}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:7px 10px;border-bottom:1px solid var(--grid);text-align:center;vertical-align:top}
thead th{font-size:13px;color:var(--ink2);font-weight:600;background:var(--surface);position:sticky;top:0}
tbody th{text-align:left;font-weight:500;color:var(--ink2);white-space:nowrap}td.txt{text-align:left}
td.quote{font-style:italic;min-width:260px}
.sc{font-weight:600;min-width:56px;border:2px solid var(--surface);border-radius:6px}
.sc.n2,.sc.p2{background:var(--n2);color:var(--on-strong)}.sc.p2{background:var(--p2)}
.sc.n1{background:var(--n1);color:var(--on-soft)}.sc.p1{background:var(--p1);color:var(--on-soft)}
.sc.z{background:var(--z);color:var(--ink)}.sc.na{color:var(--muted);font-weight:400}
tr.skipped td{color:var(--muted)}tr.skipped td strong{font-weight:500}
details{background:var(--surface);border:1px solid var(--grid);border-radius:8px;margin:8px 0;padding:8px 12px}
details>.scroll{margin-top:8px;border:0}summary{cursor:pointer}
.badge{display:inline-block;font-size:12px;padding:1px 8px;border-radius:10px;border:1px solid var(--ink2);color:var(--ink2);margin-left:6px}
.legend{display:flex;gap:6px;align-items:center;flex-wrap:wrap;font-size:13px;color:var(--ink2);margin:8px 0 0}
.legend span.sw{display:inline-block;min-width:34px;text-align:center;border-radius:5px;padding:1px 4px;font-weight:600}
.notes{background:var(--surface);border:1px solid var(--grid);border-left:4px solid var(--accent);border-radius:8px;padding:4px 18px}
.hint{display:inline-block;width:15px;height:15px;line-height:13px;border:1px solid var(--muted);border-radius:50%;
font:italic 600 11px/13px Georgia,serif;text-align:center;color:var(--muted);cursor:help;vertical-align:1px}
code{font-size:13px;background:var(--z);padding:0 4px;border-radius:4px}footer{margin-top:40px;font-size:13px;color:var(--muted)}
"""


def _legend() -> str:
    sw = "".join(f"<span class='sw sc {c}'>{t}</span>" for c, t in
                 (("n2", "−2"), ("n1", "−1"), ("z", "0"), ("p1", "+1"), ("p2", "+2")))
    return (f"<div class='legend'>Escala: {sw} <span>tono: muy negativo → muy positivo · orientación: izquierda → "
            "derecha · «–» no aplica</span></div>")


def build_day_report(cfg: Config, conn: sqlite3.Connection, date: str) -> Path | None:
    eps = _programs(conn, date)
    if not eps:
        log.info("%s: sin programas segmentados; no se genera informe", date)
        return None
    prog_label = {e["id"]: f"{e['program']}" for e in eps}
    metrics = {e["id"]: _program_metrics(conn, e) for e in eps}

    franjas: dict[str, list] = defaultdict(list)
    for e in eps:
        franjas[_franja(e["aired_at"])].append(e)
    def franja_block(f: str) -> str:
        html = _franja_table(conn, f, franjas[f], metrics)
        fpath = franja_notes_path(cfg, date, f)
        if fpath.exists():
            html += (f"<h3 id='obs-{_e(f.lower())}'>Observaciones · {_e(f)}</h3>"
                     f"<div class='notes'>{markdown_to_html(fpath.read_text(encoding='utf-8'))}</div>")
        return html

    tables = "".join(franja_block(f) for f in ("Mediodía", "Noche", "Sin hora") if f in franjas)
    events_html, events = _events_section(conn, date, prog_label)
    highlights = _highlights(conn, date, eps, prog_label, events)

    npath = notes_path(cfg, date)
    notes = (f"<h2 id='observaciones'>Observaciones escritas a mano</h2><div class='notes'>{markdown_to_html(npath.read_text(encoding='utf-8'))}</div>"
             if npath.exists() else "")

    models = [r[0] for r in conn.execute(
        f"SELECT DISTINCT analysis_model FROM stories WHERE analysis_model IS NOT NULL AND episode_id IN ({','.join('?' * len(eps))})",
        [e["id"] for e in eps])]
    versions = [r[0] for r in conn.execute(
        f"SELECT DISTINCT prompt_version FROM stories WHERE prompt_version IS NOT NULL AND episode_id IN ({','.join('?' * len(eps))})",
        [e["id"] for e in eps])]
    total_v = sum(m["verified"] for m in metrics.values())
    total_q = sum(m["quotes"] for m in metrics.values())
    nav = " ".join(f"<a href='#{a}'>{t}</a>" for a, t in (
        *[(f.lower(), f) for f in franjas], ("observaciones", "Observaciones") if notes else ("", ""),
        ("noticias", "Misma noticia"), ("hallazgos", "Hallazgos"), ("programas", "Detalle")) if a)

    page = f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>telesesgo · {_e(date)}</title><style>{CSS}</style></head>
<body><main>
<p class="small"><a href="index.html">← Todos los informes</a></p>
<h1>Telediarios del {_e(date)}</h1>
<p class="muted">{len(eps)} ediciones: {_e(', '.join(f"{e['channel']} — {e['program']}" for e in eps))}</p>
<nav>{nav}</nav>
{_legend()}
{tables}
{notes}
{events_html}
{highlights}
{_program_details(conn, eps, prog_label)}
<footer>
<p><strong>Método.</strong> Transcripción automática con Whisper; segmentación y puntuación por pieza con
{_e(', '.join(models) or '–')} (versión de prompt {_e(', '.join(versions) or '–')}). Citas verificadas contra la transcripción:
{total_v} de {total_q}. Las medias de cada programa solo incluyen las piezas puntuadas; sumarios y autopromociones no
se puntúan.</p>
<p><strong>Cautelas.</strong> Las puntuaciones son un instrumento de medida, no un dato: valida una muestra con codificación
humana antes de sacar conclusiones. Un solo día no basta para caracterizar a una cadena.</p>
<p>Generado el {datetime.now().strftime('%Y-%m-%d %H:%M')} · observaciones automáticas en
<code>{_e(npath.parent)}/{_e(date)}-mediodia.md</code> y <code>-noche.md</code>; notas a mano en <code>{_e(npath.name)}</code>.
Las citas de las observaciones automáticas que no aparecen en la transcripción se marcan como no verificadas.</p>
</footer></main></body></html>"""
    out = reports_dir(cfg) / f"{date}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    build_index(cfg, conn)
    log.info("Informe HTML: %s", out)
    return out


def build_index(cfg: Config, conn: sqlite3.Connection) -> Path:
    rdir = reports_dir(cfg)
    rdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in sorted(rdir.glob("????-??-??.html"), reverse=True):
        date = f.stem
        n = conn.execute("SELECT count(DISTINCT e.id) FROM episodes e JOIN stories s ON s.episode_id = e.id"
                         " WHERE e.air_date = ?", (date,)).fetchone()[0]
        ev = conn.execute("SELECT count(*) FROM events WHERE air_date = ?", (date,)).fetchone()[0]
        has_notes = "sí" if notes_path(cfg, date).exists() or any(
            franja_notes_path(cfg, date, f).exists() for f in ("Mediodía", "Noche")) else ""
        rows.append(f"<tr><td class='txt'><a href='{_e(f.name)}'>{_e(date)}</a></td><td>{n}</td><td>{ev}</td><td>{has_notes}</td></tr>")
    page = f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>telesesgo · informes</title><style>{CSS}</style></head>
<body><main><h1>Informes de telesesgo</h1>
<div class="scroll"><table><thead><tr><th>Fecha</th><th>Ediciones</th><th>Noticias cruzadas</th><th>Observaciones</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div></main></body></html>"""
    out = rdir / "index.html"
    out.write_text(page, encoding="utf-8")
    return out
