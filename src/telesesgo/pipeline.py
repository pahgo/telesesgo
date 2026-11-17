"""`telesesgo run`: el pipeline completo de un día (y una franja) con progreso legible.

Orden y solapes:
  1. Busca las ediciones del día en cada fuente.
  2. Descarga en paralelo (a una carpeta temporal del disco local) y la GPU transcribe cada episodio
     en cuanto termina su descarga.
  3. En cuanto un episodio está transcrito, Gemini lo segmenta y analiza en otro hilo, mientras la
     GPU sigue con el siguiente.
  4. Cruza la misma noticia entre cadenas y genera el informe HTML.
"""

from __future__ import annotations

import re
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta

from . import db
from .config import Config
from .sources import MADRID

EDITION_ALIASES = {
    "mediodia": "mediodia", "mediodía": "mediodia", "mañana": "mediodia", "manana": "mediodia", "tarde": "mediodia",
    "14h": "mediodia", "15h": "mediodia", "td1": "mediodia",
    "noche": "noche", "20h": "noche", "21h": "noche", "td2": "noche",
}
FULL_DAY = {"dia", "día", "completo", "todo", "todas", "ambas", "all"}
EDITION_LABEL = {"mediodia": "mediodía", "noche": "noche"}


def parse_edition(text: str | None) -> set[str] | None:
    """None = día completo."""
    if text is None or text.strip().lower() in FULL_DAY:
        return None
    key = text.strip().lower()
    if key not in EDITION_ALIASES:
        raise ValueError(f"Edición no reconocida: {text!r}. Usa mañana/mediodía, noche o día.")
    return {EDITION_ALIASES[key]}


def parse_date(text: str | None, today: date | None = None) -> date:
    """None o 'ayer' = ayer (hora de Madrid). Acepta YYYY-MM-DD, DD/MM/YYYY, DD-MM-YYYY, 'hoy' y 'anteayer'."""
    today = today or datetime.now(MADRID).date()
    if text is None:
        return today - timedelta(days=1)
    t = text.strip().lower()
    rel = {"hoy": 0, "ayer": 1, "anteayer": 2}
    if t in rel:
        return today - timedelta(days=rel[t])
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", t):
        return date.fromisoformat(t)
    m = re.fullmatch(r"(\d{1,2})[/-](\d{1,2})[/-](\d{2}|\d{4})", t)
    if m:
        d, mth, y = map(int, m.groups())
        return date(y + 2000 if y < 100 else y, mth, d)
    raise ValueError(f"Fecha no reconocida: {text!r}. Usa AAAA-MM-DD (p. ej. 2026-11-16), DD/MM/AAAA, hoy o ayer.")


def resolve_args(first: str | None, second: str | None) -> tuple[date, set[str] | None]:
    """Los dos argumentos posicionales en cualquier orden: `run 2026-11-16 noche`, `run noche`, `run`."""
    day_txt, ed_txt = first, second
    if first is not None and second is None:
        try:
            parse_edition(first)
            day_txt, ed_txt = None, first
        except ValueError:
            pass
    elif first is not None and second is not None:
        try:
            parse_edition(first)
            day_txt, ed_txt = second, first
        except ValueError:
            pass
    return parse_date(day_txt), parse_edition(ed_txt)


class Progress:
    """Salida de progreso legible y segura entre hilos."""

    def __init__(self, stream=None):
        self.stream = stream or sys.stdout
        self.t0 = time.monotonic()
        self.lock = threading.Lock()

    def _clock(self) -> str:
        m, s = divmod(int(time.monotonic() - self.t0), 60)
        return f"{m:02d}:{s:02d}"

    def say(self, msg: str, mark: str = " ") -> None:
        with self.lock:
            print(f"[{self._clock()}] {mark} {msg}", file=self.stream, flush=True)

    def phase(self, msg: str) -> None:
        with self.lock:
            print(f"\n[{self._clock()}] == {msg}", file=self.stream, flush=True)

    def ok(self, msg): self.say(msg, "✓")
    def fail(self, msg): self.say(msg, "✗")
    def warn(self, msg): self.say(msg, "!")


def _name(ep) -> str:
    return f"{ep['program']} ({ep['channel']}, id {ep['id']})"


def _hint(error: str | None) -> str:
    if not error:
        return ""
    if "registered users" in error or "--cookies" in error:
        return " → Atresplayer exige iniciar sesión para esta edición (ver cookies_file en la guía del README)"
    if "GEMINI_API_KEY" in error:
        return " → falta la clave de Gemini en .env"
    return f" → {error.splitlines()[0][:160]}"


def _row(conn, ep_id):
    return conn.execute("SELECT * FROM episodes WHERE id = ?", (ep_id,)).fetchone()


def _llm_episode(cfg: Config, llm, ep_id: int, p: Progress) -> None:
    """Segmenta y analiza un episodio en su propio hilo (con su propia conexión SQLite)."""
    from .analyze import analyze_episode, needs_analysis, segment_episode
    from .cli import _step

    conn = db.connect(cfg.db_path)
    try:
        ep = _row(conn, ep_id)
        name = _name(ep)
        if ep["status"] == "transcribed":
            p.say(f"Gemini segmenta {name}…")
            if not _step(conn, ep, "segment", lambda: segment_episode(cfg, conn, llm, ep)):
                p.fail(f"Segmentación de {name}{_hint(_row(conn, ep_id)['error'])}")
                return
        ep = _row(conn, ep_id)
        if ep["status"] != "segmented":
            return
        rows = conn.execute("SELECT * FROM stories WHERE episode_id = ?", (ep_id,)).fetchall()
        todo = sum(needs_analysis(cfg, r) for r in rows)
        p.say(f"Gemini analiza {name}: {len(rows)} piezas, {todo} a analizar")
        step = max(1, todo // 4)

        def on_progress(done, total):
            if done == total or done % step == 0:
                p.say(f"  {ep['program']}: {done}/{total} piezas analizadas")

        if _step(conn, ep, "analyze", lambda: analyze_episode(cfg, conn, llm, ep, on_progress=on_progress)):
            ep = _row(conn, ep_id)
            if ep["status"] == "analyzed":
                p.ok(f"Análisis de {name} terminado")
            else:
                p.warn(f"{name}: {ep['error']} (se reintentarán en la próxima ejecución)")
        else:
            p.fail(f"Análisis de {name}{_hint(_row(conn, ep_id)['error'])}")
    finally:
        conn.close()


def reset_analysis(conn: sqlite3.Connection, ids: list[int]) -> int:
    """Borra segmentación y análisis (y los cruces del día) de esos episodios y los deja como transcritos."""
    if not ids:
        return 0
    m = ",".join("?" * len(ids))
    days = [r[0] for r in conn.execute(f"SELECT DISTINCT air_date FROM episodes WHERE id IN ({m})", ids)]
    with conn:
        conn.execute(f"DELETE FROM evidence WHERE story_id IN (SELECT id FROM stories WHERE episode_id IN ({m}))", ids)
        conn.execute(f"DELETE FROM story_actors WHERE story_id IN (SELECT id FROM stories WHERE episode_id IN ({m}))", ids)
        conn.execute(f"DELETE FROM stories WHERE episode_id IN ({m})", ids)
        for d in days:
            conn.execute("UPDATE stories SET event_id = NULL WHERE event_id IN (SELECT id FROM events WHERE air_date = ?)", (d,))
            conn.execute("DELETE FROM events WHERE air_date = ?", (d,))
        n = conn.execute(f"UPDATE episodes SET status = 'transcribed', error = NULL WHERE id IN ({m})"
                         " AND status IN ('segmented', 'analyzed')", ids).rowcount
    return n


def run_day(cfg: Config, conn: sqlite3.Connection, day: date, editions: set[str] | None, *,
            use_llm: bool = True, redo: bool = False, p: Progress | None = None) -> int:
    """Devuelve 0 si todo fue bien, 1 si hubo fallos o no había nada que procesar."""
    from .cli import _download_and_transcribe, write_reports
    from .ingest import discover_date

    p = p or Progress()
    run_started = datetime.now(MADRID).isoformat(timespec="seconds")
    label = " y ".join(EDITION_LABEL[e] for e in sorted(editions)) if editions else "día completo"
    print(f"telesesgo · {day.isoformat()} · {label}", flush=True)

    # 0. Cliente de Gemini antes de descargar nada: si falta la clave, se avisa ya.
    llm = None
    if use_llm:
        from .llm import Llm, LlmError
        try:
            llm = Llm(cfg.llm)
        except LlmError as exc:
            p.warn(f"{exc}. Se descargará y transcribirá, pero no se analizará.")

    # 1. Descubrimiento
    p.phase("1/5 Buscando ediciones")

    def on_source(src, n, err):
        if err:
            p.fail(f"{src.program}: no se pudo consultar ({err[:120]})")
        elif n:
            p.ok(f"{src.program}: encontrada")
        else:
            p.warn(f"{src.program}: no hay edición de ese día en la web de la cadena")

    ids = discover_date(cfg, conn, day, editions, on_source)
    if not ids:
        p.fail("No hay ninguna edición que procesar.")
        return 1
    if redo:
        n = reset_analysis(conn, ids)
        p.say(f"--rehacer: borrado el análisis de {n} ediciones (se conservan audio y transcripción)")

    # 2-3. Descarga + transcripción (GPU) con el análisis (Gemini) solapado
    p.phase(f"2/5 Descarga, transcripción{' y análisis' if llm else ''} de {len(ids)} ediciones")
    for ep_id in ids:
        ep = _row(conn, ep_id)
        if ep["status"] == "analyzed":
            p.ok(f"{_name(ep)}: ya estaba analizada")
        elif ep["status"] != "discovered":
            p.say(f"{_name(ep)}: ya en estado «{ep['status']}», se continúa desde ahí")

    pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="llm") if llm else None
    futures = []
    submitted: set[int] = set()

    def submit(ep_id: int) -> None:
        if pool and ep_id not in submitted:
            submitted.add(ep_id)
            futures.append(pool.submit(_llm_episode, cfg, llm, ep_id, p))

    from .cli import diarize_one
    for ep_id in ids:  # lo que ya estaba transcrito: diarizar si falta y empezar a analizar ya
        if _row(conn, ep_id)["status"] in ("transcribed", "segmented"):
            diarize_one(cfg, conn, ep_id, p)
            submit(ep_id)
    try:
        _download_and_transcribe(cfg, conn, ids, transcribe=True, on_transcribed=submit, progress=p)
        for f in futures:
            f.result()
    finally:
        if pool:
            pool.shutdown(wait=True)

    # 4. Cruce e informe
    p.phase("3/5 Cruce entre cadenas")
    analyzed = conn.execute(
        "SELECT count(DISTINCT program) FROM episodes WHERE air_date = ? AND status = 'analyzed'", (day.isoformat(),)
    ).fetchone()[0]
    if not llm:
        p.warn("Sin Gemini: no se cruza.")
    elif analyzed < 2:
        p.warn(f"Solo hay {analyzed} programa(s) analizado(s) ese día: no hay nada que cruzar.")
    else:
        from .analyze import match_events
        t0, started, ok = time.monotonic(), datetime.now(MADRID).isoformat(timespec="seconds"), False
        try:
            n = match_events(cfg, conn, llm, day.isoformat())
            ok = True
            p.ok(f"{n} noticias comunes entre {analyzed} programas")
        except Exception as exc:
            p.fail(f"Cruce: {exc}")
        db.log_timing(conn, "match", time.monotonic() - t0, ok, air_date=day.isoformat(), started_at=started)

    p.phase("4/5 Observaciones por franja")
    write_notes_for(cfg, conn, llm, day.isoformat(), editions, p)

    p.phase("5/5 Informe")
    write_reports(cfg, conn, [day.isoformat()])

    return _summary(conn, ids, run_started, p, done_status="analyzed" if llm else "transcribed")


def write_notes_for(cfg: Config, conn, llm, day: str, editions: set[str] | None, p: Progress) -> None:
    """Observaciones de Gemini para cada franja pedida que tenga al menos un programa analizado."""
    from .notes import FRANJAS, write_edition_notes
    if not llm:
        p.warn("Sin Gemini: no se escriben observaciones.")
        return
    if not cfg.llm.write_notes:
        p.say("Observaciones desactivadas (llm.write_notes: false).")
        return
    for slug in sorted(editions or FRANJAS, key=list(FRANJAS).index):
        franja = FRANJAS[slug]
        t0, started, ok = time.monotonic(), datetime.now(MADRID).isoformat(timespec="seconds"), False
        try:
            path = write_edition_notes(cfg, conn, llm, day, franja)
            ok = True
            if path:
                p.ok(f"Observaciones de {franja.lower()}: {path.name}")
            else:
                p.warn(f"{franja}: ningún programa analizado; sin observaciones.")
        except Exception as exc:
            p.fail(f"Observaciones de {franja.lower()}: {exc}")
        db.log_timing(conn, "notes", time.monotonic() - t0, ok, air_date=day, started_at=started, detail=franja)


def _summary(conn, ids, run_started, p: Progress, done_status: str = "analyzed") -> int:
    finished = {"transcribed", "segmented", "analyzed"} if done_status == "transcribed" else {"analyzed"}
    marks = ",".join("?" * len(ids))
    times = {}
    for r in conn.execute(f"SELECT episode_id, step, sum(seconds) s FROM step_timings WHERE started_at >= ?"
                          f" AND episode_id IN ({marks}) GROUP BY episode_id, step", [run_started, *ids]):
        times.setdefault(r["episode_id"], {})[r["step"]] = r["s"]
    print(f"\nResumen (total {p._clock()}):", flush=True)
    print(f"  {'programa':<30} {'estado':<12} {'piezas':>6}  descarga  transcr.  voces     análisis", flush=True)
    bad = 0
    for ep_id in ids:
        ep = _row(conn, ep_id)
        t = times.get(ep_id, {})
        n = conn.execute("SELECT count(*), count(analysis_json) FROM stories WHERE episode_id = ?", (ep_id,)).fetchone()

        def fmt(*steps):
            s = sum(t.get(x, 0) for x in steps)
            return f"{s / 60:5.1f} min" if s else "      –  "

        ok = ep["status"] in finished
        bad += not ok
        print(f"  {ep['program'][:30]:<30} {ep['status']:<12} {n[1]:>3}/{n[0]:<3} {fmt('download')} {fmt('transcribe')}"
              f" {fmt('diarize')} {fmt('segment', 'analyze')}", flush=True)
        if not ok and ep["error"]:
            print(f"      {_hint(ep['error']).removeprefix(' → ')}", flush=True)
    return 1 if bad else 0
