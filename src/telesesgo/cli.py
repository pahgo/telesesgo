"""CLI: telesesgo {run,llm-check,discover,download,transcribe,analyze,match,status,report,html,timings}."""

from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
import time
from datetime import datetime

from . import db
from .config import Config, load_config
from .sources import MADRID

log = logging.getLogger("telesesgo")


def _pending(conn: sqlite3.Connection, status: str, ids: list[int] | None) -> list[sqlite3.Row]:
    q = "SELECT * FROM episodes WHERE status = ?"
    args: list = [status]
    if ids:
        q += f" AND id IN ({','.join('?' * len(ids))})"
        args += ids
    return conn.execute(q + " ORDER BY aired_at", args).fetchall()


def _now() -> str:
    return datetime.now(MADRID).isoformat(timespec="seconds")


def _step(conn, ep, label, fn) -> bool:
    started, t0 = _now(), time.monotonic()
    ok = False
    try:
        fn()
        ok = True
    except Exception as exc:
        log.exception("Fallo en %s del episodio %s", label, ep["id"])
        # Se mantiene el estado: la próxima ejecución reintenta este paso.
        conn.execute("UPDATE episodes SET error = ?, updated_at = datetime('now') WHERE id = ?",
                     (f"{label}: {exc}"[:2000], ep["id"]))
        conn.commit()
    finally:
        dt = time.monotonic() - t0
        db.log_timing(conn, label, dt, ok, episode_id=ep["id"], air_date=ep["air_date"], started_at=started)
        log.info("%s del episodio %s: %.1f s%s", label, ep["id"], dt, "" if ok else " (fallo)")
    return ok


def _download_worker(cfg: Config, episode_id: int, progress=None) -> tuple[int, bool]:
    """Descarga en un hilo propio, con su propia conexión SQLite (no se comparten entre hilos)."""
    from .ingest import download_episode
    tconn = db.connect(cfg.db_path)
    try:
        ep = tconn.execute("SELECT * FROM episodes WHERE id = ?", (episode_id,)).fetchone()
        if progress:
            progress.say(f"Descargando {ep['program']}…")
        t0 = time.monotonic()
        ok = _step(tconn, ep, "download", lambda: download_episode(cfg, tconn, ep))
        if progress:
            if ok:
                progress.ok(f"Descargado {ep['program']} en {time.monotonic() - t0:.0f} s")
            else:
                from .pipeline import _hint
                err = tconn.execute("SELECT error FROM episodes WHERE id = ?", (episode_id,)).fetchone()[0]
                progress.fail(f"Descarga de {ep['program']}{_hint(err)}")
        return episode_id, ok
    finally:
        tconn.close()


def diarize_one(cfg: Config, conn, ep_id: int, progress=None) -> bool:
    """Diariza (GPU) si está activado y aún no se hizo. Un fallo no para el pipeline: el análisis sigue
    funcionando sin hablantes, con la atribución por actores."""
    from . import transcribe as tr
    if not tr.needs_diarization(cfg, conn, ep_id):
        return True
    ep = conn.execute("SELECT * FROM episodes WHERE id = ?", (ep_id,)).fetchone()
    if progress:
        progress.say(f"Identificando voces de {ep['program']}…")
    t0 = time.monotonic()
    ok = _step(conn, ep, "diarize", lambda: tr.diarize_episode(cfg, conn, ep))
    if progress:
        if ok:
            rows = conn.execute("SELECT count(*), sum(id_source = 'voz') FROM episode_speakers WHERE episode_id = ?",
                                (ep_id,)).fetchone()
            progress.ok(f"Voces de {ep['program']}: {rows[0]} hablantes, {rows[1] or 0} reconocidos por la voz"
                        f" ({time.monotonic() - t0:.0f} s)")
        else:
            err = conn.execute("SELECT error FROM episodes WHERE id = ?", (ep_id,)).fetchone()[0] or ""
            progress.warn(f"Sin diarización en {ep['program']} ({(err.splitlines() or [''])[0][:140]}); se analiza igualmente")
    # el error de la diarización no debe quedar como error del episodio
    conn.execute("UPDATE episodes SET error = NULL WHERE id = ? AND error LIKE 'diarize:%'", (ep_id,))
    conn.commit()
    return ok


def _download_and_transcribe(cfg: Config, conn, ids: list[int] | None, transcribe: bool,
                             on_transcribed=None, progress=None) -> None:
    """Descargas en paralelo; si `transcribe`, la GPU transcribe cada episodio en cuanto termina su descarga.
    `on_transcribed(id)` se llama al terminar cada transcripción (p. ej., para lanzar el análisis)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from . import transcribe as tr

    def transcribe_one(ep_id: int) -> None:
        ep = conn.execute("SELECT * FROM episodes WHERE id = ?", (ep_id,)).fetchone()
        if progress is None:
            fn = lambda: tr.transcribe_episode(cfg, conn, ep)
        else:
            mins = round((ep["duration_s"] or 0) / 60)
            progress.say(f"Transcribiendo {ep['program']} ({mins} min de audio) en la GPU…")

            def report(done, total, speed):
                if total and int(done) % 600 < 60:  # cada ~10 min de audio
                    progress.say(f"  {ep['program']}: {int(done // 60)}/{int(total // 60)} min ({speed:.1f}x tiempo real)")

            fn = lambda: tr.transcribe_episode(cfg, conn, ep, on_progress=report)
        t0 = time.monotonic()
        ok = _step(conn, ep, "transcribe", fn)
        if progress:
            if ok:
                progress.ok(f"Transcrito {ep['program']} en {(time.monotonic() - t0) / 60:.1f} min")
            else:
                progress.fail(f"Transcripción de {ep['program']} falló (ver data/logs)")
        if ok:
            diarize_one(cfg, conn, ep_id, progress)
        if ok and on_transcribed:
            on_transcribed(ep_id)

    to_download = [ep["id"] for ep in _pending(conn, "discovered", ids)]
    workers = max(1, min(cfg.download.workers, len(to_download) or 1))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="dl") as pool:
        futures = [pool.submit(_download_worker, cfg, i, progress) for i in to_download]
        if to_download:
            log.info("Descargando %d episodios con %d en paralelo", len(to_download), workers)
        if transcribe:
            # Mientras se descarga, se adelanta lo que ya estaba descargado.
            for ep in _pending(conn, "downloaded", ids):
                if ep["id"] not in to_download:
                    transcribe_one(ep["id"])
        for fut in as_completed(futures):
            ep_id, ok = fut.result()
            if ok and transcribe:
                transcribe_one(ep_id)


def cmd_discover(cfg: Config, conn, args):
    from .ingest import discover_all
    n = discover_all(cfg, conn, args.source)
    print(f"{n} episodios nuevos")


def cmd_download(cfg, conn, args):
    _download_and_transcribe(cfg, conn, args.id, transcribe=False)


def cmd_transcribe(cfg, conn, args):
    from .transcribe import transcribe_episode
    for ep in _pending(conn, "downloaded", args.id):
        if _step(conn, ep, "transcribe", lambda: transcribe_episode(cfg, conn, ep)):
            diarize_one(cfg, conn, ep["id"])


def _llm(cfg):
    from .llm import Llm
    return Llm(cfg.llm)


def cmd_analyze(cfg, conn, args):
    from .analyze import analyze_episode, segment_episode
    llm = _llm(cfg)
    for ep in _pending(conn, "transcribed", args.id):
        _step(conn, ep, "segment", lambda: segment_episode(cfg, conn, llm, ep))
    statuses = ["segmented"] + (["analyzed"] if args.force else [])
    dates: set[str] = set()
    for st in statuses:
        for ep in _pending(conn, st, args.id):
            ep = conn.execute("SELECT * FROM episodes WHERE id = ?", (ep["id"],)).fetchone()
            if _step(conn, ep, "analyze", lambda: analyze_episode(cfg, conn, llm, ep, force=args.force)):
                dates.add(ep["air_date"])
    write_reports(cfg, conn, sorted(dates))


def cmd_match(cfg, conn, args):
    from .analyze import match_events
    dates = args.date or [r[0] for r in conn.execute(
        "SELECT DISTINCT air_date FROM episodes WHERE status = 'analyzed' ORDER BY air_date DESC LIMIT ?", (cfg.max_age_days,))]
    llm = _llm(cfg)
    for d in dates:
        started, t0, ok = _now(), time.monotonic(), False
        try:
            match_events(cfg, conn, llm, d)
            ok = True
        except Exception:
            log.exception("Fallo al cruzar %s", d)
        db.log_timing(conn, "match", time.monotonic() - t0, ok, air_date=d, started_at=started)
    write_reports(cfg, conn, dates)


def write_reports(cfg, conn, dates) -> None:
    """Regenera el informe HTML de cada fecha. Un fallo aquí no debe tumbar el pipeline."""
    from .htmlreport import build_day_report
    for d in dates:
        started, t0, ok = _now(), time.monotonic(), False
        try:
            path = build_day_report(cfg, conn, d)
            ok = True
            if path:
                print(f"Informe: {path}")
        except Exception:
            log.exception("Fallo al generar el informe HTML de %s", d)
        db.log_timing(conn, "html", time.monotonic() - t0, ok, air_date=d, started_at=started)


def cmd_timings(cfg, conn, args):
    """Tiempos por paso y episodio de una fecha (o de los últimos registros)."""
    where, params = ("WHERE t.air_date = ?", [args.date]) if args.date else ("", [])
    _table(conn.execute(
        "SELECT t.step paso, e.id ep, coalesce(e.source_key, '–') fuente, round(t.seconds, 1) s, t.ok, t.started_at inicio"
        f" FROM step_timings t LEFT JOIN episodes e ON e.id = t.episode_id {where} ORDER BY t.id DESC LIMIT ?",
        params + [args.limit]).fetchall())
    print("\nResumen por paso:")
    _table(conn.execute(
        "SELECT step paso, count(*) n, round(sum(seconds)/60, 1) total_min, round(avg(seconds), 1) media_s,"
        f" round(max(seconds), 1) max_s, sum(ok = 0) fallos FROM step_timings t {where} GROUP BY step ORDER BY min(id)",
        params).fetchall())


def cmd_html(cfg, conn, args):
    dates = args.date or [r[0] for r in conn.execute(
        "SELECT DISTINCT e.air_date FROM episodes e JOIN stories s ON s.episode_id = e.id ORDER BY 1 DESC")]
    write_reports(cfg, conn, dates)


def cmd_run(cfg, conn, args):
    """Pipeline completo de un día/franja: descubrir, descargar, transcribir, analizar, cruzar e informe."""
    from .pipeline import resolve_args, run_day
    first = args.fecha or args.pos1
    second = args.edicion or args.pos2
    try:
        day, editions = resolve_args(first, second)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return run_day(cfg, conn, day, editions, use_llm=not args.no_llm, redo=args.rehacer)


def cmd_notes(cfg, conn, args):
    """(Re)escribe las observaciones automáticas de una fecha y regenera su informe."""
    from .pipeline import Progress, parse_date, parse_edition, write_notes_for
    try:
        day, editions = parse_date(args.fecha), parse_edition(args.edicion)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    write_notes_for(cfg, conn, _llm(cfg), day.isoformat(), editions, Progress())
    write_reports(cfg, conn, [day.isoformat()])
    return 0


def cmd_voces(cfg, conn, args):
    """Biblioteca de voces: listar, ver los hablantes de un episodio, renombrar o borrar una voz."""
    if args.renombrar:
        old, new = args.renombrar
        if conn.execute("SELECT 1 FROM voices WHERE name = ?", (new,)).fetchone():
            print(f"Ya existe una voz llamada {new!r}: borra una de las dos primero.", file=sys.stderr)
            return 1
        n = conn.execute("UPDATE voices SET name = ? WHERE name = ?", (new, old)).rowcount
        conn.execute("UPDATE episode_speakers SET name = ? WHERE name = ?", (new, old))
        conn.commit()
        print(f"{n} voz renombrada. Los episodios ya analizados conservan el nombre antiguo en sus citas "
              "hasta que los reanalices (run FECHA --rehacer).")
        return 0 if n else 1
    if args.borrar:
        n = conn.execute("DELETE FROM voices WHERE name = ?", (args.borrar,)).rowcount
        conn.commit()
        print(f"{n} voz borrada de la biblioteca.")
        return 0 if n else 1
    if args.episodio:
        _table(conn.execute(
            "SELECT label hablante, round(seconds) s, role rol, coalesce(name, '–') nombre, coalesce(id_source, '–') como,"
            " coalesce(round(similarity, 2), '') similitud FROM episode_speakers WHERE episode_id = ? ORDER BY seconds DESC",
            (args.episodio,)).fetchall())
        return 0
    _table(conn.execute(
        "SELECT v.name nombre, v.role rol, coalesce(v.channel, '–') cadena, v.samples episodios, round(v.seconds / 60, 1) min_voz,"
        " (SELECT count(*) FROM episode_speakers es WHERE es.voice_id = v.id AND es.id_source = 'voz') reconocida,"
        " v.updated_at actualizada FROM voices v ORDER BY v.role, v.name").fetchall())
    return 0


def cmd_hf_check(cfg, conn, args):
    """Comprueba HF_TOKEN y el acceso al modelo de diarización sin descargarlo."""
    import json
    import urllib.error
    import urllib.request

    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        print("✗ No hay HF_TOKEN: añade una línea HF_TOKEN=hf_... al fichero .env del proyecto.", file=sys.stderr)
        return 1
    if not token.startswith("hf_"):
        print("! El token no empieza por «hf_»: revisa que lo copiaste entero y sin comillas ni espacios.")

    def get(url, extra=None):
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}", "User-Agent": "telesesgo",
                                                   **(extra or {})})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    code, body = get("https://huggingface.co/api/whoami-v2")
    if code != 200:
        print(f"✗ Hugging Face no acepta el token (HTTP {code}). Crea uno nuevo de tipo Read y cópialo otra vez.",
              file=sys.stderr)
        return 1
    print(f"✓ Token válido: cuenta «{json.loads(body).get('name', '?')}»")
    model = cfg.stt.diarization.model
    code, _ = get(f"https://huggingface.co/{model}/resolve/main/config.yaml", {"Range": "bytes=0-0"})
    if code in (200, 206):
        print(f"✓ Acceso a {model}: condiciones aceptadas")
    elif code in (401, 403):
        print(f"✗ Sin acceso a {model} (HTTP {code}). Abre https://huggingface.co/{model}, rellena el formulario y pulsa "
              "«Agree and access repository». Si tu token es Fine-grained, márcale «Read access to contents of all "
              "public gated repos you can access».", file=sys.stderr)
        return 1
    else:
        print(f"✗ Respuesta inesperada al comprobar {model} (HTTP {code})", file=sys.stderr)
        return 1
    if not cfg.stt.diarization.enabled:
        print("! La diarización sigue desactivada: pon enabled: true en config.yaml (stt → diarization).")
    else:
        print("✓ Diarización activada en config.yaml. Todo listo.")
    return 0


def cmd_llm_check(cfg, conn, args):
    from .llm import LlmError, check
    try:
        check(cfg.llm)
    except LlmError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # clave inválida, sin red, modelo inexistente…
        print(f"Error al llamar a Gemini: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_status(cfg, conn, args):
    rows = conn.execute(
        "SELECT id, air_date, source_key, status, round(duration_s/60) AS min, substr(coalesce(error,''),1,60) AS err,"
        " (SELECT count(*) FROM stories s WHERE s.episode_id = e.id) AS piezas FROM episodes e ORDER BY aired_at DESC LIMIT ?",
        (args.limit,),
    ).fetchall()
    _table(rows)
    cost = conn.execute("SELECT stage, count(*) n, sum(input_tokens) inp, sum(cache_read_tokens) cached,"
                        " sum(output_tokens) outp FROM llm_usage GROUP BY stage").fetchall()
    if cost:
        print("\nUso de tokens por etapa:")
        _table(cost)


def cmd_report(cfg, conn, args):
    date = args.date[0] if args.date else datetime.now(MADRID).date().isoformat()
    print(f"== Resumen por cadena ({date}) ==")
    _table(conn.execute(
        "SELECT channel, program, count(*) piezas, round(sum(duration_s)/60,1) min,"
        " round(avg(ideological_lean),2) lean, round(avg(tone_government),2) tono_gob,"
        " round(avg(tone_opposition),2) tono_opo, sum(editorializing) editorial"
        " FROM v_stories WHERE air_date = ? GROUP BY channel, program", (date,)).fetchall())
    print(f"\n== Tiempo de palabra por bloque (s) ==")
    _table(conn.execute("SELECT channel, bloc, round(voice_s) voz_s, round(mention_s) mencion_s FROM v_bloc_exposure"
                        " WHERE air_date = ? ORDER BY channel, voz_s DESC", (date,)).fetchall())
    print(f"\n== Misma noticia, distintas cadenas ==")
    _table(conn.execute("SELECT event_id, substr(label,1,50) acontecimiento, channel, position pos, duration_s dur,"
                        " tone_government tono_gob, tone_opposition tono_opo, primary_frame encuadre FROM v_event_comparison"
                        " WHERE air_date = ? ORDER BY event_id, channel", (date,)).fetchall())


def _table(rows):
    if not rows:
        print("(sin datos)")
        return
    cols = rows[0].keys()
    data = [[("" if r[c] is None else str(r[c])) for c in cols] for r in rows]
    widths = [min(60, max(len(c), *(len(d[i]) for d in data))) for i, c in enumerate(cols)]
    print("  ".join(c.ljust(w) for c, w in zip(cols, widths)))
    for d in data:
        print("  ".join(v[:w].ljust(w) for v, w in zip(d, widths)))


def main(argv=None):
    p = argparse.ArgumentParser(prog="telesesgo", description=__doc__)
    p.add_argument("-c", "--config", default="config.yaml")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("discover", help="buscar nuevas ediciones")
    s.add_argument("--source", nargs="*", help="claves de fuente (por defecto todas)")
    s.set_defaults(fn=cmd_discover)
    for name, fn, help_ in (("download", cmd_download, "descargar audio"),
                            ("transcribe", cmd_transcribe, "transcribir con Whisper")):
        s = sub.add_parser(name, help=help_)
        s.add_argument("--id", type=int, nargs="*")
        s.set_defaults(fn=fn)
    s = sub.add_parser("analyze", help="segmentar y analizar con el LLM")
    s.add_argument("--id", type=int, nargs="*")
    s.add_argument("--force", action="store_true", help="reanalizar piezas ya analizadas")
    s.set_defaults(fn=cmd_analyze)
    s = sub.add_parser("match", help="cruzar la misma noticia entre cadenas")
    s.add_argument("--date", nargs="*", help="YYYY-MM-DD")
    s.set_defaults(fn=cmd_match)
    s = sub.add_parser("run", help="pipeline completo de un día: telesesgo run [FECHA] [mañana|noche]",
                       description="Sin argumentos procesa el día de ayer completo. Ejemplos: "
                                   "`run 2026-11-16 noche`, `run noche`, `run 16/11/2026`, `run hoy mañana`.")
    s.add_argument("pos1", nargs="?", metavar="FECHA", help="AAAA-MM-DD, DD/MM/AAAA, hoy, ayer (por defecto: ayer)")
    s.add_argument("pos2", nargs="?", metavar="EDICION", help="mañana (= mediodía), noche o día (por defecto: día completo)")
    s.add_argument("--fecha", help="igual que FECHA")
    s.add_argument("--edicion", help="igual que EDICION")
    s.add_argument("--no-llm", "--sin-ia", dest="no_llm", action="store_true", help="solo descargar y transcribir")
    s.add_argument("--rehacer", action="store_true",
                   help="borrar el análisis de esas ediciones y volver a hacerlo (conserva audio y transcripción)")
    s.set_defaults(fn=cmd_run)
    s = sub.add_parser("notes", help="reescribir las observaciones automáticas: telesesgo notes FECHA [mañana|noche]")
    s.add_argument("fecha", nargs="?", help="AAAA-MM-DD (por defecto: ayer)")
    s.add_argument("edicion", nargs="?", help="mañana, noche o día (por defecto: las dos franjas)")
    s.set_defaults(fn=cmd_notes)
    s = sub.add_parser("voces", help="biblioteca de voces conocidas (presentadores, políticos...)")
    s.add_argument("--episodio", type=int, help="ver los hablantes identificados en un episodio")
    s.add_argument("--renombrar", nargs=2, metavar=("ACTUAL", "NUEVO"), help="corregir el nombre de una voz")
    s.add_argument("--borrar", metavar="NOMBRE", help="quitar una voz mal aprendida")
    s.set_defaults(fn=cmd_voces)
    s = sub.add_parser("hf-check", help="comprobar HF_TOKEN y el acceso al modelo de diarización")
    s.set_defaults(fn=cmd_hf_check)
    s = sub.add_parser("llm-check", help="comprobar la clave de Gemini y los modelos disponibles")
    s.set_defaults(fn=cmd_llm_check)
    s = sub.add_parser("status", help="estado de los episodios")
    s.add_argument("--limit", type=int, default=20)
    s.set_defaults(fn=cmd_status)
    s = sub.add_parser("html", help="generar los informes HTML (por defecto, todas las fechas con piezas)")
    s.add_argument("--date", nargs="*", help="YYYY-MM-DD")
    s.set_defaults(fn=cmd_html)
    s = sub.add_parser("timings", help="tiempos de cada paso del pipeline")
    s.add_argument("--date", help="YYYY-MM-DD (fecha de emisión)")
    s.add_argument("--limit", type=int, default=40)
    s.set_defaults(fn=cmd_timings)
    s = sub.add_parser("report", help="resumen comparativo de un día")
    s.add_argument("--date", nargs=1)
    s.set_defaults(fn=cmd_report)

    args = p.parse_args(argv)
    cfg_path = os.path.abspath(args.config)
    load_dotenv(os.path.join(os.path.dirname(cfg_path), ".env"))
    cfg = load_config(cfg_path)
    _setup_logging(cfg, args)
    conn = db.connect(cfg.db_path)
    rc = args.fn(cfg, conn, args)
    sys.exit(rc or 0)


def load_dotenv(path: str) -> None:
    """Carga KEY=VALOR de .env sin pisar variables ya definidas en el entorno."""
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.removeprefix("export ").partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


class _BriefFormatter(logging.Formatter):
    """En consola, sin trazas: el detalle completo queda en data/logs/AAAA-MM-DD.log."""

    def format(self, record):
        return f"  {record.levelname.lower()}: {record.getMessage()}"


def _setup_logging(cfg, args) -> None:
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    console = logging.StreamHandler(sys.stderr)
    if args.cmd == "run" and not args.verbose:
        # En `run` el progreso lo imprime el propio pipeline; en consola solo avisos y errores.
        console.setLevel(logging.WARNING)
        console.setFormatter(_BriefFormatter())
        # Los fallos de cada paso ya los muestra la línea de progreso (✗); la traza queda en el fichero de log.
        console.addFilter(lambda r: not (r.name == "telesesgo" and r.getMessage().startswith("Fallo en")))
    else:
        console.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.addHandler(console)
    logdir = cfg.data_dir / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(logdir / f"{datetime.now(MADRID).date().isoformat()}.log", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.addHandler(fh)
    for noisy in ("httpx", "httpx2", "httpcore", "google_genai", "google.genai", "urllib3", "faster_whisper"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


if __name__ == "__main__":
    main()
