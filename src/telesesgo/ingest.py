"""Ingesta: descubrir episodios nuevos y descargar solo la pista de audio."""

from __future__ import annotations

import logging
import shutil
import sqlite3
import subprocess
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path

from . import db
from .config import Config
from .sources import MADRID, discover

log = logging.getLogger(__name__)


def ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def discover_all(cfg: Config, conn: sqlite3.Connection, only: list[str] | None = None) -> int:
    cutoff = (datetime.now(MADRID) - timedelta(days=cfg.max_age_days)).date().isoformat()
    new = 0
    for src in cfg.sources:
        if not src.enabled or (only and src.key not in only):
            continue
        try:
            episodes = discover(src, cfg.discover_limit)
        except Exception as exc:  # una fuente caída no debe parar el resto
            log.error("Descubrimiento fallido en %s: %s", src.key, exc)
            continue
        for ep in episodes:
            if ep.air_date < cutoff:
                continue
            if _insert(conn, ep):
                new += 1
                log.info("Nuevo: [%s] %s %s", ep.source_key, ep.air_date, ep.title)
        conn.commit()
    return new


def _insert(conn: sqlite3.Connection, ep) -> bool:
    cur = conn.execute(
        "INSERT OR IGNORE INTO episodes (source_key, channel, program, external_id, title, air_date, aired_at, url, duration_s)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (ep.source_key, ep.channel, ep.program, ep.external_id, ep.title, ep.air_date,
         ep.aired_at.isoformat(), ep.url, ep.duration_s),
    )
    return bool(cur.rowcount)


def discover_date(cfg: Config, conn: sqlite3.Connection, day: date, editions: set[str] | None = None,
                  on_source=None) -> list[int]:
    """Busca las ediciones emitidas el día `day` (en las franjas `editions`, o todas) y devuelve sus ids.

    Cuanto más antigua es la fecha, más episodios recientes hay que recorrer en cada fuente.
    `on_source(source, encontrados, error)` informa del resultado de cada fuente.
    """
    target = day.isoformat()
    days_back = max(0, (datetime.now(MADRID).date() - day).days)
    limit = min(40, max(cfg.discover_limit, 2 * days_back + 3))
    keys = []
    for src in cfg.sources:
        if not src.enabled or (editions and src.edition not in editions):
            continue
        keys.append(src.key)
        try:
            found = [ep for ep in discover(src, limit) if ep.air_date == target]
        except Exception as exc:  # una fuente caída no debe parar el resto
            log.error("Descubrimiento fallido en %s: %s", src.key, exc)
            if on_source:
                on_source(src, 0, str(exc))
            continue
        for ep in found:
            if _insert(conn, ep):
                log.info("Nuevo: [%s] %s %s", ep.source_key, ep.air_date, ep.title)
        conn.commit()
        if on_source:
            on_source(src, len(found), None)
    if not keys:
        return []
    marks = ",".join("?" * len(keys))
    return [r[0] for r in conn.execute(
        f"SELECT id FROM episodes WHERE air_date = ? AND source_key IN ({marks}) ORDER BY aired_at", [target, *keys])]


def staging_dir(cfg: Config) -> Path:
    base = Path(cfg.download.staging_dir) if cfg.download.staging_dir else Path(tempfile.gettempdir()) / "telesesgo"
    base.mkdir(parents=True, exist_ok=True)
    return base


class _YtdlpLogger:
    """Redirige los mensajes de yt-dlp al log (el error ya llega como excepción al pipeline)."""

    def debug(self, msg):
        log.debug(msg)

    info = debug

    def warning(self, msg):
        log.debug("yt-dlp: %s", msg)

    def error(self, msg):
        log.info("yt-dlp: %s", msg)


def download_episode(cfg: Config, conn: sqlite3.Connection, ep: sqlite3.Row) -> Path:
    """Descarga y recodifica en una carpeta temporal local y mueve el resultado a data/audio de una sola vez."""
    import yt_dlp

    out_dir = cfg.audio_dir / ep["source_key"]
    out_dir.mkdir(parents=True, exist_ok=True)
    work = staging_dir(cfg) / ep["source_key"]
    work.mkdir(parents=True, exist_ok=True)
    stem = f"{ep['air_date']}_{ep['external_id']}"
    opts = {
        "format": cfg.download.format,
        "outtmpl": str(work / f"{stem}.raw.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "ffmpeg_location": ffmpeg_exe(),
        "retries": 5,
        "fragment_retries": 10,
        "concurrent_fragment_downloads": 4,
        "allow_unplayable_formats": False,  # nunca intentar contenidos con DRM
        # Certificados del sistema en vez del paquete certifi: tras un proxy corporativo con CA propia,
        # certifi no la conoce y falla con CERTIFICATE_VERIFY_FAILED.
        "compat_opts": {"no-certifi"},
        "logger": _YtdlpLogger(),
    }
    if cfg.download.cookies_file:
        opts["cookiefile"] = cfg.download.cookies_file

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(ep["url"], download=True)
        raw = Path(info["requested_downloads"][0]["filepath"])

    if cfg.download.transcode_bitrate:
        staged = work / f"{stem}.opus"
        cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(raw),
               "-vn", "-ac", "1", "-ar", "16000", "-c:a", "libopus", "-b:a", cfg.download.transcode_bitrate,
               "-application", "voip", str(staged)]
        subprocess.run(cmd, check=True)
        raw.unlink(missing_ok=True)
    else:
        staged = raw.with_name(raw.name.replace(".raw", ""))
        raw.rename(staged)
    final = out_dir / staged.name
    shutil.move(str(staged), str(final))

    duration = info.get("duration") or ep["duration_s"]
    db.set_status(conn, ep["id"], "downloaded", audio_path=str(final), duration_s=duration)
    log.info("Descargado %s (%.1f MB)", final.name, final.stat().st_size / 1e6)
    return final
