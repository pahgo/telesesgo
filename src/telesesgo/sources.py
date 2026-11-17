"""Descubrimiento de las últimas ediciones completas de cada informativo.

Solo se usan endpoints públicos que también consume la web de cada cadena.
Los contenidos con DRM se descartan: no se intenta eludir ninguna protección.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

from .config import SourceConfig

log = logging.getLogger(__name__)
MADRID = ZoneInfo("Europe/Madrid")
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130 Safari/537.36"


@dataclass
class Episode:
    source_key: str
    channel: str
    program: str
    external_id: str
    title: str
    url: str
    aired_at: datetime
    duration_s: float | None

    @property
    def air_date(self) -> str:
        return self.aired_at.astimezone(MADRID).date().isoformat()


def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


# --- RTVE --------------------------------------------------------------------

def discover_rtve(src: SourceConfig, limit: int) -> list[Episode]:
    data = _get_json(f"https://www.rtve.es/api/programas/{src.program_id}/videos.json?size=60&page=1")
    out: list[Episode] = []
    for it in data["page"]["items"]:
        kind = (it.get("type") or {}).get("name")
        if kind != "Completo":
            continue  # fragmentos, resúmenes "en 4 minutos", etc.
        if it.get("hasDRM") or it.get("paidContent"):
            log.info("RTVE %s: %s con DRM/pago, se omite", src.key, it["id"])
            continue
        aired = datetime.strptime(it["dateOfEmission"], "%d-%m-%Y %H:%M:%S").replace(tzinfo=MADRID)
        out.append(
            Episode(
                source_key=src.key,
                channel=src.channel,
                program=src.program,
                external_id=str(it["id"]),
                title=it.get("longTitle") or it["title"],
                url=it["htmlUrl"],
                aired_at=aired,
                duration_s=(it.get("duration") or 0) / 1000 or None,
            )
        )
        if len(out) >= limit:
            break
    return out


# --- Atresplayer (Antena 3, laSexta) -----------------------------------------

ATRES_API = "https://api.atresplayer.com/client/v1"


def discover_atresplayer(src: SourceConfig, limit: int) -> list[Episode]:
    rows = _get_json(
        f"{ATRES_API}/row/search?entityType=ATPEpisode&formatId={src.format_id}&size={max(limit, 3)}&sortType=THE_LAST"
    )
    out: list[Episode] = []
    for it in rows.get("itemRows", [])[:limit]:
        if it.get("visibility") not in (None, "FREE"):
            log.info("Atresplayer %s: %s no es gratuito (%s), se omite", src.key, it["contentId"], it.get("visibility"))
            continue
        page = _get_json(it["link"]["href"])
        pub = page.get("publicationDate")
        aired = datetime.fromtimestamp(pub / 1000, tz=MADRID) if pub else _date_from_title(it["title"])
        out.append(
            Episode(
                source_key=src.key,
                channel=src.channel,
                program=src.program,
                external_id=it["contentId"],
                title=page.get("title") or it["title"],
                url="https://www.atresplayer.com" + it["link"]["url"],
                aired_at=aired,
                duration_s=page.get("duration"),
            )
        )
    return out


def _date_from_title(title: str) -> datetime:
    m = re.search(r"\((\d{2})-(\d{2})-(\d{2})\)", title)
    if not m:
        return datetime.now(MADRID)
    d, mth, y = map(int, m.groups())
    return datetime(2000 + y, mth, d, 15, 0, tzinfo=MADRID)


# --- Genérico vía yt-dlp -----------------------------------------------------

def discover_ytdlp(src: SourceConfig, limit: int) -> list[Episode]:
    import yt_dlp

    opts = {"extract_flat": "in_playlist", "playlistend": max(limit * 10, 20), "quiet": True, "skip_download": True,
            "compat_opts": {"no-certifi"}}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(src.url, download=False)
    pattern = re.compile(src.title_regex) if src.title_regex else None
    out: list[Episode] = []
    for e in info.get("entries") or []:
        title = e.get("title") or ""
        if pattern and not pattern.search(title):
            continue
        if e.get("duration") and e["duration"] < src.min_duration_s:
            continue
        ts = e.get("timestamp") or e.get("release_timestamp")
        if ts:
            aired = datetime.fromtimestamp(ts, tz=MADRID)
        elif e.get("upload_date"):
            aired = datetime.strptime(e["upload_date"], "%Y%m%d").replace(tzinfo=MADRID)
        else:
            aired = datetime.now(MADRID)
        out.append(
            Episode(
                source_key=src.key,
                channel=src.channel,
                program=src.program,
                external_id=str(e.get("id")),
                title=title,
                url=e.get("url") or e.get("webpage_url"),
                aired_at=aired,
                duration_s=e.get("duration"),
            )
        )
        if len(out) >= limit:
            break
    return out


DISCOVERERS = {"rtve": discover_rtve, "atresplayer": discover_atresplayer, "ytdlp": discover_ytdlp}


def discover(src: SourceConfig, limit: int) -> list[Episode]:
    return DISCOVERERS[src.type](src, limit)
