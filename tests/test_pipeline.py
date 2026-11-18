"""Tests sin red ni API: lógica de post-proceso y flujo completo con un LLM simulado."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from telesesgo import db
from telesesgo.analyze import analyze_episode, line_seconds, match_events, normalize_stories, segment_episode, verify_quote
from telesesgo.config import Config
from telesesgo.schemas import EventMatching, Segmentation, StoryAnalysis
from telesesgo.transcribe import Line

LINES = [
    Line(0, 0.0, 5.0, "Buenas tardes, comenzamos el Telediario."),
    Line(1, 5.0, 12.0, "El Gobierno defiende al ministro Marlaska tras las cargas policiales."),
    Line(2, 12.0, 20.0, "Tiene que caer mucha gente, empezando por el ministro Marlaska.", "SPEAKER_01"),
    Line(3, 20.0, 26.0, "Moncloa enmarca esta petición en la campaña electoral y cierra filas."),
    Line(4, 26.0, 33.0, "Y en deportes, el Real Madrid gana su partido."),
]


def test_schemas_are_api_compatible():
    import json

    from telesesgo.llm import gemini_schema
    for model in (Segmentation, StoryAnalysis, EventMatching):
        schema = gemini_schema(model)
        assert schema["type"] == "object"
        assert "$ref" not in json.dumps(schema) and "$defs" not in schema  # referencias resueltas en línea


def test_normalize_stories_removes_overlap_and_clamps():
    seg = Segmentation.model_validate({"stories": [
        _story(3, 10), _story(0, 3),
    ]})
    out = normalize_stories(seg.stories, n_lines=5)
    assert [(s.start_line, s.end_line) for s in out] == [(0, 3), (4, 4)]


def test_verify_quote_tolerates_punctuation_and_accents():
    ok, score = verify_quote("tiene que caer mucha gente empezando por el ministro marlaska", LINES, 2, 1, 3)
    assert ok and score >= 95
    ok, _ = verify_quote("el ministro dimitirá mañana por la tarde", LINES, 2, 1, 3)
    assert not ok


def test_line_seconds_ignores_out_of_range_and_duplicates():
    assert line_seconds(LINES, [2, 2, 4], lo=1, hi=3) == 8.0


# --- Flujo completo con LLM simulado ----------------------------------------

def _story(a, b, section="politica_nacional", fmt="pieza_editada", headline="x"):
    return {"start_line": a, "end_line": b, "headline": headline, "summary": "s", "section": section,
            "story_format": fmt, "tags": ["t"], "is_political": section != "deportes"}


def _dim(score, quote=None, line=None):
    ev = [{"quote": quote, "line": line, "speaker": "presentador", "voice": "medio", "note": "n"}] if quote else []
    return {"score": score, "rationale": "r", "evidence": ev}


ANALYSIS = {
    "actors": [
        {"name": "Fernando Grande-Marlaska", "kind": "persona", "affiliation": "PSOE", "bloc": "gobierno",
         "role": "objeto_de_critica", "speaks_on_air": False, "voice_lines": [], "mention_lines": [1, 2, 9],
         "tone": -1, "tone_rationale": "r"},
        {"name": "Ione Belarra", "kind": "persona", "affiliation": "Podemos", "bloc": "socio_gobierno",
         "role": "fuente_con_voz", "speaks_on_air": True, "voice_lines": [2], "mention_lines": [],
         "tone": 0, "tone_rationale": "r"},
    ],
    "ideological_lean": _dim(0),
    "lean_confidence": "media",
    "tone_government": _dim(-1, "cierra filas", 3),
    "tone_opposition": _dim(None),
    "primary_frame": "conflicto",
    "secondary_frames": ["estrategia_juego_politico"],
    "frame_rationale": "r",
    "frame_evidence": [{"quote": "enmarca esta petición en la campaña electoral", "line": 3, "speaker": "presentador",
                        "voice": "medio", "note": "n"}],
    "source_balance": "desequilibrado",
    "missing_perspectives": ["sindicatos policiales"],
    "loaded_language": [{"quote": "frase inventada que no está", "line": 1, "speaker": "presentador", "voice": "medio",
                         "connotation": "negativa", "target": "Gobierno"}],
    "editorializing": False,
    "editorializing_evidence": [],
    "selection_notes": "",
}


class FakeLlm:
    def __init__(self, story_ids_for_match=None):
        self.calls = []
        self.match_ids = story_ids_for_match

    def structured(self, *, system, content, output, max_tokens=64000):
        self.calls.append(output.__name__)
        if output is Segmentation:
            data = {"stories": [_story(0, 0, fmt="sumario_titulares"), _story(1, 3), _story(4, 4, section="deportes")]}
        elif output is StoryAnalysis:
            data = ANALYSIS
        else:
            data = {"events": [{"label": "Cargas policiales", "section": "politica_nacional", "story_ids": self.match_ids}]}
        msg = SimpleNamespace(model="fake", usage=SimpleNamespace(input_tokens=1, output_tokens=1), stop_reason="end_turn")
        return output.model_validate(data), msg


@pytest.fixture
def env(tmp_path):
    cfg = Config(data_dir=tmp_path, db_path=tmp_path / "t.db", political_context="ctx")
    conn = db.connect(cfg.db_path)
    for i, (key, ch) in enumerate((("rtve_td1", "La 1"), ("a3", "Antena 3")), start=1):
        conn.execute("INSERT INTO episodes (id, source_key, channel, program, external_id, air_date, url, status)"
                     " VALUES (?, ?, ?, ?, ?, '2026-11-12', 'u', 'transcribed')", (i, key, ch, f"P{i}", f"x{i}"))
        conn.executemany("INSERT INTO transcript_lines VALUES (?, ?, ?, ?, ?, ?)",
                         [(i, l.idx, l.start, l.end, l.speaker, l.text) for l in LINES])
    conn.commit()
    return cfg, conn


def test_end_to_end_with_fake_llm(env):
    cfg, conn = env
    llm = FakeLlm()
    for eid in (1, 2):
        ep = conn.execute("SELECT * FROM episodes WHERE id = ?", (eid,)).fetchone()
        assert segment_episode(cfg, conn, llm, ep) == 3
        ep = conn.execute("SELECT * FROM episodes WHERE id = ?", (eid,)).fetchone()
        # Solo la pieza política se analiza: se omiten sumario y deportes.
        assert analyze_episode(cfg, conn, llm, ep) == 1

    st = conn.execute("SELECT * FROM stories WHERE episode_id = 1 AND ord = 2").fetchone()
    assert st["duration_s"] == 21.0 and st["tone_government"] == -1 and st["tone_opposition"] is None

    ev = {r["quote"]: r["verified"] for r in conn.execute("SELECT * FROM evidence WHERE story_id = ?", (st["id"],))}
    assert ev["cierra filas"] == 1 and ev["frase inventada que no está"] == 0

    exp = {r["actor"]: r for r in conn.execute("SELECT * FROM v_actor_exposure WHERE channel = 'La 1'")}
    assert exp["Ione Belarra"]["voice_s"] == 8.0
    assert exp["Fernando Grande-Marlaska"]["mentions"] == 2  # la línea 9 está fuera de la pieza

    political = [r["id"] for r in conn.execute("SELECT id FROM stories WHERE ord = 2")]
    llm.match_ids = political + [9999]  # id inexistente: se descarta
    assert match_events(cfg, conn, llm, "2026-11-12") == 1
    rows = conn.execute("SELECT channel FROM v_event_comparison").fetchall()
    assert sorted(r["channel"] for r in rows) == ["Antena 3", "La 1"]


# --- Informe HTML --------------------------------------------------------------

def test_html_report_after_match(env):
    from telesesgo.htmlreport import build_day_report, notes_path

    cfg, conn = env
    llm = FakeLlm()
    for eid in (1, 2):
        ep = conn.execute("SELECT * FROM episodes WHERE id = ?", (eid,)).fetchone()
        segment_episode(cfg, conn, llm, ep)
        ep = conn.execute("SELECT * FROM episodes WHERE id = ?", (eid,)).fetchone()
        analyze_episode(cfg, conn, llm, ep)
    llm.match_ids = [r["id"] for r in conn.execute("SELECT id FROM stories WHERE ord = 2")]
    match_events(cfg, conn, llm, "2026-11-12")

    notes = notes_path(cfg, "2026-11-12")
    notes.parent.mkdir(parents=True, exist_ok=True)
    notes.write_text("## Nota\n- **Clave**: <script>x</script>\n", encoding="utf-8")

    out = build_day_report(cfg, conn, "2026-11-12")
    page = out.read_text(encoding="utf-8")
    assert "Cargas policiales" in page            # acontecimiento cruzado
    assert "«cierra filas»" in page                # cita verificada destacada
    assert "frase inventada" not in page           # las no verificadas no se muestran
    assert "<strong>Clave</strong>" in page and "<script>x" not in page  # notas escapadas
    assert (out.parent / "index.html").exists()


def test_markdown_to_html_escapes_and_formats():
    from telesesgo.htmlreport import markdown_to_html

    out = markdown_to_html("## Título\nTexto con *énfasis* y `código` <b>\n- uno\n- dos")
    assert "<h3>Título</h3>" in out and "<em>énfasis</em>" in out and "<code>código</code>" in out
    assert "&lt;b&gt;" in out and out.count("<li>") == 2



# --- Descargas en paralelo y tiempos ---------------------------------------------

def test_parallel_download_then_transcribe_records_timings(tmp_path, monkeypatch):
    import threading
    import time as _time

    from telesesgo import cli, ingest, transcribe

    cfg = Config(data_dir=tmp_path, db_path=tmp_path / "t.db", political_context="ctx")
    cfg.download.workers = 3
    conn = db.connect(cfg.db_path)
    for i in range(1, 6):
        conn.execute("INSERT INTO episodes (id, source_key, channel, program, external_id, air_date, url)"
                     " VALUES (?, 's', 'c', 'p', ?, '2026-11-10', 'u')", (i, f"x{i}"))
    conn.commit()

    active, peak, lock = [0], [0], threading.Lock()

    def fake_download(cfg_, tconn, ep):
        assert threading.current_thread().name.startswith("dl")  # en un hilo del pool
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        _time.sleep(0.2)
        with lock:
            active[0] -= 1
        if ep["id"] == 4:
            raise RuntimeError("CDN caído")
        db.set_status(tconn, ep["id"], "downloaded", audio_path=f"/a/{ep['id']}.opus")

    transcribed = []

    def fake_transcribe(cfg_, c, ep):
        assert threading.current_thread() is threading.main_thread()  # la GPU, solo en el hilo principal
        transcribed.append(ep["id"])
        db.set_status(c, ep["id"], "transcribed")

    monkeypatch.setattr(ingest, "download_episode", fake_download)
    monkeypatch.setattr(transcribe, "transcribe_episode", fake_transcribe)
    cli._download_and_transcribe(cfg, conn, None, transcribe=True)

    assert peak[0] == 3                                   # respeta el número de hilos
    assert sorted(transcribed) == [1, 2, 3, 5]            # el que falla no se transcribe
    st = dict(conn.execute("SELECT id, status FROM episodes").fetchall())
    assert st[4] == "discovered" and "CDN caído" in conn.execute("SELECT error FROM episodes WHERE id = 4").fetchone()[0]
    steps = conn.execute("SELECT step, count(*), sum(ok) FROM step_timings GROUP BY step").fetchall()
    assert {r[0]: (r[1], r[2]) for r in steps} == {"download": (5, 4), "transcribe": (4, 4)}


# --- Gemini (cliente simulado, sin red) -------------------------------------

class _FakeModels:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def generate_content(self, **kw):
        self.calls.append(kw)
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def _resp(text, finish="STOP"):
    return SimpleNamespace(
        text=text, prompt_feedback=None, model_version="gemini-test",
        candidates=[SimpleNamespace(finish_reason=SimpleNamespace(value=finish))],
        usage_metadata=SimpleNamespace(prompt_token_count=100, candidates_token_count=20, thoughts_token_count=5,
                                       cached_content_token_count=60),
    )


def test_gemini_structured_parses_and_maps_usage():
    from telesesgo.config import LlmConfig
    from telesesgo.llm import Llm
    models = _FakeModels([_resp('{"events": [{"label": "x", "section": "otros", "story_ids": [1, 2]}]}')])
    llm = Llm(LlmConfig(model="gemini-3.5-flash"), SimpleNamespace(models=models))
    out, meta = llm.structured(system="sys", content=[{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
                               output=EventMatching)
    assert out.events[0].story_ids == [1, 2]
    assert meta.model == "gemini-test" and meta.stop_reason == "STOP"
    assert (meta.usage.input_tokens, meta.usage.output_tokens, meta.usage.cache_read_input_tokens) == (100, 25, 60)
    call = models.calls[0]
    assert call["contents"] == ["a", "b"] and call["config"].system_instruction == "sys"
    assert call["config"].thinking_config.thinking_level is not None  # Gemini 3.x: thinking_level


def test_gemini_retries_quota_errors_then_fails_on_truncation(monkeypatch):
    from google.genai import errors

    from telesesgo import llm as llm_mod
    from telesesgo.config import LlmConfig
    monkeypatch.setattr(llm_mod.time, "sleep", lambda s: None)
    quota = errors.ClientError(429, {"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED"}})
    models = _FakeModels([quota, _resp("{}", finish="MAX_TOKENS")])
    llm = llm_mod.Llm(LlmConfig(model="gemini-2.5-flash", max_retries=2), SimpleNamespace(models=models))
    with pytest.raises(llm_mod.LlmError, match="truncada"):
        llm.structured(system="s", content=[{"type": "text", "text": "a"}], output=EventMatching)
    assert len(models.calls) == 2                                     # un reintento tras el 429
    assert models.calls[1]["config"].thinking_config.thinking_budget  # Gemini 2.x: thinking_budget


def test_gemini_does_not_retry_bad_request():
    from google.genai import errors

    from telesesgo.config import LlmConfig
    from telesesgo.llm import Llm, LlmError
    bad = errors.ClientError(400, {"error": {"code": 400, "message": "bad schema", "status": "INVALID_ARGUMENT"}})
    models = _FakeModels([bad])
    with pytest.raises(LlmError, match="400"):
        Llm(LlmConfig(), SimpleNamespace(models=models)).structured(
            system="s", content=[{"type": "text", "text": "a"}], output=EventMatching)
    assert len(models.calls) == 1


# --- Fechas, franjas y descubrimiento por fecha -----------------------------

def test_parse_date_and_edition():
    from datetime import date

    from telesesgo.pipeline import parse_date, parse_edition, resolve_args
    today = date(2026, 11, 17)
    assert parse_date(None, today) == date(2026, 11, 16)
    assert parse_date("hoy", today) == today
    assert parse_date("16/11/2026", today) == parse_date("2026-11-16", today) == date(2026, 11, 16)
    assert parse_edition(None) is None and parse_edition("día") is None
    assert parse_edition("mañana") == parse_edition("Mediodía") == {"mediodia"}
    assert parse_edition("noche") == {"noche"}
    with pytest.raises(ValueError):
        parse_edition("madrugada")
    with pytest.raises(ValueError):
        parse_date("16 de noviembre")
    d, ed = resolve_args("noche", None)
    assert ed == {"noche"}
    d2, ed2 = resolve_args("noche", "2026-11-13")  # en cualquier orden
    assert (d2, ed2) == (date(2026, 11, 13), {"noche"})
    assert resolve_args("2026-11-13", None) == (date(2026, 11, 13), None)


def test_discover_date_filters_by_day_and_edition(tmp_path, monkeypatch):
    from datetime import date, datetime

    from telesesgo import ingest
    from telesesgo.config import SourceConfig
    from telesesgo.sources import MADRID, Episode

    cfg = Config(data_dir=tmp_path, db_path=tmp_path / "t.db", sources=[
        SourceConfig(key="td1", channel="La 1", program="TD1", type="rtve", program_id=1, edition="mediodia"),
        SourceConfig(key="td2", channel="La 1", program="TD2", type="rtve", program_id=2, edition="noche"),
    ])
    conn = db.connect(cfg.db_path)
    seen_limits = []

    def fake_discover(src, limit):
        seen_limits.append(limit)
        return [Episode(src.key, src.channel, src.program, f"{src.key}-{d}", "t", "u",
                        datetime(2026, 11, d, 21 if src.edition == "noche" else 15, tzinfo=MADRID), 1800.0)
                for d in (16, 15, 13)]

    monkeypatch.setattr(ingest, "discover", fake_discover)
    ids = ingest.discover_date(cfg, conn, date(2026, 11, 13), {"noche"})
    rows = conn.execute("SELECT source_key, air_date FROM episodes").fetchall()
    assert [tuple(r) for r in rows] == [("td2", "2026-11-13")] and len(ids) == 1  # solo ese día y esa franja
    assert len(seen_limits) == 1 and seen_limits[0] >= 3
    both = ingest.discover_date(cfg, conn, date(2026, 11, 13), None)
    assert len(both) == 2


def test_download_stages_locally_then_moves(tmp_path, monkeypatch):
    import sys as _sys

    from telesesgo import ingest
    cfg = Config(data_dir=tmp_path / "data", db_path=tmp_path / "t.db")
    cfg.download.staging_dir = str(tmp_path / "staging")
    cfg.download.transcode_bitrate = None
    conn = db.connect(cfg.db_path)
    conn.execute("INSERT INTO episodes (id, source_key, channel, program, external_id, air_date, url)"
                 " VALUES (1, 's', 'c', 'p', 'x1', '2026-11-16', 'u')")
    conn.commit()
    written = []

    class FakeYDL:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download):
            path = self.opts["outtmpl"].replace("%(ext)s", "m4a")
            written.append(path)
            open(path, "wb").write(b"audio")
            return {"requested_downloads": [{"filepath": path}], "duration": 60}

    monkeypatch.setitem(_sys.modules, "yt_dlp", SimpleNamespace(YoutubeDL=FakeYDL))
    ep = conn.execute("SELECT * FROM episodes WHERE id = 1").fetchone()
    final = ingest.download_episode(cfg, conn, ep)
    assert written[0].startswith(str(tmp_path / "staging"))        # se descarga en la carpeta temporal
    assert final.parent == cfg.audio_dir / "s" and final.read_bytes() == b"audio"
    assert not list((tmp_path / "staging").rglob("*.m4a"))           # y no deja restos
    assert conn.execute("SELECT status FROM episodes WHERE id = 1").fetchone()[0] == "downloaded"


def test_analyze_skips_non_political_when_configured(env):
    cfg, conn = env
    cfg.llm.analyze_non_political = False
    llm = FakeLlm()
    ep = conn.execute("SELECT * FROM episodes WHERE id = 1").fetchone()
    segment_episode(cfg, conn, llm, ep)
    conn.execute("UPDATE stories SET is_political = 0 WHERE episode_id = 1")
    conn.commit()
    ep = conn.execute("SELECT * FROM episodes WHERE id = 1").fetchone()
    progress = []
    assert analyze_episode(cfg, conn, llm, ep, on_progress=lambda d, t: progress.append((d, t))) == 0
    assert conn.execute("SELECT status FROM episodes WHERE id = 1").fetchone()[0] == "analyzed"
    cfg.llm.analyze_non_political = True
    assert analyze_episode(cfg, conn, llm, ep, force=True, on_progress=lambda d, t: progress.append((d, t))) == 1
    assert progress == [(1, 1)]


def test_gemini_switches_to_fallback_on_overload(monkeypatch):
    from google.genai import errors

    from telesesgo import llm as llm_mod
    from telesesgo.config import LlmConfig
    monkeypatch.setattr(llm_mod.time, "sleep", lambda s: None)
    busy = lambda: errors.ServerError(503, {"error": {"code": 503, "message": "The model is overloaded.", "status": "UNAVAILABLE"}})
    ok = '{"events": []}'
    models = _FakeModels([busy(), busy(), _resp(ok), _resp(ok)])
    llm = llm_mod.Llm(LlmConfig(model="gemini-a", fallback_models=["gemini-2.5-b"], overload_retries=2),
                      SimpleNamespace(models=models))
    for _ in range(2):
        llm.structured(system="s", content=[{"type": "text", "text": "a"}], output=EventMatching)
    assert [c["model"] for c in models.calls] == ["gemini-a", "gemini-a", "gemini-2.5-b", "gemini-2.5-b"]  # se queda en la reserva
    assert models.calls[2]["config"].thinking_config.thinking_budget  # config propia del modelo de reserva


def test_edition_notes_dossier_quote_check_and_report(env):
    from telesesgo.htmlreport import build_day_report
    from telesesgo.notes import build_dossier, write_edition_notes
    from telesesgo.schemas import EditionNotes

    cfg, conn = env
    conn.execute("UPDATE episodes SET aired_at = '2026-11-12T21:00:00+01:00'")
    llm = FakeLlm(story_ids_for_match=[2, 5])
    for eid in (1, 2):
        ep = conn.execute("SELECT * FROM episodes WHERE id = ?", (eid,)).fetchone()
        segment_episode(cfg, conn, llm, ep)
        analyze_episode(cfg, conn, llm, conn.execute("SELECT * FROM episodes WHERE id = ?", (eid,)).fetchone())
    match_events(cfg, conn, llm, "2026-11-12")

    dossier, allowed = build_dossier(cfg, conn, "2026-11-12", "Noche")
    assert "MÉTRICAS POR PROGRAMA" in dossier and "Cargas policiales" in dossier and "P1" in dossier and "P2" in dossier
    assert allowed  # citas verificadas disponibles para el modelo

    good = allowed[0]
    data = {"title": "Marlaska en el centro", "findings": [
        {"lead": "Ambos abren con Marlaska", "text": f"Las dos cadenas citan «{good}»."},
        {"lead": "Cita inventada", "text": "Un programa dice «esto no lo ha dicho nunca nadie en antena»."}],
        "same_story": [], "caveats": ["Solo dos ediciones."]}

    class NotesLlm:
        def structured(self, *, system, content, output, max_tokens=64000):
            assert output is EditionNotes and "SOLO lo que hay en el dossier" in system
            return output.model_validate(data), SimpleNamespace(model="gemini-test", usage=None, stop_reason="STOP")

    path = write_edition_notes(cfg, conn, NotesLlm(), "2026-11-12", "Noche")
    text = path.read_text(encoding="utf-8")
    assert path.name == "2026-11-12-noche.md" and text.startswith("## Noche: Marlaska en el centro")
    assert f"«{good}»" in text and f"«{good}» *(cita no verificada)*" not in text
    assert "«esto no lo ha dicho nunca nadie en antena» *(cita no verificada)*" in text
    assert write_edition_notes(cfg, conn, NotesLlm(), "2026-11-12", "Mediodía") is None  # franja sin programas

    html = build_day_report(cfg, conn, "2026-11-12").read_text(encoding="utf-8")
    assert "Observaciones · Noche" in html and "Marlaska en el centro" in html


# --- Quién habla: atribución medio/fuente, diarización y biblioteca de voces ----------

def _q(quote, line, speaker="presentador", voice="medio"):
    return {"quote": quote, "line": line, "speaker": speaker, "voice": voice, "note": "n"}


def test_source_quotes_are_not_editorializing_of_the_program(env):
    """Si el modelo atribuye al medio una frase que dice un actor con voz propia, se corrige a «fuente»;
    y si toda la «opinión propia» es de fuentes, la pieza no cuenta como editorializada."""
    import copy

    cfg, conn = env
    llm = FakeLlm()
    ep = conn.execute("SELECT * FROM episodes WHERE id = 1").fetchone()
    segment_episode(cfg, conn, llm, ep)
    data = copy.deepcopy(ANALYSIS)
    # línea 2: la dice Ione Belarra (voice_lines=[2]); el modelo se equivoca y dice que es del medio
    data["loaded_language"] = [{"quote": "Tiene que caer mucha gente", "line": 2, "speaker": "presentador", "voice": "medio",
                                "connotation": "negativa", "target": "Marlaska"}]
    data["editorializing"] = True
    data["editorializing_evidence"] = [_q("empezando por el ministro Marlaska", 2)]

    class L(FakeLlm):
        def structured(self, *, system, content, output, max_tokens=64000):
            if output is StoryAnalysis:
                return output.model_validate(data), SimpleNamespace(model="m", usage=None, stop_reason="STOP")
            return super().structured(system=system, content=content, output=output)

    analyze_episode(cfg, conn, L(), conn.execute("SELECT * FROM episodes WHERE id = 1").fetchone())
    rows = {r["dimension"]: r for r in conn.execute("SELECT * FROM evidence WHERE dimension IN ('loaded_language', 'editorializing')")}
    assert rows["loaded_language"]["voice"] == "fuente" and rows["loaded_language"]["speaker"] == "Ione Belarra"
    assert rows["loaded_language"]["voice_source"] == "actores"
    assert conn.execute("SELECT editorializing FROM stories WHERE episode_id = 1 AND ord = 2").fetchone()[0] == 0


def _speakers(conn, ep_id, rows):
    from telesesgo import voices
    voices.store_episode_speakers(conn, ep_id, {lbl: (secs, emb) for lbl, secs, emb in rows})


def test_diarization_roles_drive_attribution(env):
    import copy

    from telesesgo import voices
    cfg, conn = env
    # línea 1 y 3: presentador (SPEAKER_00); línea 2: una fuente (SPEAKER_01)
    conn.executemany("UPDATE transcript_lines SET speaker = ? WHERE episode_id = 1 AND idx = ?",
                     [("SPEAKER_00", 0), ("SPEAKER_00", 1), ("SPEAKER_01", 2), ("SPEAKER_00", 3), ("SPEAKER_00", 4)])
    _speakers(conn, 1, [("SPEAKER_00", 30.0, None), ("SPEAKER_01", 8.0, None)])
    assert voices.speaker_map(conn, 1)["SPEAKER_00"]["role"] == "presentador"  # el que más habla
    from telesesgo.schemas import SpeakerId
    voices.apply_llm_ids(conn, 1, [SpeakerId(label="H01", role="fuente", name="Ione Belarra", evidence_line=2)])
    legend = voices.legend(conn, 1)
    assert "H00: sin identificar · presentador" in legend and "H01: Ione Belarra · fuente" in legend

    data = copy.deepcopy(ANALYSIS)
    data["actors"] = []  # sin pistas de actores: solo cuenta la diarización
    data["loaded_language"] = [
        {"quote": "Tiene que caer mucha gente", "line": 2, "speaker": "presentador", "voice": "medio",
         "connotation": "negativa", "target": "Marlaska"},
        {"quote": "cierra filas", "line": 3, "speaker": "Ione Belarra", "voice": "fuente",
         "connotation": "negativa", "target": "Moncloa"}]

    class L(FakeLlm):
        def structured(self, *, system, content, output, max_tokens=64000):
            if output is StoryAnalysis:
                assert "H01: Ione Belarra" in content[0]["text"] and "] 00:12 H01| Tiene que caer" in content[0]["text"]
                return output.model_validate(data), SimpleNamespace(model="m", usage=None, stop_reason="STOP")
            return super().structured(system=system, content=content, output=output)

    ep = conn.execute("SELECT * FROM episodes WHERE id = 1").fetchone()
    segment_episode(cfg, conn, FakeLlm(), ep)
    analyze_episode(cfg, conn, L(), conn.execute("SELECT * FROM episodes WHERE id = 1").fetchone())
    got = {r["quote"]: (r["speaker"], r["voice"], r["voice_source"])
           for r in conn.execute("SELECT * FROM evidence WHERE dimension = 'loaded_language'")}
    assert got["Tiene que caer mucha gente"] == ("Ione Belarra", "fuente", "diarizacion")
    assert got["cierra filas"] == ("presentador", "medio", "diarizacion")  # la línea 3 es del presentador


def test_voice_library_learns_recognizes_and_is_not_poisoned(env):
    import numpy as np

    from telesesgo import voices
    from telesesgo.schemas import SpeakerId
    cfg, conn = env
    rng = np.random.default_rng(0)
    anchor, pol, other = (rng.normal(size=256) for _ in range(3))
    noisy = lambda v: v + rng.normal(scale=0.15, size=256)

    # Episodio 1: Gemini identifica al presentador y a un político; se aprenden
    _speakers(conn, 1, [("SPEAKER_00", 900.0, noisy(anchor)), ("SPEAKER_01", 40.0, noisy(pol)), ("SPEAKER_02", 5.0, noisy(other))])
    voices.apply_llm_ids(conn, 1, [SpeakerId(label="H00", role="presentador", name="Rodrigo Blázquez", evidence_line=0),
                                   SpeakerId(label="H01", role="fuente", name="Alberto Núñez Feijóo", evidence_line=5),
                                   SpeakerId(label="H02", role="fuente", name="Vecina", evidence_line=9)])
    assert voices.learn(cfg, conn, 1, "laSexta") == 2  # la vecina habla 5 s: no se aprende
    lib = {r["name"]: r for r in conn.execute("SELECT * FROM voices")}
    assert set(lib) == {"Rodrigo Blázquez", "Alberto Núñez Feijóo"} and lib["Rodrigo Blázquez"]["channel"] == "laSexta"
    assert lib["Alberto Núñez Feijóo"]["channel"] is None  # los políticos no pertenecen a una cadena

    # Episodio 2: se reconocen por la voz aunque nadie diga su nombre; el desconocido no
    _speakers(conn, 2, [("SPEAKER_00", 30.0, noisy(pol)), ("SPEAKER_01", 800.0, noisy(anchor)), ("SPEAKER_02", 30.0, noisy(other))])
    assert voices.match_library(cfg, conn, 2) == 2
    m = voices.speaker_map(conn, 2)
    assert (m["SPEAKER_01"]["name"], m["SPEAKER_01"]["role"], m["SPEAKER_01"]["id_source"]) == ("Rodrigo Blázquez", "presentador", "voz")
    assert m["SPEAKER_00"]["name"] == "Alberto Núñez Feijóo" and m["SPEAKER_02"]["name"] is None
    # Gemini contradice a la voz: se mantiene la voz
    voices.apply_llm_ids(conn, 2, [SpeakerId(label="H01", role="presentador", name="Otra Persona", evidence_line=0)])
    assert voices.speaker_map(conn, 2)["SPEAKER_01"]["name"] == "Rodrigo Blázquez"
    assert voices.learn(cfg, conn, 2) == 2
    assert conn.execute("SELECT samples FROM voices WHERE name = 'Rodrigo Blázquez'").fetchone()[0] == 2

    # Una voz distinta etiquetada con el mismo nombre (sustituto un fin de semana) no contamina la huella guardada
    before = conn.execute("SELECT embedding FROM voices WHERE name = 'Rodrigo Blázquez'").fetchone()[0]
    conn.execute("INSERT INTO episodes (id, source_key, channel, program, external_id, air_date, url) VALUES (3, 'x', 'c', 'p', 'x3', '2026-11-13', 'u')")
    _speakers(conn, 3, [("SPEAKER_00", 900.0, noisy(other))])
    voices.apply_llm_ids(conn, 3, [SpeakerId(label="H00", role="presentador", name="Rodrigo Blázquez", evidence_line=0)])
    assert voices.learn(cfg, conn, 3) == 0
    assert conn.execute("SELECT embedding FROM voices WHERE name = 'Rodrigo Blázquez'").fetchone()[0] == before


def test_diarize_assigns_speakers_and_embeddings(tmp_path, monkeypatch):
    import numpy as np

    from telesesgo import transcribe as tr
    cfg = Config(data_dir=tmp_path, db_path=tmp_path / "t.db")
    cfg.stt.diarization.enabled = True

    class Seg:
        def __init__(self, a, b):
            self.start, self.end = a, b

    class Ann:
        def __init__(self, turns):
            self.turns = turns

        def itertracks(self, yield_label=True):
            return [(Seg(a, b), None, spk) for a, b, spk in self.turns]

        def labels(self):
            return sorted({spk for *_, spk in self.turns})

    ann = Ann([(0, 12, "SPEAKER_00"), (12, 20, "SPEAKER_01"), (20, 33, "SPEAKER_00")])
    out = SimpleNamespace(speaker_diarization=ann, exclusive_speaker_diarization=ann,
                          speaker_embeddings=np.array([[1.0, 0, 0], [0, 1.0, 0]]))
    moves = []

    class Pipe:
        def __call__(self, inp):
            moves.append(("run", bool(tr._model_cache)))
            return out

        def to(self, device):
            moves.append(("to", str(device)))

    tr._model_cache["whisper"] = object()  # Whisper cargado de una transcripción anterior
    monkeypatch.setattr(tr, "_diarization_pipeline", lambda dc: Pipe())
    monkeypatch.setattr(tr, "_diar_device", lambda dc: "cuda")
    import torch
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: moves.append(("empty", None)))
    import faster_whisper
    monkeypatch.setattr(faster_whisper, "decode_audio", lambda path, sampling_rate: np.zeros(16000, dtype=np.float32))
    lines = [Line(l.idx, l.start, l.end, l.text) for l in LINES]
    speakers = tr.diarize(tmp_path / "a.opus", lines, cfg.stt.diarization)
    assert [l.speaker for l in lines] == ["SPEAKER_00", "SPEAKER_00", "SPEAKER_01", "SPEAKER_00", "SPEAKER_00"]
    assert speakers["SPEAKER_01"][0] == 8.0 and list(speakers["SPEAKER_01"][1]) == [0, 1.0, 0]
    # GPU en exclusiva: Whisper liberado antes, diarización a la GPU, y de vuelta a la CPU al terminar
    assert moves == [("to", "cuda"), ("run", False), ("to", "cpu"), ("empty", None)]
    assert not tr._model_cache


def test_reset_analysis_keeps_transcripts(env):
    from telesesgo.pipeline import reset_analysis
    cfg, conn = env
    llm = FakeLlm(story_ids_for_match=[2, 5])
    for eid in (1, 2):
        segment_episode(cfg, conn, llm, conn.execute("SELECT * FROM episodes WHERE id = ?", (eid,)).fetchone())
        analyze_episode(cfg, conn, llm, conn.execute("SELECT * FROM episodes WHERE id = ?", (eid,)).fetchone())
    match_events(cfg, conn, llm, "2026-11-12")
    assert reset_analysis(conn, [1, 2]) == 2
    assert conn.execute("SELECT count(*) FROM stories").fetchone()[0] == 0
    assert conn.execute("SELECT count(*) FROM events").fetchone()[0] == 0
    assert conn.execute("SELECT count(*) FROM transcript_lines").fetchone()[0] == 2 * len(LINES)
    assert {r[0] for r in conn.execute("SELECT status FROM episodes")} == {"transcribed"}


def test_downloads_start_before_pending_diarization(tmp_path, monkeypatch):
    import threading
    import time as _time

    from telesesgo import cli, ingest
    cfg = Config(data_dir=tmp_path, db_path=tmp_path / "t.db")
    conn = db.connect(cfg.db_path)
    conn.execute("INSERT INTO episodes (id, source_key, channel, program, external_id, air_date, url) VALUES (1, 's', 'c', 'p', 'x1', '2026-11-16', 'u')")
    conn.commit()
    started = threading.Event()
    seen = []

    def fake_download(cfg_, tconn, ep):
        started.set()
        _time.sleep(0.2)
        db.set_status(tconn, ep["id"], "downloaded", audio_path="/a.opus")

    monkeypatch.setattr(ingest, "download_episode", fake_download)
    monkeypatch.setattr(cli, "diarize_one", lambda *a, **k: True)
    cli._download_and_transcribe(cfg, conn, [1], transcribe=False,
                                 while_downloading=lambda: seen.append(started.wait(2)))
    assert seen == [True]  # la descarga ya estaba en marcha mientras se hacía el trabajo de GPU pendiente
