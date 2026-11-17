# telesesgo

Pipeline ETL local para analizar el encuadre y el sesgo editorial de los telediarios españoles:

```
discover ─► download (solo audio) ─► transcribe (faster-whisper) ─► segment (Gemini) ─► analyze (Gemini) ─► match (Gemini)
   API pública       yt-dlp + ffmpeg         local, CPU/GPU            piezas              sesgo por pieza      misma noticia
   de cada cadena    → Opus 16 kHz mono      + diarización opcional                                              entre cadenas
                                                 └──────────────────────── SQLite (data/telesesgo.db) ─────────────────────┘
```

**Uso diario: un solo comando** (en Linux/WSL, desde la carpeta del proyecto):

```bash
./scripts/telesesgo run                     # ayer, día completo (mediodía + noche)
./scripts/telesesgo run 2026-11-16 noche    # una fecha y una franja
./scripts/telesesgo run 16/11/2026 mañana   # «mañana» = ediciones de mediodía (14-15 h)
./scripts/telesesgo run noche               # ayer, solo la noche
```

Imprime el progreso de cada fase y termina con un resumen por programa y la ruta del informe HTML. El log
completo, con trazas, queda en `data/logs/AAAA-MM-DD.log`. Mientras la GPU transcribe una edición, Gemini ya
analiza las que están transcritas.

Cada episodio avanza por los estados `discovered → downloaded → transcribed → segmented → analyzed`.
`telesesgo run` es idempotente: si un paso falla, el error queda en `episodes.error` y la siguiente
ejecución lo reintenta desde ese punto.

## Fuentes incluidas (verificadas en noviembre de 2026)

| clave          | cadena  | programa                 | franja   | cómo se descubre                                        |
|----------------|---------|--------------------------|----------|---------------------------------------------------------|
| `rtve_td1`     | La 1    | Telediario 1 (15h)       | mediodía | `rtve.es/api/programas/45030/videos.json`, `type=Completo` |
| `rtve_td2`     | La 1    | Telediario 2 (21h)       | noche    | `rtve.es/api/programas/135930/videos.json`              |
| `a3_noticias1` | Antena 3| Antena 3 Noticias 1      | mediodía | `api.atresplayer.com` (formatId), episodios `FREE`      |
| `a3_noticias2` | Antena 3| Antena 3 Noticias 2      | noche    | `api.atresplayer.com` (formatId)                        |
| `lasexta_14h`  | laSexta | laSexta Noticias 14:00   | mediodía | `api.atresplayer.com` (formatId)                        |
| `lasexta_20h`  | laSexta | laSexta Noticias 20:00   | noche    | `api.atresplayer.com` (formatId)                        |

La franja (`edition: mediodia|noche` en `config.yaml`) es la que filtra `run ... mañana|noche`. Para otras
cadenas existe la fuente genérica `ytdlp` (cualquier listado que yt-dlp sepa recorrer, filtrado por título y
duración). Mitele (Telecinco/Cuatro) no está incluido: no está verificado, y el contenido con DRM se
descarta a propósito porque el pipeline no intenta eludir protecciones.

## Instalación (Linux o WSL2 Ubuntu)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # si no tienes uv
echo 'GEMINI_API_KEY=...' > .env                    # clave de Google AI Studio (https://aistudio.google.com/apikey)
./scripts/telesesgo llm-check                       # crea el entorno la primera vez y prueba la clave y el modelo
```

`scripts/telesesgo` crea y usa el entorno de Python en el disco de Linux (`~/.cache/telesesgo/venv`), no en
la carpeta del proyecto: si el proyecto está en `C:` (montado vía 9p/drvfs), arrancar Python desde ahí
cuesta minutos por comando. Si hay GPU NVIDIA (`nvidia-smi`), instala también el extra `cuda`. No hace falta
un ffmpeg del sistema: si no lo hay, se usa el binario estático de `imageio-ffmpeg`.

**GPU:** con una NVIDIA y el driver de Windows con soporte WSL, el extra `cuda` instala cuBLAS y cuDNN para
faster-whisper. Con `device: auto` y `compute_type: default`, la GPU se usa en int8_float16 y la CPU en int8.
Referencias medidas: T1200 (4 GB) ≈ 12× tiempo real; CPU de 16 hilos ≈ 0,3×.

**Quién habla (diarización y biblioteca de voces, opcional):** con `stt.diarization.enabled: true`, `HF_TOKEN`
en `.env` y las condiciones de `pyannote/speaker-diarization-community-1` aceptadas en Hugging Face, cada línea de
la transcripción lleva su hablante. Gemini identifica a cada voz por el contexto (presentador por la entradilla,
reporteros por su despedida, políticos por cómo se presentan sus totales) y las voces con nombre se guardan en una
biblioteca de huellas de voz: en los episodios siguientes se reconocen aunque nadie diga su nombre.
`./scripts/telesesgo voces` lista la biblioteca; `voces --episodio ID` muestra los hablantes de un episodio;
`voces --renombrar` y `voces --borrar` corrigen errores. Sin diarización, el análisis distingue las voces por el
contexto y por las líneas en que habla cada actor.

**Ediciones antiguas de Atresplayer:** pasados unos 3 días, Antena 3 y laSexta exigen iniciar sesión.
Exporta las cookies del navegador con sesión iniciada en atresplayer.com a un `cookies.txt` (formato
Netscape) y pon su ruta en `download.cookies_file`.

## Otros comandos

```bash
./scripts/telesesgo run 2026-11-16 --sin-ia     # solo descargar y transcribir
./scripts/telesesgo status                      # estado de los episodios + tokens consumidos
./scripts/telesesgo report --date 2026-11-12
./scripts/telesesgo analyze --id 4 --force      # reanalizar un episodio
./scripts/telesesgo match --date 2026-11-12
./scripts/telesesgo html                        # regenerar los informes HTML (todas las fechas, o --date ...)
./scripts/telesesgo timings --date 2026-11-10   # cuánto ha tardado cada paso
./scripts/telesesgo run 2026-11-16 --rehacer    # volver a analizar (p. ej. tras cambiar prompts o modelo)
./scripts/telesesgo voces                       # biblioteca de voces conocidas
```

### Rendimiento

* **Entorno de Python en el disco de Linux** (ver Instalación): el arranque pasa de minutos a menos de 1 s.
* **Descargas en paralelo** (`download.workers`, 4 por defecto) **en una carpeta temporal local**
  (`download.staging_dir`, por defecto `/tmp/telesesgo`). Solo el audio final (~10 MB) se mueve a `data/audio`,
  para no frenar la transcripción con escrituras en `C:`.
* **Solapes**: la GPU transcribe cada episodio en cuanto termina su descarga, y Gemini analiza cada episodio en
  cuanto termina su transcripción.
* **Análisis selectivo** (`llm.analyze_non_political: false`): solo las piezas políticas reciben el análisis
  detallado; cultura, sucesos, etc. quedan segmentadas y sin puntuar (~40% menos llamadas).
* **Transcripción por lotes** (`stt.batch_size`). Con un valor mayor que 0 se usa el `BatchedInferencePipeline` de
  faster-whisper; 0 vuelve al modo secuencial.
* **Tiempos**: cada paso (download, transcribe, segment, analyze, match, html) queda en la tabla `step_timings`.
  `telesesgo timings` los resume.

### Informes HTML

Al terminar `analyze` o `match`, se regenera `data/reports/<fecha>.html` (con un índice en
`data/reports/index.html`). Cada informe incluye:

* una tabla por franja (mediodía y noche) con duración, orientación, tonos, piezas con opinión propia, noticia
  de apertura, tiempo de palabra por bloque y citas verificadas;
* las noticias que comparten varias cadenas, ordenadas por divergencia, con la cita más significativa de cada una;
* hallazgos calculados: mayores diferencias, temas que solo cubre un programa, opinión propia, lenguaje cargado
  y piezas con fuentes desequilibradas (solo con citas verificadas);
* el detalle de cada programa. Al pasar el ratón por una puntuación se ve su justificación.

**Observaciones por franja.** Al final de `run`, Gemini escribe unas observaciones para cada franja procesada
(`data/notes/<fecha>-mediodia.md` y `-noche.md`), que aparecen bajo la tabla de su franja. Solo recibe un dossier con
los datos ya medidos (métricas por programa, noticias comunes y piezas destacadas con sus citas verificadas). Las
citas «…» de su texto se contrastan con esas citas y con la transcripción; las que no aparecen se marcan *(cita no
verificada)*. Para rehacerlas: `./scripts/telesesgo notes 2026-11-16 [mañana|noche]`. Se desactivan con
`llm.write_notes: false`. Las notas escritas a mano siguen en `data/notes/<fecha>.md` y salen aparte. Se respeta el modo claro u oscuro del sistema.

### Programación diaria

`scripts/run_daily.sh` procesa el día de ayer completo y añade la salida a `data/logs/run_daily.log`.
Conviene lanzarlo cada mañana: así se descargan las ediciones antes de que Atresplayer pida sesión.

* **cron dentro de WSL** (requiere `systemd=true` en `/etc/wsl.conf` o arrancar cron):
  `30 7 * * * /ruta/telesesgo/scripts/run_daily.sh`
* **Programador de tareas de Windows** (no depende de que WSL esté abierto): acción
  `wsl.exe`, argumentos `-d Ubuntu -- /ruta/telesesgo/scripts/run_daily.sh`, diaria a las 7:30.

## Análisis con LLM

Proveedor: Gemini (Google AI Studio), con la clave en `GEMINI_API_KEY`. Modelo en `llm.model`, con reserva en `llm.fallback_models`: si un modelo da 503 repetidos (saturado) o 404,
se pasa al siguiente y se sigue con él. `telesesgo llm-check` lista los modelos de tu clave y prueba los configurados. Razonamiento según `llm.effort`
(`thinking_level` en Gemini 3.x, `thinking_budget` en 2.x). La salida JSON se pide con el esquema de
`src/telesesgo/schemas.py` y se valida con Pydantic. Los 429 (cuota) y 5xx se reintentan con espera
exponencial (`llm.max_retries`), y `llm.max_concurrency` limita las peticiones simultáneas.

1. **Segmentación**: la transcripción entera, con líneas numeradas `[idx] mm:ss HABLANTE| texto`, se
   divide en piezas (líneas de inicio y fin, titular neutro, sección, formato). Los tiempos se calculan
   en código a partir de las marcas de Whisper, no los inventa el modelo.
2. **Análisis por pieza** (se omiten sumario, tiempo, deportes y autopromociones). La transcripción
   completa va como prefijo común y cada pieza añade solo su instrucción, para aprovechar la caché
   implícita de Gemini; la primera llamada va sola y el resto se lanza en paralelo. Por pieza se obtiene:
   * actores (nombre canónico, bloque político, rol, si habla con voz propia, tono hacia él);
   * tono hacia el Gobierno y hacia la oposición, y orientación izquierda-derecha (escala -2..+2 o null,
     con confianza);
   * encuadre principal y secundarios (taxonomía de Semetko y Valkenburg ampliada);
   * equilibrio de fuentes y perspectivas ausentes, lenguaje cargado y editorialización.
3. **Quién dice cada cita**: cada cita se atribuye a la voz del *medio* (presentador, redacción, reporteros) o
   a una *fuente* (total, entrevistado, manifestante). La atribución se comprueba en código: manda el rol del
   hablante de esa línea según la diarización; si no hay, las líneas en que habla cada actor; y si no, lo que diga
   el modelo (`evidence.speaker`, `voice`, `voice_source`). Solo la voz del medio cuenta como lenguaje cargado u
   opinión propia del informativo: si todas las pruebas de opinión de una pieza son de fuentes, no se marca como
   editorializada.
4. **Verificación de citas**: cada fragmento literal que da el modelo se contrasta con la transcripción
   (fuzzy ≥ 88, sin tildes ni puntuación) y se guarda en `evidence.verified`. Filtra por `verified = 1`
   antes de citar resultados.
5. **Exposición**: los segundos de voz propia y de mención por actor se calculan sumando la duración de
   las líneas que indica el modelo (`v_actor_exposure`, `v_bloc_exposure`).
6. **Cruce entre cadenas**: las piezas de un mismo día se agrupan por acontecimiento (`events`,
   `v_event_comparison`), lo que permite comparar orden, duración, tono y encuadre de una misma noticia.

`political_context` en `config.yaml` define quién es Gobierno, socio u oposición. **Revísalo**: de él
dependen las clasificaciones. Al cambiar prompts o contexto, sube `llm.prompt_version`; las piezas con
otra versión se reanalizan en el siguiente `analyze`.

### Coste

`telesesgo status` muestra los tokens por etapa (tabla `llm_usage`, con lecturas de caché). Un
telediario de 35-50 min son unos 15-25k tokens de transcripción. Si necesitas abaratarlo: baja `effort`
a `medium` o usa un modelo más ligero en `llm.model`, validando antes la calidad con una muestra codificada
a mano. Al cambiar de modelo, sube `llm.prompt_version` para no mezclar resultados.

## Limitaciones metodológicas

* Las puntuaciones del LLM son un instrumento de medida, no un dato. Antes de sacar conclusiones,
  valida una muestra con codificación humana (por ejemplo, el alfa de Krippendorff entre el modelo y dos
  codificadores) y fija `prompt_version` durante todo el periodo que compares.
* Los errores de Whisper en nombres propios se propagan; `stt.initial_prompt` ayuda con los habituales.
* La exposición de cada actor depende de que el modelo asigne bien las líneas. Con diarización activada
  es más fiable.
* En campaña electoral el reparto de tiempos en los medios públicos está regulado (LOREG). Tenlo en
  cuenta al comparar RTVE con las cadenas privadas.

## Consultas útiles (SQLite)

```sql
-- Tiempo de palabra por bloque y cadena en la última semana
SELECT channel, bloc, sum(voice_s) FROM v_bloc_exposure
WHERE air_date >= date('now','-7 day') GROUP BY 1,2 ORDER BY 1,3 DESC;

-- Una misma noticia en las distintas cadenas
SELECT label, channel, position, duration_s, tone_government, primary_frame
FROM v_event_comparison WHERE air_date = '2026-11-12' ORDER BY event_id, channel;

-- Evidencias verificadas de lenguaje cargado
SELECT e.channel, s.headline, ev.quote, ev.note FROM evidence ev
JOIN stories s ON s.id = ev.story_id JOIN episodes e ON e.id = s.episode_id
WHERE ev.dimension = 'loaded_language' AND ev.verified = 1;
```
