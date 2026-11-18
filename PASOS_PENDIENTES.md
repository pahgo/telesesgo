# Pasos pendientes de configuración (para hacer a mano)

Todo se ejecuta en **Linux (WSL)**, desde la carpeta del proyecto:

```bash
cd /mnt/c/vhv/cloud/telesesgo
```

La primera ejecución en WSL tarda unos minutos más de lo normal: crea el entorno de Python en
`~/.cache/telesesgo/venv` (con CUDA, ~3 GB) y descarga el modelo de Whisper (~1,6 GB). Las siguientes
arrancan en menos de un segundo.

## 1. Clave de Gemini — HECHO

Clave creada en https://aistudio.google.com/apikey.

## 2. Guardar la clave en el proyecto — HECHO

```bash
echo 'GEMINI_API_KEY=tu-clave' > .env
```

(`.env` está en `.gitignore`.)

## 3. Comprobar la clave y los modelos — HECHO (17/11: 3.5-flash saturado, 2.5-flash da 404, 3.1-flash-lite funciona)

```bash
./scripts/telesesgo llm-check
```

Lista los modelos que admite tu clave y prueba, con una sola llamada cada uno, el modelo principal
(`llm.model`) y los de reserva (`llm.fallback_models`) de `config.yaml`.

* `✓` = responde. Con que responda uno basta: si el principal está saturado, el pipeline pasa solo al
  siguiente de la lista.
* `503 … overloaded / UNAVAILABLE` = Google tiene ese modelo saturado en ese momento. No es un fallo tuyo
  ni de la clave. Prueba más tarde o cambia el orden de los modelos.
* `sin respuesta en 60 s` = el modelo no contesta (a veces, modelos que no están en el plan gratuito).
* `429` = has agotado la cuota por minuto o por día de tu cuenta. Baja `llm.max_concurrency` a 1 o 2.
* `400` o `403` con la palabra `API key` = la clave está mal copiada en `.env`.

## 4. Primera ejecución real — HECHO (16/11 mediodía analizado en 3,5 min con gemini-3.1-flash-lite)

```bash
./scripts/telesesgo run 2026-11-16 mañana
```

Pendiente: generar las observaciones automáticas del 16/11 (nuevo paso del pipeline):

```bash
./scripts/telesesgo notes 2026-11-16
```

## 4b. Reanalizar con la versión 2 (quién dice cada cita)

Los análisis hechos antes del 17/11 tarde son de la versión 1 (no distinguían la voz del informativo de la de sus
fuentes). Para rehacerlos, sin volver a descargar ni transcribir:

```bash
./scripts/telesesgo run 2026-11-16 --rehacer
```

## 4c. (Recomendado) Activar «quién habla»: diarización y reconocimiento de voces

### A. Cuenta de Hugging Face (navegador)

1. Ve a https://huggingface.co/join, regístrate con tu correo y una contraseña (o entra en
   https://huggingface.co/login si ya tienes cuenta). Elige un nombre de usuario cualquiera.
2. Abre el correo de verificación que te envían («Confirm your email») y pulsa el enlace. Sin verificar el correo
   no te dejarán aceptar las condiciones del modelo.

### B. Aceptar las condiciones del modelo (navegador)

3. Con la sesión iniciada, abre https://huggingface.co/pyannote/speaker-diarization-community-1
4. Verás un recuadro «You need to agree to share your contact information to access this model». Rellena los
   campos (empresa/organización y web; vale «particular» y tu web o LinkedIn, o lo que te pida) y pulsa
   **«Agree and access repository»**.
5. Al recargar la página debe desaparecer el recuadro y verse la pestaña «Files and versions» con los ficheros.

### C. Crear el token (navegador)

6. Abre https://huggingface.co/settings/tokens (menú de tu avatar → Settings → Access Tokens).
7. Pulsa **«Create new token»**.
8. Arriba, en «Token type», elige **Read** (es lo más sencillo y basta para descargar el modelo).
   - Si prefieres **Fine-grained**, marca al menos «Read access to contents of all public gated repos you can
     access»; si no, el token no podrá descargar el modelo.
9. Ponle un nombre (por ejemplo `telesesgo`) y pulsa **«Create token»**.
10. Copia el valor que aparece (empieza por `hf_…`). Solo se muestra entero una vez: si lo pierdes, crea otro.
    Trátalo como una contraseña: no lo pegues en chats ni lo subas a git.

### D. Guardarlo en el proyecto (terminal de WSL)

11. Abre la terminal de WSL y ve al proyecto:
    ```bash
    cd /mnt/c/vhv/cloud/telesesgo
    ```
12. Añade el token al final de `.env` (con `>>`, dos signos; uno solo borraría la clave de Gemini). Sustituye
    `hf_tu-token` por el tuyo, sin espacios alrededor del `=`:
    ```bash
    echo 'HF_TOKEN=hf_tu-token' >> .env
    ```
13. Comprueba que `.env` tiene las dos claves (muestra solo los nombres, no los valores):
    ```bash
    cut -d= -f1 .env
    ```
    Debe salir `GEMINI_API_KEY` y `HF_TOKEN`, una vez cada una. Si `HF_TOKEN` sale dos veces, abre el fichero
    (`nano .env`) y borra la línea mala (Ctrl+K borra la línea, Ctrl+O guarda, Ctrl+X sale).

### E. Comprobar y activar

14. Comprueba el token y el acceso al modelo (no descarga nada):
    ```bash
    ./scripts/telesesgo hf-check
    ```
    - `✓ Token válido` y `✓ Acceso a … condiciones aceptadas` → sigue.
    - `HTTP 401` → el token está mal copiado: repite los pasos 6-13 con uno nuevo.
    - `Sin acceso … HTTP 403` → falta el paso 4 (aceptar condiciones) o el token Fine-grained no tiene permiso
      para repositorios «gated».
15. Activa la diarización en `config.yaml`: busca la sección `stt:` → `diarization:` y cambia `enabled: false`
    por `enabled: true`. Puedes hacerlo con un editor o con:
    ```bash
    sed -i '/^  diarization:/,/^    enabled:/ s/^    enabled: false/    enabled: true/' config.yaml
    ./scripts/telesesgo hf-check     # ahora debe terminar con «Diarización activada … Todo listo»
    ```
16. Rehaz un día para probarlo. La primera vez instala la diarización (~3 GB, unos minutos) y descarga el modelo:
    ```bash
    ./scripts/telesesgo run 2026-11-16 --rehacer
    ```
    En el progreso verás «Voces de …: N hablantes, M reconocidos por la voz». El primer día casi nadie se
    reconoce por la voz (la biblioteca está vacía): se aprende de los nombres que identifica Gemini. Si aparece
    «Sin diarización en …», el mensaje dice por qué; el análisis sigue igualmente.

### F. Revisar lo aprendido

17. ```bash
    ./scripts/telesesgo voces                      # biblioteca
    ./scripts/telesesgo voces --episodio 23        # hablantes de un episodio (id en el informe o en `status`)
    ./scripts/telesesgo voces --renombrar "Nombre mal" "Nombre bien"
    ./scripts/telesesgo voces --borrar "Nombre"
    ```
    El umbral de parecido (`match_threshold: 0.75`, en la parte exigente del rango habitual 0.60-0.80) es un punto
    de partida: si ves voces confundidas, súbelo a 0.80; si no reconoce a presentadores que salen a diario, bájalo
    a 0.70.

## 5. Uso diario

Con la diarización activada, cada edición tarda unos 3 minutos más (medido: ~4,8 s por minuto de audio en la
T1200). Si un día quieres ir rápido, sin identificar voces: `./scripts/telesesgo run --sin-voces`.

Limpieza recomendada de la biblioteca (nombres aprendidos antes de exigir nombre y apellido):

```bash
./scripts/telesesgo voces --renombrar "Vicente_Valles" "Vicente Vallés"
./scripts/telesesgo voces --renombrar "Sandra" "Sandra Golpe"    # solo si es la presentadora de A3 Noticias 1
./scripts/telesesgo voces --borrar "Sonia"                       # una profesora en huelga: no es útil reconocerla
```

```bash
./scripts/telesesgo run            # ayer, día completo
./scripts/telesesgo run noche      # ayer, solo la noche
./scripts/telesesgo run 2026-11-16 mañana
```

## 6. (Opcional) Ediciones antiguas de Antena 3 y laSexta

Pasados unos 3 días, Atresplayer exige iniciar sesión (por ejemplo, para las noches del 13/11).

1. Inicia sesión en https://www.atresplayer.com en tu navegador.
2. Exporta las cookies de ese sitio en formato Netscape (extensión «Get cookies.txt LOCALLY» o similar).
3. Guarda el fichero como `cookies.txt` en la carpeta del proyecto (no lo subas a git: contiene tu sesión).
4. En `config.yaml`, sección `download:`, cambia `cookies_file: null` por `cookies_file: cookies.txt`.

## 7. (Opcional) Ejecución automática cada mañana

Así se descargan las ediciones antes de que Atresplayer pida sesión. En el **Programador de tareas de
Windows**: Crear tarea básica → Diariamente, 7:30 → Iniciar un programa:

* Programa: `wsl.exe`
* Argumentos: `-d Ubuntu -- /mnt/c/vhv/cloud/telesesgo/scripts/run_daily.sh`

(Si tu distribución de WSL no se llama `Ubuntu`, mira el nombre con `wsl -l -v` en PowerShell.) La salida
queda en `data/logs/run_daily.log`.

## 8. (Opcional) Liberar 2,8 GB en C:

El entorno antiguo, dentro del proyecto, ya no se usa:

```bash
rm -rf /mnt/c/vhv/cloud/telesesgo/.venv
```

## Pendientes de revisión (no son configuración técnica)

* `political_context` en `config.yaml` está desfasado (Frente Amplio, generales del 29N, dimisión de Óscar
  Puente, Irene Montero candidata de Podemos). Si lo cambias, sube `llm.prompt_version`.
* Las ediciones del fin de semana (14 y 15/11) no aparecen en la búsqueda; está sin investigar.
* `llm.analyze_non_political: false`: las piezas no políticas no se analizan en detalle. Ponlo a `true` si
  las quieres puntuadas.
