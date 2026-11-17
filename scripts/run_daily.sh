#!/usr/bin/env bash
# Ejecución diaria desatendida: procesa el día de ayer completo y deja el registro en data/logs/.
# Para cron en WSL, o para el Programador de tareas de Windows vía:
#   wsl.exe -d Ubuntu -- /ruta/a/telesesgo/scripts/run_daily.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/data/logs"
"$ROOT/scripts/telesesgo" run "$@" >>"$ROOT/data/logs/run_daily.log" 2>&1
