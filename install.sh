#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
VENV=${EVOLOOP_VENV:-"$HOME/.local/share/evoloop/venv"}
BIN="$HOME/.local/bin"
APPLICATIONS="$HOME/.local/share/applications"

command -v python3 >/dev/null || { echo "Erro: Python 3.10+ é necessário." >&2; exit 1; }
python3 - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit("Erro: Evoloop requer Python 3.10 ou superior.")
PY

echo "Instalando Evoloop em $VENV"
python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install --quiet --upgrade pip
"$VENV/bin/python" -m pip install --quiet "$ROOT"
mkdir -p "$BIN" "$APPLICATIONS" "$HOME/Workspace"
ln -sfn "$VENV/bin/evoloop" "$BIN/evoloop"

cat > "$APPLICATIONS/evoloop.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=Evoloop
Comment=Assistente local e evolutivo
Exec=$VENV/bin/evoloop app
Terminal=true
Categories=Development;Utility;
StartupNotify=true
EOF
chmod 755 "$APPLICATIONS/evoloop.desktop"

echo
echo "Evoloop instalado. Antes de abrir:"
echo "  1. Instale/inicie um servidor OpenAI-compatible (Ollama por padrão)."
echo "  2. Baixe um modelo, por exemplo: ollama pull qwen3:4b"
echo "  3. Abra 'Evoloop' no menu ou execute: $BIN/evoloop app"
