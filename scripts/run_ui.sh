#!/usr/bin/env bash
# The browser UI on http://127.0.0.1:7860. Extra args go to ui/server.py, e.g. --tts-device 1
cd "$(dirname "$0")/../ui"
exec "${PYTHON:-python}" -I -u server.py "$@"
