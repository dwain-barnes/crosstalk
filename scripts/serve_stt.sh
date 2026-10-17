#!/usr/bin/env bash
# Nemotron 3.5 ASR Streaming on NeMo-Speech.cpp at ws://127.0.0.1:9100
#   STT_MODEL=/path/to/nemotron-3.5-asr-streaming-0.6b.q8_0.gguf ./scripts/serve_stt.sh
# Right context 1 is 160 ms steps. Endpointing has to be on for the finals the loop waits for;
# the client asks for 500 ms of silence when it opens a session.
set -e
: "${STT_MODEL:?set STT_MODEL to the Nemotron streaming GGUF}"
exec "${NEMO_SPEECH:-nemo-speech}" serve --host 127.0.0.1 --port "${PORT:-9100}" --no-ui --device cuda:0 \
  --asr-model "$STT_MODEL" --asr.streaming.rnnt_right_context "${RIGHT_CONTEXT:-1}" \
  --asr.endpointing.enable=true "$@"
