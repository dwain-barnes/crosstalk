#!/usr/bin/env bash
# Crosstalk on llama-server at http://127.0.0.1:8080
#   LLM_MODEL=/path/to/Crosstalk-35B-A3B-UD-Q4_K_M.gguf ./scripts/serve_llm.sh
# Fully on the GPU it takes about 23 GB. N_CPU_MOE=12 keeps the experts of the first 12 layers
# on the CPU and brings that to about 16.5 GB, for about 70 ms more per tick.
set -e
: "${LLM_MODEL:?set LLM_MODEL to the Crosstalk GGUF}"
args=(-m "$LLM_MODEL" -ngl 99 -c 32768 -ub 512 -np "${SLOTS:-4}" -kvu --host 127.0.0.1 --port "${PORT:-8080}")
if [ -n "$N_CPU_MOE" ]; then
  args+=(--n-cpu-moe "$N_CPU_MOE")
fi
exec "${LLAMA_SERVER:-llama-server}" "${args[@]}" "$@"
