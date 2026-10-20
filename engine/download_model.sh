#!/usr/bin/env bash
# Downloads LibertAI's EXL3 3.5 bpw pack of DeepSeek-V4.1-Flash-1017 and the 1017 checkpoint's two engram shards
# with its index, at the revisions served here (another revision through PACK_REV / BASE_REV in the environment).
# The pack holds no engram tables and no chat template: tools/make_view.py then builds the folder TabbyAPI loads.
# Resumable: run it again after an interruption. On this machine the base checkpoint was downloaded whole; in this
# form the script has not run.
set -euo pipefail
HF=${HF:-$HOME/workspace/hf-venv/bin/hf}
PACK_REPO=LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw
BASE_REPO=deepseek-ai/DeepSeek-V4.1-Flash-1017
PACK_REV=${PACK_REV:-e55cd9e0b923e8a0c7895afb379d4d28fa503bbd}
BASE_REV=${BASE_REV:-1bae0e1088993b9b82de08f59de5be5a34e42170}
echo "$(date -Is) pack $PACK_REPO @ $PACK_REV"
"$HF" download "$PACK_REPO" --revision "$PACK_REV" --local-dir "$HOME/workspace/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw"
echo "$(date -Is) engram shards and index from $BASE_REPO @ $BASE_REV"
"$HF" download "$BASE_REPO" model-00047-of-00048.safetensors model-00048-of-00048.safetensors \
     model.safetensors.index.json --revision "$BASE_REV" --local-dir "$HOME/workspace/DeepSeek-V4.1-Flash-1017-engram"
echo "$(date -Is) DONE"
