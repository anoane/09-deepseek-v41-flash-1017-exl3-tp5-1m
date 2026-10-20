#!/usr/bin/env bash
# Builds the engine's native extension for sm_80 (CMP 170HX) and sm_120a (RTX PRO 6000).
# It reads $REL/src-$SHA (get_engine.sh) and writes $REL/native-$SHA; the build directory is kept, so a later run
# only recompiles what changed. No GPU is visible during the build. No +PTX in the arch list: each card gets its own
# machine code. The source tree is read-only, so the build runs on a copy of it.
# The served extension was built in the test folders (arch list 8.0;12.0a) from the same native sources (no native
# file changed between #449's rebase and this commit) and copied to native-5c0da061daa7; the script in this form,
# with the paths below, has not run.
# (The decode logs in results/ print a TORCH_CUDA_ARCH_LIST with +PTX: the engine sets that variable itself at import
# from the visible cards when it is unset; it is not the list the extension was built with.)
set -euo pipefail
SHA=${SHA:-5c0da061daa7}
REL=${REL:-$HOME/rnd/release/all}
SRC=$REL/src-$SHA; NATIVE=$REL/native-$SHA; BUILD=$REL/build-$SHA
[[ -f $SRC/setup.py && -f $SRC/.release-commit ]] || { echo "no complete engine source at $SRC" >&2; exit 1; }
source "$HOME/workspace/ds41-serve-venv/bin/activate"
export CUDA_HOME=/usr/local/cuda-13.4 PATH=/usr/local/cuda-13.4/bin:$PATH
export TORCH_CUDA_ARCH_LIST="8.0;12.0a" MAX_JOBS=${MAX_JOBS:-16} CUDA_VISIBLE_DEVICES=
mkdir -p "$BUILD/src"
cp -a "$SRC/." "$BUILD/src/"
chmod -R u+w "$BUILD/src"
cd "$BUILD/src"
date -Is
nice -n 15 python setup.py build_ext --build-lib "$NATIVE" --build-temp "$BUILD/tmp"
date -Is
ls -la "$NATIVE" | tail -n 3
sha256sum "$NATIVE"/exllamav3_ext*.so
echo BUILD-DONE
