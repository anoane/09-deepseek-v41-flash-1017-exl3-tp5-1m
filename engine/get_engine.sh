#!/usr/bin/env bash
# Makes the immutable engine copy the layouts name:
#   $REL/src-309ef6d11920       the source tree of the commit (a "git archive", no .git)
#   $REL/autotune-309ef6d11920  its own kernel-tuning cache (EXLLAMAV3_TUNE_CACHE in the layouts)
# The commit is upstream exllamav3 dev at f157fc0b (2026-10-18) plus the open pull requests #449 (DeepSeek-V4.1,
# with tensor parallel and the draft head under tensor parallel), #468 and #472 (placement arguments and text), and
# the one-file fix of #532 (a tensor-parallel load ran out of file descriptors since f157fc0b) -- that is commit
# 5c0da061, the fork's tag recipe-09 -- plus one commit: the lightning indexer scores in int8, fp8 or fp4 per GPU
# architecture (EXL3_DSA_INDEXER_DTYPE, --indexer_dtype). The fork's branch dsv41f-1017 keeps it; it is fetched by
# its id. This script made the copy that serves on this machine (2026-10-20).
set -euo pipefail
SHA=309ef6d11920fb2b0df9b64bece4d70646fa23b3
REL=${REL:-$HOME/rnd/release/all}; S=${SHA:0:12}
[[ ! -e $REL/src-$S ]] || { echo "already there: $REL/src-$S" >&2; exit 1; }
GIT=$(mktemp -d); trap 'rm -rf "$GIT"' EXIT
git -C "$GIT" init --quiet
git -C "$GIT" fetch --quiet --depth 1 https://github.com/anoane/exllamav3-anemone "$SHA"
[[ $(git -C "$GIT" rev-parse FETCH_HEAD) == "$SHA" ]] || { echo "fetched another commit than $SHA" >&2; exit 1; }
# into a temporary folder first: a copy that stopped half way must not pass for the tree
rm -rf "$REL/src-$S.tmp"; mkdir -p "$REL/src-$S.tmp" "$REL/autotune-$S"
git -C "$GIT" archive FETCH_HEAD | tar -x -C "$REL/src-$S.tmp"
echo "$SHA" > "$REL/src-$S.tmp/.release-commit"
chmod -R a-w "$REL/src-$S.tmp"
mv "$REL/src-$S.tmp" "$REL/src-$S"
echo "src-$S: $(find "$REL/src-$S" -type f | wc -l) files"
