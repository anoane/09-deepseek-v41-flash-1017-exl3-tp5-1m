# DeepSeek-V4.1-Flash-1017 EXL3 3.5 bpw at 1M context, tensor parallel over five cards (exllamav3 + open pull requests)

> **State of 2026-10-20. Layout `tp5-1017-35` of this repository is what serves on this machine** (with the
> lightning indexer in INT8 since 10:15 that day and `EXL3_INT8_GEMV=0` since 11:10), started at boot by its systemd
> unit. Every figure below is a single run unless two are given, measured from the client unless
> it says server; what was not measured is in [Limits](#limits-and-what-was-not-measured).

DeepSeek-V4.1-Flash-1017 (the 17 October checkpoint) quantized to EXL3 3.5 bpw by LibertAI, at **1,048,576 tokens of
FP16 context**, one request at a time, on four CMP 170HX and one RTX PRO 6000 in **tensor parallel**: every layer is
split over all five cards. The engine is upstream exllamav3 `dev` plus open pull requests and one commit of ours
that scores the lightning indexer (the part of the attention that picks which earlier tokens each token reads) in
INT8: a 1M-token prefill is 9% faster with the model's output unchanged within the measurement noise
([Lightning indexer in INT8](#lightning-indexer-in-int8)). Served by a TabbyAPI fork. Use: C coding (review, bug
finding, fixes, new code, agent turns with tools).

```
CMP 170HX x4  (64 GB, sm_80, PCIe Gen2 x16, 150 W)   tensor parallel, at most 58 GB each
RTX PRO 6000  (96 GB, sm_120, PCIe x16, 400 W)       tensor parallel, at most 70 GB, plus the model's draft head (20 GB)
host RAM      (499 GiB)                              the two engram tables (2 x 94.6 GiB) in the page cache
```

## Results

### Huge C prompts, cold, 1M FP16 context

One request at a time against the served layout (INT8 indexer; measured at 10:19, before `EXL3_INT8_GEMV=0`, so the
decode columns are with upstream's INT8 decode GEMVs). Each prompt is a review request over the first N
tokens of a C / CUDA / C++ source corpus (concatenated sources and C headers, not published; 3.155 characters per
token with the model's tokenizer), with a unique first line so nothing comes from the prompt cache: **every prefill
is cold**. The engram tables were read into the page cache first, as the prewarm unit does after boot. Greedy, 600
new tokens, reasoning on (the template's default). [tools/bench_long.py](tools/bench_long.py),
[results/long_c_prompts_tp5-int8.txt](results/long_c_prompts_tp5-int8.txt) and the server's figures in
[results/long_c_prompts_server_metrics_int8.txt](results/long_c_prompts_server_metrics_int8.txt); the same run with
the FP16 indexer the night before: [results/long_c_prompts_tp5-dyn.txt](results/long_c_prompts_tp5-dyn.txt),
[results/long_c_prompts_server_metrics.txt](results/long_c_prompts_server_metrics.txt).

| Prompt tokens | Time to first token | Prefill, client (prompt / TTFT) | Prefill, server | Decode, client | Decode, server | TTFT with the FP16 indexer |
|---:|---:|---:|---:|---:|---:|---:|
| 26,296 | 12.4 s | 2,116 tok/s | 2,138 tok/s | 84.2 tok/s | 83.4 tok/s | 14.2 s |
| 127,604 | 58.7 s | 2,173 | 2,177 | 89.2 | 88.4 | 62.3 s |
| 253,055 | 124.2 s | 2,037 | 2,039 | 67.9 | 67.4 | 131.6 s |
| 513,791 | 274.1 s | 1,874 | 1,876 | 100.3 | 99.0 | 298.1 s |
| 1,039,782 | 632.1 s (10.5 min) | 1,645 | 1,646 | 72.3 | 73.4 | 680.0 s for 1,002,677 tokens (1,475 tok/s) |

One run each. The corpus's tail was extended between the two runs, so the last prompt is 37,105 tokens longer
here; the four shorter prompts are the same tokens. Below about 128K tokens the difference is mostly run-to-run
variation (the indexer is a small part there); from 512K up it is the indexer
([Lightning indexer in INT8](#lightning-indexer-in-int8)). Decode is drafted by the model's own head (dynamic
draft length; 63-85% of the drafted tokens accepted here); it depends on how predictable the reply is, so another
text gives another figure, and the INT8 indexer does not touch it. Client decode is tokens after the first divided
by the time after the first; the server's figure also counts the first token.

### Lightning indexer in INT8

The indexer scores every earlier token of the prompt against each new one to choose the 512 compressed entries the
attention reads; during a long prefill that scoring grows with the depth and is what makes the last tokens of a 1M
prompt slower than the first. The engine commit on top of `recipe-09` lets it run in a lower precision, chosen per
GPU architecture, through `EXL3_DSA_INDEXER_DTYPE` (or `--indexer_dtype` on the command line):

- `fp16` (the default when unset), `int8` (sm_80 and newer), `fp8` (e4m3, sm_89 and newer) or `fp4` (e2m1 with a
  power-of-two scale per 32 values, the format of DeepSeek's reference indexer; sm_100 and newer);
- one value for every card, or `ARCH:DTYPE` items such as `sm_80:int8,sm_120:fp4`, where a bare item covers the
  architectures not listed; a type a card cannot run is refused when the model loads;
- only the prefill-sized scoring changes: queries and keys are quantized per row (keys once per pass), the
  multiply runs on the cards' INT8 / FP8 / FP4 tensor cores, and calls under 64 query rows (decode) stay FP16.

**Served: `sm_80:int8,sm_120:int8`.** Measured with [tools/kv_eval.py](tools/kv_eval.py) on one prompt of 1,016,849
tokens of the C corpus with three planted keys, one load per row, no draft head, every figure against an FP16 run of
the same layout ([results/indexer_dtype.txt](results/indexer_dtype.txt)). Changing only the prefill chunk size of the
FP16 run (16,384 to 8,192) already moves KL by 0.0036 / 0.0030 / 0.0012 / 0.0002 at 16K / 131K / 524K / 1M: that is
the noise floor.

| Layout | Indexer | Prefill (prompt) | KL vs FP16 at 16K / 131K / 524K / 1M | Top-1 at 16K / 1M | Keys recalled |
|---|---|---:|---|---|---:|
| tensor parallel 5 | FP16 | 662 s (1,016,849 tokens) | - | - | 3/3 |
| tensor parallel 5 | **INT8 on every card (served)** | **603 s (-9%)** | 0.0034 / 0.0029 / 0.0032 / 0.0004 | 98.0% / 99.9% | 3/3 |
| tensor parallel 5 | INT8 on the CMP 170HX, FP4 on the PRO | 600 s (-9%) | 0.0048 / 0.0043 / 0.0020 / 0.0001 | 97.1% / 99.9% | 3/3 |
| layer split, 3 x CMP 170HX + PRO | FP16 | 285 s (525,329 tokens) | - | - | 3/3 |
| layer split, 3 x CMP 170HX + PRO | INT8 on the CMP 170HX, FP4 on the PRO | 274 s (-4%) | 0.0033 / 0.0035 / 0.0018 / - | 97.8% / - | 3/3 |

The corpus NLL changed by less than two standard errors at every depth, in both directions. The gain grows with the
depth: the last 512K tokens of the 1M prompt took 326 s instead of 372 s (-12%); below about 64K tokens there is
nothing to win. Under tensor parallel every card scores every key, so the CMP 170HX set the time and FP4 on the PRO
buys nothing; in a layer split each card scores the layers it holds, and there the PRO's own type counts.

The scoring kernel alone, one 256-row prefill slab against 1M keys ([tools/bench_indexer.py](tools/bench_indexer.py)):

| Card | FP16 | INT8 | FP8 | FP4 |
|---|---:|---:|---:|---:|
| CMP 170HX (sm_80) | 22.3 ms | 12.3 ms | - | - |
| RTX PRO 6000 (sm_120) | 7.1 ms | 3.4 ms | 3.4 ms | 2.6 ms |

On the PRO, FP8 is no faster than INT8 and keeps less of the scores (top-512 equal to FP16 on random data: INT8
99.2%, FP8 97.0%, FP4 86.4%), so INT8 is the choice there. FP4, the reference's own format, is another quarter
faster in the kernel; it pays only where the PRO scores its own layers (a layer split).

### The nine C-coding requests, cold

Recipe 08's request set for the drafter tuning (review of a C file, bug finding, a fix as a diff, functions
rewritten, a whole file returned, two agent turns with tool schemas, new code, an explanation), prompts of 139 to
9,426 tokens. It quotes GPL sources, so it is not published here; [tools/bench_stream.py](tools/bench_stream.py)
reads it from `WORKLOADS_DIR`. The served layout as of 2026-10-20 11:10 (INT8 indexer, `EXL3_INT8_GEMV=0`), fresh
server, one warm-up request, one stream, greedy, each request's own token limit, 1M FP16 cache, every prompt cold.
Server figures from [results/coding_server_metrics_served-1020.txt](results/coding_server_metrics_served-1020.txt),
client figures from [results/raw/decode-gemv0-live-1020-1109/served-1.log](results/raw/decode-gemv0-live-1020-1109/served-1.log).

| Request | Prompt tokens | TTFT | Prefill, server | Tokens out | Decode, client |
|---|---:|---:|---:|---:|---:|
| review | 5,171 | 2.97 s | 1,802 tok/s | 1,500 | 88.9 tok/s |
| bug finding | 3,479 | 2.14 s | 1,714 | 621 | 95.7 |
| fix as a diff | 3,404 | 2.06 s | 1,719 | 350 | 112.9 |
| fix, functions rewritten | 3,405 | 2.06 s | 1,720 | 1,200 | 103.0 |
| whole file returned | 1,788 | 1.27 s | 1,503 | 2,600 | 99.5 |
| agent: edit call (tools) | 4,033 | 2.38 s | 1,753 | 154 | 109.8 |
| agent: later turn (tools) | 9,426 | 4.61 s | 2,085 | 212 | 119.9 |
| new code | 139 | 0.31 s | 579 | 1,500 | 84.2 |
| explanation | 5,158 | 2.77 s | 1,917 | 1,000 | 85.5 |

Mean decode 99.9 tok/s (100.7 on a second pass); the nine requests together gave 9,137 tokens in 117 s of wall time
(77.8 tok/s with prefill and gaps). The mean moves by a few percent between server starts: ten passes on 2026-10-20
gave 96.9 to 103.6 tok/s whatever the settings, and the night before, with the FP16 indexer and upstream's INT8
decode GEMVs, one pass gave 103.5 ([results/coding_server_metrics_tp5m-dyn.txt](results/coding_server_metrics_tp5m-dyn.txt)).
Tool calls come back as OpenAI `tool_calls` (checked on the tensor-parallel layouts and the layer split, results/raw/*/toolcall.log).

### INT8 in decode: off

Upstream's `EXL3_INT8_GEMV=2` (the default) runs the weight GEMVs of this pack (codebook mul1) with INT8 activations
up to 5-bit tensors on sm_80 and 6-bit on sm_120. On these cards it does not pay
([results/decode_int8_gemv.txt](results/decode_int8_gemv.txt)), on the nine requests above, two passes per fresh
server:

| `EXL3_INT8_GEMV` | Mean decode, passes 1 / 2 |
|---|---:|
| `2` (upstream default) | 97.5 / 99.9 tok/s |
| `2` with `EXL3_INT8_GEMV_MAX_K=6` (the CMP 170HX's 6-bit tensors too) | 97.1 / 96.9 |
| **`0`** (served since 2026-10-20 11:10) | **100.7 / 103.6, and 99.9 / 100.7 after the switch** |

About +2.5% (101.2 against 98.7), close to the spread between server starts, so it is served for the higher
precision at no measured cost rather than for speed. Decode here is bound by the per-step latency of five cards
synchronizing every layer, not by the weight arithmetic, so more INT8 does not help it; the INT8 indexer above is a
prefill gain only.

### Draft settings, on the same nine requests

One run each, fresh server, layout `tp5-1017-35` ([results/raw/mtp-1020-0126](results/raw/mtp-1020-0126)):

| TabbyAPI `draft_model` setting | Mean decode |
|---|---:|
| default (`draft_mode: mtp`) | 98.6 tok/s |
| `draft_num_tokens: 2` | 78.8 |
| `draft_num_tokens: 4` | 94.1 |
| `draft_num_tokens: 6` | 100.6 |
| **`dynamic_draft: true`** (served) | **103.5** |
| `hybrid_draft_tokens: 32` | 98.0 |

Dynamic drafting was faster on 7 of the 9 requests and 3-4% slower on two (functions rewritten, whole file). These
settings were compared the night before the INT8 changes, with the FP16 indexer and upstream's INT8 decode GEMVs.

### Layouts tried, same model family, two 34K-token prompts and three 900-token answers

[tools/bench_log.py](tools/bench_log.py) requests (two different ~34K-token prompts of documentation text, then three 900-token
technical answers), warm-up first, fresh server each, server figures; prose rather than code, so these compare the
layouts with each other, not with the tables above ([results/raw](results/raw)). All with the FP16 indexer:

| Layout | Model | Prefill, 34K prompts | Decode, 900-token answers |
|---|---|---:|---:|
| layer split, four cards (`n4x-1017-35`) | 1017 3.5 bpw | 1,826 / 1,885 tok/s | 50.7 / 42.7 / 44.0 |
| tensor parallel 4: four CMP 170HX, draft head on the PRO | 1017 2.9 bpw | 1,955 / 2,052 (256K ctx) | 72.7 / 60.1 / 64.6 |
| **tensor parallel 5** (`tp5-1017-35`) | **1017 3.5 bpw** | 1,944 / 2,159 | 85.8 / 62.8 / 73.1 |

The 3.5 bpw pack does not fit tensor parallel over the four CMP 170HX alone (4 x 64 GB). Tensor parallel 4 with
the NCCL backend (the cards' peer path) was slower than the native backend (65.0 / 52.9 / 57.2).

### Concurrency

Two slots of 1M each (`cache_size` 2,097,152, `max_batch_size` 2) load; four do not ("Insufficient VRAM in split for
model and cache"). On the nine requests, one stream 95.4 tok/s; two at once 34.4 tok/s each, 58 tok/s together:
less than one after the other, because two rows per step leave the drafting little to win. The served layout keeps
one slot, so concurrent requests queue.

## Hardware, model, software

- **Machine**: AMD EPYC 9J14 (96 cores), 499 GiB RAM, one NVMe SSD. CPU package limit 260 W
  ([serving/root/cpu-power-limit](serving/root/cpu-power-limit), applied after the HSMP driver is up), CPU governor
  `performance` ([serving/root/cpu-governor-performance.service](serving/root/cpu-governor-performance.service)).
- **Cards**: RTX PRO 6000 Blackwell Workstation (96 GB) at 400 W; four CMP 170HX (64 GB each) at 150 W, in PCIe
  Gen2 x16 through a patched open kernel module set ([hardware/p2p](hardware/p2p): Morrowmake's cmpunlocker fork,
  which also makes peer-to-peer copies work). [serving/root/gpu-power-caps](serving/root/gpu-power-caps) sets the caps
  by PCI device id and refuses to pass with an unexpected card count; the serving unit runs it before it starts.
- **Model**: [LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw)
  at `e55cd9e0` (routed experts 4 bpw on layers 0-9 and 30-39, 3 bpw on 10-29; attention and shared experts 6 bpw;
  the vision tower and the draft head included), and the two engram shards of
  [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) at `1bae0e10`
  (the pack has no engram tables). [engine/download_model.sh](engine/download_model.sh), then
  [tools/make_view.py](tools/make_view.py) links both into the folder TabbyAPI loads and adds the chat template.
- **Chat template**: [serving/chat_template.jinja](serving/chat_template.jinja) is the template of
  [pull request 11](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/discussions/11) on the 1017
  repository (by qgallouedec, open on 2026-10-20), served since that day. Rendered against the 1017
  `encoding/encoding.py` on 11 conversations ([results/template_comparison.txt](results/template_comparison.txt),
  [tools/compare_templates.py](tools/compare_templates.py)) it matches the reference on all but one; our earlier port
  of the pre-1017 encoder matched on plain chat, system prompts, thinking off, the reasoning levels and multi-turn,
  but rendered two newlines instead of four before the tool list, and dropped the reasoning of a tool-call turn that
  1017 keeps when no tools are declared. The one case the PR's template does not match: tool-call arguments given as
  a JSON string raise instead of being parsed; TabbyAPI parses such strings into objects before it renders the
  template, so only arguments that are not valid JSON would reach it.
- **Engine**: exllamav3 fork [anoane/exllamav3-anemone](https://github.com/anoane/exllamav3-anemone), branch
  `dsv41f-1017` = commit `309ef6d1`: the indexer precision commit (above) on top of tag `recipe-09` = commit
  `5c0da061`, which is upstream `dev` at `f157fc0b` plus the open pull requests
  [#449](https://github.com/turboderp-org/exllamav3/pull/449) (DeepSeek-V4.1, with tensor parallel and the draft head
  beside a tensor-parallel model), [#468](https://github.com/turboderp-org/exllamav3/pull/468) and
  [#472](https://github.com/turboderp-org/exllamav3/pull/472) (placement arguments and text), and the file of
  [#532](https://github.com/turboderp-org/exllamav3/pull/532) (a tensor-parallel load ran out of file descriptors on
  this `dev`). [engine/get_engine.sh](engine/get_engine.sh), [engine/build_native.sh](engine/build_native.sh)
  (sm_80 and sm_120a).
- **Server**: TabbyAPI fork [anoane/tabbyAPI](https://github.com/anoane/tabbyAPI), branch `deepseek-v41` (`ce250ff`):
  upstream `main` at `e086a65` plus the V4.1 tool-call format, `TABBY_EXL3_GENERATOR_CHUNK`, and API keys kept out of
  the log.
- **Software**: Ubuntu 26.04, kernel 7.0.0-38-generic, NVIDIA open driver 610.57.04 (patched, see above), CUDA 13.4,
  Python 3.12.15, torch 2.13.0+cu132.

## The layout, line by line

[serving/layouts/tp5-1017-35.env](serving/layouts/tp5-1017-35.env) and
[serving/ds41-exl3-tp5-1017-35.yml](serving/ds41-exl3-tp5-1017-35.yml):

- `CUDA_VISIBLE_DEVICES=0,1,3,4,2`: the four CMP 170HX first, the PRO last (PCI order 01, 41, 81 PRO, 82, C1).
- `gpu_split: [58, 58, 58, 58, 70]` with `tensor_parallel: true`, `tensor_parallel_backend: native`: every layer is
  split over the five cards in proportion to these limits; the native backend reduces through host memory.
- `draft_mode: mtp`, `draft_gpu_split: [0, 0, 0, 0, 20]`, `dynamic_draft: true`: the model's own draft head
  (DSpark), loaded by layers on the PRO beside the tensor-parallel model.
- `max_seq_len` and `cache_size` 1,048,576, FP16 cache (no KV quantization), `max_batch_size: 1`.
- `TABBY_EXL3_GENERATOR_CHUNK=16384` (prefill chunk), `EXLLAMA_NO_P2P_COPY=1`, `EXLLAMAV3_TUNE_CACHE` per engine.
- `EXL3_DSA_INDEXER_DTYPE=sm_80:int8,sm_120:int8`: the lightning indexer's prefill scoring in INT8 on every card
  (see [Lightning indexer in INT8](#lightning-indexer-in-int8) for the other values).
- `EXL3_INT8_GEMV=0`: the weight GEMVs with FP16 activations ([INT8 in decode: off](#int8-in-decode-off)); the
  fallback layouts keep upstream's default.
- Fallback: [serving/layouts/n4x-1017-35.env](serving/layouts/n4x-1017-35.env), the same model split by layers over
  the four cards that existed before the fifth (decode about two thirds of the above), also on engine `309ef6d1`
  with the INT8 indexer (checked: loads, tool calls, a 21K-token C prompt at 1,744 tok/s).
- FP16 indexer: [tp5-1017-35-fp16.env](serving/layouts/tp5-1017-35-fp16.env) (engine `5c0da061`, as served until
  2026-10-20) and [n4x-1017-35-fp16.env](serving/layouts/n4x-1017-35-fp16.env) (engine `16d01882`); unsetting the
  variable on engine `309ef6d1` gives the same FP16 scoring.

## Deploy

1. **Driver**: build and install the patched module set as in [hardware/p2p](hardware/p2p), then
   [hardware/p2p/cmp-pcie-gen2.conf](hardware/p2p/cmp-pcie-gen2.conf) in `/etc/modprobe.d/`; hold the kernel and
   driver packages. Check: all CMP 170HX at Gen2 x16.
2. **Power and boot units** (root): [serving/root](serving/root) to `/usr/local/sbin/` and `/etc/systemd/system/`
   (`gpu-power-caps`, `cpu-power-limit` with its drop-in, `cpu-governor-performance`, `ds41-ngram-prewarm`), then
   `systemctl enable` them. Adjust `EXPECT` and the caps in `gpu-power-caps` to your cards.
3. **Engine**: `engine/get_engine.sh`, then `engine/build_native.sh`.
4. **Model**: `engine/download_model.sh`, then
   `tools/make_view.py ~/workspace/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw ~/workspace/DeepSeek-V4.1-Flash-1017-engram ~/workspace/ds41-1017-3.5-view`
   and link it as `~/serve/models/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw`.
5. **Server**: clone `anoane/tabbyAPI` at `deepseek-v41` to `~/rnd/tabby-fresh`, create its `api_tokens.yml`.
6. **Serving**: the files of [serving](serving) to `~/serve` (layouts under `~/serve/layouts`), link
   `~/serve/ds41-exl3.env` to `layouts/tp5-1017-35.env`, install [serving/ds41-exl3.service](serving/ds41-exl3.service)
   (its first check expects exactly this card order by PCI device id: edit it for other cards), enable and start.
   The load takes about 3 minutes; the prewarm unit then reads the engram tables into the page cache.
7. **Proxy** (optional): [serving/proxy](serving/proxy) is the nginx site on :8443 in front of TabbyAPI, with Caddy
   only renewing the certificate (names replaced by `example.com`; the DNS token lives in an environment file).

Switch layouts: `ln -sfn layouts/<name>.env ~/serve/ds41-exl3.env && sudo systemctl restart ds41-exl3`.

## Limits and what was not measured

- **Not rebooted with this layout.** The machine was rebooted to add the fifth card (with the four-card layout
  linked); the driver options and power caps came back and are checked. The tensor-parallel layout has started from
  the unit many times, but not yet from a boot.
- **First long request after a load decodes slower** (about 47 tok/s once, then normal); a short warm-up request
  avoids it.
- **Stopping the server** logs `Timed out waiting for worker` while the tensor-parallel workers shut down; harmless
  here, and the exit error behind it is fixed upstream-side in
  [#534](https://github.com/turboderp-org/exllamav3/pull/534) (not in this engine commit).
- **Every restart of the tensor-parallel server writes tens of thousands of lines to the journal**: tracebacks
  of Python's multiprocessing resource tracker ending in `KeyError: '/<shared memory segment>'`, from the
  tensor-parallel shared-memory transfers. They name memory segments, not keys, and serving is unaffected.
- **Sampled decoding, other prompt types, and the vision tower were not measured** with this layout.
- **The INT8 indexer was judged on logits, not on task scores**: KL, top-1 and NLL against FP16 on one C corpus up
  to 1M tokens, and recall of three planted keys; the nine C-coding requests were not rerun with it.
- **One stream.** See Concurrency.
- **vLLM routes** for this model on these cards were looked at and not run: they need an sm_80 attention path for
  V4.1 and an EXL3 expert split on 128-channel blocks that are not public.

## Licence and credits

MIT for the files of this repository. DeepSeek-V4.1-Flash-1017 by DeepSeek (MIT); the EXL3 pack by LibertAI;
exllamav3 by turboderp (MIT); TabbyAPI by theroyallab (AGPL-3.0, used as a separate program); the CMP 170HX module
patches by Morrowmake and amoghmunikote (see their repositories).
