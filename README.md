# DeepSeek-V4.1-Flash-1017 EXL3 3.5 bpw at 1M context, tensor parallel over five cards (exllamav3 + open pull requests)

> **State of 2026-10-20. Layout `tp5-1017-35` of this repository is what serves on this machine**, started at boot
> by its systemd unit. Every figure below is a single run unless two are given, measured from the client unless
> it says server; what was not measured is in [Limits](#limits-and-what-was-not-measured).

DeepSeek-V4.1-Flash-1017 (the 17 October checkpoint) quantized to EXL3 3.5 bpw by LibertAI, at **1,048,576 tokens of
FP16 context**, one request at a time, on four CMP 170HX and one RTX PRO 6000 in **tensor parallel**: every layer is
split over all five cards. The engine is upstream exllamav3 `dev` plus open pull requests, served by a TabbyAPI
fork. Use: C coding (review, bug finding, fixes, new code, agent turns with tools).

```
CMP 170HX x4  (64 GB, sm_80, PCIe Gen2 x16, 150 W)   tensor parallel, at most 58 GB each
RTX PRO 6000  (96 GB, sm_120, PCIe x16, 400 W)       tensor parallel, at most 70 GB, plus the model's draft head (20 GB)
host RAM      (499 GiB)                              the two engram tables (2 x 94.6 GiB) in the page cache
```

## Results

### Huge C prompts, cold, 1M FP16 context

One request at a time against the served layout. Each prompt is a review request over the first N tokens of a C /
CUDA / C++ source corpus (concatenated sources and C headers, not published; 3.155 characters per token with the
model's tokenizer), with a unique first line so nothing comes from the prompt cache: **every prefill is cold**. The
engram tables were read into the page cache first, as the prewarm unit does after boot. Greedy, 600 new tokens,
reasoning on (the template's default). [tools/bench_long.py](tools/bench_long.py),
[results/long_c_prompts_tp5-dyn.txt](results/long_c_prompts_tp5-dyn.txt) and the server's figures in
[results/long_c_prompts_server_metrics.txt](results/long_c_prompts_server_metrics.txt).

| Prompt tokens | Time to first token | Prefill, client (prompt / TTFT) | Prefill, server | Decode, client | Decode, server |
|---:|---:|---:|---:|---:|---:|
| 26,296 | 14.2 s | 1,850 tok/s | 1,934 tok/s | 79.7 tok/s | 81.0 tok/s |
| 127,604 | 62.3 s | 2,049 | 2,062 | 75.8 | 73.0 |
| 253,055 | 131.6 s | 1,923 | 1,929 | 87.5 | 83.6 |
| 513,791 | 298.1 s | 1,724 | 1,728 | 94.6 | 88.2 |
| 1,002,677 | 680.0 s (11.3 min) | 1,475 | 1,476 | 83.0 | 81.1 |

Decode is drafted by the model's own head (dynamic draft length); it depends on how predictable the reply is, so
another text gives another figure. Client decode is tokens after the first divided by the time after the first;
the server's figure also counts the first token.

### The nine C-coding requests, cold

Recipe 08's request set for the drafter tuning (review of a C file, bug finding, a fix as a diff, functions
rewritten, a whole file returned, two agent turns with tool schemas, new code, an explanation), prompts of 139 to
9,426 tokens. It quotes GPL sources, so it is not published here; [tools/bench_stream.py](tools/bench_stream.py)
reads it from `WORKLOADS_DIR`. Fresh server, one stream, greedy, each request's own token limit, 1M FP16 cache.
Server figures from [results/coding_server_metrics_tp5m-dyn.txt](results/coding_server_metrics_tp5m-dyn.txt),
client figures from [results/raw/mtp-1020-0126/bench-tp5m-dyn.log](results/raw/mtp-1020-0126/bench-tp5m-dyn.log).

| Request | Prompt tokens | TTFT | Prefill, server | Tokens out | Decode, client |
|---|---:|---:|---:|---:|---:|
| review | 5,171 | 3.28 s | 1,647 tok/s | 1,500 | 91.9 tok/s |
| bug finding | 3,479 | 2.14 s | 1,689 | 615 | 105.7 |
| fix as a diff | 3,404 | 2.07 s | 1,702 | 403 | 120.0 |
| fix, functions rewritten | 3,405 | 2.09 s | 1,703 | 1,200 | 99.1 |
| whole file returned | 1,788 | 1.29 s | 1,478 | 2,600 | 99.1 |
| agent: edit call (tools) | 4,033 | 2.39 s | 1,754 | 208 | 115.0 |
| agent: later turn (tools) | 9,426 | 4.66 s | 2,067 | 136 | 124.7 |
| new code | 139 | 0.32 s | 579 | 1,500 | 83.2 |
| explanation | 5,158 | 2.76 s | 1,925 | 1,000 | 92.5 |

Mean decode 103.5 tok/s; the nine requests together gave 9,162 tokens in 116 s of wall time (78.6 tok/s with
prefill and gaps). Tool calls come back as OpenAI `tool_calls` (checked on the tensor-parallel layouts and the layer split, results/raw/*/toolcall.log).

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

Dynamic drafting was faster on 7 of the 9 requests and 3-4% slower on two (functions rewritten, whole file).

### Layouts tried, same model family, two 34K-token prompts and three 900-token answers

[tools/bench_log.py](tools/bench_log.py) requests (two different ~34K-token prompts of documentation text, then three 900-token
technical answers), warm-up first, fresh server each, server figures; prose rather than code, so these compare the
layouts with each other, not with the tables above ([results/raw](results/raw)):

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
- **Chat template**: [serving/chat_template.jinja](serving/chat_template.jinja), a port of the model's
  `encoding/encoding.py`; the 1017 encoder differs from the earlier one only for tool results and content blocks.
- **Engine**: exllamav3 fork [anoane/exllamav3-anemone](https://github.com/anoane/exllamav3-anemone), tag
  `recipe-09` = commit `5c0da061`: upstream `dev` at `f157fc0b` plus the open pull requests
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
- Fallback: [serving/layouts/n4x-1017-35.env](serving/layouts/n4x-1017-35.env), the same model split by layers over
  the four cards that existed before the fifth (decode about two thirds of the above).

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
- **Sampled decoding, other prompt types, and the vision tower were not measured** with this layout.
- **One stream.** See Concurrency.
- **vLLM routes** for this model on these cards were looked at and not run: they need an sm_80 attention path for
  V4.1 and an EXL3 expert split on 128-channel blocks that are not public.

## Licence and credits

MIT for the files of this repository. DeepSeek-V4.1-Flash-1017 by DeepSeek (MIT); the EXL3 pack by LibertAI;
exllamav3 by turboderp (MIT); TabbyAPI by theroyallab (AGPL-3.0, used as a separate program); the CMP 170HX module
patches by Morrowmake and amoghmunikote (see their repositories).
