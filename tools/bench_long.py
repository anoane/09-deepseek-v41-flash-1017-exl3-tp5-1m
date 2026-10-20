#!/usr/bin/env python3
"""bench_long.py <label> <port> <corpus file> <sizes in K tokens, e.g. 32,128,256,512,950>: huge C-code prompts,
measured from the client as a user sees them.

Each prompt is a review request over the first N tokens of a C / CUDA source corpus (a plain text file of
concatenated sources). The first line is unique per request, so nothing comes from the server's prompt cache: every
figure is a cold prefill (only the engram tables may sit in the page cache). Greedy, 600 new tokens, reasoning as the
server's template sets it. Printed per request: prompt tokens (the server's count), time to first token, effective
prefill (prompt tokens / time to first token: queue, tokenization and prefill together), decode tok/s after the first
token, and the tokens generated."""
import json, os, sys, time, urllib.request

label, port, corpus = sys.argv[1], sys.argv[2], sys.argv[3]
sizes = [int(x) for x in sys.argv[4].split(",")]
key = [l.split()[1] for l in open(os.path.expanduser(os.environ.get("TABBY_TOKENS", "~/rnd/tabby-fresh/api_tokens.yml")))
       if l.startswith("api_key:")][0]
text = open(corpus, errors = "replace").read()
CHARS_PER_TOKEN = 3.155    # measured with the model's tokenizer on the corpus used here; the server's count is reported

def run(n_k):
    body_text = text[: int(n_k * 1000 * CHARS_PER_TOKEN)]
    prompt = (f"Review session {time.time_ns()}.\n\nBelow is part of a C and CUDA code base.\n\n```c\n{body_text}\n```\n\n"
              "Find the three most likely real bugs (out-of-bounds access, unchecked return values, integer overflow in "
              "sizes or offsets). For each, quote the lines and explain the failure in two sentences.")
    body = {"model": "x", "max_tokens": 600, "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data = json.dumps(body).encode(),
                                 headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    t0, first, usage = time.time(), None, None
    with urllib.request.urlopen(req, timeout = 7200) as r:
        for line in r:
            line = line.decode(errors = "replace").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                usage = d["usage"]
            ch = (d.get("choices") or [{}])[0].get("delta") or {}
            if first is None and (ch.get("content") or ch.get("reasoning_content") or ch.get("reasoning")):
                first = time.time()
    t1 = time.time()
    p, n = (usage or {}).get("prompt_tokens", 0), (usage or {}).get("completion_tokens", 0)
    ttft = (first or t1) - t0
    dec = (n - 1) / max(t1 - (first or t1), 1e-9) if n > 1 else 0.0
    print(f"{label} prompt {p:>9,} tokens (cold) | ttft {ttft:7.1f} s | effective prefill {p / max(ttft, 1e-9):7.1f} tok/s"
          f" | decode {dec:5.1f} tok/s | {n} generated", flush = True)

for n_k in sizes:
    run(n_k)
