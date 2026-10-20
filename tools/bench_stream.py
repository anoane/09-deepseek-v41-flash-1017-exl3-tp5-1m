#!/usr/bin/env python3
"""bench_stream.py <label> <port> <model> [streams]: the nine C-coding requests of draft_workloads.py (the module named
by WORKLOADS_DIR, default ~/exp4090/cmds: recipe 08's drafter tuning set, not published here: it quotes GPL sources),
greedy, streamed. Per request: time to first token and decode tok/s after it (completion tokens from the server's
usage). With streams > 1 the requests run that many at a time and the aggregate tok/s is printed too."""
import json, os, sys, threading, time, urllib.request
sys.path.insert(0, os.path.expanduser(os.environ.get("WORKLOADS_DIR", "~/exp4090/cmds")))
import draft_workloads
label, port, model = sys.argv[1], sys.argv[2], sys.argv[3]
streams = int(sys.argv[4]) if len(sys.argv) > 4 else 1
key = [l.split()[1] for l in open(os.path.expanduser(os.environ.get("TABBY_TOKENS", "~/rnd/tabby-fresh/api_tokens.yml")))
       if l.startswith("api_key:")][0]
reqs = list(draft_workloads.requests())
lock, results = threading.Lock(), []

def one(name, messages, tools, max_tokens):
    for m in messages:
        for c in m.get("tool_calls") or []:
            if not isinstance(c["function"].get("arguments"), str):
                c["function"]["arguments"] = json.dumps(c["function"]["arguments"])
    body = {"model": model, "max_tokens": max_tokens, "temperature": 0, "messages": messages, "stream": True,
            "stream_options": {"include_usage": True}}
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data = json.dumps(body).encode(),
                                 headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    t0, first, usage = time.time(), None, None
    with urllib.request.urlopen(req, timeout = 3600) as r:
        for line in r:
            line = line.decode(errors = "replace").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                usage = d["usage"]
            ch = (d.get("choices") or [{}])[0].get("delta") or {}
            if first is None and (ch.get("content") or ch.get("reasoning_content") or ch.get("reasoning") or ch.get("tool_calls")):
                first = time.time()
    t1 = time.time()
    n = (usage or {}).get("completion_tokens", 0)
    dec = (n - 1) / max(t1 - (first or t1), 1e-9) if n > 1 else 0.0
    with lock:
        results.append((name, n, (first or t1) - t0, dec, t0, t1))
        print(f"{label} {name:12s} ttft {(first or t1) - t0:6.2f} s | {n:5d} tokens, decode {dec:6.1f} tok/s", flush = True)

queue = list(reqs) * (1 if streams == 1 else max(1, -(-streams * 2 // len(reqs))))
def worker():
    while True:
        with lock:
            if not queue:
                return
            name, kind, messages, tools, mt = queue.pop(0)
        one(name, json.loads(json.dumps(messages)), tools, mt)
t = time.time()
th = [threading.Thread(target = worker) for _ in range(streams)]
[x.start() for x in th]; [x.join() for x in th]
wall = time.time() - t
decs = [r[3] for r in results if r[3] > 0]
print(f"{label} streams={streams}: mean per-stream decode {sum(decs) / max(len(decs), 1):.1f} tok/s over {len(decs)} requests; "
      f"aggregate {sum(r[1] for r in results) / wall:.1f} tok/s ({sum(r[1] for r in results)} tokens in {wall:.0f} s)", flush = True)
