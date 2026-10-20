#!/usr/bin/env python3
"""bench_log.py <label> <server log> <port>: ~/serve/bench_api.py's requests (warm-up, a short request, two different
long prompts of about 34K tokens, three 900-token answers) against a server started by hand; speeds are the server's
own figures, read from its log instead of the unit's journal."""
import json, os, re, sys, time, urllib.request
label, log, port = sys.argv[1], sys.argv[2], sys.argv[3]
home = os.path.expanduser("~")
key = [l.split()[1] for l in open(home + "/rnd/tabby-fresh/api_tokens.yml") if l.startswith("api_key:")][0]
doc = home + "/rnd/release/all/src-527900a320a8/doc/"
text = "".join(open(doc + f).read() for f in sorted(os.listdir(doc)) if f.endswith(".md"))
A, B = text[:118000], text[150000:268000]
RE = re.compile(r"(\d+) tokens generated in ([\d.]+) seconds \(Queue: [\d.]+ s, Process: (\d+) cached tokens and (\d+) new "
                r"tokens at ([\d.]+) T/s, Generate: ([\d.]+) T/s, Context: (\d+) tokens")
def ask(name, content, max_tokens):
    start = os.path.getsize(log)
    body = json.dumps({"model": "x", "max_tokens": max_tokens, "temperature": 0,
                       "messages": [{"role": "user", "content": content}]}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data = body,
                                 headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout = 1800))
    dt = time.time() - t0
    time.sleep(1.5)
    with open(log, errors = "replace") as f:
        f.seek(start)
        m = RE.findall(" ".join(f.read().split()))
    reply = (r["choices"][0]["message"].get("content") or "")[:70].replace("\n", " ")
    if not m:
        # TabbyAPI main also writes: "N tokens generated at G T/s · prompt P tokens, none cached, P new in S s"
        with open(log, errors = "replace") as f:
            f.seek(start)
            t = " ".join(f.read().split()).replace(",", "")
        m2 = re.findall(r"(\d+) tokens generated at ([\d.]+) T/s · prompt (\d+) tokens (?:none cached|(\d+) cached) (\d+) new in ([\d.]+) s", t)
        if m2:
            gen, gps, _, cached, new, psecs = m2[-1]
            m = [(gen, "0", cached or "0", new, str(int(new) / max(float(psecs), 1e-9)), gps, "0")]
    if m:
        gen, secs, cached, new, pps, gps, ctx = m[-1]
        print(f"{label:9s} {name:14s} wall {dt:6.1f} s | prompt {new:>6s} new + {cached} cached, prefill {float(pps):8.1f} tok/s"
              f" | {gen:>4s} generated, decode {float(gps):5.1f} tok/s | {reply}", flush = True)
    else:
        print(f"{label:9s} {name:14s} wall {dt:6.1f} s | no metrics line | {reply}", flush = True)
ask("warm-up", "Reply with exactly: ok", 32)
ask("short", "Reply with exactly: ok", 32)
ask("long prompt A", A + "\n\nSummarize the text above in five bullet points.", 300)
ask("long prompt B", B + "\n\nSummarize the text above in five bullet points.", 300)
ask("long answer", "Write a detailed, step-by-step explanation of how TCP congestion control works (slow start, congestion avoidance, fast retransmit, fast recovery).", 900)
ask("long answer 2", "Explain how a modern CPU executes instructions out of order: renaming, the reorder buffer, speculation and recovery. Be detailed.", 900)
ask("long answer 3", "Describe the Raft consensus algorithm in detail: leader election, log replication, safety, membership changes.", 900)
