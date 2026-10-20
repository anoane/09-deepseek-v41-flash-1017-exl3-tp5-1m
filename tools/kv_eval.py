#!/usr/bin/env python3
"""kv_eval.py: what lower-bit context formats do to DeepSeek-V4.1-Flash-1017 at up to 1M tokens, on C code.

One model load (model_init arguments, as serving loads it) runs one or more modes over the same ~1M-token chat:
a user turn holding the C / CUDA corpus with three planted facts, then a question about them.
    f16    the pools as serving keeps them (the baseline; its log-probabilities are kept for the others)
    floor  f16 with 8,192-token prefill chunks instead of 16,384: a numerical change serving already lives with
    ref    DeepSeek's reference formats, simulated (kv_sim.py)
    prov   the serving stacks' FP8 cache, simulated (kv_sim.py)
    real   whatever the loaded engine and cache are (-cq 8 / -cq 4 packed pools, EXL3_DSA_INDEXER_DTYPE, ...)
ref and prov need a test copy of the engine with kv_sim.py and a two-line hook in DSV4Attention, which this
repository does not carry; results/indexer_dtype.txt uses f16 (the baseline) and real only.
Example (the tensor-parallel rows): kv_eval.py -m <model folder> -tp -tpb native -gs 58,58,58,58,70 -cs 1048576
-ambs 1 -chunk_size 4096 --modes real --label int8 --out <dir> --corpus <C corpus> -idt sm_80:int8,sm_120:int8
Per mode: at four depths the log-probabilities of the next 1,024 corpus tokens (teacher forced) against f16
(KL, top-1 agreement) and the corpus (NLL, paired difference with its standard error); prefill time per segment;
the greedy answer to the question at ~1M and its decode time per token; a short-context (16K) answer and its
decode time. Results go to <out>/<mode>.json, the f16 log-probabilities to <out>/f16_<depth>.pt."""
import argparse, json, os, random, subprocess, sys, time
import torch

CHUNK = 16384
EVAL_LEN = 1024
DEPTHS = [16384, 131072, 524288, 1015808]
NEEDLE_FRACTIONS = [0.1, 0.5, 0.9]
GEN_LONG = 160
GEN_SHORT = 128
SHORT_CTX = 16384
QUESTION_LONG = ("Three comments in the code above each give a release key for a module. List every module with its "
                 "release key, one per line, as `module: key`. Nothing else.")
MODEL_DIR = None
QUESTION_SHORT = "In two sentences: what does the first file above define, and what is its contract?"


def build_text(corpus_path, seed):
    text = open(corpus_path, errors = "replace").read()
    rng = random.Random(seed)
    needles = []
    for k, f in enumerate(NEEDLE_FRACTIONS):
        module = ["lane_scheduler", "page_mirror", "draft_verifier"][k]
        key = "%04X-%04X" % (rng.randrange(65536), rng.randrange(65536))
        needles.append((module, key, f))
    return text, needles


def render_template(model_dir, content):
    """The model folder's chat template, rendered as TabbyAPI renders it (jinja2 sandbox), thinking off."""
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    from jinja2.ext import loopcontrols
    env = ImmutableSandboxedEnvironment(trim_blocks = True, lstrip_blocks = True, extensions = [loopcontrols])
    env.filters["tojson"] = lambda x, ensure_ascii = False, indent = None, separators = None, sort_keys = False: \
        json.dumps(x, ensure_ascii = ensure_ascii, indent = indent, separators = separators, sort_keys = sort_keys)
    def raise_exception(m):
        raise ValueError(m)
    env.globals["raise_exception"] = raise_exception
    tpl = env.from_string(open(os.path.join(model_dir, "chat_template.jinja")).read())
    return tpl.render(messages = [{"role": "user", "content": content}], add_generation_prompt = True,
                      bos_token = "<｜begin▁of▁sentence｜>", eos_token = "<｜end▁of▁sentence｜>", enable_thinking = False)


def make_ids(tokenizer, text, needles, n_corpus_tokens, question):
    """Template head + corpus (first n tokens, needles planted at their fractions on line boundaries) + tail."""
    probe = "⁣PLACEHOLDER⁣"
    rendered = render_template(MODEL_DIR, probe)
    assert rendered.count(probe) == 1, "template rendering does not hold the placeholder once"
    head_s, tail_s = rendered.split(probe)
    head = tokenizer.encode(head_s, encode_special_tokens = True)[0]
    intro = tokenizer.encode("Below is part of a C and CUDA code base.\n\n```c\n")[0]
    tail = torch.cat([tokenizer.encode("\n```\n\n" + question)[0], tokenizer.encode(tail_s, encode_special_tokens = True)[0]])
    body = tokenizer.encode(text[: int(n_corpus_tokens * 4.5)])[0]
    assert body.shape[0] >= n_corpus_tokens, f"corpus too short: {body.shape[0]} tokens < {n_corpus_tokens}"
    body = body[:n_corpus_tokens]
    planted = []
    if needles:
        # plant on the token stream: each needle as its own line after the first newline past its fraction
        nl = set(tokenizer.encode("\n")[0].tolist())
        pieces, last = [], 0
        for module, key, f in needles:
            at = int(f * n_corpus_tokens)
            while at < n_corpus_tokens and int(body[at]) not in nl:
                at += 1
            pieces.append(body[last:at + 1])
            pieces.append(tokenizer.encode(f"/* release key for the {module} module: {key} */\n")[0])
            planted.append((module, key, int(head.shape[0] + intro.shape[0] + sum(p.shape[0] for p in pieces[:-1]))))
            last = at + 1
        pieces.append(body[last:])
        body = torch.cat(pieces)[:n_corpus_tokens]
    prefix = torch.cat([head, intro, body])
    return prefix, tail, planted


def gpu_memory():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                         capture_output = True, text = True).stdout
    return {int(a): int(b) for a, b in (l.split(",") for l in out.strip().splitlines())}


def main():
    from exllamav3 import model_init
    parser = argparse.ArgumentParser()
    model_init.add_args(parser, cache = True)
    parser.add_argument("--modes", required = True)
    parser.add_argument("--out", required = True)
    parser.add_argument("--corpus", required = True)
    parser.add_argument("--dry", action = "store_true")
    parser.add_argument("--label", default = None, help = "result file name (default: the mode)")
    parser.add_argument("--depths", default = None, help = "comma list of eval depths (default: 16K, 131K, 524K, 1M)")
    args = parser.parse_args()
    global MODEL_DIR, DEPTHS
    MODEL_DIR = args.model_dir
    if args.depths:
        DEPTHS = [int(x) for x in args.depths.split(",")]
    os.makedirs(args.out, exist_ok = True)
    text, needles = build_text(args.corpus, 20261020)

    if args.dry:
        from exllamav3 import Config, Tokenizer
        tokenizer = Tokenizer.from_config(Config.from_directory(args.model_dir))
        prefix, tail, planted = make_ids(tokenizer, text, needles, DEPTHS[-1] + EVAL_LEN, QUESTION_LONG)
        print("prefix", prefix.shape[0], "tail", tail.shape[0], "planted", planted, "needles", needles)
        print("tail text:", repr(tokenizer.decode(tail, decode_special_tokens = True)))
        s, st, _ = make_ids(tokenizer, text, [], SHORT_CTX, QUESTION_SHORT)
        print("short", s.shape[0], st.shape[0], repr(tokenizer.decode(s[:40], decode_special_tokens = True)))
        return

    t_load = time.time()
    model, config, cache, tokenizer, *_ = model_init.init(args)
    t_load = time.time() - t_load
    mem_loaded = gpu_memory()
    pool_bytes = sum(l.storage_size() for l in cache.layers.values() if hasattr(l, "storage_size"))
    print(f"loaded in {t_load:.0f} s, cache layers {len(cache.layers)}, pool storage {pool_bytes / 2**30:.3f} GiB per copy,"
          f" memory used {mem_loaded}", flush = True)
    prefix, tail, planted = make_ids(tokenizer, text, needles, DEPTHS[-1] + EVAL_LEN, QUESTION_LONG)
    head_len = prefix.shape[0] - (DEPTHS[-1] + EVAL_LEN)
    short, short_tail, _ = make_ids(tokenizer, text, [], SHORT_CTX, QUESTION_SHORT)
    eos = set(config.eos_token_id_list if isinstance(config.eos_token_id_list, list) else [config.eos_token_id_list])
    num_pages = cache.max_num_tokens // 256
    block_table = torch.arange(num_pages, dtype = torch.int32).unsqueeze(0)

    def params_for(state, pos, mode):
        p = {"attn_mode": "flash_attn", "block_table": block_table, "cache": cache,
             "cache_seqlens": torch.tensor([pos], dtype = torch.int32), "recurrent_states": [state]}
        if mode in ("ref", "prov"):
            p["kv_sim"] = mode
        return p

    def greedy(state, pos, ids_tail, mode, n):
        logits = model.forward(input_ids = ids_tail.unsqueeze(0), params = params_for(state, pos, mode))
        pos += ids_tail.shape[0]
        out, times = [], []
        tok = int(logits[0, -1].argmax())
        for _ in range(n):
            out.append(tok)
            if tok in eos:
                break
            t0 = time.time()
            logits = model.forward(input_ids = torch.tensor([[tok]]), params = params_for(state, pos, mode))
            tok = int(logits[0, -1].argmax())
            times.append(time.time() - t0)
            pos += 1
        return out, times

    baseline = {}
    for mode in args.modes.split(","):
        chunk = 8192 if mode == "floor" else CHUNK
        res = {"mode": mode, "label": args.label, "indexer_dtype": os.environ.get("EXL3_DSA_INDEXER_DTYPE"),
               "cache_quant": args.cache_quant, "load_s": t_load, "pool_gib_per_copy": pool_bytes / 2**30,
               "memory_used_mib_after_load": mem_loaded, "chunk": chunk, "segments": [], "depths": []}
        state = cache.get_new_state()
        pos = 0
        t_mode = time.time()
        for depth in DEPTHS:
            stop = head_len + depth
            t0 = time.time()
            while pos < stop:
                n = min(chunk, stop - pos)
                model.prefill(input_ids = prefix[pos:pos + n].unsqueeze(0), params = params_for(state, pos, mode))
                pos += n
            ev = prefix[pos:pos + EVAL_LEN]
            logits = model.forward(input_ids = ev.unsqueeze(0), params = params_for(state, pos, mode))
            t_seg = time.time() - t0
            lp = torch.log_softmax(logits[0, :-1].float().cpu(), dim = -1)
            pos += EVAL_LEN
            target = ev[1:].long()
            nll = -lp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
            d = {"depth": depth, "context_at_eval": stop, "nll": float(nll.mean()), "segment_s": t_seg}
            if mode == "f16":
                baseline[depth] = (lp, nll)
                torch.save({"lp": lp, "nll": nll}, os.path.join(args.out, f"f16_{depth}.pt"))
            else:
                if depth not in baseline:
                    b = torch.load(os.path.join(args.out, f"f16_{depth}.pt"))
                    baseline[depth] = (b["lp"], b["nll"])
                blp, bnll = baseline[depth]
                kl = (blp.exp() * (blp - lp)).sum(-1)
                dn = nll - bnll
                d.update({"kl_mean": float(kl.mean()), "kl_p99": float(kl.quantile(0.99)), "kl_max": float(kl.max()),
                          "top1_agree": float((blp.argmax(-1) == lp.argmax(-1)).float().mean()),
                          "dnll_mean": float(dn.mean()), "dnll_se": float(dn.std() / dn.numel() ** 0.5),
                          "nll_f16": float(bnll.mean())})
            res["depths"].append(d)
            print(mode, json.dumps(d), flush = True)
        res["prefill_s_total"] = time.time() - t_mode
        res["context_tokens"] = pos
        out, times = greedy(state, pos, tail, mode, GEN_LONG)
        answer = tokenizer.decode(torch.tensor(out), decode_special_tokens = False)
        res["long"] = {"context": pos + tail.shape[0], "answer": answer, "generated": len(out),
                       "decode_tok_s": (len(times) / sum(times)) if times else 0.0,
                       "needles_found": [k for _, k, _ in planted if k in answer], "needles": planted}
        res["memory_used_mib_at_1m"] = gpu_memory()
        print(mode, "long:", json.dumps(res["long"]), flush = True)
        state.free()
        state = cache.get_new_state()
        p = 0
        while p < short.shape[0]:
            n = min(chunk, short.shape[0] - p)
            model.prefill(input_ids = short[p:p + n].unsqueeze(0), params = params_for(state, p, mode))
            p += n
        out, times = greedy(state, p, short_tail, mode, GEN_SHORT)
        res["short"] = {"context": p + short_tail.shape[0], "answer": tokenizer.decode(torch.tensor(out)),
                        "generated": len(out), "decode_tok_s": (len(times) / sum(times)) if times else 0.0,
                        "tokens": out}
        print(mode, "short:", json.dumps({k: v for k, v in res["short"].items() if k != "tokens"}), flush = True)
        state.free()
        name = args.label or f"{mode}{'' if mode != 'real' else '_q' + str(args.cache_quant)}"
        json.dump(res, open(os.path.join(args.out, f"{name}.json"), "w"), indent = 1)
    print("KV-EVAL-DONE", flush = True)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
