# bench_indexer.py <engine source dir>: the lightning indexer's prefill scoring as DSV4Attention runs it (keys
# quantized once per pass, one 256-row slab call timed), for every EXL3_DSA_INDEXER_DTYPE type the visible card runs,
# at 512K and 1M paged keys. Run with one card visible (CUDA_VISIBLE_DEVICES) and the native module on PYTHONPATH.
import os, sys, torch, triton
sys.path.insert(0, sys.argv[1])
import exllamav3.modules.attention_fn.dsa_triton as dt
dev = torch.device("cuda", 0)
cc = torch.cuda.get_device_capability(dev)
arch = f"sm_{cc[0]}{cc[1]}"
R, H, D, EPP = 256, 32, 128, 256

def timed(fn, reps = 10):
    fn(); torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
    a.record()
    for _ in range(reps):
        fn()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / reps

for T in (524288, 1048576):
    pool = (torch.randn(T, D, device = dev) * 0.7).half()
    bt = torch.randperm(T // EPP, device = dev).int().unsqueeze(0)
    q = (torch.randn(R, H, D, device = dev) * 0.7).half()
    w = torch.randn(R, H, device = dev).half()
    S = torch.empty((R, triton.next_power_of_2(T)), dtype = torch.half, device = dev)
    line = f"{torch.cuda.get_device_name(dev)[:22]:22s} T {T:>8}"
    for d in ["fp16"] + [x for x, need in dt.INDEXER_DTYPES.items() if x != "fp16" and cc >= need]:
        os.environ["EXL3_DSA_INDEXER_DTYPE"] = f"{arch}:{d}"
        dt._indexer_dtypes.clear()
        tk = timed(lambda: dt.dsa_indexer_keys(pool, T, bt, EPP)) if d != "fp16" else 0.0
        kq = dt.dsa_indexer_keys(pool, T, bt, EPP)
        ts = timed(lambda: dt.dsa_indexer_scores(q, w, pool, T - R, 1, T, scores = S, block_table = bt, epp = EPP, k_quant = kq))
        line += f" | {d} slab {ts:6.2f} ms keys {tk:5.2f} ms"
    print(line, flush = True)
    del pool, S
    torch.cuda.empty_cache()
