# compare.py: renders the same conversations with DeepSeek's 1017 encoding.py (the reference), our chat template and
# HF PR #11's chat template (jinja2 as transformers/TabbyAPI use it), and reports exact matches and first differences
# Usage: in a folder holding ours_chat_template.jinja and pr11_chat_template.jinja, python compare_templates.py
# (ENCODING_PY: the 1017 repository's encoding/encoding.py). Output: results/template_comparison.txt
import copy, importlib.util, json, os
from jinja2.sandbox import ImmutableSandboxedEnvironment
from jinja2.ext import loopcontrols

ENC = os.path.expanduser(os.environ.get("ENCODING_PY", "~/anemone/models/DeepSeek-V4.1-Flash-1017/encoding/encoding.py"))
spec = importlib.util.spec_from_file_location("enc", ENC)
enc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(enc)
BOS, EOS = "<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>"


def raise_exception(m):
    raise ValueError(m)


def env():
    e = ImmutableSandboxedEnvironment(trim_blocks = True, lstrip_blocks = True, extensions = [loopcontrols])
    e.filters["tojson"] = lambda x, ensure_ascii = False, indent = None, separators = None, sort_keys = False: \
        json.dumps(x, ensure_ascii = ensure_ascii, indent = indent, separators = separators, sort_keys = sort_keys)
    e.globals["raise_exception"] = raise_exception
    return e


T = {n: env().from_string(open(f).read()) for n, f in (("ours", "ours_chat_template.jinja"), ("pr11", "pr11_chat_template.jinja"))}
TOOLS = [{"type": "function", "function": {"name": "read", "description": "Read a file",
          "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]


def call(args):
    return [{"id": "c1", "type": "function", "function": {"name": "read", "arguments": args}}]


U = {"role": "user", "content": "Fix the bug in sum.c"}
A_TOOL = lambda args: {"role": "assistant", "reasoning_content": "Read it.", "content": "", "tool_calls": call(args)}
R_TOOL = {"role": "tool", "tool_call_id": "c1", "content": "int sum(...)"}
CASES = [
    ("plain", [U], {}, {}),
    ("system", [{"role": "system", "content": "You review C code."}, U], {}, {}),
    ("thinking off", [U], {"thinking_mode": "chat"}, {"enable_thinking": False}),
    ("effort low", [U], {"reasoning_effort": "low"}, {"reasoning_effort": "low"}),
    ("effort max", [U], {"reasoning_effort": "max"}, {"reasoning_effort": "max"}),
    ("effort 30", [U], {"reasoning_effort": 30}, {"reasoning_effort": 30}),
    ("multi-turn", [U, {"role": "assistant", "reasoning_content": "Off by one.", "content": "Use i < n."},
                    {"role": "user", "content": "Now add a test."}], {}, {}),
    ("tool, object args", [U, A_TOOL({"path": "sum.c"}), R_TOOL], {"tools": TOOLS}, {"tools": TOOLS}),
    ("tool, string args", [U, A_TOOL('{"path": "sum.c"}'), R_TOOL], {"tools": TOOLS}, {"tools": TOOLS}),
    ("tool result, no tools declared", [U, A_TOOL({"path": "sum.c"}), R_TOOL], {}, {}),
    ("tool result then answer", [U, A_TOOL({"path": "sum.c"}), R_TOOL,
                                 {"role": "assistant", "reasoning_content": "Found it.", "content": "Use i < n."},
                                 {"role": "user", "content": "Thanks, now a test."}], {"tools": TOOLS}, {"tools": TOOLS}),
]

for name, msgs, enc_kw, tpl_kw in CASES:
    m = copy.deepcopy(msgs)
    kw = dict(enc_kw)
    tools = kw.pop("tools", None)
    if tools:
        if m[0]["role"] == "system":
            m[0]["tools"] = tools
        else:
            m.insert(0, {"role": "system", "content": "", "tools": tools})
    try:
        ref = enc.encode_messages(m, thinking_mode = kw.pop("thinking_mode", "thinking"), **kw)
    except Exception as e:
        ref = f"<encoder raised {type(e).__name__}: {e}>"
    out = []
    for tn, t in T.items():
        try:
            r = t.render(messages = copy.deepcopy(msgs), add_generation_prompt = True, bos_token = BOS, eos_token = EOS, **tpl_kw)
        except Exception as e:
            r = f"<template raised {type(e).__name__}: {str(e)[:70]}>"
        if r == ref:
            out.append(f"{tn}=MATCH")
        else:
            i = next((k for k in range(min(len(r), len(ref))) if r[k] != ref[k]), min(len(r), len(ref)))
            out.append(f"{tn}=DIFF@{i} ref:{ref[max(0, i - 15):i + 35]!r} got:{r[max(0, i - 15):i + 35]!r}")
    print(f"{name:30s} " + " | ".join(out))
