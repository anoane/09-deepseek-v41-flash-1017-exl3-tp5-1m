#!/usr/bin/env python3
"""make_view.py <pack dir> <base dir> <out dir>: the folder TabbyAPI loads, made of links only.

LibertAI's EXL3 pack of DeepSeek-V4.1-Flash-1017 holds no engram tables and no chat template. The folder links every
file of the pack, the 1017 checkpoint's two engram shards (model-00047/00048-of-00048), adds the four engram tensors
to the pack's index, and copies this repository's chat template and tokenizer config (serving/: a port of the
model's encoding/encoding.py; the tokenizer itself is the pack's). The downloaded files are not changed.

On this machine the folder was made by an earlier script with the same steps (~/workspace/ds41-1017-3.5-view).
"""
import json, os, shutil, sys

pack, base, out = sys.argv[1:4]
here = os.path.dirname(os.path.abspath(__file__))
serving = os.path.join(here, "..", "serving")

index = json.load(open(os.path.join(pack, "model.safetensors.index.json")))
engram = {k: v for k, v in json.load(open(os.path.join(base, "model.safetensors.index.json")))["weight_map"].items()
          if ".engram.embed." in k}
assert len(engram) == 4, f"expected the 4 engram tensors in the base index, found {len(engram)}"
assert set(engram.values()) == {"model-00047-of-00048.safetensors", "model-00048-of-00048.safetensors"}
assert not set(engram) & set(index["weight_map"]), "the pack already holds engram tensors"
index["weight_map"].update(engram)

os.makedirs(out, exist_ok = True)
for name in os.listdir(pack):
    if name in ("model.safetensors.index.json", ".cache"):
        continue
    if not os.path.lexists(os.path.join(out, name)):
        os.symlink(os.path.join(os.path.abspath(pack), name), os.path.join(out, name))
for name in sorted(set(engram.values())):
    if not os.path.lexists(os.path.join(out, name)):
        os.symlink(os.path.join(os.path.abspath(base), name), os.path.join(out, name))
for name in ("chat_template.jinja", "tokenizer_config.json"):
    dst = os.path.join(out, name)
    if os.path.lexists(dst):
        os.remove(dst)
    shutil.copyfile(os.path.join(serving, name), dst)
json.dump(index, open(os.path.join(out, "model.safetensors.index.json"), "w"), indent = 2)
print(f"{out}: {len(index['weight_map'])} tensors, engram from {base}")
