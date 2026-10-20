#!/bin/bash
# Reads DeepSeek's two engram table files into the page cache (no anonymous memory is committed: under pressure the
# kernel just drops pages). After it most rows of a new prompt come from the page cache instead of the SSD.
# The files are read with read-ahead off (POSIX_FADV_RANDOM), on purpose. A plain sequential read (cat) did not
# stay in the cache on this machine: while it ran, kswapd reclaimed page cache as fast as it was read although about
# 100 GiB of RAM were free, and of 189 GiB only the last 49 to 83 GiB were left. Read this way the cache fills and
# stays (about 176 of 189 GiB, which is all the memory there is beside the model).
# PREWARM_DROP=1 drops the whole page cache first (sudo), so that the time printed is the time from cold.
# The served model: the layout the unit reads names its config, the config names the model; every pack served here
# keeps the tables as model-00047 / 00048 of its base checkpoint (the 1017 views link the 1017 base files)
CFG=$(sed -n "s/^DS41_CONFIG=//p" $HOME/serve/ds41-exl3.env)
M=$HOME/serve/models/$(sed -n "s/^  model_name: *//p" "$CFG")
[ -e "$M/model-00047-of-00048.safetensors" ] || { echo "no n-gram tables under $M" >&2; exit 1; }
if [[ ${PREWARM_DROP:-0} == 1 ]]; then
  sync; echo 1 | sudo -n tee /proc/sys/vm/drop_caches > /dev/null || { echo "could not drop the page cache" >&2; exit 1; }
fi
echo "before: $(free -g | awk '/Mem:/ {print "used " $3 " GiB, cache " $6 " GiB, available " $7 " GiB"}')"
python3 - "$M"/model-00047-of-00048.safetensors "$M"/model-00048-of-00048.safetensors <<'PY' || exit 1
import os, sys, time
t0, total = time.time(), 0
for path in sys.argv[1:]:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
        while True:
            block = os.read(fd, 8 << 20)
            if not block:
                break
            total += len(block)
    finally:
        os.close(fd)
print(f"read {total} bytes in {time.time() - t0:.0f} s")
PY
echo "after:  $(free -g | awk '/Mem:/ {print "used " $3 " GiB, cache " $6 " GiB, available " $7 " GiB"}')"
