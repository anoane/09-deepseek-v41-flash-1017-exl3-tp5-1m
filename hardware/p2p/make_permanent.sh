#!/bin/bash
# make_permanent.sh: run as root ON THE TRIAL BOOT, only after the gate passed on every CMP 170HX pair. Makes the staged
# peer-capable set the boot default: its five modules over updates/cmpunlocker (the stock ones are in
# /var/backups/anemone-p2p/stock-6c442ee), the ForceP2P option line, depmod, initramfs; serving keeps off peer copies
# (EXLLAMA_NO_P2P_COPY=1 in its layout file) until its golden answers match the stock module's. Checks every step;
# the undo is rollback.sh.
set -euo pipefail
KERNEL=7.0.0-38-generic
STAGE=/opt/anemone-p2p/610.57.04-p2p-b
LIVE=/lib/modules/$KERNEL/updates/cmpunlocker
BACKUP=/var/backups/anemone-p2p/stock-6c442ee
OPTION_FILE=/etc/modprobe.d/cmp-pcie-gen2.conf
LAYOUT=/home/ai/serve/layouts/n4x.env
MODULES=(nvidia nvidia-modeset nvidia-uvm nvidia-drm nvidia-peermem)
fail() { echo "STOPPED: $*" >&2; exit 1; }

[ "$(id -u)" = 0 ] || fail "run as root"
[ "$(uname -r)" = "$KERNEL" ] || fail "kernel $(uname -r)"
(cd "$BACKUP" && sha256sum --quiet -c SHA256SUMS) || fail "the stock backup does not verify"
(cd "$STAGE" && sha256sum --quiet -c SHA256SUMS) || fail "the staged set does not verify"
for module in "${MODULES[@]}"; do                                  # the trial's bind mounts first, if any
    if mountpoint -q "$LIVE/$module.ko"; then umount "$LIVE/$module.ko"; fi
done
if mountpoint -q "$OPTION_FILE"; then umount "$OPTION_FILE"; fi
for module in "${MODULES[@]}"; do
    install -m 644 "$STAGE/$module.ko" "$LIVE/$module.ko.new"
    mv -f "$LIVE/$module.ko.new" "$LIVE/$module.ko"
    cmp -s "$STAGE/$module.ko" "$LIVE/$module.ko" || fail "$module.ko not in place"
done
install -m 644 "$STAGE/cmp-pcie-gen2.conf" "$OPTION_FILE.new" && mv -f "$OPTION_FILE.new" "$OPTION_FILE"
cmp -s "$STAGE/cmp-pcie-gen2.conf" "$OPTION_FILE" || fail "option file not in place"
depmod -a "$KERNEL"
[ "$(modinfo -n nvidia)" = "$LIVE/nvidia.ko" ] || fail "modinfo -n nvidia names $(modinfo -n nvidia)"
update-initramfs -u -k "$KERNEL"
SCRATCH=$(mktemp -d)
trap 'rm -rf "$SCRATCH"' EXIT
unmkinitramfs "/boot/initrd.img-$KERNEL" "$SCRATCH"
found=$(find "$SCRATCH" -path "*updates/cmpunlocker/nvidia.ko" | head -n 1)
# the initramfs keeps modules stripped of debug data: compare build ids. On a boot whose command line blacklists the
# nvidia modules (the trial boot) the builder leaves them out; nvidia then loads from disk, and a rebuild on a normal
# boot puts them back.
staged_id=$(readelf -n "$STAGE/nvidia.ko" | awk '/Build ID/ { print $3; exit }')
if [ -z "$found" ] && modprobe -c | grep -qx 'blacklist nvidia'; then
    echo "NOTE: nvidia is blacklisted on this boot's command line, so the initramfs leaves it out; rebuild it after a normal boot"
else
    [ -n "$found" ] && [ "$(readelf -n "$found" | awk '/Build ID/ { print $3; exit }')" = "$staged_id" ] \
        || fail "the new initramfs does not carry the staged nvidia.ko"
fi
grep -rqs 'ForceP2P=0x11' "$SCRATCH"/*/etc/modprobe.d/cmp-pcie-gen2.conf "$SCRATCH"/etc/modprobe.d/cmp-pcie-gen2.conf \
    || fail "the new initramfs lacks the ForceP2P option line"
if ! grep -qx 'EXLLAMA_NO_P2P_COPY=1' "$LAYOUT"; then
    cp -a "$LAYOUT" "$BACKUP/n4x.env.before-p2p"
    printf '%s\n' '# P2P module since 2026-10-18: serving bounces device copies through host memory until its golden answers' \
        '# are compared with peer copies on (design/peer_to_peer.md 6, D5).' 'EXLLAMA_NO_P2P_COPY=1' >> "$LAYOUT"
fi
echo "PERMANENT: the peer-capable set is the boot default from the next boot; serving with EXLLAMA_NO_P2P_COPY=1"
