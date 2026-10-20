#!/bin/bash
# rollback.sh: run as root. Puts today's stock set (backup of 2026-10-18) back as the boot default: the five modules,
# the Gen2-only option line, depmod, initramfs (the backup's own file), the layout file; removes the trial GRUB entry.
# Checks every step. Reboot afterwards.
set -euo pipefail
KERNEL=7.0.0-38-generic
LIVE=/lib/modules/$KERNEL/updates/cmpunlocker
BACKUP=/var/backups/anemone-p2p/stock-6c442ee
OPTION_FILE=/etc/modprobe.d/cmp-pcie-gen2.conf
MODULES=(nvidia nvidia-modeset nvidia-uvm nvidia-drm nvidia-peermem)
fail() { echo "STOPPED: $*" >&2; exit 1; }
[ "$(id -u)" = 0 ] || fail "run as root"
(cd "$BACKUP" && sha256sum --quiet -c SHA256SUMS) || fail "the stock backup does not verify"
for module in "${MODULES[@]}"; do
    if mountpoint -q "$LIVE/$module.ko"; then umount "$LIVE/$module.ko"; fi
    install -m 644 "$BACKUP/updates-cmpunlocker/$module.ko" "$LIVE/$module.ko.new" && mv -f "$LIVE/$module.ko.new" "$LIVE/$module.ko"
    cmp -s "$BACKUP/updates-cmpunlocker/$module.ko" "$LIVE/$module.ko" || fail "$module.ko not restored"
done
if mountpoint -q "$OPTION_FILE"; then umount "$OPTION_FILE"; fi
install -m 644 "$BACKUP/cmp-pcie-gen2.conf" "$OPTION_FILE.new" && mv -f "$OPTION_FILE.new" "$OPTION_FILE"
cmp -s "$BACKUP/cmp-pcie-gen2.conf" "$OPTION_FILE" || fail "option file not restored"
depmod -a "$KERNEL"
install -m 644 "$BACKUP/initrd.img-$KERNEL" "/boot/initrd.img-$KERNEL.new" && mv -f "/boot/initrd.img-$KERNEL.new" "/boot/initrd.img-$KERNEL"
cmp -s "$BACKUP/initrd.img-$KERNEL" "/boot/initrd.img-$KERNEL" || fail "initramfs not restored"
[ -f "$BACKUP/n4x.env.before-p2p" ] && cp -a "$BACKUP/n4x.env.before-p2p" /home/ai/serve/layouts/n4x.env
rm -f /etc/grub.d/42_anemone_p2p_trial && update-grub
echo "ROLLED BACK: stock set restored; reboot, then check the loaded build id is 443f3716b8f3494ca5ce1caa1c8013aaa6996c5c"
