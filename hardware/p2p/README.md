# CMP 170HX: Gen2 x16 links and PCIe peer-to-peer

The four CMP 170HX run at PCIe Gen2 x16 and can copy to each other directly, through an NVIDIA open kernel module
set patched by [Morrowmake/cmpunlocker](https://github.com/Morrowmake/cmpunlocker) (its PCIe P2P fork of
[amoghmunikote/cmpunlocker](https://github.com/amoghmunikote/cmpunlocker)).

- Build: Morrowmake/cmpunlocker `1f2c498`, the patches in its `PATCH_ORDER`, profile `8gb` (the 64 GB geometry),
  on NVIDIA's open-gpu-kernel-modules 610.57.04, kernel 7.0.0-38-generic. [PROVENANCE.txt](PROVENANCE.txt) has the
  tarball and patch-set hashes and the module checksums. `nvidia.ko`'s srcversion equals the stock one (it hashes
  only the open interface sources): identify the module by its sha256, not by srcversion.
- Driver options: [cmp-pcie-gen2.conf](cmp-pcie-gen2.conf) in `/etc/modprobe.d/`:
  `RmForceEnableGen2=1;RMPcieLinkSpeed=0x1` (the Gen2 link) and `ForceP2P=0x11` (peer reads and writes).
- The upstream way is the fork's installer, `sudo ./install.sh --p2p --profile=8gb --no-iommu --no-passthrough`.
  Here the modules were built with its `driver/build_only.sh`, loaded once on a trial boot, checked with a content
  gate on every 170HX pair, and then made the boot default by [make_permanent.sh](make_permanent.sh);
  [rollback.sh](rollback.sh) puts the stock modules back. Both carry this machine's paths (the layout they name,
  `n4x`, was the layout served on 2026-10-18).
- What serving uses: nothing of it directly. The layouts set `EXLLAMA_NO_P2P_COPY=1`, and tensor parallel runs on
  exllamav3's native backend, which reduces through host memory; the NCCL backend, which would use the peer path,
  measured slower here (results/raw/tp4c-1019-2146). The Gen2 x16 links are what matters for this recipe.
- At boot nothing extra runs: the kernel's module loader takes the modules from
  `/lib/modules/7.0.0-38-generic/updates/cmpunlocker/` and the options from `/etc/modprobe.d/`. Checked after the
  reboot of 2026-10-19 that added the fourth 170HX: the loaded `nvidia.ko` has the patched build's sha256
  (`08050d7e…`, see PROVENANCE.txt), `/proc/driver/nvidia/params` shows the three options, and all four 170HX link at
  Gen2 x16. The kernel meta-packages (`linux-generic-hwe-26.04`, `linux-image-generic-hwe-26.04`, ...) and the NVIDIA
  610 driver packages are held (`apt-mark hold`): a new kernel or driver would boot the stock modules, which have
  neither the Gen2 link nor peer access, so a rebuild comes first.
  Check: `sha256sum $(modinfo -n nvidia)` against PROVENANCE.txt, and
  `nvidia-smi --query-gpu=name,pcie.link.gen.current,pcie.link.width.current --format=csv`.
