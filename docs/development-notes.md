# Development Notes

## Template provenance

Vendored from https://github.com/obsproject/obs-plugintemplate at upstream commit
`3e7d7ac3b5342cd7d9b88890b9c70b472d1520fc` (shallow clone, 2026-07-26). For future template upgrades, three-way
merge against this base.

## Design amendments (2026-07-26)

- **Linux-only scope** — Windows/macOS support dropped by user decision. Spec and
  Plan 1 updated. Consequences for the vendored template: non-Linux CMake
  configure presets, CI jobs, helper scripts, and buildspec Win/mac keys
  (windows-*, macos-*) were removed in the Linux-only cleanup commit. The macOS
  `bundleId` in `buildspec.json` is retained only because the vendored
  `cmake/common/bootstrap.cmake` reads it unconditionally.
- **Transparent replace mode** — background replacement supports two modes:
  transparent alpha (scene sources below show through) or image replacement.

## Follow-ups for Plan 5 (packaging & release)

- `buildspec.json`: `displayName` still "Plugin Template for OBS" — rebrand
  before first release (it drives CI artifact naming).
- Template source files (`src/plugin-support.*`, `src/plugin-main.c`) still carry
  GPLv2 copyright header comments while the project LICENSE is MIT — relicense
  headers when those files are substantively rewritten (Plan 2).
- CI presets (`*-ci-*`) share `binaryDir` with local presets; running a CI preset
  locally leaves `BUILD_TESTING=FALSE` sticky in the shared cache. Give CI presets
  their own binaryDir if this bites.
- GitHub Actions CI is configured but never executed: first push to GitHub will
  validate the Linux leg (ubuntu-24.04 build) + check-format gate — the only
  remaining jobs after the cleanup above.
- Non-English systems: the OBS smoke log shows a benign
  `Failed to load '<locale>' text for module: 'obs-cam-effects.so'` warning
  (e.g. fr-FR) — locale fallback, not a load failure.

## Plan 2 state (segmentation pipeline)

- Background modes working: off / transparent / image / blur (single mode
  selector — combinable toggles deferred deliberately).
- CPU-only inference (PP-HumanSeg, ~7ms/frame; soak test ~140fps sustained
  with drop policy engaged). CUDA execution provider: Plan 3 with the
  Quality tier.
- Guided filter runs at 192x192; GPU bilinear upscale at composite.
- Failure modes implemented per spec: passthrough (default) / freeze-last-
  frame (black before first processed frame; frozen output drawn scaled on
  source resize).
- User visual verification (2026-07-26): transparent compositing OK, image
  replace OK, quality rated "decent", no flicker/ghosting/wrong-position/
  fps issues observed. Blur mode implemented (Kawase, strength 1-4) and
  verified crash-free on the render path; user visual confirmation pending.
- Known limitations:
  - Model load + ORT session creation happen under the graphics lock on
    first non-off activation (one-time render stall, tens of ms).
  - Staging a SCENE target whose items use item_render (crop/scale filters,
    non-default blend) drops those items from the 192x192 segmentation
    input only (libobs effect-loop nesting guard); main composite
    unaffected. Same limitation exists in stock OBS filter-bypass paths.
  - blur_strength is a plain int written on UI thread / read on graphics
    thread (benign on x86_64: worst case one stale frame).
  - Golden constants tied to ORT 1.28.0 + pinned model + this CPU arch;
    re-record on bumps.
  - Follow-ups for Plan 3+:
    - Status property row is never written; worker exception-skip does not
      log (spec: "catch + log"); spec's "30 consecutive failures -> disable"
      was replaced by 1s staleness fallback (undeclared spec deviation).
    - Background image is re-decoded from disk on every settings update
      (no path-change detection) — UI jank in image mode.
    - obs_source_process_filter_begin failure draws black for one frame
      instead of passthrough.
    - Kawase pass loop lacks ONE/ZERO blend setup (identity for opaque
      frames; slightly darkens semi-transparent sources).
    - Freeze-before-first-frame outputs transparency, not spec-literal
      black (privacy intent satisfied).
    - get_name hardcodes "Camera Effects"; locale Name= key unused.
    - mask_tex is hard-coded 192x192 and would silently keep a stale mask
      if a future model's mask size differs — revisit with RVM tier.

## Plan 3 state (model manager + quality tier)

- Tier picker: Auto / Lite (MediaPipe 256x256, bundled) / Standard
  (PP-HumanSeg, bundled) / Quality (RVM, runtime download w/ GPL notice).
  Auto = Quality if downloaded, else Standard. Worker hot-swap via
  setProcessor (no filter recreation on tier change).
- RVM: recurrent states, ~17.7ms/frame (57fps) on CPU. PP-HumanSeg
  6.3ms (154fps). Benchmarks on this machine (RTX 5070 present).
- Model manager: curl subprocess + sha256sum + tar extract (staged,
  atomic dir swap); manifest in data/models/manifest.json; cache
  ~/.config/obs-cam-effects/{models,providers}.
- Advanced mask settings (amendment 9): threshold / contour / feather /
  temporal smoothing, all wired through the pipeline.
- Status line: composed at dialog open (get_properties); shows quality
  model state, download state (+ error text), backend, fps. The refresh
  button and modified_callback live-refresh were tried and REMOVED —
  OBS cannot reliably repaint an open properties dialog; status refreshes
  on dialog open only (user decision).
- CUDA (final, 2026-08-06): the ORT 1.28 plugin-EP-V2 path
  (RegisterExecutionProviderLibrary + AppendExecutionProvider_V2)
  SIGSEGVs on inswapper/RVM on both GPUs here — abandoned. The classic
  API (OrtSessionOptionsAppendExecutionProvider_CUDA, dlsym-probed: the
  symbol exists only in the GPU build's main lib) + the full ORT GPU
  build runs EVERYTHING (inswapper 15.75ms, RVM 2.62ms, PP-HumanSeg
  2.05ms, MediaPipe 1.96ms, YuNet 1.58ms on the RTX 5070). The GPU build
  is an optional 240MB manifest download that drops into the plugin bin
  dir (same SONAME = drop-in); one OBS restart enables it. The RVM CPU
  pin is lifted (BUG-1 in rvm.cpp marked historical). See
  docs/benchmarks.md "CUDA path history" for the full timeline.
- CUDA runtime requirement for users: CUDA 13 runtime (cudart/cublas/
  curand) + cuDNN 9 on the loader path (system-wide on this machine via
  the NVIDIA debs; the extracted ~/.local/lib/cuda13 copy was removed).
  The 240MB ORT download alone is NOT enough.
- The ~/.config/obs-cam-effects/providers dir is now write-only scratch
  (the old plugin-EP provider-libs flow is gone; the dir may still hold
  stale provider libs from it — harmless; cleanup candidate).
- Follow-ups:
  - Surface ORT/cuDNN error text through EpProbe to the status line
    (currently catch(...) swallows the reason).
  - Rvm test depends on a manually staged model file in
    build_x86_64/models (not CMake-downloaded) — re-stage after wiping
    the build dir.
  - fpsLast cross-thread read is benign-on-x86 but non-atomic.
  - blur_strength plain int cross-thread (carried from Plan 2 notes).
  - Downloader cancel is only checked after curl exits (bounded to ≤30s
    stall by --speed-time/--limit + 30min cap); true kill-on-cancel
    (fork/exec + kill) is a follow-up.
  - CUDA does not auto-activate when the provider download completes
    in-session — requires a tier toggle or OBS restart (probe is
    once-latched by design).
  - OrtModel::cuda_ flag: worker-thread write / UI-thread read (benign
    on x86, same class as fpsLast).
  - Spec drift to reconcile at next spec revision: §7.1 Auto rule says
    "usable GPU EP" but shipped Auto = Quality when RVM exists regardless
    of GPU (CPU-viable, deliberate); §7.3 says HuggingFace but shipped
    downloads use GitHub raw/releases; §7.3 "consent dialog" shipped as
    persistent license notice + click-to-download.
  - mask threshold binarize has no direct unit test (contour/feather/EMA
    do); trivial code, user-verified.
  - Plan 5 notes (from Plan 4 final review): no CMake install() rules
    yet for libonnxruntime.so.1.28.0 or the CMake-downloaded models
    (pphumanseg/selfie/yunet) — a .deb today would ship only the
    manifest. YuNet lacks a .license companion download (other bundled
    models have one). The GPU-build download overlays into the plugin
    bin dir — root-owned under .deb, read-only under Flatpak (fails
    cleanly with "overlay install failed"); needs a packaging-time
    decision (system CUDA packaging vs user-space download). CUDA 13 +
    cuDNN 9 host requirement must land in release notes.
  - Face-swap 2-stage download chain advances only when the properties
    dialog is open (poll-driven); w600k stage waits for a dialog open.
  - Downloader dtor can hang filter removal until curl exits (bounded
    ≤30s stall / 30min cap); true kill-on-cancel is a follow-up.

## Plan 4 state (face swap)

- Pipeline: YuNet (bundled, Apache-2.0) -> umeyama align -> inswapper
  (runtime download, non-commercial) -> DLC-style feathered-ellipse
  paste-back -> segmentation. Amendment-9 settings shipped: swap
  intensity, sharpness, preserve-mouth (geometric), AI watermark
  (default ON).
- Models: **inswapper_128_fp16 is the default download** (278 MB,
  1.65x faster on CUDA than fp32; fp32 graph IO and a bit-identical
  embedded emap = zero-change drop-in, same InsightFace non-commercial
  license). inswapper_128 (fp32) is kept in the manifest as fallback —
  resolution prefers fp16 when present, and a user who already has the
  fp32 file is not re-downloaded. ArcFace w600k_r50 provides the source
  embedding (once per source image, not a per-frame cost). Download
  chain: inswapper_128_fp16 -> w600k_r50 (~450 MB total).
- Detection decimation: YuNet runs every 2nd frame by default
  (FaceSwapParams.detectEveryN = 2; <=1 restores detect-every-frame).
  Skipped frames reuse the EMA-smoothed previous box; a no-face detect
  frame invalidates it (next frame forces a re-detect). Align/swap/
  paste-back still run EVERY frame — only detection is decimated.
  Internal default, no UI setting (YAGNI).
- Bench (RTX 5070, classic CUDA API, 2026-08-06): inswapper fp16
  9.54 ms (fp32 15.75 ms); e2e fp16 34.47 ms detect-every-frame,
  31.10 ms (32.2 fps) with decimation — see docs/benchmarks.md.
- Known gaps: single face only; no GFPGAN (not real-time); inswapper
  quality is 128px (soft by design, sharpness slider mitigates).

## Plan 5 state (packaging & release)

- Distribution: Linux `.tar.gz` (per-user, extract into
  ~/.config/obs-studio/plugins/obs-cam-effects/) + `.deb` (system
  install, Depends: libobs-dev >= 31). Both via CPack from the same
  CMake install rules.
- CMake install() rules with GPU-preservation guard (nm-based;
  CPU ORT bundled; GPU build downloaded at runtime).
- install-local.sh uses `cmake --install` with manual fallback.
- GitHub Actions: ubuntu-24.04 build + cpack TGZ+DEB + release
  on tag push with installation/GPU/face-swap instructions.
- Release notes include CUDA 13 + cuDNN 9 requirement for GPU.
- Known: CI full validation requires a push to GitHub (local only
  here); the workflow is YAML-valid and both artifacts build + load
  locally. The template's `push.yaml` has a permissions cap
  (contents: read) that would block the release step and a duplicate
  release job that may overwrite the rich body — both pre-existing
  template behaviors to resolve before the first tagged release.
- The `.deb` system install path lacks co-located ORT (ORT is in the
  per-user layout only); a functional system .deb would need an
  additional ORT install rule for /usr/lib/. Accepted limitation; the
  `.tar.gz` is the primary distribution artifact.
