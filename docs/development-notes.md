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
- Status line: composed at dialog open (get_properties) + Refresh
  button; shows active model, tier setting, model state, download state,
  backend, fps. fps counter verified live (29-35fps with camera running).
- CUDA: provider package (240MB, MIT) downloads + registers on RTX 5070
  via ORT 1.28 plugin-EP API. HOWEVER: actual GPU inference fails on
  this machine — cuDNN FE HEURISTIC_QUERY_FAILED on Blackwell sm_120
  with a hand-assembled CUDA 13.3 runtime (deb-extracted, no proper
  toolkit). Judged environment, not code. Lazy CPU fallback (added
  beyond plan): first CUDA run failure permanently degrades the session
  to CPU — users with broken CUDA get CPU speed, never a crash.
  CUDA runtime requirement for users: CUDA 13 toolkit + cuDNN on the
  loader path (the 240MB ORT provider package alone is NOT enough).
- Follow-ups:
  - Surface ORT/cuDNN error text through EpProbe to the status line
    (currently catch(...) swallows the reason).
  - Rvm test depends on a manually staged model file in
    build_x86_64/models (not CMake-downloaded) — re-stage after wiping
    the build dir.
  - fpsLast cross-thread read is benign-on-x86 but non-atomic.
  - blur_strength plain int cross-thread (carried from Plan 2 notes).
