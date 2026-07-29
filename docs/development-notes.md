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
