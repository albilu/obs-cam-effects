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
