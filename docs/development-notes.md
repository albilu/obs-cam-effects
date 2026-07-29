# Development Notes

## Template provenance

Vendored from https://github.com/obsproject/obs-plugintemplate at upstream commit
`3e7d7ac3b5342cd7d9b88890b9c70b472d1520fc` (shallow clone, 2026-07-26). For future template upgrades, three-way
merge against this base.

## Follow-ups for Plan 5 (packaging & release)

- `buildspec.json`: `displayName` still "Plugin Template for OBS" and
  `platformConfig.macos.bundleId` still "com.example.plugintemplate-for-obs" —
  rebrand before first release.
- `buildspec.json`: `dependencies.obs-studio` hashes (macos/windows-x64) still
  correspond to 31.1.1 archives; version is pinned to 32.1.2. CI downloads are
  hash-verified and will fail until hashes are updated to the 32.1.2 archives.
- Template source files (`src/plugin-support.*`, `src/plugin-main.c`) still carry
  GPLv2 copyright header comments while the project LICENSE is MIT — relicense
  headers when those files are substantively rewritten (Plan 2).
- CI presets (`*-ci-*`) share `binaryDir` with local presets; running a CI preset
  locally leaves `BUILD_TESTING=FALSE` sticky in the shared cache. Give CI presets
  their own binaryDir if this bites.
- GitHub Actions CI is configured but never executed: first push to GitHub will
  validate the full matrix (Linux/Windows/macOS + check-format gate).
