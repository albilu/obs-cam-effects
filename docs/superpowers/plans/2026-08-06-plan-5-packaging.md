# Plan 5: Packaging & Release — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a distributable Linux tarball (bin + data) that users extract into `~/.config/obs-studio/plugins/obs-cam-effects/`, with CMake install rules, CPack tarball generation, and GitHub Actions release CI.

**Architecture:** CMake `install()` rules lay out the per-user plugin structure (bin/64bit/*.so + data/...). CPack packages that into a `.tar.gz`. The CI workflow builds on ubuntu-24.04, runs the test suite, and on tag push produces the tarball as a GitHub release asset. The CPU ORT runtime is bundled in the tarball; the GPU build remains a runtime in-plugin download (not packaged).

**Tech Stack:** CMake (existing presets), CPack, GitHub Actions (existing workflow from obs-plugintemplate, Linux-only).

## Scope

- CMake install rules: plugin .so + CPU ORT lib (with symlink) + all data files (models, effects, locale, manifest)
- CPack tarball configuration
- install-local.sh updated to use `cmake --install` (or a DESTDIR staging)
- GitHub Actions: ubuntu-24.04 build + ctest + tarball on tag; release notes with CUDA/cuDNN requirement
- Closeout: dev-notes + final verification

---

### Task 1: CMake install rules

**Files:**
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Add install rules to CMakeLists.txt**

The per-user layout is:
```
obs-cam-effects/
├── bin/64bit/obs-cam-effects.so
├── bin/64bit/libonnxruntime.so.1.28.0
├── bin/64bit/libonnxruntime.so.1 -> libonnxruntime.so.1.28.0
└── data/
    ├── effects/mask_composite.effect
    ├── effects/kawase_blur.effect
    ├── locale/en-US.ini
    └── models/
        ├── manifest.json
        ├── pphumanseg_fp32.onnx + .license
        ├── selfie_segmentation.onnx + .license
        └── face_detection_yunet_2023mar.onnx
```

Add to CMakeLists.txt (after the existing target definitions, before the tests block):

```cmake
# --- Install rules (per-user plugin layout) ---
install(TARGETS ${CMAKE_PROJECT_NAME}
	RUNTIME DESTINATION bin/64bit
	LIBRARY DESTINATION bin/64bit)

# CPU ORT runtime (next to the plugin .so, $ORIGIN RPATH finds it)
install(FILES "${ORT_ROOT}/lib/libonnxruntime.so.${ORT_VERSION}"
	DESTINATION bin/64bit)
install(CODE "execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
	libonnxruntime.so.${ORT_VERSION}
	\"\${CMAKE_INSTALL_PREFIX}/bin/64bit/libonnxruntime.so.1\")")

# Effect shaders
install(DIRECTORY data/effects/
	DESTINATION data/effects
	FILES_MATCHING PATTERN "*.effect")

# Locale
install(FILES data/locale/en-US.ini
	DESTINATION data/locale)

# Manifest
install(FILES data/models/manifest.json
	DESTINATION data/models)

# Bundled models (CMake-downloaded at configure time)
install(DIRECTORY "${FX_MODEL_DIR}/"
	DESTINATION data/models
	FILES_MATCHING PATTERN "*.onnx" PATTERN "*.license")
```

NOTE: verify the install rules produce the correct layout with `cmake --install /tmp/staging --prefix /tmp/p5-test` and `find /tmp/p5-test -type f | sort`. The ORT symlink creation must use the installed path, not the build path — the `CMAKE_INSTALL_PREFIX` variable is available in install(CODE) via `${CMAKE_INSTALL_PREFIX}` but inside the install script it's `${CMAKE_INSTALL_PREFIX}` (set by cmake --install --prefix). Use `$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}` in the install(CODE) for DESTDIR staging compatibility. Test both: `cmake --install build_x86_64 --prefix /tmp/p5-test` and `DESTDIR=/tmp/p5-destdir cmake --install build_x86_64` (the latter produces /tmp/p5-destdir/<prefix>/...).

- [ ] **Step 2: Verify the install layout**

```bash
cmake --install build_x86_64 --prefix /tmp/p5-test
find /tmp/p5-test -type f | sort
find /tmp/p5-test -type l | sort
```

Expected: obs-cam-effects.so, libonnxruntime.so.1.28.0, libonnxruntime.so.1 (symlink), mask_composite.effect, kawase_blur.effect, en-US.ini, manifest.json, pphumanseg_fp32.onnx, pphumanseg_fp32.onnx.license, selfie_segmentation.onnx, selfie_segmentation.onnx.license, face_detection_yunet_2023mar.onnx.

- [ ] **Step 3: Verify the installed plugin loads in OBS**

```bash
rm -rf ~/.config/obs-studio/plugins/obs-cam-effects
cp -r /tmp/p5-test/obs-cam-effects ~/.config/obs-studio/plugins/
timeout 25 obs --verbose > /tmp/opencode/obs-smoke-p5t1.log 2>&1; true
grep -c "plugin loaded successfully" /tmp/opencode/obs-smoke-p5t1.log
```

Expected: plugin loads cleanly from the installed layout (not the dev install-local.sh).

- [ ] **Step 4: Commit**

```bash
git add CMakeLists.txt
git commit -m "feat: CMake install rules for per-user plugin layout"
```

---

### Task 2: CPack tarball

**Files:**
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Add CPack configuration**

Append to CMakeLists.txt (after the install rules):

```cmake
# --- CPack tarball ---
set(CPACK_GENERATOR "TGZ")
set(CPACK_PACKAGE_NAME "obs-cam-effects")
set(CPACK_PACKAGE_VERSION "${CMAKE_PROJECT_VERSION}")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY
	"Real-time camera effects for OBS Studio: background blur, replacement, face swap")
set(CPACK_PACKAGE_FILE_NAME
	"obs-cam-effects-${CMAKE_PROJECT_VERSION}-linux")
set(CPACK_SOURCE_GENERATOR "")
set(CPACK_ARCHIVE_COMPONENT_INSTALL OFF)
set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY ON)
include(CPack)
```

The tarball will contain `obs-cam-effects-<version>-linux/obs-cam-effects/{bin,data}/...` — wait, that's one level too deep. The user extracts the tarball and gets `obs-cam-effects-<version>-linux/` with the plugin inside? No — the user wants to extract INTO `~/.config/obs-studio/plugins/obs-cam-effects/`. So the tarball should contain `obs-cam-effects/{bin,data}/...` directly (CPACK_PACKAGE_FILE_NAME sets the archive name; CPACK_INCLUDE_TOPLEVEL_DIRECTORY adds a top dir — set it OFF so the tarball's root IS bin/ + data/). But then the archive's internal name is lost. Compromise: keep the top dir as `obs-cam-effects/` (matching the plugin dir name) so `tar xzf obs-cam-effects-1.0.0-linux.tar.gz -C ~/.config/obs-studio/plugins/` works directly.

Adjust: `set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY ON)` and ensure the top-level dir is named `obs-cam-effects` (not `obs-cam-effects-1.0.0-linux`). The install prefix is `obs-cam-effects/` — hmm, the top dir name comes from CPACK_PACKAGE_FILE_NAME by default. Override: set `CPACK_INCLUDE_TOPLEVEL_DIRECTORY ON` and the top dir = CPACK_PACKAGE_NAME = "obs-cam-effects". The archive file name = CPACK_PACKAGE_FILE_NAME. Let the implementer test the actual layout and adjust.

- [ ] **Step 2: Build the tarball**

```bash
cd build_x86_64 && cpack -G TGZ
ls -la obs-cam-effects-*.tar.gz
tar tzf obs-cam-effects-*.tar.gz | head -15
```

Expected: tarball contains `obs-cam-effects/bin/64bit/obs-cam-effects.so` etc.

- [ ] **Step 3: Verify the tarball installs**

```bash
rm -rf ~/.config/obs-studio/plugins/obs-cam-effects
mkdir -p ~/.config/obs-studio/plugins
tar xzf build_x86_64/obs-cam-effects-*.tar.gz -C ~/.config/obs-studio/plugins/
timeout 25 obs --verbose > /tmp/opencode/obs-smoke-p5t2.log 2>&1; true
grep -c "plugin loaded successfully" /tmp/opencode/obs-smoke-p5t2.log
```

Expected: plugin loads from the tarball-extracted layout.

- [ ] **Step 4: Commit**

```bash
git add CMakeLists.txt
git commit -m "feat: CPack tarball generation"
```

---

### Task 3: install-local.sh update

**Files:**
- Modify: `build-aux/install-local.sh`

- [ ] **Step 1: Replace manual copies with cmake --install**

The current install-local.sh does ~15 manual cp/ln commands. Replace with:

```bash
#!/usr/bin/env bash
set -euo pipefail
BUILD_DIR="${1:-build_x86_64}"
DEST="$HOME/.config/obs-studio/plugins"

cmake --install "$BUILD_DIR" --prefix "$DEST" 2>/dev/null || {
	echo "cmake --install failed; falling back to manual copy"
	# Keep the manual fallback below for environments without cmake --install
	PLUGIN_DIR="$DEST/obs-cam-effects"
	mkdir -p "$PLUGIN_DIR/bin/64bit" "$PLUGIN_DIR/data/effects" \
		 "$PLUGIN_DIR/data/locale" "$PLUGIN_DIR/data/models"
	cp "$BUILD_DIR/obs-cam-effects.so" "$PLUGIN_DIR/bin/64bit/"
	# GPU build preservation (from Plan 4)
	if ls "$PLUGIN_DIR/bin/64bit"/libonnxruntime_providers_*.so 2>/dev/null; then
		echo "Keeping existing GPU ORT build"
	else
		cp "$BUILD_DIR/onnxruntime/lib/libonnxruntime.so.1.28.0" \
			"$PLUGIN_DIR/bin/64bit/"
		ln -sf libonnxruntime.so.1.28.0 \
			"$PLUGIN_DIR/bin/64bit/libonnxruntime.so.1"
	fi
	cp data/effects/*.effect "$PLUGIN_DIR/data/effects/"
	cp data/locale/en-US.ini "$PLUGIN_DIR/data/locale/"
	cp data/models/manifest.json "$PLUGIN_DIR/data/models/"
	cp "$BUILD_DIR/models/"*.onnx "$BUILD_DIR/models/"*.license \
		"$PLUGIN_DIR/data/models/" 2>/dev/null || true
}
echo "Installed to $DEST/obs-cam-effects"
```

NOTE: `cmake --install --prefix ~/.config/obs-studio/plugins` produces `~/.config/obs-studio/plugins/obs-cam-effects/bin/64bit/...` — wait, does it? The install(TARGETS ... DESTINATION bin/64bit) produces `<prefix>/bin/64bit/...` — but we need `<prefix>/obs-cam-effects/bin/64bit/...` (the plugin dir). Two options: (a) the install DESTINATION includes the plugin name: `install(... DESTINATION obs-cam-effects/bin/64bit)` — then cmake --install --prefix ~/.config/obs-studio/plugins produces the right layout. (b) The user passes --prefix ~/.config/obs-studio/plugins/obs-cam-effects. Option (a) is cleaner (the install rules are self-contained). Adjust the install rules in Task 1 to include `obs-cam-effects/` in the DESTINATION paths if the implementer finds the layout wrong. The implementer should verify and fix.

- [ ] **Step 2: Verify**

```bash
chmod +x build-aux/install-local.sh
./build-aux/install-local.sh
ls ~/.config/obs-studio/plugins/obs-cam-effects/bin/64bit/obs-cam-effects.so
timeout 25 obs --verbose 2>&1 | grep -c "plugin loaded successfully"
```

- [ ] **Step 3: Commit**

```bash
git add build-aux/install-local.sh
git commit -m "feat: install-local.sh uses cmake --install with fallback"
```

---

### Task 4: GitHub Actions release CI

**Files:**
- Modify: `.github/workflows/build-project.yaml` (already Linux-only from Plan 1 cleanup)

- [ ] **Step 1: Add tarball packaging to the release workflow**

The existing workflow (from obs-plugintemplate, trimmed to Linux-only in Plan 1) already builds on tag push. Add a step after the build to run `cpack -G TGZ` and upload the tarball as a release asset. Also add the release notes (CUDA 13 + cuDNN 9 requirement).

Inspect the existing workflow for the release job structure. Add after the build step:

```yaml
      - name: Package tarball
        run: |
          cd build
          cpack -G TGZ
      - name: Upload tarball
        uses: softprops/action-gh-release@v2
        if: startsWith(github.ref, 'refs/tags/')
        with:
          files: build/obs-cam-effects-*.tar.gz
          body: |
            ## obs-cam-effects ${{ github.ref_name }}

            ### Installation
            1. Download `obs-cam-effects-<version>-linux.tar.gz`
            2. Extract into your OBS plugins directory:
               ```bash
               tar xzf obs-cam-effects-*-linux.tar.gz -C ~/.config/obs-studio/plugins/
               ```
            3. Restart OBS. Add the "Camera Effects" filter to any source.

            ### GPU acceleration (optional)
            For CUDA-accelerated segmentation and face swap on NVIDIA GPUs:
            1. Install CUDA 13 runtime + cuDNN 9 system-wide
            2. In the filter properties, click "Download GPU acceleration (MIT, ~240 MB)"
            3. Restart OBS

            ### Face swap (optional, non-commercial)
            1. In the filter properties, click "Download face swap models (non-commercial, ~450 MB)"
            2. Pick a source face image
            3. Enable Face swap

            Models are licensed for NON-COMMERCIAL research use only (InsightFace).
            An AI disclosure badge is applied by default (EU AI Act Art. 50).
```

NOTE: verify the workflow file structure; the template may use a different job/step layout. Adapt. Also: the `softprops/action-gh-release` action needs `permissions: contents: write` on the job — add if missing.

- [ ] **Step 2: Verify the workflow file is valid YAML**

```bash
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/build-project.yaml'))"
```

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/build-project.yaml
git commit -m "feat: release CI produces distributable tarball"
```

---

### Task 5: Closeout

**Files:**
- Modify: `docs/development-notes.md`

- [ ] **Step 1: Append Plan 5 state**

```markdown
## Plan 5 state (packaging & release)

- Distribution: Linux tarball (bin + data); extract into
  ~/.config/obs-studio/plugins/obs-cam-effects/. CPU ORT bundled;
  GPU build is a runtime in-plugin download (240MB, MIT).
- CMake install() rules + CPack TGZ; install-local.sh uses
  cmake --install with a manual fallback.
- GitHub Actions: ubuntu-24.04, tag push → tarball release with
  installation + GPU/face-swap instructions in the release body.
- Release notes include CUDA 13 + cuDNN 9 requirement for GPU.
- Known: CI full validation requires a push to GitHub (local only
  here); the workflow is YAML-valid and the tarball builds + loads
  locally.
```

- [ ] **Step 2: Final verification**

```bash
git status --short
cmake --build --preset ubuntu-x86_64 && ctest --test-dir build_x86_64
cd build_x86_64 && cpack -G TGZ && ls obs-cam-effects-*.tar.gz && cd ..
tar tzf build_x86_64/obs-cam-effects-*.tar.gz | head -10
```

- [ ] **Step 3: Commit**

```bash
git add docs/development-notes.md
git commit -m "docs: plan 5 closeout notes"
```

---

## Plan 5 Definition of Done

- [ ] CMake install rules produce the correct per-user layout
- [ ] CPack generates a tarball that extracts and loads in OBS
- [ ] install-local.sh uses cmake --install with fallback
- [ ] GitHub Actions workflow produces a release tarball on tag push (YAML-valid; full CI verification deferred to first push)
- [ ] Release notes include installation + GPU + face-swap instructions + CUDA/cuDNN requirement
- [ ] Git history: one commit per task, clean tree
