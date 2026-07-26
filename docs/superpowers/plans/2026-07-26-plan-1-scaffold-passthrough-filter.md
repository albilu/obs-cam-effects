# Plan 1: Scaffold + Passthrough Filter — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bootstrap obs-cam-effects from obs-plugintemplate so it builds on Linux, loads into OBS 32.1.2 as a "Camera Effects" video filter that passes video through untouched, and has a working C++ test harness (GoogleTest) for the libfx core that later plans build on.

**Architecture:** Single C++ OBS module built with CMake presets from obs-plugintemplate. A thin OBS adapter (`src/plugin-main.c`, `src/cam-effects-filter.c`) plus a static OBS-free core library `fx` (`src/fx/`) that all inference code will live in from Plan 2 onward. Spec: `docs/superpowers/specs/2026-07-26-obs-cam-effects-design.md`.

**Tech Stack:** C (OBS adapter) / C++17 (libfx), CMake ≥ 3.16 presets, obs-plugintemplate (master), libobs 32.1.2 (Debian `libobs-dev`), GoogleTest 1.15.2 via FetchContent, Ninja, GCC 15.

**Environment (this machine):** Kali GNU/Linux Rolling, cmake 4.3.1, gcc 15.2.0, ninja, git, OBS Studio 32.1.2+ds-1 installed. `libobs-dev` 32.1.2+ds-1 available in apt but not yet installed.

**License note (decided here per spec §"License"): MIT.** libobs is GPLv2; Apache-2.0 is incompatible with GPLv2 (patent clause) while MIT is GPL-compatible, so MIT is the only valid permissive choice for code linking libobs. The combined distributed binary remains effectively GPL via libobs, as with all OBS plugins.

---

### Task 1: Install libobs development headers

**Files:** none (system package)

- [ ] **Step 1: Install libobs-dev**

Ask the user to run (needs sudo):

```bash
sudo apt install -y libobs-dev
```

- [ ] **Step 2: Verify headers and CMake config are present**

```bash
ls /usr/include/obs/obs-module.h && find /usr/lib -name "libobsConfig*.cmake" -o -name "libobs-config*.cmake" 2>/dev/null | head -3
```

Expected: prints `/usr/include/obs/obs-module.h` and at least one CMake config path (e.g. `/usr/lib/x86_64-linux-gnu/cmake/libobs/libobsConfig.cmake`). If the CMake config is missing, STOP and report to the user before continuing — the fallback (building against an obs-studio source tree) changes Task 5 and needs a plan amendment.

- [ ] **Step 3: Commit** — nothing to commit (system state only). Skip.

---

### Task 2: Import obs-plugintemplate into the repository

**Files:**
- Create: all template files (copied into repo root, excluding template's `.git` and `README.md`)

- [ ] **Step 1: Check current repo state**

```bash
git status --short && git log --oneline -5
```

Expected: repo contains only `README.md`, the two screenshots, and `docs/` (clean or nearly clean tree). If there are uncommitted changes, commit or stash them first.

- [ ] **Step 2: Clone the template to a temp dir**

```bash
git clone --depth 1 https://github.com/obsproject/obs-plugintemplate.git /tmp/opencode/obs-plugintemplate
```

Expected: `Cloning into '/tmp/opencode/obs-plugintemplate'... done.`

- [ ] **Step 3: Copy template files into the repo (keep our README, docs, and git history)**

```bash
rsync -a --exclude='.git' --exclude='README.md' /tmp/opencode/obs-plugintemplate/ .
ls
```

Expected listing includes: `CMakeLists.txt`, `CMakePresets.json`, `buildspec.json`, `cmake/`, `src/`, `data/`, `.github/`, `.gitignore`, `LICENSE` (template's), plus our original `README.md`, `docs/`, `deep-live-cam-ui.png`, `kdenlive-sam2-settings.png`.

- [ ] **Step 4: Replace the template LICENSE with MIT**

Delete the template's `LICENSE` and create a new one:

```bash
rm LICENSE
```

Create `LICENSE` with this exact content:

```text
MIT License

Copyright (c) 2026 obs-cam-effects contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore: import obs-plugintemplate scaffold (MIT license)"
```

Expected: one commit adding the template tree.

---

### Task 3: Rename the plugin to obs-cam-effects

**Files:**
- Modify: `buildspec.json`
- Modify: `src/plugin-main.c`
- Modify: `data/locale/en-US.ini`

- [ ] **Step 1: Edit `buildspec.json` metadata**

Open `buildspec.json`. In the top-level object, set:

```json
"name": "obs-cam-effects",
"version": "0.1.0",
"author": "obs-cam-effects contributors",
"website": "",
"email": "",
```

Also find `dependencies."obs-studio".version` and set it to `"32.1.2"` (matches the locally installed OBS; Windows/macOS CI will fetch that version's prebuilts). Leave every other key untouched.

- [ ] **Step 2: Edit `src/plugin-main.c` strings**

Open `src/plugin-main.c`. Find the `OBS_MODULE_USE_DEFAULT_LOCALE(...)` line and set its first argument to `"obs-cam-effects"`. Find `obs_module_description()` and make it return `"Real-time camera effects: background blur, background replacement, face swap"`. Do not change anything else yet.

- [ ] **Step 3: Edit `data/locale/en-US.ini`**

Replace the whole file content with:

```ini
Name="Camera Effects"
```

- [ ] **Step 4: Verify the rename is coherent**

```bash
grep -rn "obs-plugintemplate" buildspec.json src/ data/ || echo "OK: no stale template name"
```

Expected: `OK: no stale template name` (matches elsewhere, e.g. `.github/`, are fine for now).

- [ ] **Step 5: Commit**

```bash
git add buildspec.json src/plugin-main.c data/locale/en-US.ini
git commit -m "chore: rename plugin to obs-cam-effects 0.1.0"
```

---

### Task 4: Implement the passthrough "Camera Effects" filter

**Files:**
- Create: `src/cam-effects-filter.h`
- Create: `src/cam-effects-filter.c`
- Modify: `src/plugin-main.c`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Create `src/cam-effects-filter.h`**

```c
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/* Registers the "Camera Effects" video filter with libobs.
   Call once from obs_module_load(). */
void cam_effects_register_filter(void);

#ifdef __cplusplus
}
#endif
```

- [ ] **Step 2: Create `src/cam-effects-filter.c`**

```c
#include "cam-effects-filter.h"

#include <obs-module.h>

struct cam_effects_filter {
	obs_source_t *source;
};

static const char *cam_effects_get_name(void *type_data)
{
	UNUSED_PARAMETER(type_data);
	return "Camera Effects";
}

static void *cam_effects_create(obs_data_t *settings, obs_source_t *source)
{
	UNUSED_PARAMETER(settings);
	struct cam_effects_filter *filter =
		bzalloc(sizeof(struct cam_effects_filter));
	filter->source = source;
	return filter;
}

static void cam_effects_destroy(void *data)
{
	struct cam_effects_filter *filter = data;
	bfree(filter);
}

static void cam_effects_video_render(void *data, gs_effect_t *effect)
{
	UNUSED_PARAMETER(effect);
	struct cam_effects_filter *filter = data;

	/* Plan 1: passthrough only. Real compositing arrives in Plan 2. */
	obs_source_skip_video_filter(filter->source);
}

static obs_properties_t *cam_effects_properties(void *data)
{
	UNUSED_PARAMETER(data);
	return obs_properties_create();
}

static struct obs_source_info cam_effects_filter_info = {
	.id = "obs_cam_effects_filter",
	.type = OBS_SOURCE_TYPE_FILTER,
	.output_flags = OBS_SOURCE_VIDEO,
	.get_name = cam_effects_get_name,
	.create = cam_effects_create,
	.destroy = cam_effects_destroy,
	.video_render = cam_effects_video_render,
	.get_properties = cam_effects_properties,
};

void cam_effects_register_filter(void)
{
	obs_register_source(&cam_effects_filter_info);
}
```

- [ ] **Step 3: Register the filter in `src/plugin-main.c`**

Open `src/plugin-main.c`. Add near the top, after the existing includes:

```c
#include "cam-effects-filter.h"
```

Inside `obs_module_load()`, immediately before `return true;`, add:

```c
	cam_effects_register_filter();
```

- [ ] **Step 4: Add the new source file to the build in `CMakeLists.txt`**

Open `CMakeLists.txt`. Find the line `target_sources(${PROJECT_NAME} PRIVATE src/plugin-main.c)` (or equivalent existing `target_sources` call) and change it to:

```cmake
target_sources(${PROJECT_NAME} PRIVATE src/plugin-main.c src/cam-effects-filter.c)
```

- [ ] **Step 5: Commit**

```bash
git add src/cam-effects-filter.h src/cam-effects-filter.c src/plugin-main.c CMakeLists.txt
git commit -m "feat: register passthrough Camera Effects filter"
```

---

### Task 5: Build the plugin with CMake presets

**Files:** none (build only)

- [ ] **Step 1: Discover the Linux preset name**

```bash
cmake --list-presets
```

Expected: a list including a Linux configure preset — typically `linux-x86_64` (some template versions call it `ubuntu-x86_64`). Note the exact name; use it in place of `<PRESET>` below. If no Linux preset exists, STOP and report — the template changed its preset layout.

- [ ] **Step 2: Configure**

```bash
cmake --preset <PRESET>
```

Expected: configure completes; the final line prints `Build files have been written to: <BINARY_DIR>`. Note `<BINARY_DIR>` (commonly `build_x86_64`). If configure fails with "could not find libobs", revisit Task 1 Step 2.

- [ ] **Step 3: Build**

```bash
cmake --build --preset <PRESET>
```

Expected: build completes with no errors; the module exists:

```bash
find <BINARY_DIR> -name "obs-cam-effects.so"
```

Expected: prints exactly one path, e.g. `<BINARY_DIR>/obs-cam-effects.so`.

- [ ] **Step 4: Commit** — build artifacts are covered by the template's `.gitignore`; nothing to commit. Skip.

---

### Task 6: Install locally and verify OBS loads the plugin

**Files:** none (install + smoke test)

- [ ] **Step 1: Install into the per-user OBS plugin directory**

```bash
mkdir -p ~/.config/obs-studio/plugins/obs-cam-effects/bin/64bit
cp "$(find build_x86_64 -name obs-cam-effects.so | head -1)" ~/.config/obs-studio/plugins/obs-cam-effects/bin/64bit/
```

(If `<BINARY_DIR>` from Task 5 differs, substitute it.) Expected: no output; `ls ~/.config/obs-studio/plugins/obs-cam-effects/bin/64bit/` shows `obs-cam-effects.so`.

- [ ] **Step 2: Run OBS and capture the startup log**

If `echo $DISPLAY` prints something (desktop session), run:

```bash
timeout 25 obs --verbose > /tmp/opencode/obs-smoke.log 2>&1; true
```

If `$DISPLAY` is empty (headless), first `sudo apt install -y xvfb`, then:

```bash
xvfb-run -a timeout 25 obs --verbose > /tmp/opencode/obs-smoke.log 2>&1; true
```

OBS will be killed by `timeout` after 25s — that is expected. (On its next launch OBS may mention an unclean shutdown; harmless.)

- [ ] **Step 3: Verify the module loaded**

```bash
grep -E "obs-cam-effects" /tmp/opencode/obs-smoke.log
grep -iE "fail.*obs-cam-effects|obs-cam-effects.*fail" /tmp/opencode/obs-smoke.log || echo "OK: no load failures"
```

Expected: first command shows a load line for `obs-cam-effects.so` and/or our `plugin loaded successfully` blog line; second command prints `OK: no load failures`. If the module is missing from the log, check `grep -i "error" /tmp/opencode/obs-smoke.log` and report.

- [ ] **Step 4: Commit** — nothing to commit (runtime state only). Skip.

---

### Task 7: libfx skeleton with GoogleTest harness (TDD)

This establishes the OBS-free core library and the test loop that all later plans use.

**Files:**
- Create: `tests/test_version.cpp`
- Create: `src/fx/version.h`
- Create: `src/fx/version.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write the failing test — `tests/test_version.cpp`**

```cpp
#include <gtest/gtest.h>

#include "fx/version.h"

TEST(FxVersion, ReturnsCurrentVersion) {
	EXPECT_STREQ(fx::version(), "0.1.0");
}
```

- [ ] **Step 2: Declare (but do not define) the API — `src/fx/version.h`**

```cpp
#pragma once

namespace fx {

/* Returns the libfx core version string, e.g. "0.1.0". */
const char *version();

} // namespace fx
```

- [ ] **Step 3: Wire the fx library and tests into `CMakeLists.txt`**

Open `CMakeLists.txt`. Append at the end of the file:

```cmake
# --- libfx: OBS-free core library (all inference code lives here from Plan 2) ---
add_library(fx STATIC src/fx/version.cpp)
target_include_directories(fx PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_features(fx PUBLIC cxx_std_17)
set_target_properties(fx PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(${PROJECT_NAME} PRIVATE fx)

# --- Tests ---
include(CTest)
if(BUILD_TESTING)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
  )
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  include(GoogleTest)
  add_executable(fx_tests tests/test_version.cpp)
  target_link_libraries(fx_tests PRIVATE fx GTest::gtest_main)
  gtest_discover_tests(fx_tests)
endif()
```

Optional hardening: pin the GoogleTest tarball hash by adding `URL_HASH SHA256=<hash>` to the `FetchContent_Declare` call, where `<hash>` comes from:

```bash
curl -sL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz | sha256sum
```

- [ ] **Step 4: Create an empty `src/fx/version.cpp` so the target exists, then run the test and watch it FAIL**

Create `src/fx/version.cpp`:

```cpp
#include "fx/version.h"

namespace fx {

const char *version()
{
	return ""; // deliberately wrong: TDD red state
}

} // namespace fx
```

Then reconfigure and run:

```bash
cmake --preset <PRESET> && cmake --build --preset <PRESET>
ctest --test-dir <BINARY_DIR> --output-on-failure
```

Expected: `FxVersion.ReturnsCurrentVersion` FAILS with `Which is: ""` vs expected `"0.1.0"`. FetchContent will download GoogleTest on the first configure — allow a minute.

- [ ] **Step 5: Make the test pass**

Edit `src/fx/version.cpp`, changing the return to:

```cpp
	return "0.1.0";
```

- [ ] **Step 6: Run tests again — all pass**

```bash
cmake --build --preset <PRESET> && ctest --test-dir <BINARY_DIR> --output-on-failure
```

Expected: `100% tests passed, 0 tests failed out of 1`.

- [ ] **Step 7: Commit**

```bash
git add CMakeLists.txt src/fx/version.h src/fx/version.cpp tests/test_version.cpp
git commit -m "feat: libfx skeleton with GoogleTest harness (fx::version)"
```

---

### Task 8: Verify CI wiring and close out

**Files:**
- Read-only: `.github/workflows/build-project.yaml`

- [ ] **Step 1: Confirm the template's CI workflow exists and references the right project**

```bash
ls .github/workflows/
grep -n "buildspec" .github/workflows/*.yaml | head -5
```

Expected: at least `build-project.yaml` (and likely `push.yaml`/`pr-pull.yaml`); the grep shows the workflow reads `buildspec.json` — no edits needed since CI derives name/version from it.

- [ ] **Step 2: Record the out-of-scope CI verification**

Full CI validation (Windows/macOS/Linux builds, signing on tags) requires pushing to GitHub, which is a user action outside this plan. No local step can substitute; flag this as the plan's one unverified item in the final summary.

- [ ] **Step 3: Final commit (if anything was adjusted) and status check**

```bash
git status --short
git log --oneline
```

Expected: clean tree; commit history shows the five commits from Tasks 2–7 (`import obs-plugintemplate scaffold`, `rename plugin`, `passthrough Camera Effects filter`, `libfx skeleton`, plus any fix-ups).

---

## Plan 1 Definition of Done

- [ ] `cmake --build --preset <linux preset>` completes clean
- [ ] `ctest` passes 1/1
- [ ] OBS 32.1.2 loads `obs-cam-effects.so` with no failures in the log
- [ ] "Camera Effects" filter registered (load line present; filter info compiled in)
- [ ] Git history: 4–5 clean commits
- [ ] Known unverified: GitHub CI (needs a push); Windows/macOS builds (CI-only)
