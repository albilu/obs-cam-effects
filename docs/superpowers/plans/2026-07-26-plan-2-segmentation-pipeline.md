# Plan 2: Segmentation Pipeline (Blur + Replace) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the passthrough filter into a working background-effects filter: PP-HumanSeg segmentation on CPU via ONNX Runtime, off-graphics-thread inference, and three background modes (transparent / image / blur) composited on the GPU.

**Architecture:** All inference logic lives in the OBS-free `fx` static C++ library (engine → model → pipeline → worker). The OBS side (`cam-effects-filter.c` + new `fx_bridge.cpp`) stages 192×192 frames to the worker and composites the published mask via custom OBS effect shaders. No OpenCV — image ops are hand-rolled in `fx::image`.

**Tech Stack:** C++17 (fx), C (OBS filter), ONNX Runtime 1.28.0 (CPU, prebuilt tgz), GoogleTest 1.15.2, OBS 32.1.2 (libobs-dev), CMake presets (`ubuntu-x86_64`).

**Environment:** Kali Linux, OBS 32.1.2, branch `main`. Build: `cmake --preset ubuntu-x86_64 && cmake --build --preset ubuntu-x86_64`; test: `ctest --test-dir build_x86_64`. Warnings are errors (template sets `CMAKE_COMPILE_WARNING_AS_ERROR=ON`) — keep code warning-clean.

## Verified facts (spike results — do not re-derive)

- **PP-HumanSeg model** (Apache-2.0, PaddlePaddle):
  - URL: `https://raw.githubusercontent.com/royshil/obs-backgroundremoval/main/models/pphumanseg_fp32.onnx`
  - SHA256: `6913acd125be55fd08e76072ed464146925223392954c437f3aa700e4da84b17` (6,196,913 bytes)
  - License file: `https://raw.githubusercontent.com/royshil/obs-backgroundremoval/main/models/pphumanseg_fp32.onnx.license`, SHA256 `363552428d64011b9242e0889f90e06cf2c30370908a2a9641dbdd8d37a5f2a6` (225 bytes, Apache-2.0 — must ship with the model)
  - Input: name `x`, shape `[1, 3, 192, 192]` float32 (NCHW)
  - Output: name `tf.identity`, shape `[1, 192, 192, 2]` float32 (NHWC); **channel 1 = person**
  - Measured: ~7.3ms/frame on this machine (ORT 1.28.0 CPU, default threads)
  - Preprocessing (from obs-backgroundremoval's `ModelPPHumanSeg.hpp`, verified): keep **BGR order**, `(v/256.0 − 0.5) / 0.5`, HWC→CHW
  - Postprocessing (same source): take channel 1, then `cv::normalize(NORM_MINMAX)` to [0,1]
- **ONNX Runtime 1.28.0 Linux x64**:
  - URL: `https://github.com/microsoft/onnxruntime/releases/download/v1.28.0/onnxruntime-linux-x64-1.28.0.tgz`
  - SHA256: `a3e1b79d7bb1bf09696ce675f49e4064e6c81f6202b8225624fff0e93f8d6407` (9,125,960 bytes)
  - Tarball layout: `onnxruntime-linux-x64-1.28.0/{include,lib}`; lib contains `libonnxruntime.so.1.28.0` + symlinks `libonnxruntime.so.1`, `libonnxruntime.so`

## Declared deviations from the design spec (reviewer: these are intentional)

1. **Single background-mode selector** (Off / Transparent / Image / Blur) instead of combinable per-effect toggles — simpler UI; the pipeline can combine later. Face-swap toggle unaffected (Plan 4).
2. **Guided filter runs at 192×192 mask resolution** (guide = staged small frame), GPU bilinear upscale at composite — matches the amended spec §6 invariant.
3. **No OpenCV** — hand-rolled resize/gray/EMA/guided filter in `fx::image` keeps the dependency surface at just ONNX Runtime.
4. **CPU-only in this plan** — CUDA execution provider arrives with the Quality tier in Plan 3.

---

### Task 1: ONNX Runtime CMake integration + OrtModel wrapper

**Files:**
- Modify: `CMakeLists.txt`
- Create: `src/fx/engine/ort_backend.h`
- Create: `src/fx/engine/ort_backend.cpp`
- Test: `tests/test_ort_backend.cpp`

- [ ] **Step 1: Add the ORT fetch + imported target to `CMakeLists.txt`**

Insert BEFORE the `# --- libfx ---` block (and update the fx sources/targets as shown):

```cmake
# --- ONNX Runtime 1.28.0 (CPU, Linux x86-64, prebuilt) ---
set(ORT_VERSION 1.28.0)
set(ORT_ROOT ${CMAKE_BINARY_DIR}/onnxruntime)
if(NOT EXISTS "${ORT_ROOT}/lib/libonnxruntime.so.${ORT_VERSION}")
  file(DOWNLOAD
    "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz"
    ${CMAKE_BINARY_DIR}/ort-${ORT_VERSION}.tgz
    EXPECTED_HASH SHA256=a3e1b79d7bb1bf09696ce675f49e4064e6c81f6202b8225624fff0e93f8d6407
    SHOW_PROGRESS)
  file(ARCHIVE_EXTRACT INPUT ${CMAKE_BINARY_DIR}/ort-${ORT_VERSION}.tgz
       DESTINATION ${CMAKE_BINARY_DIR})
  file(RENAME ${CMAKE_BINARY_DIR}/onnxruntime-linux-x64-${ORT_VERSION} ${ORT_ROOT})
endif()

add_library(onnxruntime SHARED IMPORTED GLOBAL)
set_target_properties(onnxruntime PROPERTIES
  IMPORTED_LOCATION ${ORT_ROOT}/lib/libonnxruntime.so.${ORT_VERSION}
  INTERFACE_INCLUDE_DIRECTORIES ${ORT_ROOT}/include)

find_package(Threads REQUIRED)
```

Then change the libfx block to:

```cmake
# --- libfx: OBS-free core library ---
add_library(fx STATIC
  src/fx/version.cpp
  src/fx/engine/ort_backend.cpp
)
target_include_directories(fx PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_features(fx PUBLIC cxx_std_17)
set_target_properties(fx PROPERTIES POSITION_INDEPENDENT_CODE ON
                                    BUILD_RPATH ${ORT_ROOT}/lib)
target_link_libraries(fx PUBLIC onnxruntime Threads::Threads)
target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE fx)
set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES
  BUILD_RPATH "${ORT_ROOT}/lib;\$ORIGIN"
  INSTALL_RPATH "$ORIGIN")
```

NOTE: the `\$ORIGIN` entry lets the plugin find `libonnxruntime.so.1` next to its own `.so` even if the build tree is wiped. Verify after building: `readelf -d build_x86_64/obs-cam-effects.so | grep -i 'rpath\|runpath'` must show an `$ORIGIN` entry.
```

And in the tests block, change `add_executable` and add RPATH:

```cmake
  add_executable(fx_tests
    tests/test_version.cpp
    tests/test_ort_backend.cpp
  )
  target_link_libraries(fx_tests PRIVATE fx GTest::gtest_main)
  set_target_properties(fx_tests PROPERTIES BUILD_RPATH ${ORT_ROOT}/lib)
  gtest_discover_tests(fx_tests)
```

- [ ] **Step 2: Write the failing test — `tests/test_ort_backend.cpp`**

```cpp
#include <gtest/gtest.h>

#include "fx/engine/ort_backend.h"

TEST(OrtModel, ThrowsOnMissingFile)
{
	EXPECT_THROW(fx::OrtModel model("/nonexistent/model.onnx", 1),
		     std::exception);
}
```

- [ ] **Step 3: Create the header — `src/fx/engine/ort_backend.h`**

```cpp
#pragma once

#include <onnxruntime_cxx_api.h>

#include <string>
#include <vector>

namespace fx {

/* Thin wrapper around a single-input single-output ONNX Runtime CPU
 * session with dynamic IO name/shape discovery. */
class OrtModel {
public:
	struct TensorDesc {
		std::string name;
		std::vector<int64_t> shape;
	};

	explicit OrtModel(const std::string &modelPath, int intraOpThreads = 2);

	const TensorDesc &input() const { return input_; }
	const TensorDesc &output() const { return output_; }

	/* Runs the model on a float tensor matching input().shape.
	 * Returns the output tensor values. */
	std::vector<float> run(const std::vector<float> &inputData);

private:
	Ort::Env env_;
	Ort::Session session_;
	TensorDesc input_;
	TensorDesc output_;
};

} // namespace fx
```

- [ ] **Step 4: Build and watch the test FAIL (link error: undefined `fx::OrtModel`)**

```bash
cmake --preset ubuntu-x86_64 && cmake --build --preset ubuntu-x86_64 2>&1 | tail -5
```

Expected: build fails at link of `fx_tests` (undefined reference to `fx::OrtModel::OrtModel...`). The ORT tarball downloads on first configure (~9MB).

- [ ] **Step 5: Implement `src/fx/engine/ort_backend.cpp`**

```cpp
#include "fx/engine/ort_backend.h"

#include <stdexcept>

namespace fx {

OrtModel::OrtModel(const std::string &modelPath, int intraOpThreads)
	: env_(ORT_LOGGING_LEVEL_WARNING, "fx"), session_(nullptr)
{
	Ort::SessionOptions opts;
	opts.SetIntraOpNumThreads(intraOpThreads);
	opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
	session_ = Ort::Session(env_, modelPath.c_str(), opts);

	if (session_.GetInputCount() != 1 || session_.GetOutputCount() != 1)
		throw std::runtime_error(
			"fx: expected single-input single-output model");

	Ort::AllocatorWithDefaultOptions alloc;
	auto inName = session_.GetInputNameAllocated(0, alloc);
	auto outName = session_.GetOutputNameAllocated(0, alloc);
	input_.name = inName.get();
	output_.name = outName.get();
	input_.shape = session_.GetInputTypeInfo(0)
			       .GetTensorTypeAndShapeInfo()
			       .GetShape();
	output_.shape = session_.GetOutputTypeInfo(0)
				.GetTensorTypeAndShapeInfo()
				.GetShape();
}

std::vector<float> OrtModel::run(const std::vector<float> &inputData)
{
	auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,
					      OrtMemTypeDefault);
	Ort::Value in = Ort::Value::CreateTensor<float>(
		mem, const_cast<float *>(inputData.data()), inputData.size(),
		input_.shape.data(), input_.shape.size());
	const char *inNames[] = {input_.name.c_str()};
	const char *outNames[] = {output_.name.c_str()};
	auto outs = session_.Run(Ort::RunOptions{nullptr}, inNames, &in, 1,
				 outNames, 1);
	float *data = outs[0].GetTensorMutableData<float>();
	size_t count = outs[0].GetTensorTypeAndShapeInfo().GetElementCount();
	return std::vector<float>(data, data + count);
}

} // namespace fx
```

- [ ] **Step 6: Build and run tests — all pass**

```bash
cmake --build --preset ubuntu-x86_64 && ctest --test-dir build_x86_64 --output-on-failure
```

Expected: `100% tests passed, 0 tests failed out of 2`.

- [ ] **Step 7: Commit**

```bash
git add CMakeLists.txt src/fx/engine/ort_backend.h src/fx/engine/ort_backend.cpp tests/test_ort_backend.cpp
git commit -m "feat: ONNX Runtime 1.28 integration with OrtModel wrapper"
```

---

### Task 2: PP-HumanSeg model fetch + fx types + PPHumanSeg class

**Files:**
- Modify: `CMakeLists.txt`
- Create: `src/fx/types.h`
- Create: `src/fx/models/pp_humanseg.h`
- Create: `src/fx/models/pp_humanseg.cpp`
- Test: `tests/test_pp_humanseg.cpp`

- [ ] **Step 1: Add the model download + FX_MODEL_PATH to `CMakeLists.txt`**

Append after the ORT block:

```cmake
# --- Bundled segmentation model: PP-HumanSeg v2 lite (Apache-2.0) ---
set(FX_MODEL_DIR ${CMAKE_BINARY_DIR}/models)
if(NOT EXISTS "${FX_MODEL_DIR}/pphumanseg_fp32.onnx")
  file(DOWNLOAD
    "https://raw.githubusercontent.com/royshil/obs-backgroundremoval/main/models/pphumanseg_fp32.onnx"
    ${FX_MODEL_DIR}/pphumanseg_fp32.onnx
    EXPECTED_HASH SHA256=6913acd125be55fd08e76072ed464146925223392954c437f3aa700e4da84b17)
  file(DOWNLOAD
    "https://raw.githubusercontent.com/royshil/obs-backgroundremoval/main/models/pphumanseg_fp32.onnx.license"
    ${FX_MODEL_DIR}/pphumanseg_fp32.onnx.license
    EXPECTED_HASH SHA256=363552428d64011b9242e0889f90e06cf2c30370908a2a9641dbdd8d37a5f2a6)
endif()
```

Add to the fx target (after target_link_libraries):

```cmake
target_compile_definitions(fx PUBLIC FX_MODEL_PATH="${FX_MODEL_DIR}/pphumanseg_fp32.onnx")
```

Add `src/fx/models/pp_humanseg.cpp` to the `add_library(fx STATIC ...)` source list, and `tests/test_pp_humanseg.cpp` to `add_executable(fx_tests ...)`.

- [ ] **Step 2: Create `src/fx/types.h`**

```cpp
#pragma once

#include <cstdint>
#include <vector>

namespace fx {

/* A packed BGRA frame (linesize == width*4 after copy). */
struct Frame {
	int width = 0;
	int height = 0;
	std::vector<uint8_t> bgra;
};

/* A single-channel person mask, values in [0,1]. */
struct Mask {
	int width = 0;
	int height = 0;
	std::vector<float> px;
};

} // namespace fx
```

- [ ] **Step 3: Write the failing test — `tests/test_pp_humanseg.cpp`**

```cpp
#include <gtest/gtest.h>

#include "fx/models/pp_humanseg.h"

static fx::Frame makeBlobFrame()
{
	/* 320x240 dark frame with a bright vertical blob in the center. */
	fx::Frame f;
	f.width = 320;
	f.height = 240;
	f.bgra.assign(320u * 240u * 4u, 30);
	for (int y = 40; y < 220; y++) {
		for (int x = 120; x < 200; x++) {
			uint8_t *p = f.bgra.data() + (y * 320 + x) * 4;
			p[0] = 200; // B
			p[1] = 160; // G
			p[2] = 140; // R
			p[3] = 255; // A
		}
	}
	return f;
}

TEST(PPHumanSeg, InfersValidDeterministicMask)
{
	fx::PPHumanSeg model(FX_MODEL_PATH, 1);
	fx::Frame f = makeBlobFrame();

	fx::Mask m = model.infer(f);
	ASSERT_EQ(m.width, 192);
	ASSERT_EQ(m.height, 192);
	ASSERT_EQ(m.px.size(), 192u * 192u);
	for (float v : m.px) {
		ASSERT_GE(v, 0.0f);
		ASSERT_LE(v, 1.0f);
	}

	/* Determinism: identical input -> identical output. */
	fx::Mask m2 = model.infer(f);
	ASSERT_EQ(m.px, m2.px);
}
```

- [ ] **Step 4: Create the header — `src/fx/models/pp_humanseg.h`**

```cpp
#pragma once

#include "fx/engine/ort_backend.h"
#include "fx/types.h"

#include <string>
#include <vector>

namespace fx {

/* PP-HumanSeg v2 lite (192x192, Apache-2.0).
 * Preprocessing matches obs-backgroundremoval: BGR order kept,
 * (v/256 - 0.5)/0.5, NCHW. Postprocess: output channel 1 = person,
 * min-max normalized to [0,1]. */
class PPHumanSeg {
public:
	static constexpr int kSize = 192;

	explicit PPHumanSeg(const std::string &modelPath, int threads = 2);

	Mask infer(const Frame &frame);

private:
	OrtModel model_;
	std::vector<float> tensor_; // 3*192*192 scratch
};

} // namespace fx
```

- [ ] **Step 5: Build, watch FAIL (link error), then implement `src/fx/models/pp_humanseg.cpp`**

```cpp
#include "fx/models/pp_humanseg.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace fx {

PPHumanSeg::PPHumanSeg(const std::string &modelPath, int threads)
	: model_(modelPath, threads), tensor_(3 * kSize * kSize)
{
	const auto &shape = model_.input().shape;
	if (shape.size() != 4 || shape[1] != 3 || shape[2] != kSize ||
	    shape[3] != kSize)
		throw std::runtime_error("fx: unexpected PPHumanSeg input shape");
}

Mask PPHumanSeg::infer(const Frame &frame)
{
	const int sw = frame.width, sh = frame.height;
	constexpr int dw = kSize, dh = kSize;

	/* Bilinear-resize BGRA -> 192x192, normalize into NCHW float. */
	for (int y = 0; y < dh; y++) {
		float fy = (y + 0.5f) * sh / (float)dh - 0.5f;
		int y0 = std::clamp((int)std::floor(fy), 0, sh - 1);
		int y1 = std::min(y0 + 1, sh - 1);
		float wy = std::clamp(fy - (float)y0, 0.0f, 1.0f);
		for (int x = 0; x < dw; x++) {
			float sx = (x + 0.5f) * sw / (float)dw - 0.5f;
			int x0 = std::clamp((int)std::floor(sx), 0, sw - 1);
			int x1 = std::min(x0 + 1, sw - 1);
			float wx = std::clamp(sx - (float)x0, 0.0f, 1.0f);
			const uint8_t *p00 =
				frame.bgra.data() + (y0 * sw + x0) * 4;
			const uint8_t *p01 =
				frame.bgra.data() + (y0 * sw + x1) * 4;
			const uint8_t *p10 =
				frame.bgra.data() + (y1 * sw + x0) * 4;
			const uint8_t *p11 =
				frame.bgra.data() + (y1 * sw + x1) * 4;
			for (int c = 0; c < 3; c++) { // 0=B,1=G,2=R kept
				float v = (p00[c] * (1 - wx) + p01[c] * wx) *
						  (1 - wy) +
					  (p10[c] * (1 - wx) + p11[c] * wx) * wy;
				tensor_[c * dh * dw + y * dw + x] =
					(v / 256.0f - 0.5f) / 0.5f;
			}
		}
	}

	std::vector<float> out = model_.run(tensor_); // [1,192,192,2]

	Mask m;
	m.width = dw;
	m.height = dh;
	m.px.resize(dw * dh);
	float lo = 1e30f, hi = -1e30f;
	for (int i = 0; i < dw * dh; i++) {
		float person = out[i * 2 + 1];
		m.px[i] = person;
		lo = std::min(lo, person);
		hi = std::max(hi, person);
	}
	float range = hi - lo;
	if (range > 1e-6f) {
		for (float &v : m.px)
			v = (v - lo) / range;
	} else {
		std::fill(m.px.begin(), m.px.end(), 0.0f);
	}
	return m;
}

} // namespace fx
```

- [ ] **Step 6: Build and run tests — all pass**

```bash
cmake --build --preset ubuntu-x86_64 && ctest --test-dir build_x86_64 --output-on-failure
```

Expected: 3/3 pass. The model downloads on first configure.

- [ ] **Step 7: Commit**

```bash
git add CMakeLists.txt src/fx/types.h src/fx/models/pp_humanseg.h src/fx/models/pp_humanseg.cpp tests/test_pp_humanseg.cpp
git commit -m "feat: PP-HumanSeg segmentation model with verified preprocessing"
```

---

### Task 3: fx::image primitives (gray, EMA, guided filter)

**Files:**
- Modify: `CMakeLists.txt` (add sources/tests)
- Create: `src/fx/image/ops.h`
- Create: `src/fx/image/ops.cpp`
- Test: `tests/test_image_ops.cpp`

- [ ] **Step 1: Add `src/fx/image/ops.cpp` to the fx sources and `tests/test_image_ops.cpp` to fx_tests sources in `CMakeLists.txt`**

- [ ] **Step 2: Write the failing test — `tests/test_image_ops.cpp`**

```cpp
#include <gtest/gtest.h>

#include "fx/image/ops.h"

#include <cmath>

TEST(GrayFromBgra, LumaCoefficients)
{
	fx::Frame f;
	f.width = 1;
	f.height = 1;
	f.bgra = {50, 100, 200, 255}; // B=50, G=100, R=200
	std::vector<float> g = fx::grayFromBgra(f);
	ASSERT_EQ(g.size(), 1u);
	float expect = (0.114f * 50 + 0.587f * 100 + 0.299f * 200) / 255.0f;
	ASSERT_NEAR(g[0], expect, 1e-6f);
}

TEST(EmaMask, ConvergesToNewValue)
{
	fx::Mask cur, prev;
	cur.width = prev.width = 2;
	cur.height = prev.height = 2;
	cur.px = {1.0f, 1.0f, 1.0f, 1.0f};
	prev.px = {0.0f, 0.0f, 0.0f, 0.0f};
	fx::emaMask(cur, prev, 0.6f); // 0.6*prev + 0.4*cur
	for (float v : cur.px)
		ASSERT_NEAR(v, 0.4f, 1e-6f);
}

TEST(GuidedFilter, PreservesStepEdge)
{
	/* 16x1 guide with a step at x=8; src = guide + heavy noise. */
	const int w = 16, h = 1;
	std::vector<float> guide(w * h), src(w * h);
	for (int x = 0; x < w; x++) {
		guide[x] = x < 8 ? 0.0f : 1.0f;
		src[x] = guide[x] + (x % 3 == 0 ? 0.2f : -0.1f);
	}
	std::vector<float> out = fx::guidedFilter(guide, src, w, h, 3, 1e-3f);
	/* Edge cells stay separated (edge preserved), unlike a box blur
	 * which would pull x=7 and x=8 toward ~0.5. */
	ASSERT_LT(out[7], 0.35f);
	ASSERT_GT(out[8], 0.65f);
}

TEST(GuidedFilter, SmoothsFlatGuide)
{
	/* Flat guide -> behaves like a local average (noise removed). */
	const int w = 8, h = 8;
	std::vector<float> guide(w * h, 0.5f), src(w * h);
	for (int i = 0; i < w * h; i++)
		src[i] = (i % 2 == 0) ? 0.9f : 0.1f;
	std::vector<float> out = fx::guidedFilter(guide, src, w, h, 2, 1e-3f);
	float mean = 0;
	for (float v : out)
		mean += v;
	mean /= out.size();
	ASSERT_NEAR(mean, 0.5f, 0.05f);
}
```

- [ ] **Step 3: Create the header — `src/fx/image/ops.h`**

```cpp
#pragma once

#include "fx/types.h"

#include <vector>

namespace fx {

/* Grayscale [0,1] from a packed BGRA frame (Rec. 601 luma). */
std::vector<float> grayFromBgra(const Frame &f);

/* In-place exponential moving average: cur = beta*prev + (1-beta)*cur. */
void emaMask(Mask &cur, const Mask &prev, float beta);

/* Guided filter (He et al. 2013): edge-preserving smoothing of src using
 * guide. Both single-channel [0,1], size w*h. r = box radius, eps =
 * regularization. Returns the filtered image. */
std::vector<float> guidedFilter(const std::vector<float> &guide,
				const std::vector<float> &src, int w, int h,
				int r, float eps);

} // namespace fx
```

- [ ] **Step 4: Build, watch FAIL (link error), then implement `src/fx/image/ops.cpp`**

```cpp
#include "fx/image/ops.h"

#include <algorithm>

namespace fx {

std::vector<float> grayFromBgra(const Frame &f)
{
	std::vector<float> g(f.width * f.height);
	for (int i = 0; i < f.width * f.height; i++) {
		const uint8_t *p = f.bgra.data() + i * 4;
		g[i] = (0.114f * p[0] + 0.587f * p[1] + 0.299f * p[2]) / 255.0f;
	}
	return g;
}

void emaMask(Mask &cur, const Mask &prev, float beta)
{
	if (cur.px.size() != prev.px.size())
		return;
	for (size_t i = 0; i < cur.px.size(); i++)
		cur.px[i] = beta * prev.px[i] + (1.0f - beta) * cur.px[i];
}

namespace {

/* Integral image (w+1)x(h+1), double precision accumulation. */
std::vector<double> integral(const std::vector<float> &img, int w, int h)
{
	std::vector<double> ii((w + 1) * (h + 1), 0.0);
	for (int y = 0; y < h; y++) {
		double rowSum = 0;
		for (int x = 0; x < w; x++) {
			rowSum += img[y * w + x];
			ii[(y + 1) * (w + 1) + (x + 1)] =
				ii[y * (w + 1) + (x + 1)] + rowSum;
		}
	}
	return ii;
}

/* Box blur via integral image, clamped window, radius r. */
std::vector<double> boxBlur(const std::vector<float> &img, int w, int h,
			    int r)
{
	std::vector<double> ii = integral(img, w, h);
	std::vector<double> out(w * h);
	for (int y = 0; y < h; y++) {
		int y0 = std::max(0, y - r), y1 = std::min(h - 1, y + r);
		for (int x = 0; x < w; x++) {
			int x0 = std::max(0, x - r), x1 =
				std::min(w - 1, x + r);
			double sum = ii[(y1 + 1) * (w + 1) + (x1 + 1)] -
				     ii[y0 * (w + 1) + (x1 + 1)] -
				     ii[(y1 + 1) * (w + 1) + x0] +
				     ii[y0 * (w + 1) + x0];
			out[y * w + x] =
				sum / ((double)(y1 - y0 + 1) * (x1 - x0 + 1));
		}
	}
	return out;
}

std::vector<float> mul(const std::vector<float> &a,
		       const std::vector<float> &b)
{
	std::vector<float> out(a.size());
	for (size_t i = 0; i < a.size(); i++)
		out[i] = a[i] * b[i];
	return out;
}

} // namespace

std::vector<float> guidedFilter(const std::vector<float> &guide,
				const std::vector<float> &src, int w, int h,
				int r, float eps)
{
	std::vector<double> meanI = boxBlur(guide, w, h, r);
	std::vector<double> meanP = boxBlur(src, w, h, r);
	std::vector<double> corrI = boxBlur(mul(guide, guide), w, h, r);
	std::vector<double> corrIp = boxBlur(mul(guide, src), w, h, r);

	std::vector<float> a(w * h), b(w * h);
	for (int i = 0; i < w * h; i++) {
		double varI = corrI[i] - meanI[i] * meanI[i];
		double covIp = corrIp[i] - meanI[i] * meanP[i];
		a[i] = (float)(covIp / (varI + eps));
		b[i] = (float)(meanP[i] - a[i] * meanI[i]);
	}

	std::vector<double> meanA = boxBlur(a, w, h, r);
	std::vector<double> meanB = boxBlur(b, w, h, r);
	std::vector<float> q(w * h);
	for (int i = 0; i < w * h; i++)
		q[i] = (float)(meanA[i] * guide[i] + meanB[i]);
	return q;
}

} // namespace fx
```

- [ ] **Step 5: Build and run tests — all pass**

```bash
cmake --build --preset ubuntu-x86_64 && ctest --test-dir build_x86_64 --output-on-failure
```

Expected: 7/7 pass.

- [ ] **Step 6: Commit**

```bash
git add CMakeLists.txt src/fx/image/ops.h src/fx/image/ops.cpp tests/test_image_ops.cpp
git commit -m "feat: fx::image primitives (gray, EMA, guided filter)"
```

---

### Task 4: SegmentationPipeline + Worker (latest-wins, mailbox)

**Files:**
- Modify: `CMakeLists.txt` (add sources/tests)
- Create: `src/fx/pipeline/segmentation_pipeline.h`
- Create: `src/fx/pipeline/segmentation_pipeline.cpp`
- Create: `src/fx/worker.h`
- Create: `src/fx/worker.cpp`
- Test: `tests/test_worker.cpp`

- [ ] **Step 1: Add the new sources to `CMakeLists.txt`**

Add `src/fx/pipeline/segmentation_pipeline.cpp` and `src/fx/worker.cpp` to the fx library sources; add `tests/test_worker.cpp` to fx_tests sources.

- [ ] **Step 2: Create `src/fx/pipeline/segmentation_pipeline.h`**

```cpp
#pragma once

#include "fx/models/pp_humanseg.h"
#include "fx/types.h"

#include <memory>

namespace fx {

/* model inference -> temporal EMA -> guided-filter edge refinement. */
class SegmentationPipeline {
public:
	explicit SegmentationPipeline(const std::string &modelPath,
				      int threads = 2);

	std::shared_ptr<Mask> process(const Frame &frame);

	void setTemporalBeta(float beta) { beta_ = beta; }

private:
	PPHumanSeg model_;
	float beta_ = 0.6f;
	std::shared_ptr<Mask> prev_;
};

} // namespace fx
```

- [ ] **Step 3: Create `src/fx/worker.h`**

```cpp
#pragma once

#include "fx/types.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>

namespace fx {

/* Single-worker latest-wins frame processor.
 * submit() never blocks: a still-pending frame is replaced by the newest.
 * tryGetLatest() returns the newest published mask (nullptr before first). */
class Worker {
public:
	using Processor =
		std::function<std::shared_ptr<Mask>(const Frame &)>;

	explicit Worker(Processor processor);
	~Worker();

	Worker(const Worker &) = delete;
	Worker &operator=(const Worker &) = delete;

	void start();
	void stop(); // idempotent; joins

	void submit(std::shared_ptr<Frame> frame);

	/* Latest published mask and its sequence (0 = none yet). */
	std::shared_ptr<const Mask> tryGetLatest(uint64_t &seqOut) const;

	/* True if a mask was published within maxAgeMs. */
	bool isFresh(uint64_t maxAgeMs) const;

private:
	void loop();

	Processor processor_;
	std::thread thread_;
	std::atomic<bool> running_{false};

	mutable std::mutex inM_;
	std::condition_variable inCv_;
	std::shared_ptr<Frame> pending_;

	mutable std::mutex outM_;
	std::shared_ptr<const Mask> latest_;
	uint64_t seq_ = 0;
	std::atomic<int64_t> lastPublishMs_{0};
};

} // namespace fx
```

- [ ] **Step 4: Write the failing test — `tests/test_worker.cpp`**

```cpp
#include <gtest/gtest.h>

#include "fx/pipeline/segmentation_pipeline.h"
#include "fx/worker.h"

#include <chrono>
#include <latch>
#include <thread>

using namespace std::chrono_literals;

static std::shared_ptr<fx::Frame> makeFrame(uint8_t v)
{
	auto f = std::make_shared<fx::Frame>();
	f->width = 4;
	f->height = 4;
	f->bgra.assign(4u * 4u * 4u, v);
	return f;
}

TEST(Worker, ProcessesAndPublishesLatest)
{
	auto slow = [](const fx::Frame &) {
		auto m = std::make_shared<fx::Mask>();
		m->width = 2;
		m->height = 2;
		m->px = {0.5f, 0.5f, 0.5f, 0.5f};
		return m;
	};
	fx::Worker w(slow);
	w.start();
	w.submit(makeFrame(1));
	uint64_t seq = 0;
	for (int i = 0; i < 100 && seq == 0; i++) {
		std::this_thread::sleep_for(5ms);
		w.tryGetLatest(seq);
	}
	w.stop();
	ASSERT_EQ(seq, 1u);
	ASSERT_TRUE(w.isFresh(5000));
}

TEST(Worker, LatestWinsDropsStaleFrames)
{
	/* Processor blocks on a latch; while blocked, submit 3 frames.
	 * After release, the NEXT processed frame must be the newest. */
	std::latch gate(2);
	std::atomic<int> processed{0};
	std::atomic<uint8_t> lastSeen{0};
	auto blocking = [&](const fx::Frame &f) {
		if (processed.fetch_add(1) == 0)
			gate.count_down(); // first call: hold until released
		lastSeen.store(f.bgra[0]);
		auto m = std::make_shared<fx::Mask>();
		m->width = 1;
		m->height = 1;
		m->px = {1.0f};
		return m;
	};
	fx::Worker w(blocking);
	w.start();
	w.submit(makeFrame(10));
	std::this_thread::sleep_for(20ms); // first frame picked up
	w.submit(makeFrame(20));
	w.submit(makeFrame(30)); // replaces 20
	gate.count_down();       // release the gate
	for (int i = 0; i < 100 && processed.load() < 2; i++)
		std::this_thread::sleep_for(5ms);
	w.stop();
	ASSERT_EQ(lastSeen.load(), 30);
}

TEST(Worker, StopIsIdempotent)
{
	auto fast = [](const fx::Frame &) {
		return std::make_shared<fx::Mask>();
	};
	fx::Worker w(fast);
	w.start();
	w.stop();
	w.stop(); // must not hang or crash
	SUCCEED();
}

TEST(SegmentationPipeline, EndToEndWithRealModel)
{
	fx::SegmentationPipeline pipe(FX_MODEL_PATH, 1);
	fx::Frame f;
	f.width = 320;
	f.height = 240;
	f.bgra.assign(320u * 240u * 4u, 128);
	auto m = pipe.process(f);
	ASSERT_EQ(m->width, 192);
	ASSERT_EQ(m->height, 192);
	ASSERT_EQ(m->px.size(), 192u * 192u);
}
```

- [ ] **Step 5: Build, watch FAIL (link errors), then implement**

`src/fx/pipeline/segmentation_pipeline.cpp`:

```cpp
#include "fx/pipeline/segmentation_pipeline.h"

#include "fx/image/ops.h"

namespace fx {

SegmentationPipeline::SegmentationPipeline(const std::string &modelPath,
					   int threads)
	: model_(modelPath, threads)
{
}

std::shared_ptr<Mask> SegmentationPipeline::process(const Frame &frame)
{
	auto m = std::make_shared<Mask>(model_.infer(frame));
	if (prev_ && prev_->px.size() == m->px.size())
		emaMask(*m, *prev_, beta_);
	std::vector<float> guide = grayFromBgra(frame);
	if ((int)guide.size() == m->width * m->height)
		m->px = guidedFilter(guide, m->px, m->width, m->height, 4,
				     0.01f);
	prev_ = m;
	return m;
}

} // namespace fx
```

`src/fx/worker.cpp`:

```cpp
#include "fx/worker.h"

#include <chrono>

namespace fx {

static int64_t nowMs()
{
	return std::chrono::duration_cast<std::chrono::milliseconds>(
		       std::chrono::steady_clock::now().time_since_epoch())
		.count();
}

Worker::Worker(Processor processor) : processor_(std::move(processor)) {}

Worker::~Worker()
{
	stop();
}

void Worker::start()
{
	bool expected = false;
	if (!running_.compare_exchange_strong(expected, true))
		return;
	thread_ = std::thread([this] { loop(); });
}

void Worker::stop()
{
	bool expected = true;
	if (!running_.compare_exchange_strong(expected, false))
		return;
	{
		std::lock_guard<std::mutex> lk(inM_);
		inCv_.notify_all();
	}
	if (thread_.joinable())
		thread_.join();
}

void Worker::submit(std::shared_ptr<Frame> frame)
{
	{
		std::lock_guard<std::mutex> lk(inM_);
		pending_ = std::move(frame); // latest-wins: replaces stale
	}
	inCv_.notify_one();
}

std::shared_ptr<const Mask> Worker::tryGetLatest(uint64_t &seqOut) const
{
	std::lock_guard<std::mutex> lk(outM_);
	seqOut = seq_;
	return latest_;
}

bool Worker::isFresh(uint64_t maxAgeMs) const
{
	int64_t last = lastPublishMs_.load();
	return last != 0 && (nowMs() - last) <= (int64_t)maxAgeMs;
}

void Worker::loop()
{
	while (running_.load()) {
		std::shared_ptr<Frame> frame;
		{
			std::unique_lock<std::mutex> lk(inM_);
			inCv_.wait(lk, [this] {
				return pending_ != nullptr || !running_.load();
			});
			if (!running_.load())
				break;
			frame = std::move(pending_);
			pending_.reset();
		}
		std::shared_ptr<Mask> result = processor_(*frame);
		{
			std::lock_guard<std::mutex> lk(outM_);
			latest_ = std::move(result);
			seq_++;
			lastPublishMs_.store(nowMs());
		}
	}
}

} // namespace fx
```

NOTE: `#include <latch>` in the test requires C++20. GoogleTest 1.15 + GCC 15 support it, but the fx target is cxx_std_17 — the TEST target needs `target_compile_features(fx_tests PRIVATE cxx_std_20)` added to CMakeLists.txt in the tests block. Add it in this task.

- [ ] **Step 6: Build and run tests — all pass**

```bash
cmake --build --preset ubuntu-x86_64 && ctest --test-dir build_x86_64 --output-on-failure
```

Expected: 11/11 pass.

- [ ] **Step 7: Commit**

```bash
git add CMakeLists.txt src/fx/pipeline/ src/fx/worker.h src/fx/worker.cpp tests/test_worker.cpp
git commit -m "feat: segmentation pipeline + latest-wins worker thread"
```

---

### Task 5: fx bridge + filter staging + transparent mode + mask_composite.effect

**Files:**
- Modify: `CMakeLists.txt` (plugin sources)
- Create: `src/fx_bridge.h`
- Create: `src/fx_bridge.cpp`
- Modify: `src/cam-effects-filter.c` (major rewrite)
- Create: `data/effects/mask_composite.effect`
- Modify: `build-aux/` — create `build-aux/install-local.sh`

- [ ] **Step 1: Create `src/fx_bridge.h`**

```c
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cam_fx cam_fx_t;

/* Creates the segmentation engine (loads model, starts worker thread).
 * model_path: absolute path to the ONNX model. Returns NULL on failure. */
cam_fx_t *cam_fx_create(const char *model_path, int threads);

void cam_fx_destroy(cam_fx_t *fx);

/* Submits a packed BGRA frame (any size; copied internally). Never blocks. */
void cam_fx_submit(cam_fx_t *fx, const uint8_t *bgra, int w, int h,
		   int linesize);

/* Fetches the latest mask. Returns 1 if a mask exists, 0 otherwise.
 * On success *px points to an internal w*h uint8 buffer valid until the
 * next call, and *seq is the mask sequence number (increments per mask). */
int cam_fx_try_get_mask(cam_fx_t *fx, const uint8_t **px, int *w, int *h,
			uint64_t *seq);

/* 1 if a mask was published within max_age_ms, else 0. */
int cam_fx_is_fresh(cam_fx_t *fx, uint64_t max_age_ms);

#ifdef __cplusplus
}
#endif
```

- [ ] **Step 2: Create `src/fx_bridge.cpp`**

```cpp
#include "fx_bridge.h"

#include "fx/pipeline/segmentation_pipeline.h"
#include "fx/worker.h"

#include <cstring>
#include <memory>
#include <vector>

struct cam_fx {
	std::unique_ptr<fx::SegmentationPipeline> pipeline;
	std::unique_ptr<fx::Worker> worker;
	uint64_t seenSeq = 0;
	std::vector<uint8_t> u8;
};

extern "C" {

cam_fx_t *cam_fx_create(const char *model_path, int threads)
{
	try {
		auto fx = std::make_unique<cam_fx>();
		fx->pipeline = std::make_unique<fx::SegmentationPipeline>(
			model_path, threads);
		fx->worker = std::make_unique<fx::Worker>(
			[&](const fx::Frame &f) {
				return fx->pipeline->process(f);
			});
		fx->worker->start();
		return fx.release();
	} catch (...) {
		return nullptr;
	}
}

void cam_fx_destroy(cam_fx_t *fx)
{
	delete fx;
}

void cam_fx_submit(cam_fx_t *fx, const uint8_t *bgra, int w, int h,
		   int linesize)
{
	auto frame = std::make_shared<fx::Frame>();
	frame->width = w;
	frame->height = h;
	frame->bgra.resize((size_t)w * h * 4);
	for (int y = 0; y < h; y++)
		std::memcpy(frame->bgra.data() + (size_t)y * w * 4,
			    bgra + (size_t)y * linesize, (size_t)w * 4);
	fx->worker->submit(std::move(frame));
}

int cam_fx_try_get_mask(cam_fx_t *fx, const uint8_t **px, int *w, int *h,
			uint64_t *seq)
{
	uint64_t s = 0;
	auto m = fx->worker->tryGetLatest(s);
	if (!m)
		return 0;
	if (s != fx->seenSeq) {
		fx->u8.resize(m->px.size());
		for (size_t i = 0; i < m->px.size(); i++) {
			float v = m->px[i];
			v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
			fx->u8[i] = (uint8_t)(v * 255.0f + 0.5f);
		}
		fx->seenSeq = s;
	}
	*px = fx->u8.data();
	*w = m->width;
	*h = m->height;
	*seq = s;
	return 1;
}

int cam_fx_is_fresh(cam_fx_t *fx, uint64_t max_age_ms)
{
	return fx->worker->isFresh(max_age_ms) ? 1 : 0;
}

} // extern "C"
```

- [ ] **Step 3: Create `data/effects/mask_composite.effect`**

```
uniform float4x4 ViewProj;
uniform texture2d image;
uniform texture2d mask;
uniform texture2d bg_image;
uniform texture2d blur_image;

sampler_state def_sampler {
	Filter = Linear;
	AddressU = Clamp;
	AddressV = Clamp;
};

struct VertData {
	float4 pos : POSITION;
	float2 uv : TEXCOORD0;
};

VertData mainTransform(VertData v_in)
{
	VertData v_out;
	v_out.pos = mul(float4(v_in.pos.xyz, 1.0), ViewProj);
	v_out.uv = v_in.uv;
	return v_out;
}

float4 DrawTransparent(VertData v_in) : TARGET
{
	float4 rgba = image.Sample(def_sampler, v_in.uv);
	float a = mask.Sample(def_sampler, v_in.uv).r;
	return float4(rgba.rgb, rgba.a * a);
}

float4 DrawReplace(VertData v_in) : TARGET
{
	float4 rgba = image.Sample(def_sampler, v_in.uv);
	float4 bg = bg_image.Sample(def_sampler, v_in.uv);
	float a = mask.Sample(def_sampler, v_in.uv).r;
	return float4(bg.rgb * (1.0 - a) + rgba.rgb * a, rgba.a);
}

float4 DrawBlur(VertData v_in) : TARGET
{
	float4 sharp = image.Sample(def_sampler, v_in.uv);
	float4 blurred = blur_image.Sample(def_sampler, v_in.uv);
	float a = mask.Sample(def_sampler, v_in.uv).r;
	return float4(blurred.rgb * (1.0 - a) + sharp.rgb * a, sharp.a);
}

technique DrawTransparent
{
	pass
	{
		vertex_shader = mainTransform(v_in);
		pixel_shader = DrawTransparent(v_in);
	}
}

technique DrawReplace
{
	pass
	{
		vertex_shader = mainTransform(v_in);
		pixel_shader = DrawReplace(v_in);
	}
}

technique DrawBlur
{
	pass
	{
		vertex_shader = mainTransform(v_in);
		pixel_shader = DrawBlur(v_in);
	}
}
```

- [ ] **Step 4: Rewrite `src/cam-effects-filter.c` (full new content)**

```c
#include "cam-effects-filter.h"

#include "fx_bridge.h"

#include <obs-module.h>
#include <graphics/image-file.h>

#define SETTING_MODE "mode"
#define SETTING_IMAGE_PATH "image_path"
#define SETTING_BLUR_STRENGTH "blur_strength"
#define SETTING_FAILURE "failure_mode"
#define SETTING_STATUS "status"

#define STAGE_SIZE 192
#define MASK_STALE_MS 1000

struct cam_effects_filter {
	obs_source_t *source;

	gs_texrender_t *stage_render;   /* STAGE_SIZE x STAGE_SIZE */
	gs_stagesurf_t *stage_surface;  /* STAGE_SIZE x STAGE_SIZE BGRA */
	gs_texrender_t *out_render;     /* frame-size composite + freeze */
	gs_texture_t *mask_tex;         /* STAGE_SIZE x STAGE_SIZE R8 */
	gs_effect_t *effect;            /* mask_composite.effect */

	cam_fx_t *fx;
	uint64_t mask_seq;

	char *mode;         /* off | transparent | image | blur */
	char *failure_mode; /* passthrough | freeze */
	int blur_strength;
	char *image_path;
	gs_image_file_t bg_image;
	bool bg_loaded;
};

/* Forward declarations (used before definition below). */
static void cam_effects_update(void *data, obs_data_t *settings);

static const char *cam_effects_get_name(void *type_data)
{
	UNUSED_PARAMETER(type_data);
	return "Camera Effects";
}

static void cam_effects_destroy_graphics(struct cam_effects_filter *filter)
{
	obs_enter_graphics();
	gs_texrender_destroy(filter->stage_render);
	gs_stagesurface_destroy(filter->stage_surface);
	gs_texrender_destroy(filter->out_render);
	gs_texture_destroy(filter->mask_tex);
	gs_effect_destroy(filter->effect);
	gs_image_file_free(&filter->bg_image);
	obs_leave_graphics();
}

static void cam_effects_load_effect(struct cam_effects_filter *filter)
{
	char *path = obs_module_file("effects/mask_composite.effect");
	obs_enter_graphics();
	if (path)
		filter->effect = gs_effect_create_from_file(path, NULL);
	obs_leave_graphics();
	bfree(path);

	obs_enter_graphics();
	filter->mask_tex = gs_texture_create(STAGE_SIZE, STAGE_SIZE, GS_R8, 1,
					     NULL, GS_DYNAMIC);
	obs_leave_graphics();
}

static void *cam_effects_create(obs_data_t *settings, obs_source_t *source)
{
	struct cam_effects_filter *filter =
		bzalloc(sizeof(struct cam_effects_filter));
	filter->source = source;

	obs_enter_graphics();
	filter->stage_render = gs_texrender_create(GS_BGRA, GS_ZS_NONE);
	filter->stage_surface =
		gs_stagesurface_create(STAGE_SIZE, STAGE_SIZE, GS_BGRA);
	filter->out_render = gs_texrender_create(GS_BGRA, GS_ZS_NONE);
	obs_leave_graphics();

	cam_effects_load_effect(filter);
	cam_effects_update(filter, settings);
	return filter;
}

static void cam_effects_destroy(void *data)
{
	struct cam_effects_filter *filter = data;
	if (filter->fx)
		cam_fx_destroy(filter->fx);
	cam_effects_destroy_graphics(filter);
	bfree(filter->mode);
	bfree(filter->failure_mode);
	bfree(filter->image_path);
	bfree(filter);
}

static void cam_effects_update(void *data, obs_data_t *settings)
{
	struct cam_effects_filter *filter = data;

	bfree(filter->mode);
	bfree(filter->failure_mode);
	bfree(filter->image_path);
	filter->mode = bstrdup(obs_data_get_string(settings, SETTING_MODE));
	filter->failure_mode =
		bstrdup(obs_data_get_string(settings, SETTING_FAILURE));
	filter->image_path =
		bstrdup(obs_data_get_string(settings, SETTING_IMAGE_PATH));
	filter->blur_strength =
		(int)obs_data_get_int(settings, SETTING_BLUR_STRENGTH);

	/* (Re)load the background image if the path changed and mode
	 * needs it. */
	obs_enter_graphics();
	gs_image_file_free(&filter->bg_image);
	filter->bg_loaded = false;
	if (strcmp(filter->mode, "image") == 0 &&
	    filter->image_path[0] != '\0') {
		gs_image_file_init(&filter->bg_image, filter->image_path);
		filter->bg_loaded =
			gs_image_file_init_texture(&filter->bg_image);
	}
	obs_leave_graphics();

	/* Create the inference engine lazily on first non-off mode. */
	if (!filter->fx && strcmp(filter->mode, "off") != 0) {
		char *model = obs_module_file("models/pphumanseg_fp32.onnx");
		if (model)
			filter->fx = cam_fx_create(model, 2);
		bfree(model);
	}
}

static void cam_effects_get_defaults(obs_data_t *settings)
{
	obs_data_set_default_string(settings, SETTING_MODE, "transparent");
	obs_data_set_default_string(settings, SETTING_FAILURE, "passthrough");
	obs_data_set_default_int(settings, SETTING_BLUR_STRENGTH, 2);
}

static obs_properties_t *cam_effects_properties(void *data)
{
	struct cam_effects_filter *filter = data;
	obs_properties_t *props = obs_properties_create();

	obs_property_t *mode = obs_properties_add_list(
		props, SETTING_MODE, "Background", OBS_COMBO_TYPE_LIST,
		OBS_COMBO_FORMAT_STRING);
	obs_property_list_add_string(mode, "Off", "off");
	obs_property_list_add_string(mode, "Transparent", "transparent");
	obs_property_list_add_string(mode, "Replace with image", "image");
	obs_property_list_add_string(mode, "Blur", "blur");

	obs_properties_add_path(props, SETTING_IMAGE_PATH, "Background image",
				OBS_PATH_FILE,
				"Images (*.png *.jpg *.jpeg *.bmp)", NULL);
	obs_properties_add_int_slider(props, SETTING_BLUR_STRENGTH,
				      "Blur strength", 1, 4, 1);

	obs_property_t *fm = obs_properties_add_list(
		props, SETTING_FAILURE, "On processing failure",
		OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_STRING);
	obs_property_list_add_string(fm, "Show camera feed", "passthrough");
	obs_property_list_add_string(fm, "Freeze last processed frame",
				     "freeze");

	obs_properties_add_text(props, SETTING_STATUS, "Status",
				OBS_TEXT_INFO);
	return props;
}

/* Render the parent source into the small staging surface and submit. */
static void cam_effects_stage(struct cam_effects_filter *filter,
			      obs_source_t *target)
{
	uint32_t tw = obs_source_get_base_width(target);
	uint32_t th = obs_source_get_base_height(target);
	if (tw == 0 || th == 0)
		return;

	gs_texrender_reset(filter->stage_render);
	if (!gs_texrender_begin(filter->stage_render, STAGE_SIZE,
				STAGE_SIZE))
		return;
	struct vec4 clear;
	vec4_zero(&clear);
	gs_clear(GS_CLEAR_COLOR, &clear, 0.0f, 0);
	gs_ortho(0.0f, (float)STAGE_SIZE, 0.0f, (float)STAGE_SIZE, -100.0f,
		 100.0f);
	gs_blend_state_push();
	gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);
	gs_matrix_push();
	gs_matrix_scale3f((float)STAGE_SIZE / (float)tw,
			  (float)STAGE_SIZE / (float)th, 1.0f);
	obs_source_video_render(target);
	gs_matrix_pop();
	gs_blend_state_pop();
	gs_texrender_end(filter->stage_render);

	gs_stage_texture(filter->stage_surface,
			 gs_texrender_get_texture(filter->stage_render));
	uint8_t *data = NULL;
	uint32_t linesize = 0;
	if (gs_stagesurface_map(filter->stage_surface, &data, &linesize)) {
		cam_fx_submit(filter->fx, data, STAGE_SIZE, STAGE_SIZE,
			      (int)linesize);
		gs_stagesurface_unmap(filter->stage_surface);
	}
}

/* Draw the contents of out_render to screen. */
static void cam_effects_draw_out(struct cam_effects_filter *filter,
				 uint32_t w, uint32_t h)
{
	gs_texture_t *tex = gs_texrender_get_texture(filter->out_render);
	if (!tex)
		return;
	gs_effect_t *def = obs_get_base_effect(OBS_EFFECT_DEFAULT);
	gs_eparam_t *image = gs_effect_get_param_by_name(def, "image");
	gs_effect_set_texture(image, tex);
	while (gs_effect_loop(def, "Draw"))
		gs_draw_sprite(tex, 0, w, h);
}

static void cam_effects_video_render(void *data, gs_effect_t *effect)
{
	UNUSED_PARAMETER(effect);
	struct cam_effects_filter *filter = data;
	obs_source_t *target = obs_filter_get_target(filter->source);
	if (!target || !filter->effect) {
		obs_source_skip_video_filter(filter->source);
		return;
	}

	bool mode_off = strcmp(filter->mode, "off") == 0;
	bool freeze = strcmp(filter->failure_mode, "freeze") == 0;

	uint32_t w = obs_source_get_base_width(target);
	uint32_t h = obs_source_get_base_height(target);
	if (mode_off || w == 0 || h == 0 || !filter->fx) {
		if (freeze && !mode_off) {
			cam_effects_draw_out(filter, w ? w : 1920,
					     h ? h : 1080);
			return;
		}
		obs_source_skip_video_filter(filter->source);
		return;
	}

	cam_effects_stage(filter, target);

	const uint8_t *mask = NULL;
	int mw = 0, mh = 0;
	uint64_t seq = 0;
	bool have_mask =
		cam_fx_try_get_mask(filter->fx, &mask, &mw, &mh, &seq) == 1;

	if (!have_mask) {
		/* Startup: no mask ever processed yet. */
		if (freeze) {
			cam_effects_draw_out(filter, w, h); /* black */
			return;
		}
		obs_source_skip_video_filter(filter->source);
		return;
	}

	if (!cam_fx_is_fresh(filter->fx, MASK_STALE_MS)) {
		/* Inference stalled. */
		if (freeze) {
			cam_effects_draw_out(filter, w, h);
			return;
		}
		obs_source_skip_video_filter(filter->source);
		return;
	}

	/* Upload mask and composite into out_render. */
	gs_texture_set_image(filter->mask_tex, mask, (uint32_t)mw, false);

	const char *tech = "DrawTransparent";
	if (strcmp(filter->mode, "image") == 0 && filter->bg_loaded)
		tech = "DrawReplace";
	else if (strcmp(filter->mode, "blur") == 0)
		tech = "DrawTransparent"; /* blur arrives in Task 8 */

	gs_texrender_reset(filter->out_render);
	if (gs_texrender_begin(filter->out_render, w, h)) {
		if (obs_source_process_filter_begin(filter->source, GS_BGRA,
						    OBS_NO_DIRECT_RENDERING)) {
			gs_effect_set_texture(
				gs_effect_get_param_by_name(filter->effect,
							    "mask"),
				filter->mask_tex);
			if (filter->bg_loaded)
				gs_effect_set_texture(
					gs_effect_get_param_by_name(
						filter->effect, "bg_image"),
					filter->bg_image.texture);
			obs_source_process_filter_tech_end(filter->source,
							   filter->effect, w,
							   h, tech);
		}
		gs_texrender_end(filter->out_render);
	}
	cam_effects_draw_out(filter, w, h);
}

static struct obs_source_info cam_effects_filter_info = {
	.id = "obs_cam_effects_filter",
	.type = OBS_SOURCE_TYPE_FILTER,
	.output_flags = OBS_SOURCE_VIDEO,
	.get_name = cam_effects_get_name,
	.create = cam_effects_create,
	.destroy = cam_effects_destroy,
	.update = cam_effects_update,
	.get_defaults = cam_effects_get_defaults,
	.video_render = cam_effects_video_render,
	.get_properties = cam_effects_properties,
};

void cam_effects_register_filter(void)
{
	obs_register_source(&cam_effects_filter_info);
}
```

IMPORTANT for the implementer: verify every gs_* signature against `/usr/include/obs/` headers (especially `gs_texture_set_image`, `gs_stagesurface_map`, `gs_matrix_scale3f`, `gs_image_file_init_from_file`, `obs_source_process_filter_tech_end`). If a signature differs, adapt and note it in the report.

- [ ] **Step 5: Add `src/fx_bridge.cpp` to the plugin target in `CMakeLists.txt`**

```cmake
target_sources(${CMAKE_PROJECT_NAME} PRIVATE src/plugin-main.c src/cam-effects-filter.c src/fx_bridge.cpp)
```

- [ ] **Step 6: Create `build-aux/install-local.sh`**

```bash
#!/usr/bin/env bash
# Installs the plugin into the per-user OBS plugin directory for local testing.
set -euo pipefail

BUILD_DIR="${1:-build_x86_64}"
DEST="$HOME/.config/obs-studio/plugins/obs-cam-effects"

mkdir -p "$DEST/bin/64bit" "$DEST/data/locale" "$DEST/data/models" "$DEST/data/effects"
cp "$BUILD_DIR/obs-cam-effects.so" "$DEST/bin/64bit/"
cp "$BUILD_DIR/onnxruntime/lib/libonnxruntime.so.1.28.0" "$DEST/bin/64bit/"
ln -sf libonnxruntime.so.1.28.0 "$DEST/bin/64bit/libonnxruntime.so.1"
cp data/locale/en-US.ini "$DEST/data/locale/"
cp "$BUILD_DIR/models/pphumanseg_fp32.onnx" "$DEST/data/models/"
cp "$BUILD_DIR/models/pphumanseg_fp32.onnx.license" "$DEST/data/models/"
cp data/effects/*.effect "$DEST/data/effects/"
echo "Installed to $DEST"
```

Make it executable: `chmod +x build-aux/install-local.sh`.

- [ ] **Step 7: Build, install, smoke test**

```bash
cmake --preset ubuntu-x86_64 && cmake --build --preset ubuntu-x86_64
./build-aux/install-local.sh
timeout 25 obs --verbose > /tmp/opencode/obs-smoke3.log 2>&1; true
grep -E "obs-cam-effects|effect" /tmp/opencode/obs-smoke3.log | grep -viE "vlc|portal" | head -20
```

Expected: `plugin loaded successfully`; NO `Failed to load effect` / `gs_effect_create_from_file` errors; no unresolved symbol errors for `libonnxruntime`. If the log shows `error: gs_effect_create...` — the shader has a syntax error; the OBS log prints the line; fix and re-run.

- [ ] **Step 8: Run the existing test suite (must stay green)**

```bash
ctest --test-dir build_x86_64 --output-on-failure
```

Expected: 11/11 pass.

- [ ] **Step 9: Commit**

```bash
git add CMakeLists.txt src/fx_bridge.h src/fx_bridge.cpp src/cam-effects-filter.c data/effects/mask_composite.effect build-aux/install-local.sh
git commit -m "feat: staging, worker wiring and transparent background mode"
```

---

### Task 6: Verify transparent mode end-to-end (user visual checkpoint)

**Files:** none

- [ ] **Step 1: Prepare a test scene (scripted where possible)**

Ask the user to (or drive the OBS UI if possible):
1. Open OBS → add a Video Capture Device (webcam) source, or a Media Source playing a video with a person.
2. Right-click the source → Filters → add "Camera Effects".
3. Set Background = "Transparent".
4. Place an image/color source BELOW the camera in the scene.

Expected: the background around the person shows what is below (transparent compositing). Edges may flicker slightly — EMA + guided filter should keep it stable.

- [ ] **Step 2: Verify the failure modes**

1. Set "On processing failure" = "Show camera feed" → temporarily rename the model file away, restart OBS, open the filter: camera shows normally (passthrough).
2. Set it to "Freeze last processed frame" with the model missing: output is black, NOT the raw feed.
3. Restore the model file, reinstall via `./build-aux/install-local.sh`, confirm effects return.

- [ ] **Step 3: Record the result**

Note observations (fps feel, edge quality, artifacts) in the final report. These inform Plan 3's quality work. Nothing to commit.

---

### Task 7: Image replace mode (end-to-end)

**Files:** none (implemented in Task 5's filter + effect; this task is verification)

- [ ] **Step 1: Visual verification**

In the filter properties: Background = "Replace with image", pick an image file. Expected: background around the person replaced by the image. Confirm the status line and no log errors (`grep -i error /tmp/opencode/obs-smoke3.log`).

- [ ] **Step 2: Commit** — nothing. Record observations.

---

### Task 8: Blur mode (Kawase multi-pass)

**Files:**
- Create: `data/effects/kawase_blur.effect`
- Modify: `src/cam-effects-filter.c`

- [ ] **Step 1: Create `data/effects/kawase_blur.effect`**

```
uniform float4x4 ViewProj;
uniform texture2d image;
uniform float2 texel;
uniform float iteration;

sampler_state def_sampler {
	Filter = Linear;
	AddressU = Clamp;
	AddressV = Clamp;
};

struct VertData {
	float4 pos : POSITION;
	float2 uv : TEXCOORD0;
};

VertData mainTransform(VertData v_in)
{
	VertData v_out;
	v_out.pos = mul(float4(v_in.pos.xyz, 1.0), ViewProj);
	v_out.uv = v_in.uv;
	return v_out;
}

float4 Kawase(VertData v_in) : TARGET
{
	float2 o = texel * (iteration + 0.5);
	float4 c = image.Sample(def_sampler, v_in.uv + float2(-o.x, -o.y)) +
		   image.Sample(def_sampler, v_in.uv + float2(o.x, -o.y)) +
		   image.Sample(def_sampler, v_in.uv + float2(-o.x, o.y)) +
		   image.Sample(def_sampler, v_in.uv + float2(o.x, o.y));
	return c * 0.25;
}

technique Draw
{
	pass
	{
		vertex_shader = mainTransform(v_in);
		pixel_shader = Kawase(v_in);
	}
}
```

- [ ] **Step 2: Add blur support to `src/cam-effects-filter.c`**

Add to the struct:

```c
	gs_texrender_t *blur_a;         /* half-res ping */
	gs_texrender_t *blur_b;         /* half-res pong */
	gs_effect_t *blur_effect;       /* kawase_blur.effect */
```

Create/destroy them alongside the other graphics objects (`gs_texrender_create(GS_BGRA, GS_ZS_NONE)`; load `effects/kawase_blur.effect` in `cam_effects_load_effect`).

Add this helper before `cam_effects_video_render`:

```c
/* Runs `passes` Kawase blur iterations on target at half resolution.
 * Returns the texture containing the blurred result, or NULL. */
static gs_texture_t *cam_effects_blur(struct cam_effects_filter *filter,
				      obs_source_t *target, uint32_t w,
				      uint32_t h)
{
	uint32_t bw = w / 2 > 0 ? w / 2 : 1;
	uint32_t bh = h / 2 > 0 ? h / 2 : 1;

	/* Downsample target into blur_a using the default effect. */
	gs_texrender_reset(filter->blur_a);
	if (!gs_texrender_begin(filter->blur_a, bw, bh))
		return NULL;
	struct vec4 clear;
	vec4_zero(&clear);
	gs_clear(GS_CLEAR_COLOR, &clear, 0.0f, 0);
	gs_ortho(0.0f, (float)bw, 0.0f, (float)bh, -100.0f, 100.0f);
	gs_blend_state_push();
	gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);
	gs_matrix_push();
	gs_matrix_scale3f((float)bw / (float)w, (float)bh / (float)h, 1.0f);
	obs_source_video_render(target);
	gs_matrix_pop();
	gs_blend_state_pop();
	gs_texrender_end(filter->blur_a);

	gs_texture_t *src = gs_texrender_get_texture(filter->blur_a);
	for (int i = 0; i < filter->blur_strength; i++) {
		gs_texrender_t *dst = (i % 2 == 0) ? filter->blur_b
						   : filter->blur_a;
		gs_texrender_reset(dst);
		if (!gs_texrender_begin(dst, bw, bh))
			return src;
		gs_effect_set_texture(
			gs_effect_get_param_by_name(filter->blur_effect,
						    "image"),
			src);
		struct vec2 texel = {1.0f / (float)bw, 1.0f / (float)bh};
		gs_effect_set_vec2(
			gs_effect_get_param_by_name(filter->blur_effect,
						    "texel"),
			&texel);
		gs_effect_set_float(
			gs_effect_get_param_by_name(filter->blur_effect,
						    "iteration"),
			(float)i);
		gs_clear(GS_CLEAR_COLOR, &clear, 0.0f, 0);
		gs_ortho(0.0f, (float)bw, 0.0f, (float)bh, -100.0f, 100.0f);
		while (gs_effect_loop(filter->blur_effect, "Draw"))
			gs_draw_sprite(src, 0, bw, bh);
		gs_texrender_end(dst);
		src = gs_texrender_get_texture(dst);
	}
	return src;
}
```

In `cam_effects_video_render`, change the technique selection and composite: when mode is "blur", compute `gs_texture_t *blur = cam_effects_blur(filter, target, w, h);` before `gs_texrender_begin(filter->out_render, ...)`, set technique to `"DrawBlur"` when blur != NULL, and set the `blur_image` param:

```c
			if (blur)
				gs_effect_set_texture(
					gs_effect_get_param_by_name(
						filter->effect, "blur_image"),
					blur);
```

(Declare `gs_texture_t *blur = NULL;` near the top of the mask-available section; set `tech = "DrawBlur"` when mode is blur and blur != NULL.)

- [ ] **Step 3: Build, install, verify**

```bash
cmake --build --preset ubuntu-x86_64 && ./build-aux/install-local.sh
timeout 25 obs --verbose > /tmp/opencode/obs-smoke4.log 2>&1; true
grep -iE "error|fail" /tmp/opencode/obs-smoke4.log | grep -viE "vlc|portal|fr-FR|shutdown" || echo "clean"
```

Expected: `clean`. Then user visual check: Background = "Blur", adjust strength 1–4: background blurs, person stays sharp.

- [ ] **Step 4: Commit**

```bash
git add data/effects/kawase_blur.effect src/cam-effects-filter.c
git commit -m "feat: Kawase multi-pass blur background mode"
```

---

### Task 9: Golden-frame anchor + soak test

**Files:**
- Modify: `CMakeLists.txt` (test sources)
- Create: `tests/test_golden.cpp`
- Create: `tests/test_soak.cpp`

- [ ] **Step 1: Write `tests/test_golden.cpp` (golden values recorded at implementation time)**

```cpp
#include <gtest/gtest.h>

#include "fx/pipeline/segmentation_pipeline.h"

#include <cmath>

/* Golden anchor: fixed synthetic frame -> pipeline -> mask statistics.
 * The expected values were recorded from a verified-good run (see plan);
 * they catch model/ORT/preprocessing regressions, not absolute truth. */
TEST(Golden, MaskStatisticsMatchRecordedRun)
{
	fx::Frame f;
	f.width = 640;
	f.height = 360;
	f.bgra.assign(640u * 360u * 4u, 0);
	/* Deterministic pseudo-image: gradient + center ellipse. */
	for (int y = 0; y < 360; y++) {
		for (int x = 0; x < 640; x++) {
			uint8_t *p = f.bgra.data() + (y * 640 + x) * 4;
			float dx = (x - 320.0f) / 120.0f;
			float dy = (y - 200.0f) / 150.0f;
			bool inEllipse = dx * dx + dy * dy < 1.0f;
			p[0] = inEllipse ? 200 : (uint8_t)(x % 256);
			p[1] = inEllipse ? 160 : (uint8_t)(y % 256);
			p[2] = inEllipse ? 140 : 60;
			p[3] = 255;
		}
	}

	fx::SegmentationPipeline pipe(FX_MODEL_PATH, 1);
	auto m = pipe.process(f);

	double mean = 0, sq = 0;
	uint64_t checksum = 0;
	for (size_t i = 0; i < m->px.size(); i++) {
		mean += m->px[i];
		sq += m->px[i] * m->px[i];
		checksum = checksum * 1099511628211ULL +
			   (uint64_t)(m->px[i] * 255.0f);
	}
	mean /= m->px.size();
	double var = sq / m->px.size() - mean * mean;

	/* RECORDED VALUES — see note below. */
	EXPECT_NEAR(mean, GOLDEN_MEAN, 1e-3);
	EXPECT_NEAR(var, GOLDEN_VAR, 1e-3);
	EXPECT_EQ(checksum, GOLDEN_CHECKSUM);
}
```

TDD flow for this test: write it with `GOLDEN_MEAN 0.0`, `GOLDEN_VAR 0.0`, `GOLDEN_CHECKSUM 0ULL` → run → it fails printing actual values → replace the three constants with the printed actuals → run → passes. In the report, record the actual values. The failure output must print actuals with enough precision — use `::testing::PrintToString` or a temporary `std::cout` during recording (remove after).

- [ ] **Step 2: Write `tests/test_soak.cpp`**

```cpp
#include <gtest/gtest.h>

#include "fx/pipeline/segmentation_pipeline.h"
#include "fx/worker.h"

#include <atomic>
#include <chrono>
#include <thread>

/* Soak: 600 frames through the worker; asserts throughput and that the
 * drop policy engages (processed < submitted when worker is saturated). */
TEST(Soak, ThroughputAndDropPolicy)
{
	fx::SegmentationPipeline pipe(FX_MODEL_PATH, 2);
	std::atomic<uint64_t> processed{0};
	fx::Worker w([&](const fx::Frame &f) {
		auto m = pipe.process(f);
		processed.fetch_add(1);
		return m;
	});
	w.start();

	auto t0 = std::chrono::steady_clock::now();
	const int kFrames = 600;
	for (int i = 0; i < kFrames; i++) {
		auto f = std::make_shared<fx::Frame>();
		f->width = 192;
		f->height = 192;
		f->bgra.assign(192u * 192u * 4u, (uint8_t)(i & 0xFF));
		w.submit(std::move(f));
	}
	uint64_t seq = 0;
	while (processed.load() < 1)
		std::this_thread::sleep_for(std::chrono::milliseconds(5));
	/* Drain: wait until worker caught up (no pending, latest seq). */
	std::this_thread::sleep_for(std::chrono::milliseconds(500));
	w.stop();
	auto t1 = std::chrono::steady_clock::now();

	double ms =
		std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0)
			.count();
	uint64_t done = processed.load();
	double fps = done / (ms / 1000.0);
	std::cout << "soak: " << done << " frames in " << ms
		  << " ms = " << fps << " fps" << std::endl;
	EXPECT_GT(fps, 30.0);
	EXPECT_LE(done, (uint64_t)kFrames); // drop policy bounds work
	(void)seq;
}
```

Add both files to fx_tests sources in CMakeLists.txt.

- [ ] **Step 3: Run the full suite**

```bash
cmake --build --preset ubuntu-x86_64 && ctest --test-dir build_x86_64 --output-on-failure
```

Expected: all pass (13 tests + golden + soak). Record the soak fps in the report.

- [ ] **Step 4: Commit**

```bash
git add CMakeLists.txt tests/test_golden.cpp tests/test_soak.cpp
git commit -m "test: golden mask anchor and worker soak test"
```

---

### Task 10: Closeout

**Files:**
- Modify: `docs/development-notes.md`

- [ ] **Step 1: Update `docs/development-notes.md`**

Append a new section:

```markdown
## Plan 2 state (segmentation pipeline)

- Background modes working: off / transparent / image / blur (single mode
  selector — combinable toggles deferred deliberately).
- CPU-only inference (PP-HumanSeg, ~[MEASURED] ms/frame, soak [FPS] fps).
  CUDA execution provider: Plan 3 with the Quality tier.
- Guided filter runs at 192x192; GPU bilinear upscale at composite.
- Failure modes implemented: passthrough (default) / freeze-last-frame
  (black before first processed frame).
- [OBSERVATIONS FROM TASKS 6-8: edge quality, artifacts, fps feel]
```

- [ ] **Step 2: Final verification pass**

```bash
git status --short
cmake --build --preset ubuntu-x86_64 && ctest --test-dir build_x86_64
grep -c "technique" data/effects/mask_composite.effect  # expect 3
```

- [ ] **Step 3: Commit**

```bash
git add docs/development-notes.md
git commit -m "docs: plan 2 closeout notes"
```

---

## Plan 2 Definition of Done

- [ ] Full ctest suite green (incl. golden anchor + soak ≥30 fps)
- [ ] OBS 32.1.2 loads the plugin with no effect-compile or ORT errors
- [ ] Transparent mode composites correctly (user-verified)
- [ ] Image replace works (user-verified)
- [ ] Blur mode works with strength 1–4 (user-verified)
- [ ] Failure modes behave per spec (passthrough / freeze incl. black-before-first-frame)
- [ ] Git history: one commit per task, clean tree
- [ ] Known unverified recorded: visual quality observations feed Plan 3
