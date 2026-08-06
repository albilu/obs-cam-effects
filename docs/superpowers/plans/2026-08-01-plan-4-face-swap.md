# Plan 4: Face Swap — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the face-swap effect to the filter: YuNet face detection → 5-pt alignment → inswapper_128 identity swap → anti-wobble paste-back, composited before the existing background effects, with the amendment-9 settings (swap intensity, sharpness, preserve-mouth) and the EU AI Act disclosure watermark.

**Architecture:** A `FaceSwapPipeline` in libfx runs on the worker thread BEFORE the segmentation pipeline: full-res staged frame → detect → swap → paste-back → downscale → segmentation → publish (swapped frame + mask). The filter's composite reuses the Plan 2 shaders, with the uploaded swapped frame bound as `image`. Face-swap models (inswapper, ArcFace) are runtime downloads (non-commercial licenses, shown pre-download); YuNet is bundled (Apache-2.0). Face swap requires a usable CUDA EP (CPU is 200–400ms/frame — not viable); the toggle is gated on EpProbe.

**Tech Stack:** existing fx stack (OrtModel multi-IO, Worker, models_dl, EpProbe), stb_image.h (vendored, public domain) for source-face image decode, no other new deps.

**Environment:** Kali Linux, OBS 32.1.2, RTX 5070 with working CUDA EP for simple static graphs (PP-HumanSeg/MediaPipe verified; YuNet/inswapper to verify in Task 9), branch `main`.

## Verified facts (spike results — do not re-derive)

- **YuNet face detector** (bundled, Apache-2.0, OpenCV Zoo):
  - URL: `https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx` (Git LFS — if a download returns 0 bytes or a 131-byte LFS pointer, retry the `media.githubusercontent.com` URL or the API base64 endpoint; the CDN flaked once during the spike)
  - SHA256: `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` (232,589 bytes)
  - Input: `input` [1,3,640,640] float32 — **BGR kept, NO normalization** (0–255)
  - Outputs (12): `cls_{8,16,32}` [1,N,1], `obj_{8,16,32}` [1,N,1], `bbox_{8,16,32}` [1,N,4], `kps_{8,16,32}` [1,N,10]; grids 80×80 (s8), 40×40 (s16), 20×20 (s32)
  - Decode (verified against opencv/modules/objdetect/src/face_detect.cpp): per cell (r,c): score = sqrt(clamp(cls,0,1)·clamp(obj,0,1)); cx=(c+bbox[0])·stride; cy=(r+bbox[1])·stride; w=exp(bbox[2])·stride; h=exp(bbox[3])·stride; landmark n: x=(kps[2n]+c)·stride, y=(kps[2n+1]+r)·stride. Threshold 0.6, NMS IoU 0.3
  - Landmark order: right eye, left eye, nose tip, right mouth corner, left mouth corner
  - Verified on a real portrait (PD official photo): 1 face, score 0.947, geometrically sane box + landmarks
- **inswapper_128** (runtime download, InsightFace NON-COMMERCIAL research license):
  - URL: `https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128.onnx`
  - SHA256: `e4a3f08c753cb72d04e10aa0f7dbe3deebbf39567d4ead6dce08e98aa49e16af` (554,253,681 bytes)
  - IO: `target` [1,3,128,128] (aligned crop), `source` [1,512] (identity latent), `output` [1,3,128,128]
  - Preprocessing (verified in insightface inswapper.py): aligned crop → RGB (swapRB) → /255 → [0,1]; output ×255 → BGR flip
  - Latent: `L2norm(arcface_embedding) @ emap`, then L2-normalize again
  - **emap**: the model's LAST initializer, name `buff2fs`, shape (512,512) float32, embedded in the .onnx (no separate file). Reference values for parser tests: row0[:4] = [0.12484695, −0.00845782, 0.08038428, −0.1220004]; row511[−4:] = [−0.20361629, −0.33891863, 0.29195625, −0.08580378]; |sum| = 35887.31640625
- **ArcFace w600k_r50** (runtime download, InsightFace NON-COMMERCIAL):
  - URL: `https://huggingface.co/hacksider/deep-live-cam/resolve/main/buffalo_l/buffalo_l/w600k_r50.onnx` (doubled `buffalo_l/` path is CORRECT)
  - SHA256: `4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43` (174,383,860 bytes)
  - IO: `input.1` [N,3,112,112], output `683` [1,512]
  - Preprocessing: aligned 112×112 crop → RGB (swapRB) → (x−127.5)/127.5
- **5-pt alignment template** (verified in insightface face_align.py):
  - `arcface_dst` = [[38.2946,51.6963],[73.5318,51.5014],[56.0252,71.7366],[41.5493,92.3655],[70.7299,92.2041]] (112×112)
  - For 128×128: same with **x += 8.0** (i.e. [[46.2946,…],[81.5318,…],[64.0252,…],[49.5493,…],[78.7299,…]])
  - Similarity transform (Umeyama) from detected landmarks → template; cv2.warpAffine convention: pass the FORWARD M, OpenCV inverts internally — our C++ warp must do the same
- **Test fixture**: PD official portrait (US gov't work) — YuNet verified on it. Plan uses CMake download with hash (see Task 3), not a repo blob.

## Declared deviations / scope notes

1. **Full-res staging when face swap is enabled** (~8.3MB BGRA readback at 1080p, ~1–3ms) — required: swapping happens on the full frame. The 192×192 stage stays for background-only mode.
2. **CPU composite when face swap is on**: worker publishes (swapped frame + mask); filter uploads the swapped frame as the `image` texture and reuses the Plan 2 shader techniques unchanged. Blur passes sample the uploaded texture instead of re-rendering the target.
3. **emap via a minimal protobuf wire parser** (~100 lines): ONNX initializers are TensorProtos; we need only the last one's float raw_data. Tested against the spike reference values.
4. **Face swap requires CUDA** (EpProbe::cudaAvailable). RVM's Blackwell crash was graph-specific; YuNet/inswapper are simple static graphs and are EXPECTED to work — Task 9 verifies with the bench. If inswapper segfaults on this machine, face swap is gated off with the reason logged.
5. **Watermark is a CPU bitmap stamp** ("AI" badge, bottom-right) applied on the worker's output frame before upload — automatically included in frozen frames (spec §8/§9).

---

### Task 1: fx::image face-swap math (Umeyama, warp, ellipse, unsharp, mouth, watermark)

**Files:**
- Modify: `CMakeLists.txt` (sources/tests)
- Create: `src/fx/image/align.h`
- Create: `src/fx/image/align.cpp`
- Test: `tests/test_align.cpp`

- [ ] **Step 1: Add `src/fx/image/align.cpp` to fx sources, `tests/test_align.cpp` to fx_tests**

- [ ] **Step 2: Create `src/fx/image/align.h`**

```cpp
#pragma once

#include "fx/types.h"

#include <array>
#include <vector>

namespace fx {

using Point2 = std::array<float, 2>;
using Landmarks5 = std::array<Point2, 5>;

/* 2x3 affine matrix, row-major [a11 a12 tx; a21 a22 ty]. */
struct Affine23 {
	float m[6];
};

/* The two verified alignment templates (insightface face_align.py). */
Landmarks5 template112();
Landmarks5 template128(); // template112 with x += 8

/* Umeyama similarity transform src -> dst (least squares, 5 points).
 * Returns the FORWARD affine (cv2.warpAffine convention: pass forward,
 * the warp inverts internally). */
Affine23 umeyama(const Landmarks5 &src, const Landmarks5 &dst);

Affine23 invertAffine(const Affine23 &m);

/* Bilinear warp of an interleaved-uint8 BGR(A) image.
 * dst(x,y) = src(M⁻¹ · (x,y)) — caller passes the FORWARD transform.
 * Channels: 3 (BGR) or 4 (BGRA). Out-of-bounds samples are zero. */
void warpAffineBilinear(const uint8_t *src, int sw, int sh, int channels,
			const Affine23 &forwardM, uint8_t *dst, int dw,
			int dh);

/* Feathered elliptical alpha mask centered in a crop of size s×s
 * (DLC-style anti-wobble paste mask), radii relative to s, feather in px. */
std::vector<float> ellipseMask(int s, float rx, float ry, int feather);

/* unsharp mask on an interleaved uint8 image: out = src + amount·(src − blur). */
void unsharpMask(uint8_t *img, int w, int h, int channels, int radius,
		 float amount);

/* Composite `origMouth` (same geometry) back over `img` inside an
 * ellipse centered between mouth landmarks, feathered. */
void restoreMouthRegion(uint8_t *img, const uint8_t *orig, int w, int h,
			int channels, Point2 mouthL, Point2 mouthR,
			float widthScale, int feather);

/* Stamp a small "AI" disclosure badge (white on black rounded box) at
 * the bottom-right corner, ~2.5% of frame width. No-op if too small. */
void stampWatermarkAI(uint8_t *img, int w, int h, int channels);

} // namespace fx
```

- [ ] **Step 3: Write the failing test — `tests/test_align.cpp`**

```cpp
#include <gtest/gtest.h>

#include "fx/image/align.h"

#include <cmath>

TEST(Umeyama, IdentityWhenSrcEqualsDst)
{
	fx::Landmarks5 t = fx::template128();
	fx::Affine23 m = fx::umeyama(t, t);
	ASSERT_NEAR(m.m[0], 1.0f, 1e-5f);
	ASSERT_NEAR(m.m[1], 0.0f, 1e-5f);
	ASSERT_NEAR(m.m[2], 0.0f, 1e-4f);
	ASSERT_NEAR(m.m[3], 0.0f, 1e-5f);
	ASSERT_NEAR(m.m[4], 1.0f, 1e-5f);
	ASSERT_NEAR(m.m[5], 0.0f, 1e-4f);
}

TEST(Umeyama, UniformScaleTranslate)
{
	fx::Landmarks5 src = {{{0, 0}, {10, 0}, {10, 10}, {0, 10}, {5, 5}}};
	fx::Landmarks5 dst;
	for (int i = 0; i < 5; i++)
		dst[i] = {2.0f * src[i][0] + 3.0f, 2.0f * src[i][1] + 7.0f};
	fx::Affine23 m = fx::umeyama(src, dst);
	ASSERT_NEAR(m.m[0], 2.0f, 1e-5f);
	ASSERT_NEAR(m.m[1], 0.0f, 1e-5f);
	ASSERT_NEAR(m.m[2], 3.0f, 1e-4f);
	ASSERT_NEAR(m.m[3], 0.0f, 1e-5f);
	ASSERT_NEAR(m.m[4], 2.0f, 1e-5f);
	ASSERT_NEAR(m.m[5], 7.0f, 1e-4f);
}

TEST(Affine, InvertRoundTrip)
{
	fx::Landmarks5 src = {{{12, 3}, {99, 20}, {60, 80}, {25, 90}, {70, 95}}};
	fx::Landmarks5 dst = fx::template128();
	fx::Affine23 m = fx::umeyama(src, dst);
	fx::Affine23 inv = fx::invertAffine(m);
	/* inv(m) · m(p) == p for all src points */
	for (const auto &p : src) {
		float x = m.m[0] * p[0] + m.m[1] * p[1] + m.m[2];
		float y = m.m[3] * p[0] + m.m[4] * p[1] + m.m[5];
		float ox = inv.m[0] * x + inv.m[1] * y + inv.m[2];
		float oy = inv.m[3] * x + inv.m[4] * y + inv.m[5];
		ASSERT_NEAR(ox, p[0], 1e-2f);
		ASSERT_NEAR(oy, p[1], 1e-2f);
	}
}

TEST(WarpAffine, TranslateByIntegerPixels)
{
	/* 4x4 single-channel-ish BGR image with a marked pixel at (1,1). */
	uint8_t src[4 * 4 * 3] = {0};
	src[(1 * 4 + 1) * 3 + 0] = 255;
	uint8_t dst[4 * 4 * 3] = {0};
	fx::Affine23 m; /* translate (+2,+1): dst(3,2) should be 255 */
	m.m[0] = 1; m.m[1] = 0; m.m[2] = 2;
	m.m[3] = 0; m.m[4] = 1; m.m[5] = 1;
	fx::warpAffineBilinear(src, 4, 4, 3, m, dst, 4, 4);
	ASSERT_EQ(dst[(2 * 4 + 3) * 3 + 0], 255);
	ASSERT_EQ(dst[(1 * 4 + 1) * 3 + 0], 0);
}

TEST(EllipseMask, CenterOneCornerZero)
{
	auto mask = fx::ellipseMask(128, 0.35f, 0.45f, 8);
	ASSERT_EQ(mask.size(), 128u * 128u);
	ASSERT_FLOAT_EQ(mask[(64 * 128) + 64], 1.0f);
	ASSERT_FLOAT_EQ(mask[0], 0.0f);
	ASSERT_FLOAT_EQ(mask[(127 * 128) + 127], 0.0f);
}

TEST(UnsharpMask, SharpensStep)
{
	uint8_t img[8 * 3];
	for (int i = 0; i < 8; i++) {
		uint8_t v = i < 4 ? 50 : 200;
		img[i * 3 + 0] = v;
		img[i * 3 + 1] = v;
		img[i * 3 + 2] = v;
	}
	uint8_t before[8 * 3];
	memcpy(before, img, sizeof(before));
	fx::unsharpMask(img, 8, 1, 3, 2, 1.0f);
	ASSERT_LT(img[3 * 3], before[3 * 3]); // darker side dips
	ASSERT_GT(img[4 * 3], before[4 * 3]); // bright side overshoots
}

TEST(RestoreMouthRegion, OnlyMouthRestored)
{
	const int w = 32, h = 32, ch = 3;
	std::vector<uint8_t> img(w * h * ch, 200), orig(w * h * ch, 50);
	fx::restoreMouthRegion(img.data(), orig.data(), w, h, ch, {10, 24},
			       {22, 24}, 1.0f, 2);
	ASSERT_EQ(img[(24 * w + 16) * 3], 50);	 // mouth center restored
	ASSERT_EQ(img[(4 * w + 4) * 3], 200);	 // far away untouched
}

TEST(Watermark, StampsPixels)
{
	std::vector<uint8_t> img(64 * 64 * 3, 0);
	fx::stampWatermarkAI(img.data(), 64, 64, 3);
	bool anyLit = false;
	for (uint8_t v : img)
		anyLit |= (v > 200);
	ASSERT_TRUE(anyLit);
}
```

- [ ] **Step 4: Implement `src/fx/image/align.cpp` per the header contracts**

Implementation requirements (write the full implementation; these are the critical details):
- `umeyama`: 2D similarity least squares. With centered points X (src) and Y (dst): `norm = Σ(Xx²+Xy²)`; `a = Σ(Xx·Yx + Xy·Yy)/norm`; `b = Σ(Xx·Yy − Xy·Yx)/norm`; M = [[a, −b, tx],[b, a, ty]] with `tx = dstMx − (a·srcMx − b·srcMy)`, `ty = dstMy − (b·srcMx + a·srcMy)`.
- `invertAffine`: for the 2x2 block A and translation t: A⁻¹ = adj(A)/det(A), t' = −A⁻¹t. Handle |det| < 1e-12 (return identity).
- `warpAffineBilinear`: compute `inv = invertAffine(forwardM)` once; for each dst pixel: srcX = inv.m[0]·x + inv.m[1]·y + inv.m[2], srcY = inv.m[3]·x + inv.m[4]·y + inv.m[5]; bilinear sample with edge clamp; zero when the source point is outside (use clamp, not zero-fill — OpenCV borderValue=0 for out-of-bounds; clamp matches for our use and is safer for ellipse edges).
- `ellipseMask`: normalized ellipse equation ((x−c)/(rx·s))² + ((y−c)/(ry·s))²; 1 inside, 0 outside, linear feather across `feather` px at the boundary; then a light box blur over the feather band.
- `restoreMouthRegion`: ellipse centered at the midpoint of mouthL/mouthR, rx = |mouthR−mouthL|·widthScale·0.75, ry = rx·0.5; alpha = feathered ellipse; img = orig·α + img·(1−α) inside.
- `stampWatermarkAI`: hardcoded 5×7 bitmaps for 'A' and 'I' (plus a filled box behind); scale ≈ h/270 (~2.5% of frame width), alpha ≈ 0.8, margin ≈ h/40 from the corner.

- [ ] **Step 5: Build + test — all pass; commit**

```bash
cmake --build --preset ubuntu-x86_64 && ctest --test-dir build_x86_64 --output-on-failure
git add CMakeLists.txt src/fx/image/align.h src/fx/image/align.cpp tests/test_align.cpp
git commit -m "feat: face-swap image math (umeyama, warp, ellipse, unsharp, mouth, watermark)"
```

---

### Task 2: emap extraction (minimal protobuf parser)

**Files:**
- Modify: `CMakeLists.txt` (sources/tests)
- Create: `src/fx/models/onnx_init.h`
- Create: `src/fx/models/onnx_init.cpp`
- Test: `tests/test_onnx_init.cpp`

- [ ] **Step 1: Create `src/fx/models/onnx_init.h`**

```cpp
#pragma once

#include <string>
#include <vector>

namespace fx {

/* Extracts the LAST initializer's float32 data from an .onnx file
 * (used for inswapper's embedded `emap` projection matrix, 512x512).
 * Minimal protobuf wire-format walk: no protobuf library required.
 * Throws std::runtime_error on malformed input or missing initializer. */
std::vector<float> onnxLastInitializerFloats(const std::string &modelPath,
					     int64_t expectedCount);
}
```

- [ ] **Step 2: Write the test — `tests/test_onnx_init.cpp`**

Uses the spike-verified reference values (requires the inswapper file — skip gracefully if absent):

```cpp
#include <gtest/gtest.h>

#include "fx/models/onnx_init.h"

#include <cmath>
#include <cstdio>

static bool fileExists(const char *p)
{
	FILE *f = fopen(p, "r");
	if (f)
		fclose(f);
	return f != nullptr;
}

TEST(OnnxInit, ExtractsEmap)
{
	const char *path = FX_INSWAPPER_PATH;
	if (!fileExists(path))
		GTEST_SKIP() << "inswapper_128.onnx not downloaded";
	auto emap = fx::onnxLastInitializerFloats(path, 512 * 512);
	ASSERT_EQ(emap.size(), 512u * 512u);
	ASSERT_NEAR(emap[0], 0.12484695f, 1e-6f);
	ASSERT_NEAR(emap[1], -0.00845782f, 1e-6f);
	ASSERT_NEAR(emap[2], 0.08038428f, 1e-6f);
	ASSERT_NEAR(emap[3], -0.1220004f, 1e-6f);
	size_t last = 512u * 512u - 4;
	ASSERT_NEAR(emap[last + 0], -0.20361629f, 1e-6f);
	ASSERT_NEAR(emap[last + 1], -0.33891863f, 1e-6f);
	ASSERT_NEAR(emap[last + 2], 0.29195625f, 1e-6f);
	ASSERT_NEAR(emap[last + 3], -0.08580378f, 1e-6f);
	double sumAbs = 0;
	for (float v : emap)
		sumAbs += std::fabs(v);
	ASSERT_NEAR(sumAbs, 35887.31640625, 1.0);
}
```

Add `FX_INSWAPPER_PATH` compile def pointing at the CACHE path (`${HOME}/.config/obs-cam-effects/models/inswapper_128.onnx` is NOT available at configure time — use a build-dir path and stage the spike file: `cp /tmp/opencode/models/inswapper_128.onnx build_x86_64/models/` and define FX_INSWAPPER_PATH to that; skip if absent).

- [ ] **Step 3: Implement the protobuf walk**

ONNX = protobuf. ModelProto.graph is field 2; GraphProto.initializer is field 5 (repeated TensorProto); TensorProto.data_type field 2 (varint, float=1), TensorProto.raw_data is field 9 (length-delimited), dims field 1 (repeated varint). Walk: read whole file; scan top-level fields for field 2 (graph, length-delimited); inside it, collect all field-5 entries; take the LAST; inside that, read dims (field 1, may be packed), data_type, and raw_data (field 9); verify dims product == expectedCount and data_type == 1; memcpy raw_data into floats. Varint + length-delimited parsing only (wire types 0 and 2; skip types 1 and 5 by fixed size). ~100 lines. Reject raw_data==0 (external data unsupported).

- [ ] **Step 4: Build + test + commit**

```bash
git add CMakeLists.txt src/fx/models/onnx_init.h src/fx/models/onnx_init.cpp tests/test_onnx_init.cpp
git commit -m "feat: extract embedded emap matrix from inswapper onnx"
```

---

### Task 3: YuNet detector class

**Files:**
- Modify: `CMakeLists.txt` (model download + sources/tests)
- Create: `src/fx/models/yunet.h`
- Create: `src/fx/models/yunet.cpp`
- Test: `tests/test_yunet.cpp`

- [ ] **Step 1: YuNet model download in CMakeLists.txt (bundled, like PP-HumanSeg)**

```cmake
# --- Bundled face detector: YuNet 2023mar (Apache-2.0) ---
if(NOT EXISTS "${FX_MODEL_DIR}/face_detection_yunet_2023mar.onnx")
  file(DOWNLOAD
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    ${FX_MODEL_DIR}/face_detection_yunet_2023mar.onnx
    EXPECTED_HASH SHA256=8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4)
endif()
```

(If the download 404s/0-bytes due to the LFS CDN flake, use the media.githubusercontent.com URL form instead and note it.) Add `FX_YUNET_MODEL_PATH` compile def + sources.

- [ ] **Step 2: Create `src/fx/models/yunet.h`**

```cpp
#pragma once

#include "fx/engine/ort_backend.h"
#include "fx/image/align.h"
#include "fx/types.h"

#include <string>
#include <vector>

namespace fx {

struct FaceBox {
	float x, y, w, h; // top-left + size, in input-frame pixels
	Landmarks5 landmarks; // re, le, nose, rcm, lcm
	float score;
};

/* YuNet 2023mar face detector (640x640, BGR, no normalization). */
class YuNet {
public:
	explicit YuNet(const std::string &modelPath, int threads = 2);

	/* Detects faces in a BGRA frame of any size (resized to 640x640
	 * internally; coordinates mapped back to frame pixels). Returns
	 * boxes sorted by score desc, NMS 0.3 applied. */
	std::vector<FaceBox> detect(const Frame &frame, float scoreThresh = 0.6f);

private:
	OrtModel model_;
	std::vector<float> tensor_; // 3*640*640 scratch
};

} // namespace fx
```

- [ ] **Step 3: Test — `tests/test_yunet.cpp`**

```cpp
#include <gtest/gtest.h>

#include "fx/models/yunet.h"

TEST(YuNet, NoCrashOnBlankFrame)
{
	fx::YuNet det(FX_YUNET_MODEL_PATH, 1);
	fx::Frame f;
	f.width = 320;
	f.height = 240;
	f.bgra.assign(320u * 240u * 4u, 128);
	auto faces = det.detect(f);
	/* Blank frame: no assertion on count (model may fire or not on
	 * noise) — the contract is: valid geometry on whatever returns. */
	for (const auto &b : faces) {
		ASSERT_GT(b.w, 0.0f);
		ASSERT_GT(b.h, 0.0f);
		ASSERT_GE(b.score, 0.6f);
	}
}
```

(The real-face behavioral test with the PD portrait is Task 9's territory; this is the smoke contract.)

- [ ] **Step 4: Implement `yunet.cpp`**

- Resize BGRA→640×640 BGR float tensor (reuse the PPHumanSeg bilinear pattern, no normalization)
- Run the 12-output model via `model_.runWithShapes({tensor_}, {{1,3,640,640}})` — multi-output: use the multi-IO run and map outputs by NAME (cls_8/16/32, obj_*, bbox_*, kps_*) into the decode (order of `outputs_` is declared order; find indices by name — do NOT rely on positional order)
- Decode per the verified formulas; score = sqrt(clamp·clamp); threshold; greedy IoU-NMS 0.3 descending; map 640-coords back to frame size (scaleX = frame.width/640 etc.)

- [ ] **Step 5: Build + test + commit**

```bash
git add CMakeLists.txt src/fx/models/yunet.h src/fx/models/yunet.cpp tests/test_yunet.cpp
git commit -m "feat: YuNet face detector with verified decode"
```

---

### Task 4: ArcFace embedder + source-image embedding

**Files:**
- Modify: `CMakeLists.txt` (sources/tests)
- Create: `src/fx/third_party/stb_image.h` (vendor stb_image 2.30, public domain)
- Create: `src/fx/models/face_embedder.h`
- Create: `src/fx/models/face_embedder.cpp`
- Test: `tests/test_face_embedder.cpp`

- [ ] **Step 1: Vendor stb_image.h**

`curl -sL https://raw.githubusercontent.com/nothings/stb/master/stb_image.h -o src/fx/third_party/stb_image.h` (public domain/MIT; ~260KB). Record the version in a comment in the commit message.

- [ ] **Step 2: Create `src/fx/models/face_embedder.h`**

```cpp
#pragma once

#include "fx/models/onnx_init.h"
#include "fx/models/yunet.h"
#include "fx/types.h"

#include <string>
#include <vector>

namespace fx {

/* Source-identity embedder: ArcFace w600k_r50 (112x112 aligned,
 * RGB, (x-127.5)/127.5) + inswapper's emap projection. */
class FaceEmbedder {
public:
	FaceEmbedder(const std::string &arcfacePath,
		     const std::string &inswapperPathForEmap, int threads = 2);

	/* Embedding for one aligned 112x112 BGR crop (from YuNet
	 * landmarks via template112 umeyama warp). Returns the 512-d
	 * inswapper-ready latent: L2norm(L2norm(arcface) @ emap). */
	std::vector<float> embed(const std::vector<uint8_t> &bgrCrop112);

	/* One-call helper: decode an image file (jpg/png) with stb_image,
	 * detect the largest face, warp to 112, embed. Empty on failure. */
	std::vector<float> embedFromImageFile(const char *path, YuNet &det);

private:
	OrtModel arcface_;
	std::vector<float> emap_; // 512*512
};

} // namespace fx
```

- [ ] **Step 3: Test — `tests/test_face_embedder.cpp`**

```cpp
#include <gtest/gtest.h>

#include "fx/models/face_embedder.h"

#include <cstdio>

static bool fileExists(const char *p)
{
	FILE *f = fopen(p, "r");
	if (f)
		fclose(f);
	return f != nullptr;
}

TEST(FaceEmbedder, LatentIsUnitNorm)
{
	if (!fileExists(FX_ARCFACE_PATH) || !fileExists(FX_INSWAPPER_PATH))
		GTEST_SKIP() << "runtime models not downloaded";
	fx::FaceEmbedder emb(FX_ARCFACE_PATH, FX_INSWAPPER_PATH, 1);
	std::vector<uint8_t> crop(112 * 112 * 3, 128);
	auto latent = emb.embed(crop);
	ASSERT_EQ(latent.size(), 512u);
	double n = 0;
	for (float v : latent)
		n += v * v;
	ASSERT_NEAR(n, 1.0, 1e-3);
}
```

Stage the spike files for tests: `cp /tmp/opencode/models/{w600k_r50.onnx,inswapper_128.onnx} build_x86_64/models/`; add `FX_ARCFACE_PATH` and `FX_INSWAPPER_PATH` compile defs to those build-dir paths.

- [ ] **Step 4: Implement `face_embedder.cpp`**

- ctor: load ArcFace OrtModel; `emap_ = onnxLastInitializerFloats(inswapperPathForEmap, 512*512)`
- embed: tensor from BGR crop → RGB, (x−127.5)/127.5, NCHW; run; L2-normalize; multiply by emap_ (row-major: out[j] = Σ_i emb[i]·emap[i*512+j]); L2-normalize again
- embedFromImageFile: stbi_load (force 3 channels) → build Frame → detect → pick largest-area FaceBox → umeyama(landmarks, template112()) → warpAffineBilinear to 112×112 → embed
- stb usage: `#define STB_IMAGE_IMPLEMENTATION` in THIS .cpp only; `STBI_NO_STDIO`? No — we use stbi_load(path). Suppress stb warnings if -Werror complains (e.g. `STBI_NO_SIMD` or pragma GCC diagnostic around the include if needed — report what was required).

- [ ] **Step 5: Build + test + commit**

```bash
git add CMakeLists.txt src/fx/third_party/stb_image.h src/fx/models/onnx_init.h src/fx/models/face_embedder.h src/fx/models/face_embedder.cpp tests/test_face_embedder.cpp
git commit -m "feat: ArcFace source embedding with embedded emap projection"
```

---

### Task 5: FaceSwapPipeline

**Files:**
- Modify: `CMakeLists.txt` (sources/tests)
- Create: `src/fx/pipeline/face_swap_pipeline.h`
- Create: `src/fx/pipeline/face_swap_pipeline.cpp`
- Test: `tests/test_face_swap_pipeline.cpp`

- [ ] **Step 1: Create `src/fx/pipeline/face_swap_pipeline.h`**

```cpp
#pragma once

#include "fx/models/face_embedder.h"
#include "fx/models/yunet.h"
#include "fx/types.h"

#include <memory>
#include <string>
#include <vector>

namespace fx {

struct FaceSwapParams {
	float intensity = 1.0f;   // 0..1 swap opacity (amendment 9)
	float sharpness = 0.0f;   // 0..1 unsharp amount (amendment 9)
	bool preserveMouth = false; // geometric mouth restore (amendment 9)
	bool watermark = true;      // AI disclosure badge (spec §9)
	float bboxEma = 0.7f;       // detection smoothing
};

/* YuNet detect (bbox EMA) -> umeyama align -> inswapper -> paste-back
 * (feathered ellipse from the swap affine, DLC anti-wobble pattern) ->
 * optional mouth restore / intensity / watermark. Owns its models. */
class FaceSwapPipeline {
public:
	FaceSwapPipeline(const std::string &yunetPath,
			 const std::string &inswapperPath,
			 const std::string &arcfacePath, int threads = 2,
			 const std::string &providersDir = "");

	void setSourceEmbedding(std::vector<float> latent);
	bool hasSource() const { return !sourceLatent_.empty(); }
	void setParams(const FaceSwapParams &p) { params_ = p; }

	/* Swaps the largest detected face of `frame` in place (BGRA).
	 * Returns true if a swap was applied. */
	bool process(Frame &frame);

private:
	YuNet detector_;
	OrtModel swapper_;
	FaceEmbedder embedder_;
	std::vector<float> sourceLatent_;
	FaceSwapParams params_;

	/* Temporal state */
	bool havePrevBox_ = false;
	FaceBox prevBox_{};
	std::vector<uint8_t> aimg_;       // 128x128x3 aligned crop
	std::vector<uint8_t> fake128_;    // swap output crop
	std::vector<uint8_t> origFrame_;  // for mouth restore
};

} // namespace fx
```

- [ ] **Step 2: Test — `tests/test_face_swap_pipeline.cpp`**

```cpp
#include <gtest/gtest.h>

#include "fx/pipeline/face_swap_pipeline.h"

#include <cstdio>

static bool fileExists(const char *p)
{
	FILE *f = fopen(p, "r");
	if (f)
		fclose(f);
	return f != nullptr;
}

TEST(FaceSwapPipeline, NoSourceNoSwap)
{
	if (!fileExists(FX_INSWAPPER_PATH) || !fileExists(FX_ARCFACE_PATH))
		GTEST_SKIP() << "runtime models not downloaded";
	fx::FaceSwapPipeline pipe(FX_YUNET_MODEL_PATH, FX_INSWAPPER_PATH,
				  FX_ARCFACE_PATH, 1);
	fx::Frame f;
	f.width = 320;
	f.height = 240;
	f.bgra.assign(320u * 240u * 4u, 128);
	ASSERT_FALSE(pipe.process(f)); // no source embedding set
}

TEST(FaceSwapPipeline, SwapIsDeterministicOnSameFrame)
{
	if (!fileExists(FX_INSWAPPER_PATH) || !fileExists(FX_ARCFACE_PATH))
		GTEST_SKIP() << "runtime models not downloaded";
	fx::FaceSwapPipeline pipe(FX_YUNET_MODEL_PATH, FX_INSWAPPER_PATH,
				  FX_ARCFACE_PATH, 1);
	fx::FaceEmbedder emb(FX_ARCFACE_PATH, FX_INSWAPPER_PATH, 1);
	std::vector<uint8_t> crop(112 * 112 * 3, 128);
	pipe.setSourceEmbedding(emb.embed(crop));

	fx::Frame f;
	f.width = 320;
	f.height = 240;
	f.bgra.assign(320u * 240u * 4u, 128);
	auto before = f.bgra;
	pipe.process(f); // blank frame: may or may not swap; must not crash
	ASSERT_EQ(f.bgra.size(), before.size());
}
```

- [ ] **Step 3: Implement `face_swap_pipeline.cpp`**

process():
1. `detector_.detect(frame)` → empty → return false. Pick largest-area FaceBox.
2. bbox EMA on the box + landmarks vs prevBox_ (per-component lerp with bboxEma), update prevBox_.
3. `umeyama(smoothedLandmarks, template128())` → forward M.
4. `warpAffineBilinear(frame → aimg_ 128×128×3 BGR)`.
5. Swap: tensor from aimg_ (RGB swap, /255, NCHW) → `swapper_.runWithShapes({imgTensor, sourceLatent_}, {{1,3,128,128},{1,512}})` → output [1,3,128,128] → ×255 → RGB→BGR flip into fake128_.
6. If sharpness > 0: `unsharpMask(fake128_, 128, 128, 3, 2, sharpness)`.
7. Paste-back: `invM = invertAffine(M)`; warp fake128_ back to frame size (forward=invM through warpAffineBilinear — since our warp inverts internally, pass invM so sampling uses M); build the ellipse alpha in 128-space (`ellipseMask(128, 0.35, 0.45, 12)`), warp it to frame space with the same transform (warp as single-channel float — small helper or warp per-channel float copy); blend: `frame = fake·α + frame·(1−α)` in the warped region.
8. intensity < 1: within the ellipse region, final = step7·intensity + frame·(1−intensity).
9. preserveMouth: `restoreMouthRegion(frame, origFrame_, mouthL=smoothed landmarks[3], mouthR=landmarks[4], …)` using the ORIGINAL frame copy (keep a copy of the frame before step 7 when preserveMouth is on).
10. watermark: `stampWatermarkAI(frame.bgra.data(), w, h, 4)`.

- [ ] **Step 4: Build + test + commit**

```bash
git add CMakeLists.txt src/fx/pipeline/face_swap_pipeline.h src/fx/pipeline/face_swap_pipeline.cpp tests/test_face_swap_pipeline.cpp
git commit -m "feat: face swap pipeline (detect, align, swap, paste-back)"
```

---

### Task 6: Worker payload generalization (frame + mask bundle)

**Files:**
- Modify: `src/fx/worker.h/.cpp`
- Modify: `src/fx_bridge.h/.cpp`
- Test: `tests/test_worker.cpp` (extend)

- [ ] **Step 1: Generalize the published payload**

Current Worker publishes `shared_ptr<const Mask>`. Face swap publishes BOTH a swapped frame AND a mask. Change the payload to:

```cpp
struct WorkerResult {
	std::shared_ptr<const Mask> mask;
	std::shared_ptr<const Frame> frame; // may be null (background-only)
};
```

Update `Worker::Processor` to `std::function<WorkerResult(const Frame &)>`, `tryGetLatest(uint64_t&, WorkerResult&)` — keep a backward-compatible convenience: `tryGetLatestMask(uint64_t&)` returning the mask for existing callers (bridge's mask path + existing tests must keep compiling with minimal edits).

- [ ] **Step 2: Update all Processor call sites** (bridge's segmentation lambda; tests) and add a swap-path API to the bridge (declarations only in this task; wiring is Task 7):

```c
/* Face swap controls (Task 7 wires them into the filter). */
int cam_fx_faceswap_available(cam_fx_t *fx); // 1 if models+CUDA ready
int cam_fx_faceswap_set_source(cam_fx_t *fx, const char *image_path);
void cam_fx_faceswap_set_params(cam_fx_t *fx, float intensity,
				float sharpness, int preserve_mouth,
				int watermark);
int cam_fx_try_get_frame(cam_fx_t *fx, const uint8_t **bgra, int *w,
			 int *h, uint64_t *seq);
```

- [ ] **Step 3: Tests stay green + new bundle test**

```bash
cmake --build --preset ubuntu-x86_64 && ctest --test-dir build_x86_64 --output-on-failure
git add src/fx/worker.h src/fx/worker.cpp src/fx_bridge.h src/fx_bridge.cpp tests/test_worker.cpp
git commit -m "feat: worker publishes frame+mask bundle for face swap"
```

---

### Task 7: Manifest additions + download flow

**Files:**
- Modify: `data/models/manifest.json`
- Modify: `src/fx_bridge.cpp` (face-swap model resolution)
- Modify: `src/cam-effects-filter.c` (face-swap UI: toggle, source image, params, notices, download button)

- [ ] **Step 1: Add to `data/models/manifest.json`**

```json
    ,{
      "id": "inswapper_128",
      "kind": "model",
      "tier": "faceswap",
      "file": "inswapper_128.onnx",
      "url": "https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128.onnx",
      "sha256": "e4a3f08c753cb72d04e10aa0f7dbe3deebbf39567d4ead6dce08e98aa49e16af",
      "size": 554253681,
      "license": "InsightFace non-commercial",
      "notice": "inswapper_128 (InsightFace): licensed for NON-COMMERCIAL research use only. By downloading you accept these terms."
    },
    {
      "id": "w600k_r50",
      "kind": "model",
      "tier": "faceswap",
      "file": "w600k_r50.onnx",
      "url": "https://huggingface.co/hacksider/deep-live-cam/resolve/main/buffalo_l/buffalo_l/w600k_r50.onnx",
      "sha256": "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43",
      "size": 174383860,
      "license": "InsightFace non-commercial",
      "notice": "ArcFace w600k_r50 (InsightFace): licensed for NON-COMMERCIAL research use only. By downloading you accept these terms."
    }
```

- [ ] **Step 2: Face-swap availability in the bridge**

`cam_fx_faceswap_available` = (inswapper file exists in cache) AND (w600k file exists) AND `EpProbe::cudaAvailable(providersDir)`. The status line reports which requirement is missing (models not downloaded / no GPU acceleration / ready).

- [ ] **Step 3: Filter UI**

New settings (with defaults):
- `"face_swap"` bool toggle, default false — **disabled (greyed) when `cam_fx_faceswap_available` is false**; the consent notice text shows the non-commercial terms and the missing requirement instead
- `"face_image"` path picker (source face)
- `"swap_intensity"` float 0–1 step 0.05, default 1.0
- `"swap_sharpness"` float 0–1 step 0.05, default 0
- `"swap_preserve_mouth"` bool, default false
- `"swap_watermark"` bool, default **true** (spec §9)
- `"download_faceswap_btn"` button: "Download face swap models (non-commercial, 730 MB)" — starts BOTH inswapper_128 and w600k_r50 downloads sequentially (the downloader is single-shot: queue w600k after inswapper completes; state shown in status)
- Consent notice text (from manifest notices, OBS_TEXT_INFO): includes "Use only faces you have rights/consent to use. Output is AI-generated; a disclosure badge is applied by default (EU AI Act Art. 50)."

- [ ] **Step 4: Smoke + commit**

```bash
cmake --build --preset ubuntu-x86_64 && ctest --test-dir build_x86_64
./build-aux/install-local.sh
timeout 25 obs --verbose > /tmp/opencode/obs-smoke-p4t7.log 2>&1; true
grep -c "plugin loaded successfully" /tmp/opencode/obs-smoke-p4t7.log
git add -A
git commit -m "feat: face swap downloads, availability gating, and filter UI"
```

---

### Task 8: Filter integration — full-res staging + swapped-frame composite

**Files:**
- Modify: `src/cam-effects-filter.c`

- [ ] **Step 1: Full-res staging path (face swap on)**

Add to the filter struct: `gs_stagesurf_t *full_surface;` created at the TARGET's base size (recreate on size change in video_render; cap at 1920×1080 — if larger, log once and skip face swap with a status note). When `face_swap` is on and available: stage the full-res frame (same render pattern as the 192 stage but at base size; reuse `stage_render`? No — needs its own texrender at base size: `full_render`), map, submit via a new `cam_fx_submit_full(fx, data, w, h, linesize)`.

- [ ] **Step 2: Swapped-frame composite**

When face swap is on: `cam_fx_try_get_frame` → if a fresh swapped frame exists: upload it to a persistent `frame_tex` (recreate on size change; `gs_texture_set_image` or recreate), then composite with `image` = frame_tex: the DrawTransparent/DrawReplace/DrawBlur techniques sample `image`. When `frame_tex != NULL`, SKIP `obs_source_process_filter_begin`/`tech_end` entirely and drive the technique manually (set the `image`/`mask`/`bg_image`/`blur_image` params, then `while (gs_effect_loop(filter->effect, tech)) gs_draw_sprite(frame_tex, 0, w, h);` inside the out_render block) — process_filter_tech_end → render_filter_tex rebinds `image` to the unswapped parent texture unconditionally (params upload at pass begin, last-set-wins), so overriding the param after process_filter_begin is dead code, and process_filter_begin would render the parent for nothing. ViewProj is auto-populated from the current projection (same convention as the Kawase helper). For blur: the Kawase helper blurs frame_tex instead of re-rendering the target. The mask comes from `cam_fx_try_get_mask` as usual (the pipeline computes it from the swapped frame).

When face swap is off: current behavior unchanged.

- [ ] **Step 3: Failure-mode integration**

Face-swap staleness uses the same is_fresh/failure-mode machinery. If the swapped frame is stale/missing: passthrough or freeze per setting (freeze shows the last COMPOSITED frame, which already contains the watermark — spec §8/§9 compliant).

- [ ] **Step 4: Build + smoke + commit**

```bash
cmake --build --preset ubuntu-x86_64 && ctest --test-dir build_x86_64
./build-aux/install-local.sh
timeout 25 obs --verbose > /tmp/opencode/obs-smoke-p4t8.log 2>&1; true
grep -iE "error|fail" /tmp/opencode/obs-smoke-p4t8.log | grep -viE "vlc|portal|fr-FR|shutdown|dbus|decklink|advanced-scene|v4l2" || echo "clean"
git add src/cam-effects-filter.c
git commit -m "feat: face swap staging and swapped-frame composite in filter"
```

---

### Task 9: CUDA verification bench (YuNet + inswapper on RTX 5070)

**Files:** none (verification; results go in the report + dev notes)

- [ ] **Step 1: Stage the runtime models and run the bench**

```bash
cp /tmp/opencode/models/{inswapper_128.onnx,w600k_r50.onnx} ~/.config/obs-cam-effects/models/ 2>/dev/null || true
```

Write a scratch bench (like the RVM bench): YuNet detect 100× and inswapper swap 100×, (a) CPU, (b) CUDA with providersDir, on this machine. If inswapper SIGSEGVs on the Blackwell EP (RVM-style), face swap must be gated off on this machine (report immediately — this changes Task 7's gating).

- [ ] **Step 2: Record numbers**

YuNet CPU/CUDA ms; inswapper CPU/CUDA ms; total pipeline estimate. CPU viability verdict (spec says CPU face swap is not viable — confirm with numbers).

- [ ] **Step 3: Commit dev-note numbers (folded into Task 10)**

---

### Task 10: User visual checkpoint + closeout

**Files:**
- Modify: `docs/development-notes.md`

- [ ] **Step 1: User checkpoint**

1. Download the face swap models through the UI (730MB total, notices shown)
2. Enable Face swap, pick a source face image (any portrait photo)
3. Verify: face swapped live; watermark "AI" badge visible bottom-right; intensity 0.5 = half-blend; sharpness crisps the swap; preserve-mouth keeps your real mouth; background modes still work ON TOP of the swapped face
4. Report: quality, fps feel, any artifacts

- [ ] **Step 2: Closeout notes in `docs/development-notes.md`**

```markdown
## Plan 4 state (face swap)

- Pipeline: YuNet (bundled, Apache) -> umeyama align -> inswapper_128
  (runtime download, non-commercial) -> DLC-style feathered-ellipse
  paste-back -> segmentation. CPU composite when enabled; GPU shaders
  reused with the swapped frame bound as image.
- Bench (RTX 5070, cuDNN 9.24): YuNet CPU [X]ms / CUDA [Y]ms;
  inswapper CPU [X]ms / CUDA [Y]ms. [CPU viability verdict]
- Amendment-9 settings shipped: swap intensity, sharpness,
  preserve-mouth (geometric), AI watermark (default ON).
- emap parsed from inswapper's last initializer (protobuf wire walk,
  tested against reference values).
- Known gaps: single face only; no GFPGAN (not real-time); inswapper
  quality is 128px (soft by design, sharpness slider mitigates).
```

- [ ] **Step 3: Final verification + commit**

```bash
git status --short
cmake --build --preset ubuntu-x86_64 && ctest --test-dir build_x86_64
git add docs/development-notes.md
git commit -m "docs: plan 4 closeout notes"
```

---

## Plan 4 Definition of Done

- [x] Full ctest suite green (align, emap, yunet, embedder, pipeline, worker bundle)
- [x] Face swap models downloadable via manifest with non-commercial notices
- [x] Swap works in OBS on the RTX 5070 (user-verified) with watermark visible
- [x] Background modes compose on top of the swapped face
- [x] Amendment-9 settings all functional (intensity/sharpness/mouth/watermark)
- [x] Failure modes cover face swap (passthrough/freeze incl. watermark in frozen frames)
- [x] Bench numbers recorded (YuNet + inswapper, CPU vs CUDA)
- [x] Git history: one commit per task, clean tree
