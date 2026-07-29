# Plan 3: Model Manager + Quality Tier + CUDA — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the full model-tier system to the segmentation pipeline: a tier picker (Auto/Lite/Standard/Quality), MediaPipe Lite and RVM Quality models, runtime downloads with license consent + SHA-256 verification (model manager), CUDA acceleration via ORT plugin execution providers, and the 4 advanced mask settings from spec amendment 9.

**Architecture:** fx gains a `SegmentationModel` interface (PPHumanSeg / MediaPipeLite / RVM behind it), an `OrtModel` generalized to multi-IO (RVM has 6 inputs / 6 outputs), a `models_dl` downloader (subprocess curl/sha256sum/tar — zero new C++ deps), and an `EpProbe` (CUDA provider registration at runtime). The bridge routes tier/download/status between filter UI and worker. Model hot-swap happens worker-side under mutex.

**Tech Stack:** existing fx/gtest stack + ORT 1.28 plugin EP API (`RegisterExecutionProviderLibrary`, `GetEpDevices`, `SessionOptionsAppendExecutionProvider_V2`), curl/sha256sum/tar system binaries, obs_data JSON for manifest parsing (bridge side).

**Environment:** Kali Linux, OBS 32.1.2, branch `main`, **NVIDIA RTX 5070 (driver 580.126.09, CUDA 13.0)** — CUDA path is verifiable on this machine. libcurl dev is NOT installed and NOT needed (subprocess approach).

## Verified facts (spike results — do not re-derive)

- **RVM MobileNetV3 fp32** (Quality tier, GPL-3.0-only, Peter Lin):
  - URL: `https://raw.githubusercontent.com/royshil/obs-backgroundremoval/main/models/rvm_mobilenetv3_fp32.onnx`
  - SHA256: `88d4531297118f595bf2fd60f6f566aec2e559393802d1f436c380f0cbbd2828` (14,975,696 bytes)
  - License URL: `…/rvm_mobilenetv3_fp32.onnx.license` (SPDX `GPL-3.0-only` confirmed)
  - Inputs (6): `src` [1,3,H,W] float32 (dynamic); `r1i`/`r2i`/`r3i`/`r4i` recurrent states; `downsample_ratio` [1] float
  - Outputs (6): `fgr` [1,3,H,W] (ignored), `pha` [1,1,H,W] (**alpha matte = mask**), `r1o..r4o` (next states)
  - At 192×192 base: state shapes r1o `[1,16,96,96]`, r2o `[1,20,48,48]`, r3o `[1,40,24,24]`, r4o `[1,64,12,12]` (verified by running the model)
  - Recurrence: zero-init states; feed r*o of frame N as r*i of frame N+1; `downsample_ratio = 1.0`
  - Preprocessing: BGR kept, `/255.0` to [0,1], NCHW (bgremoval ModelBCHW pattern)
  - Measured: **~9.5ms/frame on this CPU** — Quality tier is viable even without CUDA
- **MediaPipe Selfie Segmentation** (Lite tier, Apache-2.0, Google):
  - URL: `https://raw.githubusercontent.com/royshil/obs-backgroundremoval/main/models/selfie_segmentation.onnx`
  - SHA256: `1511ab5564cb0ae179cba9bcad6ad931003ccc71656f11fd270967023678c6a5` (447,596 bytes)
  - Input: `input_1:0` [1,256,256,3] float32 (**NHWC**); output: `activation_10` [1,256,256,1] float32 (**single-channel person mask, NHWC**)
  - Preprocessing: BGR kept, `/255.0` to [0,1], NHWC (bgremoval base Model pattern)
  - Wrapper must downscale the 256×256 output mask to 192×192 (pipeline contract)
- **ONNX Runtime CUDA plugin EP** (MIT):
  - URL: `https://github.com/microsoft/onnxruntime/releases/download/v1.28.0/onnxruntime-linux-x64-gpu_cuda13-1.28.0.tgz`
  - SHA256: `84d28f27589090b280d4312743efd3d450cd4ac7d1e1d75e7d9076d9637bf9de` (240,874,643 bytes)
  - Extract only: `onnxruntime-linux-x64-gpu_cuda13-1.28.0/lib/libonnxruntime_providers_cuda.so` and `…/lib/libonnxruntime_providers_shared.so`
  - Registration API (verified in local ORT 1.28 headers): `Ort::Env::RegisterExecutionProviderLibrary("CUDA", path)`; `Ort::Env::GetEpDevices()`; C API `OrtApi::SessionOptionsAppendExecutionProvider_V2`
  - Compatible with our bundled ORT (same 1.28.0 build version); cuDNN optional at runtime in 1.28
- **Plugin EP API present in our CPU build's headers** (verified: onnxruntime_cxx_api.h:1405/1408)

## Declared deviations / scope notes

1. **HTTP/SHA/extract via system binaries** (curl, sha256sum, tar) — libcurl dev is unavailable; subprocess keeps C++ deps at zero. Tests use `file://` URLs (offline, deterministic).
2. **Manifest parsed bridge-side with obs_data JSON** — fx stays OBS-free and needs no JSON parser; the bridge converts manifest entries to plain structs for fx.
3. **Status line refreshes on properties-dialog open** — OBS properties are static once open; live status would need a timer hack. The status text (state + fps) updates in `update()`; documented limitation.
4. **RVM path keeps the uniform EMA + guided-filter post-processing** — RVM has its own recurrence (main stabilizer); EMA at beta 0.6 is mild and the pipeline stays uniform. Tunable via the temporal smoothing slider.

---

### Task 1: OrtModel multi-IO generalization

**Files:**
- Modify: `src/fx/engine/ort_backend.h`
- Modify: `src/fx/engine/ort_backend.cpp`
- Modify: `src/fx/models/pp_humanseg.cpp` (adapt to new API)
- Test: `tests/test_ort_backend.cpp` (extend)

- [ ] **Step 1: Extend the test first — `tests/test_ort_backend.cpp`**

Append:

```cpp
TEST(OrtModel, ExposesMultiIoInterface)
{
	fx::OrtModel model(FX_MODEL_PATH, 1);
	ASSERT_EQ(model.inputCount(), 1u);
	ASSERT_EQ(model.outputCount(), 1u);
	ASSERT_EQ(model.input(0).shape.size(), 4u);
	ASSERT_EQ(model.input(0).shape[1], 3);	 // NCHW
	ASSERT_EQ(model.input(0).shape[2], 192);
	ASSERT_EQ(model.output(0).shape[3], 2);	 // 2 classes
}
```

(The old `ThrowsOnMissingFile` test and the `input()`/`output()` accessors must keep compiling — see Step 3's backward-compatible accessors.)

- [ ] **Step 2: Run — watch it FAIL to compile** (`inputCount` undefined)

- [ ] **Step 3: Generalize `src/fx/engine/ort_backend.h`**

```cpp
#pragma once

#include <onnxruntime_cxx_api.h>

#include <string>
#include <vector>

namespace fx {

/* Wrapper around an ONNX Runtime session with dynamic IO discovery.
 * Supports multi-input multi-output models (RVM: 6 in / 6 out);
 * single-IO models use the convenience accessors. */
class OrtModel {
public:
	struct TensorDesc {
		std::string name;
		std::vector<int64_t> shape;
	};

	explicit OrtModel(const std::string &modelPath, int intraOpThreads = 2);

	size_t inputCount() const { return inputs_.size(); }
	size_t outputCount() const { return outputs_.size(); }
	const TensorDesc &input(size_t i) const { return inputs_.at(i); }
	const TensorDesc &output(size_t i) const { return outputs_.at(i); }

	/* Backward-compatible single-IO accessors. */
	const TensorDesc &input() const { return inputs_.at(0); }
	const TensorDesc &output() const { return outputs_.at(0); }

	/* Single-IO run (existing behavior). */
	std::vector<float> run(const std::vector<float> &inputData);

	/* Multi-IO run: one flat float tensor per declared input.
	 * Returns one flat float tensor per declared output (in order). */
	std::vector<std::vector<float>>
	run(const std::vector<std::vector<float>> &inputData);

private:
	Ort::Env env_;
	Ort::Session session_;
	std::vector<TensorDesc> inputs_;
	std::vector<TensorDesc> outputs_;
};

} // namespace fx
```

- [ ] **Step 4: Implement `src/fx/engine/ort_backend.cpp` (full replacement)**

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

	if (session_.GetInputCount() == 0 || session_.GetOutputCount() == 0)
		throw std::runtime_error("fx: model with no inputs/outputs");

	Ort::AllocatorWithDefaultOptions alloc;
	for (size_t i = 0; i < session_.GetInputCount(); i++) {
		auto name = session_.GetInputNameAllocated(i, alloc);
		TensorDesc d;
		d.name = name.get();
		d.shape = session_.GetInputTypeInfo(i)
				  .GetTensorTypeAndShapeInfo()
				  .GetShape();
		inputs_.push_back(std::move(d));
	}
	for (size_t i = 0; i < session_.GetOutputCount(); i++) {
		auto name = session_.GetOutputNameAllocated(i, alloc);
		TensorDesc d;
		d.name = name.get();
		d.shape = session_.GetOutputTypeInfo(i)
				  .GetTensorTypeAndShapeInfo()
				  .GetShape();
		outputs_.push_back(std::move(d));
	}
}

static size_t elementCount(const std::vector<int64_t> &shape)
{
	size_t n = 1;
	for (int64_t d : shape)
		n *= (d > 0) ? (size_t)d : 1;
	return n;
}

std::vector<std::vector<float>>
OrtModel::run(const std::vector<std::vector<float>> &inputData)
{
	if (inputData.size() != inputs_.size())
		throw std::runtime_error("fx: input tensor count mismatch");

	auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,
					      OrtMemTypeDefault);
	std::vector<Ort::Value> inTensors;
	std::vector<const char *> inNames;
	inTensors.reserve(inputs_.size());
	for (size_t i = 0; i < inputs_.size(); i++) {
		inTensors.push_back(Ort::Value::CreateTensor<float>(
			mem, const_cast<float *>(inputData[i].data()),
			inputData[i].size(), inputs_[i].shape.data(),
			inputs_[i].shape.size()));
		inNames.push_back(inputs_[i].name.c_str());
	}
	std::vector<const char *> outNames;
	outNames.reserve(outputs_.size());
	for (const auto &o : outputs_)
		outNames.push_back(o.name.c_str());

	auto outs = session_.Run(Ort::RunOptions{nullptr}, inNames.data(),
				 inTensors.data(), inTensors.size(),
				 outNames.data(), outNames.size());

	std::vector<std::vector<float>> result;
	result.reserve(outs.size());
	for (auto &o : outs) {
		float *data = o.GetTensorMutableData<float>();
		size_t count = o.GetTensorTypeAndShapeInfo().GetElementCount();
		result.emplace_back(data, data + count);
	}
	return result;
}

std::vector<float> OrtModel::run(const std::vector<float> &inputData)
{
	auto out = run(std::vector<std::vector<float>>{inputData});
	return std::move(out.at(0));
}

} // namespace fx
```

NOTE: `pp_humanseg.cpp` uses `model_.run(tensor_)` (single-IO) — it keeps compiling unchanged; `elementCount` helper is currently unused beyond future tasks — if `-Werror` flags it, mark it `[[maybe_unused]]`.

- [ ] **Step 5: Build + test — 13/13 pass**

```bash
cmake --build --preset ubuntu-x86_64 && ctest --test-dir build_x86_64 --output-on-failure
```

- [ ] **Step 6: Commit**

```bash
git add src/fx/engine/ort_backend.h src/fx/engine/ort_backend.cpp tests/test_ort_backend.cpp
git commit -m "feat: generalize OrtModel to multi-input multi-output sessions"
```

---

### Task 2: models_dl downloader core (curl/sha256sum subprocess)

**Files:**
- Modify: `CMakeLists.txt` (fx sources + tests)
- Create: `src/fx/models_dl/downloader.h`
- Create: `src/fx/models_dl/downloader.cpp`
- Test: `tests/test_downloader.cpp`

- [ ] **Step 1: Add `src/fx/models_dl/downloader.cpp` to fx sources and `tests/test_downloader.cpp` to fx_tests sources**

- [ ] **Step 2: Create `src/fx/models_dl/downloader.h`**

```cpp
#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace fx::models_dl {

struct DownloadRequest {
	std::string url;           // http(s):// or file:// (tests)
	std::string destPath;      // final file path
	std::string sha256;        // expected hex digest (lowercase)
	uint64_t expectedSize = 0; // bytes; 0 = unknown (progress then indeterminate)
	/* Optional: treat destPath as a tgz and extract these member paths
	 * (relative to archive root) into extractDestDir, then delete the tgz. */
	std::vector<std::string> extractMembers;
	std::string extractDestDir;
};

enum class State { Idle, Downloading, Verifying, Extracting, Done, Error };

const char *stateName(State s);

/* Single-shot background downloader using system binaries
 * (curl, sha256sum, tar). Thread-safe; progress polled from the
 * growing .part file size (no stderr parsing). */
class Downloader {
public:
	Downloader() = default;
	~Downloader();

	Downloader(const Downloader &) = delete;
	Downloader &operator=(const Downloader &) = delete;

	/* Starts the download. Throws std::runtime_error if already busy. */
	void start(const DownloadRequest &req);
	void cancel();

	State state() const { return state_.load(); }
	double progress() const; // 0..1, or -1 if expectedSize unknown
	std::string error() const;

private:
	void run(DownloadRequest req);

	std::thread thread_;
	std::atomic<State> state_{State::Idle};
	std::atomic<bool> cancel_{false};
	std::atomic<uint64_t> expectedSize_{0};
	std::string destPath_;
	std::string error_;
	mutable std::mutex errorM_;
};

} // namespace fx::models_dl
```

- [ ] **Step 3: Write the failing test — `tests/test_downloader.cpp`**

```cpp
#include <gtest/gtest.h>

#include "fx/models_dl/downloader.h"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <thread>

using namespace std::chrono_literals;
using fx::models_dl::Downloader;
using fx::models_dl::State;

namespace {

std::string tmpPath(const char *name)
{
	return std::string("/tmp/opencode/fx-dl-test-") + name;
}

std::string writeFixture(const std::string &path, const std::string &content)
{
	std::ofstream f(path, std::ios::binary);
	f << content;
	return path;
}

std::string sha256Of(const std::string &path)
{
	std::string cmd = "sha256sum " + path;
	FILE *p = popen(cmd.c_str(), "r");
	char buf[65] = {0};
	if (p) {
		if (fgets(buf, sizeof(buf), p) == nullptr)
			buf[0] = 0;
		pclose(p);
	}
	return std::string(buf, strnlen(buf, 64));
}

bool waitDone(Downloader &d, int ms)
{
	for (int i = 0; i < ms / 10; i++) {
		auto s = d.state();
		if (s == State::Done || s == State::Error)
			return true;
		std::this_thread::sleep_for(10ms);
	}
	return false;
}

} // namespace

TEST(Downloader, DownloadsAndVerifiesFileUrl)
{
	std::string src = writeFixture(tmpPath("src.bin"), "hello fx downloader\n");
	std::string dst = tmpPath("dst.bin");
	std::remove(dst.c_str());

	Downloader d;
	fx::models_dl::DownloadRequest req;
	req.url = "file://" + src;
	req.destPath = dst;
	req.sha256 = sha256Of(src);
	req.expectedSize = 19;
	d.start(req);

	ASSERT_TRUE(waitDone(d, 10000));
	ASSERT_EQ(d.state(), State::Done) << d.error();
	std::ifstream f(dst);
	std::string content((std::istreambuf_iterator<char>(f)),
			    std::istreambuf_iterator<char>());
	ASSERT_EQ(content, "hello fx downloader\n");
}

TEST(Downloader, HashMismatchIsError)
{
	std::string src = writeFixture(tmpPath("src2.bin"), "bad hash test\n");
	std::string dst = tmpPath("dst2.bin");
	std::remove(dst.c_str());

	Downloader d;
	fx::models_dl::DownloadRequest req;
	req.url = "file://" + src;
	req.destPath = dst;
	req.sha256 =
		"0000000000000000000000000000000000000000000000000000000000000000";
	d.start(req);

	ASSERT_TRUE(waitDone(d, 10000));
	ASSERT_EQ(d.state(), State::Error);
	ASSERT_FALSE(d.error().empty());
}

TEST(Downloader, ExtractsTgzMembers)
{
	/* Build a tiny tgz fixture: root/pkg/file.txt = "payload". */
	std::string root = tmpPath("tgzsrc");
	std::string mk = "mkdir -p " + root + "/pkg && printf payload > " +
			 root + "/pkg/file.txt && tar czf " + tmpPath("f.tgz") +
			 " -C " + root + " .";
	ASSERT_EQ(std::system(mk.c_str()), 0);
	std::string dst = tmpPath("f2.tgz");
	std::remove(dst.c_str());
	std::string outDir = tmpPath("tgzout");
	std::string mkOut = "mkdir -p " + outDir;
	ASSERT_EQ(std::system(mkOut.c_str()), 0);

	Downloader d;
	fx::models_dl::DownloadRequest req;
	req.url = "file://" + tmpPath("f.tgz");
	req.destPath = dst;
	req.sha256 = sha256Of(tmpPath("f.tgz"));
	req.extractMembers = {"./pkg/file.txt"};
	req.extractDestDir = outDir;
	d.start(req);

	ASSERT_TRUE(waitDone(d, 10000));
	ASSERT_EQ(d.state(), State::Done) << d.error();
	std::ifstream f(outDir + "/pkg/file.txt");
	std::string content((std::istreambuf_iterator<char>(f)),
			    std::istreambuf_iterator<char>());
	ASSERT_EQ(content, "payload");
}

TEST(Downloader, StateNamesAreStable)
{
	ASSERT_STREQ(fx::models_dl::stateName(State::Done), "done");
	ASSERT_STREQ(fx::models_dl::stateName(State::Error), "error");
	ASSERT_STREQ(fx::models_dl::stateName(State::Downloading),
		     "downloading");
}
```

- [ ] **Step 4: Build, watch FAIL, then implement `src/fx/models_dl/downloader.cpp`**

```cpp
#include "fx/models_dl/downloader.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

namespace fx::models_dl {

const char *stateName(State s)
{
	switch (s) {
	case State::Idle:
		return "idle";
	case State::Downloading:
		return "downloading";
	case State::Verifying:
		return "verifying";
	case State::Extracting:
		return "extracting";
	case State::Done:
		return "done";
	case State::Error:
		return "error";
	}
	return "unknown";
}

Downloader::~Downloader()
{
	cancel();
	if (thread_.joinable())
		thread_.join();
}

void Downloader::start(const DownloadRequest &req)
{
	State expected = State::Idle;
	if (!state_.compare_exchange_strong(expected, State::Downloading) &&
	    expected != State::Done && expected != State::Error) {
		throw std::runtime_error("fx-dl: downloader busy");
	}
	cancel_.store(false);
	error_.clear();
	expectedSize_.store(req.expectedSize);
	destPath_ = req.destPath;
	thread_ = std::thread([this, req] { run(req); });
}

void Downloader::cancel()
{
	cancel_.store(true);
}

std::string Downloader::error() const
{
	std::lock_guard<std::mutex> lk(errorM_);
	return error_;
}

double Downloader::progress() const
{
	uint64_t total = expectedSize_.load();
	if (total == 0)
		return -1.0;
	struct stat st;
	std::string part = destPath_ + ".part";
	if (stat(part.c_str(), &st) != 0)
		return 0.0;
	double p = (double)st.st_size / (double)total;
	return p > 1.0 ? 1.0 : p;
}

namespace {

int runCmd(const std::string &cmd)
{
	int rc = std::system(cmd.c_str());
	if (rc == -1)
		return -1;
	if (WIFEXITED(rc))
		return WEXITSTATUS(rc);
	return -1;
}

std::string shellQuote(const std::string &s)
{
	std::string q = "'";
	for (char c : s)
		q += (c == '\'') ? "'\\''" : std::string(1, c);
	return q + "'";
}

} // namespace

void Downloader::run(DownloadRequest req)
{
	auto fail = [&](const std::string &msg) {
		{
			std::lock_guard<std::mutex> lk(errorM_);
			error_ = msg;
		}
		state_.store(State::Error);
	};

	std::string part = req.destPath + ".part";
	std::string cmd = "curl -fSL -C - -o " + shellQuote(part) + " " +
			  shellQuote(req.url) + " 2>/dev/null";
	int rc = runCmd(cmd);
	if (cancel_.load()) {
		state_.store(State::Idle);
		return;
	}
	if (rc != 0) {
		/* 33 = resume not supported: retry from scratch. */
		if (rc == 33) {
			std::remove(part.c_str());
			rc = runCmd(cmd);
		}
		if (rc != 0) {
			fail("curl failed with exit code " + std::to_string(rc));
			return;
		}
	}

	state_.store(State::Verifying);
	std::string verify = "sha256sum " + shellQuote(part);
	FILE *p = popen(verify.c_str(), "r");
	char buf[65] = {0};
	if (!p || fgets(buf, sizeof(buf), p) == nullptr) {
		if (p)
			pclose(p);
		fail("sha256sum failed");
		return;
	}
	pclose(p);
	std::string got(buf, strnlen(buf, 64));
	if (got != req.sha256) {
		std::remove(part.c_str());
		fail("hash mismatch: got " + got);
		return;
	}

	if (!req.extractMembers.empty()) {
		state_.store(State::Extracting);
		std::string mk = "mkdir -p " + shellQuote(req.extractDestDir);
		runCmd(mk);
		std::string tar = "tar xzf " + shellQuote(part) + " -C " +
				  shellQuote(req.extractDestDir);
		for (const auto &m : req.extractMembers)
			tar += " " + shellQuote(m);
		rc = runCmd(tar);
		if (rc != 0) {
			fail("tar extract failed with exit code " +
			     std::to_string(rc));
			return;
		}
		std::remove(part.c_str());
	} else {
		std::string tmpFinal = req.destPath + ".verify";
		std::rename(part.c_str(), tmpFinal.c_str());
		if (std::rename(tmpFinal.c_str(), req.destPath.c_str()) != 0) {
			fail("atomic rename failed");
			return;
		}
	}
	state_.store(State::Done);
}

} // namespace fx::models_dl
```

- [ ] **Step 5: Build + test — 17/17 pass**

```bash
cmake --build --preset ubuntu-x86_64 && ctest --test-dir build_x86_64 --output-on-failure
```

- [ ] **Step 6: Commit**

```bash
git add CMakeLists.txt src/fx/models_dl/downloader.h src/fx/models_dl/downloader.cpp tests/test_downloader.cpp
git commit -m "feat: models_dl downloader (curl subprocess, hash verify, tgz extract)"
```

---

### Task 3: SegmentationModel interface + MediaPipe Lite

**Files:**
- Modify: `CMakeLists.txt` (sources/tests)
- Create: `src/fx/models/segmentation_model.h`
- Create: `src/fx/models/mediapipe_lite.h`
- Create: `src/fx/models/mediapipe_lite.cpp`
- Modify: `src/fx/models/pp_humanseg.h` (implement the interface)
- Modify: `src/fx/models/pp_humanseg.cpp` (ctor signature: take OrtModel by value? No — keep ctor, add interface)
- Modify: `src/fx/pipeline/segmentation_pipeline.h/.cpp` (use interface via unique_ptr)
- Test: `tests/test_mediapipe_lite.cpp`

- [ ] **Step 1: Add the MediaPipe model download to CMakeLists.txt**

Append after the PP-HumanSeg block:

```cmake
# --- Bundled segmentation model: MediaPipe Selfie Seg (Apache-2.0, Lite tier) ---
if(NOT EXISTS "${FX_MODEL_DIR}/selfie_segmentation.onnx")
  file(DOWNLOAD
    "https://raw.githubusercontent.com/royshil/obs-backgroundremoval/main/models/selfie_segmentation.onnx"
    ${FX_MODEL_DIR}/selfie_segmentation.onnx
    EXPECTED_HASH SHA256=1511ab5564cb0ae179cba9bcad6ad931003ccc71656f11fd270967023678c6a5)
  file(DOWNLOAD
    "https://raw.githubusercontent.com/royshil/obs-backgroundremoval/main/models/selfie_segmentation.onnx.license"
    ${FX_MODEL_DIR}/selfie_segmentation.onnx.license)
endif()
```

Add compile definitions to the fx target:

```cmake
target_compile_definitions(fx PUBLIC
  FX_MEDIAPIPE_MODEL_PATH="${FX_MODEL_DIR}/selfie_segmentation.onnx")
```

Also add the new sources: `src/fx/models/mediapipe_lite.cpp` to fx; `tests/test_mediapipe_lite.cpp` to fx_tests. Also install both models in install-local.sh (Step 6).

- [ ] **Step 2: Create `src/fx/models/segmentation_model.h`**

```cpp
#pragma once

#include "fx/types.h"

namespace fx {

/* Common interface for segmentation models (PP-HumanSeg, MediaPipe Lite,
 * RVM). infer() always returns a 192x192 mask (the pipeline contract). */
class SegmentationModel {
public:
	virtual ~SegmentationModel() = default;
	virtual Mask infer(const Frame &frame) = 0;
};

} // namespace fx
```

- [ ] **Step 3: Write the failing test — `tests/test_mediapipe_lite.cpp`**

```cpp
#include <gtest/gtest.h>

#include "fx/models/mediapipe_lite.h"

TEST(MediaPipeLite, Infers192MaskThroughInterface)
{
	fx::MediaPipeLite model(FX_MEDIAPIPE_MODEL_PATH, 1);
	fx::Frame f;
	f.width = 320;
	f.height = 240;
	f.bgra.assign(320u * 240u * 4u, 128);

	fx::SegmentationModel &iface = model;
	fx::Mask m = iface.infer(f);
	ASSERT_EQ(m.width, 192);
	ASSERT_EQ(m.height, 192);
	ASSERT_EQ(m.px.size(), 192u * 192u);
	for (float v : m.px) {
		ASSERT_GE(v, 0.0f);
		ASSERT_LE(v, 1.0f);
	}
	fx::Mask m2 = iface.infer(f);
	ASSERT_EQ(m.px, m2.px); // determinism
}
```

- [ ] **Step 4: Create `src/fx/models/mediapipe_lite.h` and adapt `pp_humanseg.h`**

`mediapipe_lite.h`:

```cpp
#pragma once

#include "fx/engine/ort_backend.h"
#include "fx/models/segmentation_model.h"

#include <string>
#include <vector>

namespace fx {

/* MediaPipe Selfie Segmentation (256x256 NHWC, Apache-2.0) — Lite tier.
 * Input /255 NHWC; output single-channel person mask, downscaled to
 * 192x192 to satisfy the pipeline contract. */
class MediaPipeLite : public SegmentationModel {
public:
	static constexpr int kSize = 256;

	explicit MediaPipeLite(const std::string &modelPath, int threads = 2);

	Mask infer(const Frame &frame) override;

private:
	OrtModel model_;
	std::vector<float> tensor_; // 3*256*256 scratch
	std::vector<float> mask256_;
};

} // namespace fx
```

`pp_humanseg.h` — change the class line to implement the interface (keep everything else):

```cpp
#include "fx/models/segmentation_model.h"
...
class PPHumanSeg : public SegmentationModel {
...
	Mask infer(const Frame &frame) override;
...
};
```

- [ ] **Step 5: Build, watch FAIL, implement `src/fx/models/mediapipe_lite.cpp`**

```cpp
#include "fx/models/mediapipe_lite.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace fx {

MediaPipeLite::MediaPipeLite(const std::string &modelPath, int threads)
	: model_(modelPath, threads), tensor_(3 * kSize * kSize),
	  mask256_(kSize * kSize)
{
	const auto &shape = model_.input().shape;
	if (shape.size() != 4 || shape[1] != kSize || shape[2] != kSize ||
	    shape[3] != 3)
		throw std::runtime_error(
			"fx: unexpected MediaPipe input shape (want NHWC 256)");
}

Mask MediaPipeLite::infer(const Frame &frame)
{
	const int sw = frame.width, sh = frame.height;
	constexpr int dw = kSize, dh = kSize;

	/* Bilinear-resize BGRA -> 256x256, /255, NHWC layout. */
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
			float *dst = tensor_.data() + (y * dw + x) * 3;
			for (int c = 0; c < 3; c++) {
				float v = (p00[c] * (1 - wx) + p01[c] * wx) *
						  (1 - wy) +
					  (p10[c] * (1 - wx) + p11[c] * wx) * wy;
				dst[c] = v / 255.0f;
			}
		}
	}

	std::vector<float> out = model_.run(tensor_); // [1,256,256,1]

	/* Downscale 256x256 -> 192x192 (bilinear on the mask). */
	Mask m;
	m.width = 192;
	m.height = 192;
	m.px.resize(192 * 192);
	for (int y = 0; y < 192; y++) {
		float fy = (y + 0.5f) * 256.0f / 192.0f - 0.5f;
		int y0 = std::clamp((int)std::floor(fy), 0, 255);
		int y1 = std::min(y0 + 1, 255);
		float wy = std::clamp(fy - (float)y0, 0.0f, 1.0f);
		for (int x = 0; x < 192; x++) {
			float sx = (x + 0.5f) * 256.0f / 192.0f - 0.5f;
			int x0 = std::clamp((int)std::floor(sx), 0, 255);
			int x1 = std::min(x0 + 1, 255);
			float wx = std::clamp(sx - (float)x0, 0.0f, 1.0f);
			float v = (out[y0 * 256 + x0] * (1 - wx) +
				   out[y0 * 256 + x1] * wx) *
					  (1 - wy) +
				  (out[y1 * 256 + x0] * (1 - wx) +
				   out[y1 * 256 + x1] * wx) *
					  wy;
			m.px[y * 192 + x] = std::clamp(v, 0.0f, 1.0f);
		}
	}
	return m;
}

} // namespace fx
```

- [ ] **Step 6: Update `build-aux/install-local.sh`** — add:

```bash
cp "$BUILD_DIR/models/selfie_segmentation.onnx" "$DEST/data/models/"
cp "$BUILD_DIR/models/selfie_segmentation.onnx.license" "$DEST/data/models/"
```

- [ ] **Step 7: Build + test — 18/18 pass; commit**

```bash
cmake --build --preset ubuntu-x86_64 && ctest --test-dir build_x86_64 --output-on-failure
git add CMakeLists.txt src/fx/models/ tests/test_mediapipe_lite.cpp build-aux/install-local.sh
git commit -m "feat: SegmentationModel interface + MediaPipe Lite tier"
```

---

### Task 4: RVM model class (multi-IO, recurrent states)

**Files:**
- Modify: `CMakeLists.txt` (sources/tests)
- Create: `src/fx/models/rvm.h`
- Create: `src/fx/models/rvm.cpp`
- Test: `tests/test_rvm.cpp`

NOTE: RVM is NOT downloaded by CMake (it's the runtime-download Quality model). Tests need the file present: the test checks for `${FX_MODEL_DIR}/rvm_mobilenetv3_fp32.onnx` and SKIPS gracefully if absent (GTEST_SKIP), because the model arrives via the downloader. For development on this machine, copy the spike-verified file: `cp /tmp/opencode/models/rvm_mobilenetv3_fp32.onnx build_x86_64/models/` — do this in Step 1 so tests actually exercise the model here.

- [ ] **Step 1: Stage the verified RVM file for tests + add sources**

```bash
mkdir -p build_x86_64/models
cp /tmp/opencode/models/rvm_mobilenetv3_fp32.onnx build_x86_64/models/
```

Add `src/fx/models/rvm.cpp` to fx sources, `tests/test_rvm.cpp` to fx_tests, and to the fx target:

```cmake
target_compile_definitions(fx PUBLIC
  FX_RVM_MODEL_PATH="${FX_MODEL_DIR}/rvm_mobilenetv3_fp32.onnx")
```

- [ ] **Step 2: Create `src/fx/models/rvm.h`**

```cpp
#pragma once

#include "fx/engine/ort_backend.h"
#include "fx/models/segmentation_model.h"

#include <array>
#include <string>
#include <vector>

namespace fx {

/* Robust Video Matting MobileNetV3 (192x192, GPL-3.0) — Quality tier.
 * Recurrent temporal memory: states r1..r4 are zero-initialized and fed
 * back frame-to-frame. Input BGR /255 NCHW; mask = pha output. */
class Rvm : public SegmentationModel {
public:
	static constexpr int kSize = 192;

	explicit Rvm(const std::string &modelPath, int threads = 2);

	Mask infer(const Frame &frame) override;

	/* Clears recurrent states (e.g. on source change). */
	void resetState();

private:
	OrtModel model_;
	std::vector<float> tensor_; // 3*192*192 scratch
	std::array<std::vector<float>, 4> states_;
	std::array<std::vector<int64_t>, 4> stateShapes_;
};

} // namespace fx
```

- [ ] **Step 3: Write the failing test — `tests/test_rvm.cpp`**

```cpp
#include <gtest/gtest.h>

#include "fx/models/rvm.h"

#include <cstdio>

static bool fileExists(const char *p)
{
	FILE *f = fopen(p, "r");
	if (f)
		fclose(f);
	return f != nullptr;
}

static fx::Frame makeFrame(uint8_t v)
{
	fx::Frame f;
	f.width = 320;
	f.height = 240;
	f.bgra.assign(320u * 240u * 4u, v);
	return f;
}

TEST(Rvm, InfersMaskAndFeedsBackState)
{
	if (!fileExists(FX_RVM_MODEL_PATH))
		GTEST_SKIP() << "RVM model not downloaded (runtime model)";

	fx::Rvm model(FX_RVM_MODEL_PATH, 1);
	fx::Frame f = makeFrame(128);

	fx::Mask m = model.infer(f);
	ASSERT_EQ(m.width, 192);
	ASSERT_EQ(m.height, 192);
	ASSERT_EQ(m.px.size(), 192u * 192u);
	for (float v : m.px) {
		ASSERT_GE(v, 0.0f);
		ASSERT_LE(v, 1.0f);
	}

	/* Second inference with fed-back state: deterministic given same
	 * input, but different from a fresh model (proves state flows). */
	fx::Mask m2 = model.infer(f);
	fx::Rvm fresh(FX_RVM_MODEL_PATH, 1);
	fx::Mask mFresh = fresh.infer(f);
	ASSERT_NE(m2.px, mFresh.px);

	/* After reset, the model reproduces the fresh result exactly. */
	model.resetState();
	fx::Mask m3 = model.infer(f);
	ASSERT_EQ(m3.px, mFresh.px);
}
```

- [ ] **Step 4: Build, watch FAIL, implement `src/fx/models/rvm.cpp`**

```cpp
#include "fx/models/rvm.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace fx {

Rvm::Rvm(const std::string &modelPath, int threads)
	: model_(modelPath, threads), tensor_(3 * kSize * kSize)
{
	if (model_.inputCount() != 6 || model_.outputCount() != 6)
		throw std::runtime_error("fx: unexpected RVM IO count");

	/* Recurrent state shapes: r_i channels 16/20/40/64, spatial
	 * kSize / 2^i. Verified by introspection at 192x192. */
	static const int channels[4] = {16, 20, 40, 64};
	for (int i = 0; i < 4; i++) {
		int s = kSize >> (i + 1);
		stateShapes_[i] = {1, channels[i], s, s};
		states_[i].assign((size_t)channels[i] * s * s, 0.0f);
	}
}

void Rvm::resetState()
{
	for (auto &s : states_)
		std::fill(s.begin(), s.end(), 0.0f);
}

Mask Rvm::infer(const Frame &frame)
{
	const int sw = frame.width, sh = frame.height;
	constexpr int dw = kSize, dh = kSize;

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
			for (int c = 0; c < 3; c++) {
				float v = (p00[c] * (1 - wx) + p01[c] * wx) *
						  (1 - wy) +
					  (p10[c] * (1 - wx) + p11[c] * wx) * wy;
				tensor_[c * dh * dw + y * dw + x] = v / 255.0f;
			}
		}
	}

	/* IO order (verified): inputs src, r1i..r4i, downsample_ratio;
	 * outputs fgr, pha, r1o..r4o. Dynamic dims (src input) get shape
	 * overrides; see runWithShapes note below. */
	std::vector<std::vector<float>> inputs;
	inputs.reserve(6);
	inputs.push_back(tensor_);
	for (int i = 0; i < 4; i++)
		inputs.push_back(states_[i]);
	inputs.push_back({1.0f}); // downsample_ratio

	std::vector<std::vector<int64_t>> shapes = {
		{1, 3, kSize, kSize}, stateShapes_[0], stateShapes_[1],
		stateShapes_[2], stateShapes_[3], {1}};
	auto outs = model_.runWithShapes(inputs, shapes);
	states_[0] = outs[2];
	states_[1] = outs[3];
	states_[2] = outs[4];
	states_[3] = outs[5];

	const std::vector<float> &pha = outs[1];
	Mask m;
	m.width = dw;
	m.height = dh;
	m.px.assign(pha.begin(), pha.end());
	return m;
}

} // namespace fx
```

REQUIRED OrtModel extension (add to `src/fx/engine/ort_backend.h/.cpp` in this task): the declared `src` input shape is dynamic (`[batch,3,height,width]`), and `OrtModel::run` passes declared shapes verbatim — dynamic dims (symbolic/-1) would create a wrong tensor. Add:

```cpp
public:
	/* Multi-IO run with per-input shape override (for models with
	 * dynamic dims like RVM's src input). overrides[i] empty = keep
	 * declared shape. */
	std::vector<std::vector<float>>
	runWithShapes(const std::vector<std::vector<float>> &inputData,
		      const std::vector<std::vector<int64_t>> &overrides);
```

Implementation: identical to run(), but when `overrides[i]` is non-empty use it as the tensor shape instead of `inputs_[i].shape`.

- [ ] **Step 5: Build + test — 19/19 pass (RVM test runs on this machine); commit**

```bash
cmake --build --preset ubuntu-x86_64 && ctest --test-dir build_x86_64 --output-on-failure
git add CMakeLists.txt src/fx/engine/ort_backend.h src/fx/engine/ort_backend.cpp src/fx/models/rvm.h src/fx/models/rvm.cpp tests/test_rvm.cpp
git commit -m "feat: RVM MobileNetV3 quality model with recurrent state feedback"
```

---

### Task 5: Pipeline tiers + mask params + worker hot-swap

**Files:**
- Modify: `src/fx/pipeline/segmentation_pipeline.h/.cpp` (tier factory + mask params)
- Modify: `src/fx/worker.h/.cpp` (processor hot-swap, thread-safe)
- Modify: `src/fx_bridge.h/.cpp` (tier + params + download API)
- Modify: `src/cam-effects-filter.c` (tier picker, advanced settings, download button, status)
- Create: `data/models/manifest.json`
- Test: `tests/test_pipeline_params.cpp` (+ worker swap test in test_worker.cpp)

- [ ] **Step 1: Manifest — `data/models/manifest.json`**

```json
{
  "version": 1,
  "models": [
    {
      "id": "rvm_mobilenetv3_fp32",
      "kind": "model",
      "tier": "quality",
      "file": "rvm_mobilenetv3_fp32.onnx",
      "url": "https://raw.githubusercontent.com/royshil/obs-backgroundremoval/main/models/rvm_mobilenetv3_fp32.onnx",
      "license_url": "https://raw.githubusercontent.com/royshil/obs-backgroundremoval/main/models/rvm_mobilenetv3_fp32.onnx.license",
      "sha256": "88d4531297118f595bf2fd60f6f566aec2e559393802d1f436c380f0cbbd2828",
      "size": 14975696,
      "license": "GPL-3.0-only",
      "notice": "Robust Video Matting (RVM) MobileNetV3, licensed GPL-3.0-only. By downloading you accept the license terms."
    },
    {
      "id": "ort_cuda_ep_1.28.0",
      "kind": "provider",
      "tier": "cuda",
      "file": "libonnxruntime_providers_cuda.so",
      "url": "https://github.com/microsoft/onnxruntime/releases/download/v1.28.0/onnxruntime-linux-x64-gpu_cuda13-1.28.0.tgz",
      "sha256": "84d28f27589090b280d4312743efd3d450cd4ac7d1e1d75e7d9076d9637bf9de",
      "size": 240874643,
      "license": "MIT",
      "extract": [
        "onnxruntime-linux-x64-gpu_cuda13-1.28.0/lib/libonnxruntime_providers_cuda.so",
        "onnxruntime-linux-x64-gpu_cuda13-1.28.0/lib/libonnxruntime_providers_shared.so"
      ],
      "notice": "NVIDIA CUDA execution provider for ONNX Runtime 1.28.0 (MIT). ~240 MB download, optional."
    }
  ]
}
```

- [ ] **Step 2: Worker hot-swap — modify `src/fx/worker.h/.cpp`**

Add to the public API:

```cpp
	/* Replaces the processor (model hot-swap). Clears any pending frame
	 * so no frame is processed by a stale/torn pipeline. */
	void setProcessor(Processor processor);
```

Implementation: take `inM_` lock, clear `pending_`, move the new processor in. In `loop()`, copy `processor_` into a local while holding `inM_` (same critical section that takes the frame), then call the LOCAL copy outside the lock. Add the test to `tests/test_worker.cpp`:

```cpp
TEST(Worker, SetProcessorSwapsSafely)
{
	std::atomic<int> which{0};
	auto mk = [](int id) {
		return [id](const fx::Frame &) {
			auto m = std::make_shared<fx::Mask>();
			m->width = 1;
			m->height = 1;
			m->px = {(float)id};
			return m;
		};
	};
	fx::Worker w(mk(1));
	w.start();
	std::this_thread::sleep_for(10ms);
	w.setProcessor(mk(2));
	std::this_thread::sleep_for(10ms);
	w.submit(makeFrame(1));
	uint64_t seq = 0;
	std::shared_ptr<const fx::Mask> m;
	for (int i = 0; i < 100 && seq == 0; i++) {
		std::this_thread::sleep_for(5ms);
		m = w.tryGetLatest(seq);
	}
	w.stop();
	ASSERT_EQ(seq, 1u);
	ASSERT_FLOAT_EQ(m->px[0], 2.0f);
}
```

- [ ] **Step 3: Pipeline tiers + mask params — `src/fx/pipeline/segmentation_pipeline.h` (full new content)**

```cpp
#pragma once

#include "fx/models/segmentation_model.h"
#include "fx/types.h"

#include <memory>
#include <string>

namespace fx {

enum class SegTier { Lite, Standard, Quality };

struct MaskParams {
	float threshold = 0.0f;    // 0 = off; else binarize cutoff
	float contourFrac = 0.0f;  // drop blobs < this fraction of frame
	float featherRadius = 0.0f;// mask blur radius in px (0 = off)
	float beta = 0.6f;         // temporal EMA factor
};

/* model inference -> temporal EMA -> guided filter -> threshold ->
 * contour filter -> feather. */
class SegmentationPipeline {
public:
	/* Creates a pipeline for the given tier.
	 * modelPaths: lite / standard / quality ONNX paths (quality may be
	 * empty ONLY if tier != Quality). Throws on missing/invalid model. */
	SegmentationPipeline(SegTier tier, const std::string &litePath,
			     const std::string &standardPath,
			     const std::string &qualityPath, int threads = 2);

	std::shared_ptr<Mask> process(const Frame &frame);

	void setMaskParams(const MaskParams &p) { params_ = p; }
	MaskParams maskParams() const { return params_; }

private:
	std::unique_ptr<SegmentationModel> model_;
	MaskParams params_;
	std::shared_ptr<Mask> prev_;
};

} // namespace fx
```

Implementation notes for `segmentation_pipeline.cpp`:
- Factory: Lite → `MediaPipeLite(litePath, threads)`, Standard → `PPHumanSeg(standardPath, threads)`, Quality → `Rvm(qualityPath, threads)`.
- `process()`: infer → EMA (beta from params_) → guided filter → threshold (`if (params_.threshold > 0) for px: v = v >= t ? 1 : 0`) → contour filter → feather.
- New ops in `fx/image/ops.h/.cpp` (add, with unit tests in a new `tests/test_mask_ops2.cpp`):

```cpp
/* Zeroes connected components smaller than frac * (w*h) in a binarized
 * (>= 0.5) mask. 4-connectivity, iterative flood fill. */
void contourFilter(Mask &m, float frac);

/* Blurs the mask with a box blur of the given radius (px). 0 = no-op. */
void featherMask(Mask &m, float radius);
```

contourFilter: binarize at 0.5, BFS label components, zero those with area < frac*w*h. featherMask: reuse the boxBlur approach on Mask.px with radius (clamp 0..16).

Tests for test_mask_ops2.cpp:

```cpp
#include <gtest/gtest.h>

#include "fx/image/ops.h"

TEST(ContourFilter, RemovesSmallBlobs)
{
	/* 20x20 frame: one 8x8 blob (area 64) and one 2x2 blob (area 4). */
	fx::Mask m;
	m.width = 20;
	m.height = 20;
	m.px.assign(400, 0.0f);
	for (int y = 2; y < 10; y++)
		for (int x = 2; x < 10; x++)
			m.px[y * 20 + x] = 1.0f;
	for (int y = 14; y < 16; y++)
		for (int x = 14; x < 16; x++)
			m.px[y * 20 + x] = 1.0f;
	fx::contourFilter(m, 0.05f); // 5% of 400 = 20 px threshold
	float bigSum = 0, smallSum = 0;
	for (int y = 2; y < 10; y++)
		for (int x = 2; x < 10; x++)
			bigSum += m.px[y * 20 + x];
	for (int y = 14; y < 16; y++)
		for (int x = 14; x < 16; x++)
			smallSum += m.px[y * 20 + x];
	ASSERT_EQ(bigSum, 64.0f);
	ASSERT_EQ(smallSum, 0.0f);
}

TEST(ContourFilter, ZeroFracKeepsEverything)
{
	fx::Mask m;
	m.width = 4;
	m.height = 4;
	m.px.assign(16, 1.0f);
	fx::contourFilter(m, 0.0f);
	for (float v : m.px)
		ASSERT_EQ(v, 1.0f);
}

TEST(FeatherMask, BlursStepEdge)
{
	fx::Mask m;
	m.width = 16;
	m.height = 1;
	m.px.assign(16, 0.0f);
	for (int x = 8; x < 16; x++)
		m.px[x] = 1.0f;
	fx::featherMask(m, 2.0f);
	ASSERT_GT(m.px[7], 0.0f); // edge softened leftward
	ASSERT_LT(m.px[7], 1.0f);
	ASSERT_EQ(m.px[0], 0.0f); // far field untouched
}
```

- [ ] **Step 4: Bridge tier/params/download API — extend `src/fx_bridge.h`**

```c
/* Tier: 0=auto, 1=lite, 2=standard, 3=quality. */
void cam_fx_set_tier(cam_fx_t *fx, int tier);

/* Advanced mask params (see fx::MaskParams). */
void cam_fx_set_mask_params(cam_fx_t *fx, float threshold, float contour,
			    float feather, float beta);

/* Starts a background download for the given manifest entry id
 * ("rvm_mobilenetv3_fp32" or "ort_cuda_ep_1.28.0"). Returns 0 on start,
 * -1 if busy/invalid. */
int cam_fx_start_download(cam_fx_t *fx, const char *id);

/* Download status: state string via cam_fx_download_state (one of
 * idle/downloading/verifying/extracting/done/error), progress 0..1
 * or -1. */
int cam_fx_download_state(cam_fx_t *fx, char *buf, int buf_len,
			  double *progress);
```

Bridge implementation notes:
- **New create signature** (replaces the Plan 2 one): `cam_fx_t *cam_fx_create(const char *lite_path, const char *standard_path, const char *quality_path, int threads);` — lite/standard resolve via `obs_module_file(...)` (bundled), quality resolves to the cache path `~/.config/obs-cam-effects/models/rvm_mobilenetv3_fp32.onnx` (may not exist yet — pass it anyway; Quality tier construction is deferred until the file exists). Update the filter's `cam_fx_create` call site in `cam-effects-filter.c` accordingly.
- Tier resolution in the bridge: `auto` → Quality if (quality model file exists AND construction succeeds) else Standard. A missing quality file at tier=quality → keep Standard + report via status (the filter shows the download UI).
- On tier change: build the new pipeline on the calling thread (model load ~100ms — acceptable on settings change), then `worker->setProcessor(...)` (hot-swap, no restart).
- The bridge owns a `std::unique_ptr<fx::models_dl::Downloader>`; `cam_fx_start_download` maps id → manifest entry (parsed by the bridge via obs_data — see below) → `Downloader::start`.
- Manifest parsing (bridge only): `obs_data_from_json_file(manifest_path)` → iterate `models` array via `obs_data_array_item` → fill a small struct per id. Paths: cache dir = `~/.config/obs-cam-effects/models/` (create if needed) for the RVM file; `~/.config/obs-cam-effects/providers/` for the CUDA libs.

- [ ] **Step 5: Filter UI — modify `src/cam-effects-filter.c`**

New settings (with defaults):
- `"tier"` list: `"auto"` (default) / `"lite"` / `"standard"` / `"quality"`
- `"mask_threshold"` float slider 0.00–1.00 step 0.01, default 0
- `"mask_contour"` float slider 0.00–0.50 step 0.01, default 0
- `"mask_feather"` float slider 0–8 step 0.5, default 0
- `"mask_temporal"` float slider 0.00–0.95 step 0.05, default 0.6
- `"download_btn"` button: label set at creation to "Download Quality model (GPL-3.0, 15 MB)"; callback starts the RVM download via `cam_fx_start_download(fx, "rvm_mobilenetv3_fp32")` (or CUDA: a second button `"download_cuda_btn"`, "Download GPU acceleration (MIT, ~240 MB)")
- license notice text (OBS_TEXT_INFO, from manifest `notice`)
- status text updated in `update()` with: tier in effect, model state (bundled/downloaded/missing), download state+progress, processing backend (CPU), measured fps (from bridge: publish per-second processed-frame count from the worker via `cam_fx_fps(cam_fx_t*)`)

Add `uint64_t cam_fx_fps(cam_fx_t *fx);` to the bridge (worker-side counter of masks published in the last second — rolling window: keep 60 latest publish timestamps... simpler: frames processed in the last completed 1s window, updated in Worker publish or bridge try_get_mask call site; implement in the bridge with a small ring).

update() wiring: read tier + 4 floats → `cam_fx_set_tier` + `cam_fx_set_mask_params`. update() also refreshes the status description text.

- [ ] **Step 6: Build + test — 23/23 pass (incl. new worker swap + mask ops2); commit**

```bash
cmake --build --preset ubuntu-x86_64 && ctest --test-dir build_x86_64 --output-on-failure
git add -A
git commit -m "feat: tier picker, mask tuning params, download flow, hot-swap"
```

---

### Task 6: EP probe + CUDA registration (verified on RTX 5070)

**Files:**
- Modify: `CMakeLists.txt` (sources/tests)
- Create: `src/fx/engine/ep_probe.h`
- Create: `src/fx/engine/ep_probe.cpp`
- Modify: `src/fx/engine/ort_backend.h/.cpp` (EP-aware session creation)
- Test: `tests/test_ep_probe.cpp`

- [ ] **Step 1: Create `src/fx/engine/ep_probe.h`**

```cpp
#pragma once

#include <string>

namespace fx {

/* Execution-provider probe. CUDA is used when its provider library has
 * been downloaded to providersDir (via the model manager) AND
 * registration succeeds. CPU is always the guaranteed fallback. */
class EpProbe {
public:
	/* Registers the CUDA EP library if the file exists.
	 * Returns true if CUDA is usable for new sessions. Safe to call
	 * multiple times (idempotent). */
	static bool cudaAvailable(const std::string &providersDir);

	/* Human-readable backend name for status display. */
	static const char *backendName(bool cuda);
};

} // namespace fx
```

- [ ] **Step 2: Write `tests/test_ep_probe.cpp`**

```cpp
#include <gtest/gtest.h>

#include "fx/engine/ep_probe.h"

#include <cstdio>
#include <string>

TEST(EpProbe, CpuFallbackWhenNoProviders)
{
	ASSERT_FALSE(fx::EpProbe::cudaAvailable("/nonexistent/providers"));
	ASSERT_STREQ(fx::EpProbe::backendName(false), "CPU");
}

TEST(EpProbe, CudaRegistersWhenProviderPresent)
{
	const char *dir = "/home/pain/.config/obs-cam-effects/providers";
	std::string so = std::string(dir) + "/libonnxruntime_providers_cuda.so";
	FILE *f = fopen(so.c_str(), "r");
	if (!f)
		GTEST_SKIP() << "CUDA provider not downloaded";
	fclose(f);
	ASSERT_TRUE(fx::EpProbe::cudaAvailable(dir));
	ASSERT_STREQ(fx::EpProbe::backendName(true), "CUDA");
}
```

- [ ] **Step 3: Implement `src/fx/engine/ep_probe.cpp`**

```cpp
#include "fx/engine/ep_probe.h"

#include <onnxruntime_cxx_api.h>

#include <mutex>
#include <sys/stat.h>

namespace fx {

static bool fileExists(const std::string &p)
{
	struct stat st;
	return stat(p.c_str(), &st) == 0;
}

bool EpProbe::cudaAvailable(const std::string &providersDir)
{
	static std::once_flag once;
	static bool ok = false;
	std::string so = providersDir + "/libonnxruntime_providers_cuda.so";
	if (!fileExists(so))
		return false;
	std::call_once(once, [&] {
		try {
			Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "fx-ep");
			env.RegisterExecutionProviderLibrary("CUDA", so.c_str());
			ok = !env.GetEpDevices().empty();
		} catch (...) {
			ok = false;
		}
	});
	return ok;
}

const char *EpProbe::backendName(bool cuda)
{
	return cuda ? "CUDA" : "CPU";
}

} // namespace fx
```

NOTE on correctness the implementer must resolve: registering on a throwaway `Ort::Env` registers the library process-wide on THAT env — but sessions are created on each OrtModel's OWN env. Check the ORT 1.28 semantics: registration is per-env. Therefore the clean design is: **fx uses ONE shared env for all sessions when CUDA is involved** — refactor `OrtModel` to accept a shared `Ort::Env` (default-construct its own when not provided). Apply: `OrtModel(path, threads, Ort::Env *sharedEnv = nullptr)`; the pipeline passes the shared env that EpProbe registered CUDA on. Implement `fx::engine::sharedEnv()` returning a process-wide `Ort::Env&` (function-local static), do registration on THAT, and default all OrtModel instances to it. (Keep it simple: one static env for the whole process; ORT documents env as shareable across sessions/threads.)

- [ ] **Step 4: EP-aware session creation in `ort_backend.cpp`**

Extend `OrtModel` ctor: after building SessionOptions, if `EpProbe::cudaAvailable(providersDir)` (providersDir passed to ctor or via `fx::engine::providersDir()` setter — simplest: ctor param with default empty) then append CUDA to the session options via the C API on the shared env:

```cpp
	/* after RegisterExecutionProviderLibrary on the shared env: */
	auto devices = sharedEnv.GetEpDevices(); // filter for CUDA entries
	if (!devices.empty()) {
		const OrtEpDevice *ep = devices.front();
		const OrtApi &api = Ort::GetApi();
		std::vector<const OrtEpDevice *> eps{ep};
		api.SessionOptionsAppendExecutionProvider_V2(opts, eps.data(), 1);
	}
```

Verify the exact C++ wrapper availability (`Ort::SessionOptions::AppendExecutionProvider_V2` may exist in 1.28 cxx API — check the headers; prefer it if present). If appending throws (e.g. driver mismatch), catch and fall back to CPU silently (status shows CPU).

- [ ] **Step 5: Download the CUDA provider (exercise the real flow) and benchmark**

Use the filter UI (after Task 5's download button) OR the downloader directly via a tiny throwaway test harness. Then run a benchmark comparing RVM on CPU vs CUDA:

```bash
# quick manual bench via the existing soak-style binary or python:
/tmp/opencode/ort-venv/bin/python -c "
import onnxruntime as ort, numpy as np, time
# CPU baseline for reference (measured in spike): ~9.5ms
" 
```

Better: write the benchmark as a temporary gtest filter (`--gtest_filter=RvmBench.*` style) OR just record the CUDA soak fps from `ctest -R Soak` after forcing CUDA via the pipeline's providersDir. Record both numbers in the report and dev notes.

- [ ] **Step 6: Build + test — pass; commit**

```bash
git add -A
git commit -m "feat: CUDA execution provider probe and registration (RTX 5070 verified)"
```

---

### Task 7: User visual checkpoint — Quality tier + advanced settings

**Files:** none

- [ ] **Step 1: Download Quality model through the UI**

In the filter properties: click "Download Quality model (GPL-3.0, 15 MB)". Reopen properties: status shows done. (On error, report it.)

- [ ] **Step 2: Compare tiers visually**

Set tier = Standard, look at edges (hair, hands). Switch tier = Quality (RVM). Expected: visibly steadier edges, less flicker (RVM temporal memory), possibly softer mask onset. Report: is the difference visible? Better?

- [ ] **Step 3: CUDA check**

If "Download GPU acceleration" was done: status should show CUDA backend. Note any fps/responsiveness change vs CPU.

- [ ] **Step 4: Advanced settings sanity**

- mask_threshold 0.5: edges become hard cutouts
- mask_contour 0.05: background blob flicker (if any) disappears
- mask_temporal 0.95: very laggy/smeary mask; 0.0: raw per-frame flicker visible
- mask_feather 2: softer edges

Report any surprise.

---

### Task 8: Closeout

**Files:**
- Modify: `docs/development-notes.md`

- [ ] **Step 1: Append Plan 3 state**

```markdown
## Plan 3 state (model manager + quality tier)

- Tier picker: Auto / Lite (MediaPipe 256x256, bundled) / Standard
  (PP-HumanSeg, bundled) / Quality (RVM, runtime download w/ GPL notice).
  Auto = Quality if downloaded (+ CUDA usable), else Standard.
- RVM: recurrent states, ~9.5ms/frame CPU; CUDA provider (240MB opt-in
  download, MIT) registered via ORT 1.28 plugin EP API.
  Benchmarks on RTX 5070: CPU [X] fps / CUDA [Y] fps.
- Model manager: curl subprocess + sha256sum + tar extract; manifest in
  data/models/manifest.json; cache ~/.config/obs-cam-effects/{models,providers}.
- Advanced mask settings: threshold / contour filter / feather / temporal
  smoothing (spec amendment 9).
- Worker hot-swap via setProcessor (no filter recreation on tier change).
- Status line: refresh-on-dialog-open only (OBS properties limitation).
```

- [ ] **Step 2: Final verification + commit**

```bash
git status --short
cmake --build --preset ubuntu-x86_64 && ctest --test-dir build_x86_64
git add docs/development-notes.md
git commit -m "docs: plan 3 closeout notes"
```

---

## Plan 3 Definition of Done

- [ ] Full ctest suite green (incl. downloader, RVM, worker swap, mask ops)
- [ ] Tier picker works; Quality requires downloaded RVM (license notice shown)
- [ ] RVM runs with state feedback; determinism + reset verified
- [ ] CUDA registers on RTX 5070 (or gracefully absent); CPU fallback intact
- [ ] Advanced mask settings affect output (unit tests + user visual)
- [ ] Downloads: hash verified, atomic install, error states surfaced
- [ ] Git history: one commit per task, clean tree
