# Benchmarks

Performance measurements for obs-cam-effects. Newest entries last per section.

## Environment

| | |
|---|---|
| Machine | Kali GNU/Linux Rolling (Debian-based), x86-64 |
| GPU | NVIDIA GeForce RTX 5070 12GB (Blackwell, sm_120) — driver 580.126.09 |
| CUDA | Runtime 13.3 (system) + cuDNN 9.24 (`cudnn9-cuda-13`) |
| Inference | ONNX Runtime 1.28.0 (CPU build bundled; CUDA via the classic API on the optional GPU build download) |
| Compiler | GCC 15.2.0, -O2 |

Method: C++ bench harness, 100 inferences at 192×192 (seg models) after 1 warmup, mean ms/frame. ORT intra-op threads = 2. "Pipeline" numbers include fx pre/post-processing (resize, EMA, guided filter); "model" numbers are inference-only.

## Segmentation models (2026-07-30)

| Model | CPU (pipeline) | CUDA (RTX 5070) | Notes |
|---|---|---|---|
| MediaPipe Selfie Seg (Lite) | not measured | **1.96 ms (511 fps)** | 256×256 in / 192×192 out |
| PP-HumanSeg v2-Lite (Standard) | **6.6 ms (151 fps)** | **2.05 ms (488 fps)** | 192×192; **3.2× speedup on CUDA** |
| RVM MobileNetV3 (Quality) | **17.1 ms (58 fps)** | **2.62 ms (382 fps)** | 192×192 + recurrent states; **6.5× speedup** (2026-08-06, classic API — CPU pin lifted) |

Reference spike numbers (Python ORT, model-only, this machine): PP-HumanSeg 7.3 ms, RVM 9.5 ms.

## Worker soak test (2026-07-30)

PP-HumanSeg pipeline on the worker thread, latest-wins drop policy, 600 frames offered at ~1000 fps:

- **~140 fps sustained** (91 processed / 600 submitted — drops proven)
- Inference latency never blocks the OBS graphics thread by design

## CUDA path history (2026-08-06)

- **plugin-EP-V2 (abandoned)**: ORT 1.28's plugin execution-provider API hard-segfaulted (uncatchable SIGSEGV) on inswapper_128 and RVM on Blackwell (sm_120) — the whole OBS process died. PP-HumanSeg and MediaPipe ran fine on it.
- **Classic API + full GPU build (current)**: `OrtSessionOptionsAppendExecutionProvider_CUDA`, dlsym-probed (the symbol exists only in the GPU build's main lib, not in the bundled CPU build — same SONAME, drop-in replacement). The GPU build is an optional ~240 MB in-app download that lands in the plugin bin dir next to the plugin .so; **one OBS restart** after the download enables CUDA. All models run correctly on it, including inswapper and RVM.
- **Gotcha fixed along the way**: the classic append API logs through ORT's DefaultLogger, which exists only after the first `Ort::Env`. The first CUDA session in a fresh process failed with "Attempt to use DefaultLogger but none has been registered" and silently fell back to CPU (the Env/options expression in the `OrtModel` ctor has unspecified evaluation order). `makeSessionOptions` now forces `sharedEnv()` before the append.

## Face swap models (2026-08-06, classic CUDA API)

**Models in use:** YuNet (detection, bundled) + inswapper (swap, per frame, downloaded — **fp16 is the default download**, fp32 remains as fallback) + ArcFace w600k_r50 (source identity embedding, downloaded, runs once per source-image selection — not a per-frame cost, so not in the table). Detection is decimated by default: YuNet runs every 2nd frame (`detectEveryN=2`), skipped frames reuse the EMA-smoothed box while align/swap/paste-back still run every frame.

Fixture: `tests/data/face-test.jpg` (640×799, real face, detection score 0.95). 100 iterations after 1 warmup, ORT intra-op threads = 2. "detect" = full YuNet path (resize + inference + decode/NMS); "model" = inference only.

| Stage | CPU | CUDA (RTX 5070) | Notes |
|---|---|---|---|
| YuNet 2023mar — detect (full path) | **10.21 ms** | — | fx::YuNet detect stays CPU by design |
| YuNet 2023mar — model only | 7.33 ms* | **1.58 ms** | raw OrtModel; **4.6× speedup**. *CPU figure re-measured by independent review (stable 7.33 ms); the original 4.97 ms was not reproducible |
| inswapper_128 (fp32) — model only | 1245.96 ms | **15.75 ms** | **79× speedup — works on the classic API** (plugin-EP-V2 SIGSEGV'd on first Run). Re-measured 15.72 ms in the fp16 bench run |
| inswapper_128_fp16 — model only | not measured | **9.54 ms** | **1.65× vs fp32**; fp32 graph IO + bit-identical embedded emap = zero-change drop-in; default download since 2026-08-06 |
| FaceSwapPipeline end-to-end (fp32, detect every frame) | 1271.61 ms (0.8 fps) | **41.00 ms (24.4 fps)** | detect CPU + swap CUDA + paste-back |
| FaceSwapPipeline end-to-end (fp16, detect every frame) | — | **34.47 ms (29.0 fps)** | fp16 swap alone: −6.5 ms vs fp32 e2e |
| FaceSwapPipeline end-to-end (fp16, detect every 2nd frame) | — | **31.10 ms (32.2 fps)** | **shipped default** (`detectEveryN=2`): decimation saves another ~3.4 ms of CPU detect |

**CPU face swap is not viable** (spec confirmed): 0.8 fps end-to-end — the 174.7-GFLOP inswapper dominates (~1.25 s/frame). YuNet detect alone (10.2 ms) would be fine on CPU.

**CUDA face swap is viable: 32.2 fps end-to-end** with the shipped defaults (fp16 inswapper + detection decimation), up from 24.4 fps (fp32, detect every frame). inswapper drops 1246 ms → 15.8 ms (fp32) → 9.5 ms (fp16) on the classic API; the remaining e2e budget is CPU-side detect (~10 ms every frame, ~5 ms amortized at `detectEveryN=2`) plus align/paste-back/watermark. Verified in the real plugin too: `pipeline tier in effect: quality (backend: CUDA)` + masks flowing, OBS smoke run 2026-08-06.

## Bounded paste-back (2026-08-23)

CPU: 12th Gen Intel(R) Core(TM) i9-12900KF.

Build: CMake cache `CMAKE_BUILD_TYPE=RelWithDebInfo` and `CMAKE_CXX_FLAGS_RELWITHDEBINFO=-O2 -g -DNDEBUG`.

Method: five warm-up iterations followed by the median of 31 measured iterations per implementation. Both implementations received the same deterministic frame, face, fixed feathered-ellipse mask, centered 128x128 crop, and intensity 1. Frame copies were outside the timed region.

```text
640x480 full=10.667 ms bounded=0.489 ms speedup=21.827x old_full_frame_scratch=1228800 bytes
1920x1080 full=69.939 ms bounded=0.490 ms speedup=142.829x old_full_frame_scratch=8294400 bytes
```

This microbenchmark isolates CPU paste-back. It excludes YuNet, inswapper, OBS GPU readback/upload, optional background segmentation, and compositing. The output frame remains at its source dimensions.
