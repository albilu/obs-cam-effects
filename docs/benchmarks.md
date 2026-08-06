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

Fixture: `tests/data/face-test.jpg` (640×799, real face, detection score 0.95). 100 iterations after 1 warmup, ORT intra-op threads = 2. "detect" = full YuNet path (resize + inference + decode/NMS); "model" = inference only.

| Stage | CPU | CUDA (RTX 5070) | Notes |
|---|---|---|---|
| YuNet 2023mar — detect (full path) | **10.21 ms** | — | fx::YuNet detect stays CPU by design |
| YuNet 2023mar — model only | 7.33 ms* | **1.58 ms** | raw OrtModel; **4.6× speedup**. *CPU figure re-measured by independent review (stable 7.33 ms); the original 4.97 ms was not reproducible |
| inswapper_128 — model only | 1245.96 ms | **15.75 ms** | **79× speedup — works on the classic API** (plugin-EP-V2 SIGSEGV'd on first Run) |
| FaceSwapPipeline end-to-end | 1271.61 ms (0.8 fps) | **41.00 ms (24.4 fps)** | detect CPU + swap CUDA + paste-back |

**CPU face swap is not viable** (spec confirmed): 0.8 fps end-to-end — the 174.7-GFLOP inswapper dominates (~1.25 s/frame). YuNet detect alone (10.2 ms) would be fine on CPU.

**CUDA face swap is viable: 24.4 fps end-to-end.** inswapper drops 1246 ms → 15.8 ms on the classic API; the remaining e2e budget is CPU-side detect (~10 ms) plus align/paste-back/watermark. Verified in the real plugin too: `pipeline tier in effect: quality (backend: CUDA)` + masks flowing, OBS smoke run 2026-08-06.
