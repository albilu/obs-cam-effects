# Benchmarks

Performance measurements for obs-cam-effects. Newest entries last per section.

## Environment

| | |
|---|---|
| Machine | Kali GNU/Linux Rolling (Debian-based), x86-64 |
| GPU | NVIDIA GeForce RTX 5070 12GB (Blackwell, sm_120) — driver 580.126.09 |
| CUDA | Runtime 13.3 (system) + cuDNN 9.24 (`cudnn9-cuda-13`) |
| Inference | ONNX Runtime 1.28.0 (CPU build bundled; CUDA via plugin EP) |
| Compiler | GCC 15.2.0, -O2 |

Method: C++ bench harness, 100 inferences at 192×192 (seg models) after 1 warmup, mean ms/frame. ORT intra-op threads = 2. "Pipeline" numbers include fx pre/post-processing (resize, EMA, guided filter); "model" numbers are inference-only.

## Segmentation models (2026-07-30)

| Model | CPU (pipeline) | CUDA (RTX 5070) | Notes |
|---|---|---|---|
| MediaPipe Selfie Seg (Lite) | not measured | **1.96 ms (511 fps)** | 256×256 in / 192×192 out |
| PP-HumanSeg v2-Lite (Standard) | **6.3 ms (159 fps)** | **2.05 ms (488 fps)** | 192×192; **3.1× speedup on CUDA** |
| RVM MobileNetV3 (Quality) | **17.0 ms (59 fps)** | — | 192×192 + recurrent states |

Reference spike numbers (Python ORT, model-only, this machine): PP-HumanSeg 7.3 ms, RVM 9.5 ms.

## Worker soak test (2026-07-30)

PP-HumanSeg pipeline on the worker thread, latest-wins drop policy, 600 frames offered at ~1000 fps:

- **~140 fps sustained** (91 processed / 600 submitted — drops proven)
- Inference latency never blocks the OBS graphics thread by design

## Known issue: RVM on Blackwell

RVM hard-segfaults the ORT 1.28 prebuilt CUDA EP (uncatchable SIGSEGV) on sm_120 — upstream kernel-coverage gap (microsoft/onnxruntime#26177; prebuilt EPs lack complete sm_120 kernels; source builds with `CMAKE_CUDA_ARCHITECTURES=120` work). RVM is pinned to CPU (59 fps — fine). PP-HumanSeg and MediaPipe run correctly on the same EP. Re-test on ORT/cuDNN bumps.

inswapper_128 shows the identical failure (2026-08-04): session creation on the CUDA EP succeeds (`usesCuda=1`), the first `Run()` hard-segfaults (exit 139) — see the face swap section below.

## Face swap models (2026-08-04)

Fixture: `tests/data/face-test.jpg` (640×799, real face, detection score 0.95). 100 iterations after 1 warmup, ORT intra-op threads = 2. "detect" = full YuNet path (resize + inference + decode/NMS); "model" = inference only.

| Stage | CPU | CUDA (RTX 5070) | Notes |
|---|---|---|---|
| YuNet 2023mar — detect (full path) | **10.23 ms** | — | fx::YuNet is CPU-only by design (no providersDir ctor param) |
| YuNet 2023mar — model only | 4.93 ms | **1.68 ms** | raw OrtModel; **runs correctly on the Blackwell EP** (2.9× speedup) |
| inswapper_128 — model only | **1277.21 ms** | — | **SIGSEGV on first Run() with the prebuilt CUDA EP** |
| FaceSwapPipeline end-to-end | **1288.91 ms (0.8 fps)** | — | same SIGSEGV via `process()` with providersDir |

**CPU face swap is not viable** (spec confirmed): 0.8 fps end-to-end — the 174.7-GFLOP inswapper dominates (~1.28 s/frame). YuNet detect alone (10.2 ms) would be fine on CPU.

**inswapper on Blackwell: SIGSEGV** — same failure mode as RVM (see "Known issue" above): the OrtModel exception-fallback never engages because a SEGV is uncatchable. Face swap is therefore gated off entirely on this machine (CPU too slow, CUDA crashes). Re-test on ORT/cuDNN bumps or a source-built EP with `CMAKE_CUDA_ARCHITECTURES=120`.
