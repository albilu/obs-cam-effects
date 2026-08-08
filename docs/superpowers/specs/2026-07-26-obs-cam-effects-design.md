# obs-cam-effects — Design Specification

**Date:** 2026-07-26 (amended 2026-07-26: Linux-only scope; transparent replace mode)
**Status:** Approved design
**License:** Permissive core (MIT or Apache-2.0); encumbered models distributed via runtime download, never bundled.
**Platform:** Linux only (Windows/macOS support dropped by user decision, see Decisions log #7–8)

## 1. Overview

obs-cam-effects is a Linux OBS Studio plugin providing real-time camera effects through a single combined filter:

1. **Background blur** — mask-selective multi-pass blur
2. **Background replacement** — swap background with a user-provided image, **or make it transparent** (alpha) so scene sources below the camera show through
3. **Face swap** — replace the user's face with a face from a user-provided image ("deepfake feed")

Effects are combinable within one filter instance and run in real time (≥30fps at 720p) on mainstream hardware, with GPU acceleration where available.

## 2. Goals and non-goals

### Goals
- Single combined "Camera Effects" filter with per-effect toggles
- ≥30fps at 720p on a modern CPU without discrete GPU (segmentation effects) — **a floor, not a cap**: inference runs at fixed 192×192 regardless of output resolution, so 1080p60 and 4K segmentation are realistic (measured ~140fps sustained on a 2025 desktop CPU); higher output res costs only GPU compositing
- ≥30fps at 720p for face swap when a usable GPU execution provider exists; on CPU-only machines the face-swap toggle is shown as unavailable (CPU swap is 200–400ms/frame — not viable)
- One simple install path on Linux: `.deb` package + Flatpak (Flatpak OBS cannot see system plugins)
- Offline-capable out of the box for bundled (permissively licensed) models
- No camera blackout or freeze on failure unless the user explicitly chooses freeze (privacy mode)

### Non-goals (YAGNI)
- **Windows and macOS support** — dropped by user decision (2026-07-26); the design keeps no Win/mac-specific abstractions to maintain
- Offline/video-editing segmentation (SAM2-style prompt-driven masking) — research shows it is fundamentally unsuited to live filters
- Multiple simultaneous faces swap with per-face mapping (single prominent face in v1)
- Face enhancement (GFPGAN) in the real-time path — 20–50ms/frame breaks the budget; may return as an offline/async option later
- Body/hand tracking, virtual avatar, background video replacement (image only in v1)
- Out-of-process Python worker, virtual-camera companion app

## 3. Research conclusions (basis of design)

### 3.1 Segmentation: SAM2 vs lightweight models

Investigated as the README's open question. Conclusion: **SAM2 is not viable for live filters.**

- Kdenlive 25.04 uses SAM2 strictly offline: user draws a box, clicks "Generate Mask", a background Python process grinds through frames with a progress bar. Never touches a live feed.
- SAM2 is prompt-driven (needs a click/box to initialize), needs a discrete GPU even then (39–91fps only on an A100; ~0.7fps on an iPhone 15 Pro Max), and has no production ONNX export.
- The only near-edge SAM variant (EdgeTAM) reaches ~16fps on a flagship phone NPU and remains prompt-driven.

**Chosen stack** (proven by obs-backgroundremoval, Google Meet, published benchmarks):

| Tier | Model | License | Perf evidence |
|---|---|---|---|
| Lite (weak HW) | MediaPipe Selfie Seg (448KB) / SINet (395KB) | Apache-2.0 / MIT | Ships in Google Meet; real-time on phones |
| Standard (default) | PP-HumanSeg v2-Lite (6.2MB) | Apache-2.0 | 63fps single-thread on Snapdragon 855 (2019 phone); 96.63 mIoU |
| Quality (GPU) | RVM MobileNetV3 (~15MB) | **GPL-3.0** | 100+fps @1080p on GTX 1080 Ti; recurrent temporal memory = flicker-free hair-level mattes |

Hybrid quality technique: infer mask at model-native resolution (192×192 for the shipped PP-HumanSeg build), temporal EMA smoothing, then guided-filter edge refinement at mask resolution with GPU bilinear upscale at composite time.

### 3.2 Face swap

- Pipeline: YuNet detection (Apache-2.0) → 5-point Umeyama alignment → 128×128 crop → inswapper_128 → inverse-affine paste-back with a feathered-elliptical mask derived from the swap's own affine (not re-detected landmarks — avoids wobble; lesson from Deep-Live-Cam) → optional frame EMA.
- Estimated latency at 720p: ~15–30ms/frame on discrete GPU (30–60fps); 200–400ms on CPU — **CPU-only face swap is not viable** and is not offered in the UI.
- **Licensing blocker:** no legally redistributable real-time swap model exists. inswapper_128 and the insightface buffalo_l pack are non-commercial research only; Deep-Live-Cam is AGPL-3.0 (code cannot be copied into this plugin); SimSwap is CC-BY-NC; FaceFusion assets are unlicensed/non-commercial mixed.
- **Chosen strategy (user decision):** runtime download on first use — the Deep-Live-Cam distribution pattern. The plugin ships no swap weights; a consent dialog presents license terms before downloading from HuggingFace with SHA-256 verification.
- GFPGAN (Apache-2.0) enhancement excluded from the real-time path (latency).

### 3.3 OBS integration & architecture

- Frame path: `video_render` → render parent into a small texrender at model resolution (192×192 BGRA ≈ 147KB) → `gs_stage_texture` → worker thread → composite via custom effect shaders. ~50× less readback than full-res staging. (Model-native resolution confirmed 192×192 for the chosen PP-HumanSeg ONNX during Plan 2 spike.)
- Inference must run off the OBS graphics thread. obs-backgroundremoval's synchronous `video_tick` inference is its documented weakness (the lite fork exists to fix exactly this). Worker thread from day one.
- One combined filter, not three stacked filters: OBS has no official cross-filter data-sharing API; stacking would triple readbacks, sessions, and threads.
- ONNX Runtime (v1.28): CPU-only ORT bundled; GPU execution providers registered at runtime via plugin-EP libraries. CUDA packages (230–430MB) never ship in the installer.
- Symbol hygiene (`-Bsymbolic` + version script) to avoid collisions with OBS/other plugins.

## 4. Decisions log

| # | Decision | Choice | Rationale |
|---|---|---|---|
| 1 | Session focus | Tech evaluation before design | README contained open research questions |
| 2 | Hardware targets | Both CPU and GPU tracks, swappable | User base unknown; evaluate both |
| 3 | Face-swap model distribution | Runtime download (DLC pattern) | No redistributable model exists; also solves 530MB installer size |
| 4 | Plugin license | Permissive core; RVM via runtime download with GPL terms shown | Maximizes adoption; consistent rule: encumbered assets = runtime download |
| 5 | Architecture | Single combined filter, in-process ORT, worker thread (Approach A) | Ecosystem convergence; best perf; single-installer simplicity |
| 6 | Failure mode | User-selectable: passthrough (default) or freeze-last-frame | Privacy: passthrough would expose identities the filter exists to hide |
| 7 | Platform scope (amendment) | Linux only; Windows/macOS dropped | User decision 2026-07-26: eliminates Win/mac packaging, signing/notarization, and DirectML/CoreML provider work |
| 8 | Replace modes (amendment) | Background replace supports two modes: transparent alpha, or image | User decision 2026-07-26: transparency lets scene sources below show through (the obs-backgroundremoval compositing pattern) |
| 9 | Settings surface (amendment) | Adopt 8 tweakable settings inspired by obs-backgroundremoval (threshold, contour filter, feather, temporal smoothing) and Deep-Live-Cam (swap intensity, sharpness, preserve-mouth, status fps); their mask-every-X knob superseded by our automatic drop policy; perf targets clarified as floors | User decision 2026-07-26 after reviewing both apps' settings UIs |

## 5. Architecture

### 5.1 Module breakdown

C++ codebase from obs-plugintemplate. An OBS-free core library (`libfx`) plus a thin OBS adapter, so all logic is unit-testable without OBS:

```
obs-cam-effects/
├── src/
│   ├── plugin/              # OBS adapter (the only OBS-aware code)
│   │   ├── filter.cpp       # single "Camera Effects" obs_source_info
│   │   ├── staging.cpp      # texrender @ model-res → stagesurface → queue
│   │   └── properties.cpp   # settings UI (effect toggles, model picker)
│   ├── fx/                  # libfx — pure C++, no obs.h includes
│   │   ├── engine/          # inference runtime abstraction (Model interface)
│   │   │   ├── ort_backend.cpp    # ONNX Runtime impl + EP detection
│   │   │   └── ep_probe.cpp       # CUDA / CPU probe (Linux-only)
│   │   ├── models/          # PPHumanSeg, MediaPipe, SINet, RVM, YuNet, Inswapper
│   │   ├── pipeline/        # SegmentationPipeline, FaceSwapPipeline
│   │   ├── image/           # temporal smoothing, guided filter, feathering,
│   │   │                    #   5-pt alignment (Umeyama), paste-back blend
│   │   └── worker.cpp       # worker thread + SPSC queues + drop policy
│   ├── models_dl/           # model manager: download, verify, cache, consent
│   └── effects/             # OBS shaders: mask_composite, kawase_blur, face_blend
├── data/models/bundled/     # Apache/MIT models shipped in installer
└── tests/                   # libfx unit + golden-frame tests
```

Six modules, one job each:

1. **Filter shell** — registers the filter; owns settings; never blocks.
2. **Staging** — GPU→CPU frame handoff at 192×192.
3. **Engine** — ORT behind a `Model` interface; EP probe picks the best available provider; CPU guaranteed baseline.
4. **Pipelines** — `SegmentationPipeline` (mask shared by blur+replace), `FaceSwapPipeline` (detect→align→swap→blend). Fixed order: face swap first, then background ops on the swapped frame.
5. **Worker** — one thread per filter instance; latest-frame-wins drop policy; results mailbox.
6. **Model manager** — consent dialog → HuggingFace download with resume → SHA-256 verify → atomic rename into cache → hot-load without OBS restart.

### 5.2 Settings (per filter instance)

- Background mode: **Off / Transparent / Replace with image / Blur** (single selector as implemented in Plan 2; combinable toggles deferred) + image path (replace mode) + blur strength (1–4)
- Face swap: enable + source face image; **swap intensity** (0–100% opacity), **sharpness** (unsharp mask on face ROI), **preserve mouth region** toggle (v1: geometric lower-face region, no extra model; landmark-based refinement later)
- Model tier: **user-selectable dropdown** — Auto (default) / Lite / Standard / Quality. Auto picks per §7.1; a manual choice overrides it, so users can experiment (force Lite on weak hardware, force Quality to compare edge quality)
- Advanced mask tuning: **threshold** (binarize cutoff, 0 = off/soft mask), **contour filter** (drop disconnected mask blobs < % of frame, 0 = off), **feather** (edge softening), **temporal smoothing** (EMA factor, default 0.6)
- On processing failure: **Show camera feed (default)** / **Freeze last processed frame**
- AI disclosure watermark: ON by default (visible when face swap active; see §9)
- Performance: inference threads (default 2), advanced debug logging toggle
- Status line: current state (active / downloading / degraded / error reason) **+ measured processing fps**

## 6. Data flow & threading

Three thread roles; the OBS graphics thread never blocks on inference:

```
OBS graphics thread (video_render)
  ├─ render parent → texrender @ 192×192 (BGRA)
  ├─ gs_stage_texture → map → copy 147KB → input queue (SPSC, try-lock; drop if busy)
  ├─ mailbox.try_read() → latest published result
  └─ composite via effect shader:
        blur    → multi-pass Kawase blur, mask-selective
        replace → transparent: source drawn with mask as alpha
                  image: background image × (1−mask) + source × mask
        swap    → swapped-face texture blended over source
     (no result / failure → failure-mode behavior, §8)

Worker thread (1 per filter instance)
  ├─ pop latest frame (stale frames discarded)
  ├─ FaceSwapPipeline (if enabled): YuNet detect (bbox EMA) → 5-pt align →
  │    inswapper → inverse-affine paste-back (feathered ellipse from swap affine)
  ├─ SegmentationPipeline (if blur/replace enabled): seg model → mask →
  │    temporal EMA → guided-filter upsample to output resolution
  └─ publish result (double-buffered mailbox, atomic swap)

Download thread (spawned on demand)
  └─ consent dialog (UI thread) → HF download w/ resume → SHA-256 →
     atomic rename → worker hot-load
```

Invariants:

- Inference latency never stalls rendering; the effect may lag 1–2 frames (invisible for masks, acceptable for swap).
- Model switches happen under a worker-side mutex; the graphics thread only sees immutable published results.
- Segmentation always infers at model-native resolution (192×192); guided-filter edge refinement runs at mask resolution with the small staged frame as guide; final upscale to output resolution is GPU bilinear in the composite shader (full-res guided upsampling is deferred as a quality option).

## 7. Models, tiers & execution providers

### 7.1 Tier mapping

| Tier | Segmentation | In installer? | Face swap |
|---|---|---|---|
| Lite | MediaPipe Selfie Seg or SINet | yes | not offered |
| Standard (default) | PP-HumanSeg v2-Lite | yes | not offered |
| Quality | RVM MobileNetV3 | runtime download (GPL-3.0 terms shown) | YuNet (bundled, Apache) + inswapper_128 fp32 (runtime download, non-commercial terms shown) |

Auto = Standard on CPU-only machines; Quality when a usable GPU EP is detected and the required models are downloaded. The picker is user-visible (§5.2): Auto is the default, but the user can force any tier manually.

### 7.2 Execution provider strategy (ORT v1.28, runtime probe; Linux-only)

| Baseline (bundled) | Accelerated (runtime-detected) |
|---|---|
| ORT CPU (x86-64, AVX2) | CUDA via plugin-EP library (opt-in); MIGraphX (AMD) later |

Only CPU ORT ships in the installer. GPU EP libraries are registered at runtime when present. Probe failure at any level falls back silently to the tier below, with a one-line note in filter properties ("GPU acceleration unavailable — using CPU").

### 7.3 Model manager

- Cache dir: `~/.config/obs-cam-effects/models` (Linux-only)
- Downloads from HuggingFace; SHA-256 pinned in a plugin-controlled `manifest.json`
- Consent dialog shows license terms before each first download (GPL-3.0 for RVM; personal/non-commercial for inswapper)
- Resume support; hash mismatch or partial file → purge; atomic rename so a killed download never leaves a corrupt model
- Hot-load into the worker without restarting OBS

## 8. Failure handling

**User-selectable failure mode** ("On processing failure" setting):

- **Show camera feed (default):** call `obs_source_skip_video_filter()` — the source renders as if the filter were unattached. Rationale: for most users the camera feed is essential and the effect is cosmetic.
- **Freeze last processed frame:** re-render the last fully-composited output texture while the failure persists. If no frame was ever successfully processed, output **black** (never the raw feed — in this mode showing it even once defeats the privacy purpose). If the source resolution changes while frozen, draw the frozen frame scaled (content is already the safe processed output). Frozen output already contains the disclosure watermark when face swap was active.

Both modes show the failure reason in the properties status line and log with an `[obs-cam-effects]` prefix.

| Failure | Behavior (then apply failure mode for output) |
|---|---|
| Model missing/corrupt/not downloaded | "download required" status + button in properties |
| Download failure / hash mismatch | retry with resume; after 3 failures → error status + log; partial purged |
| ORT session creation fails on EP | fall back EP → CPU; if CPU fails → error status |
| Inference throws mid-stream | catch + log; 30 consecutive frame failures (~1s) → effect disabled until settings change |
| Worker thread dies | stale mailbox (>1s no result) detected → one worker restart attempt |
| Unsupported source (sync/texture-only) | "unsupported source" status |
| Settings change mid-stream | applied on worker side under mutex; graphics thread never blocks |

## 9. Compliance & responsible design

- **EU AI Act Art. 50** (obligations from 2026-08-02): AI-generated content must be disclosed. Face swap therefore ships with a visible "AI-generated" watermark toggle, **ON by default**, composited into the output (and thus into frozen frames).
- Consent notice in the face-swap properties: use only faces you have rights/consent to use.
- US TAKE IT DOWN Act (2025) and pending NO FAKES Act noted; distribution via runtime download with license terms keeps bundled assets fully permissive.
- Downloaded models' license terms are displayed pre-download and recorded in the model manifest.

## 10. Packaging & distribution

Built on obs-plugintemplate (CMake presets, `buildspec.json` pinned to oldest supported OBS minor — plugins built against newer libobs fail to load on older OBS). GitHub Actions CI: push → Linux build; semver tag → release tarball + SLSA attestation.

| Artifact | Specifics |
|---|---|
| `.deb` (CPack DEB) | System install: `/usr/lib/obs-plugins/` + `/usr/share/obs/obs-plugins/obs-cam-effects/` data dir; ORT bundled with `$ORIGIN` RPATH (system plugin dir) |
| `.tar.gz` (bin + data) | Per-user install: extract into `~/.config/obs-studio/plugins/obs-cam-effects/`; `bin/64bit/*.so` + `data/{effects,locale,models}`. CPU ORT bundled next to the plugin .so; GPU build is a runtime download, not in either artifact. |

Symbol export hygiene: `-Bsymbolic` + version script so bundled ORT/OpenCV symbols cannot collide with OBS or other plugins. CI: ubuntu-24.04 only.

## 11. Testing

- **Unit tests (libfx, GoogleTest, no OBS):** mask temporal smoothing convergence; guided-filter edge quality vs reference; Umeyama alignment sub-pixel stability; paste-back blend bounds; model manifest parsing; hash verification; tier-selection logic; EP probe fallback (mocked).
- **Golden-frame tests:** fixed inputs → each pipeline → perceptual-hash compare vs approved references (catches model/ORT version regressions).
- **Load/soak test:** headless libfx driver, 1080p30 for 30 min: fps ≥ 30 on CI CPU, stable RSS, drop policy engages under slowed worker.
- **Manual QA matrix (release checklist):** {deb-installed OBS, Flatpak OBS} × {webcam, media source, browser source (unsupported case)} × stacking with other filters.

## 12. Open risks & uncertainties

| Risk | Mitigation |
|---|---|
| inswapper license enforcement (gray-zone download pattern; ReActor repo was disabled by GitHub) | Terms shown pre-download; monitor InsightFace licensing; fallback: negotiate license or drop feature |
| GPU acceleration on Linux is the weakest story (NVIDIA user issues upstream) | CUDA strictly opt-in; CPU path must be excellent |
| Guided-filter edge quality at 192×192 on fine hair | Quality tier (RVM) exists for exactly this; validate in golden-frame tests |
| Face-swap fps on Linux GPUs via CUDA EP (estimated 30–60fps, unverified) | Early spike: port pipeline, measure on a discrete NVIDIA GPU before committing Quality-tier defaults |
| OBS ABI drift across versions | Pin build to oldest supported minor; test-load on latest in CI |
