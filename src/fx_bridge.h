#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cam_fx cam_fx_t;

enum cam_fx_result_flags {
	CAM_FX_RESULT_MASK = 1 << 0,
	CAM_FX_RESULT_FRAME = 1 << 1,
	/* The validated frame was modified by the AI face-swap pipeline.
	 * This flag always implies CAM_FX_RESULT_FRAME. */
	CAM_FX_RESULT_AI = 1 << 2,
};

/* Creates the segmentation engine (loads model, starts worker thread).
 * lite_path / standard_path: bundled model paths. quality_path: cache
 * path of the runtime-downloaded quality model (may not exist yet;
 * Quality tier construction is deferred until the file appears).
 * Returns NULL on failure. */
cam_fx_t *cam_fx_create(const char *lite_path, const char *standard_path,
			const char *quality_path, int threads);

void cam_fx_destroy(cam_fx_t *fx);

/* Submits a packed BGRA frame for segmentation-only processing (any size;
 * copied internally). Never blocks. */
void cam_fx_submit(cam_fx_t *fx, const uint8_t *bgra, int w, int h,
		   int linesize);

/* Submits a packed BGRA frame at the target's full resolution for the
 * face-swap dataflow: the worker runs swap -> segmentation on it and
 * publishes frame+mask. Same non-blocking semantics as cam_fx_submit.
 * Use exactly one of the two per render tick (full-res when face swap is
 * active and in-cap, the segmentation-only stage otherwise — never both). */
void cam_fx_submit_full(cam_fx_t *fx, const uint8_t *bgra, int w, int h,
			int linesize);

/* Fetches validated mask/frame components from exactly one latest-worker
 * snapshot and returns their CAM_FX_RESULT_* bitmask. All output slots are
 * required. Returned components have positive dimensions and exact,
 * overflow-checked packed payload sizes; masks contain only finite values
 * before uint8 conversion. A component absent or invalid in that snapshot has
 * a NULL pointer and zero dimensions; a frame-only result is valid. On a
 * nonzero return, *seq is the one sequence shared by every returned component.
 * Returned pointers address bridge-owned packed buffers (mask: w*h bytes;
 * frame: BGRA w*h*4 bytes) valid until the next call to any result/mask/frame
 * retrieval function or cam_fx_destroy(). CAM_FX_RESULT_AI always implies
 * CAM_FX_RESULT_FRAME and records actual frame modification rather than merely
 * taking the full-resolution route. Invalid arguments and a latest result with
 * no valid components return 0 after clearing every provided output slot. */
int cam_fx_try_get_result(cam_fx_t *fx, const uint8_t **mask, int *mask_w, int *mask_h, const uint8_t **frame_bgra,
			  int *frame_w, int *frame_h, uint64_t *seq);

/* Fetches the latest mask. Returns 1 if a mask exists, 0 otherwise.
 * On success *px points to an internal w*h uint8 buffer valid until the
 * next result/mask/frame retrieval call, and *seq is the shared result
 * sequence number. */
int cam_fx_try_get_mask(cam_fx_t *fx, const uint8_t **px, int *w, int *h,
			uint64_t *seq);

/* 1 if the latest successful Worker result was published within max_age_ms,
 * else 0. This freshness is not mask-specific. */
int cam_fx_is_fresh(cam_fx_t *fx, uint64_t max_age_ms);

/* 1 if at least one valid mask has been observed since start, else 0. */
int cam_fx_has_mask(cam_fx_t *fx);

/* Tier: 0=auto, 1=lite, 2=standard, 3=quality. Cheap and idempotent:
 * rebuilds and hot-swaps the pipeline only when the resolved tier
 * changes (auto resolves to quality once the file is downloaded). */
void cam_fx_set_tier(cam_fx_t *fx, int tier);

/* Effective tier after resolution/fallback: 1=lite, 2=standard,
 * 3=quality. */
int cam_fx_tier_in_effect(cam_fx_t *fx);

/* 1 if the quality model file is present at the quality cache path. */
int cam_fx_quality_available(cam_fx_t *fx);

/* 1 if the GPU (CUDA) onnxruntime build is present in the plugin bin
 * dir. FILE presence only (deliberately NOT EpProbe::cudaAvailable):
 * the lib may be downloaded but is only loaded on the next OBS start. */
int cam_fx_gpu_build_present(cam_fx_t *fx);

/* Renders the "AI" disclosure badge into a freshly allocated,
 * tightly-packed RGBA buffer (transparent padding, semi-transparent
 * dark box, white glyphs); *w/*h receive the badge size (~52x36).
 * Free the returned buffer with bfree(). Returns NULL on failure. */
uint8_t *cam_fx_watermark_badge_rgba(int *w, int *h);

/* Advanced mask params (see fx::MaskParams). */
void cam_fx_set_mask_params(cam_fx_t *fx, float threshold, float contour,
			    float feather, float beta);

/* Starts a background download for the given manifest entry id
 * ("rvm_mobilenetv3_fp32" or "ort_cuda_ep_1.28.0"). Returns 0 on start,
 * -1 if busy/invalid. */
int cam_fx_start_download(cam_fx_t *fx, const char *id);

/* Download status: state string via buf (one of idle/downloading/
 * verifying/extracting/done/error), progress 0..1 or -1. Returns 0. */
int cam_fx_download_state(cam_fx_t *fx, char *buf, int buf_len,
			  double *progress);

/* Download error text (empty unless state == error). Returns 0. */
int cam_fx_download_error(cam_fx_t *fx, char *buf, int buf_len);

/* License/notice text for a manifest entry id (empty if unknown).
 * Returns 0 on success, -1 if the id is unknown. */
int cam_fx_notice(cam_fx_t *fx, const char *id, char *buf, int buf_len);

/* Usable results published per second (last completed 1s window).
 * Returns 0 for NULL fx. */
uint64_t cam_fx_fps(cam_fx_t *fx);

/* Active processing backend name ("CPU" or "CUDA") for status display.
 * Returns 0 on success, -1 on NULL fx. */
int cam_fx_backend(cam_fx_t *fx, char *buf, int buf_len);

/* Tells the bridge whether a background mode is active (1) or off (0).
 * When face swap runs with background off, the worker skips the
 * segmentation pass and publishes the swapped frame with a null mask
 * (the fs-only composite does not use a mask). Cheap and idempotent. */
void cam_fx_set_background_active(cam_fx_t *fx, int active);

/* --- Face swap --- */

/* Face-swap execution state for status display: "disabled" while off or
 * activation is not yet installed, "not loaded" when active without a CUDA
 * pipeline, "CUDA" while active, or "failed" after a terminal detector or
 * swapper runtime failure. A runtime failure remains enabled until explicit
 * disable. This is separate from cam_fx_backend(), which reports the
 * segmentation backend. Returns 0 on success, -1 on NULL fx. */
int cam_fx_faceswap_backend(cam_fx_t *fx, char *buf, int buf_len);

/* 1 only after a face-swap pipeline was successfully constructed and its
 * worker processor installed, else 0 (including NULL fx). Runtime Failed
 * remains enabled until the user disables face swap so worker staleness
 * drives failure UI. */
int cam_fx_faceswap_enabled(cam_fx_t *fx);

/* 1 after the active face-swap worker observes a terminal CUDA runtime
 * failure, else 0 (including NULL fx). Logical enablement remains true until
 * explicit disable. This runtime signal is distinct from static availability
 * and the face-swap backend status text. */
int cam_fx_faceswap_failed(cam_fx_t *fx);

/* Static availability: 1 when both face-swap models are in the models cache
 * AND the CUDA provider API is present, else 0. This does not imply successful
 * pipeline construction or activation; use cam_fx_faceswap_enabled for that. */
int cam_fx_faceswap_available(cam_fx_t *fx);

/* Why face swap is unavailable: "models not downloaded" /
 * "no GPU acceleration", or "" when available. Returns 1 when
 * something is missing, 0 when available. */
int cam_fx_faceswap_missing(cam_fx_t *fx, char *buf, int buf_len);

/* 1 if both face-swap models (inswapper fp16 or fp32, and w600k_r50)
 * exist in the models cache, regardless of GPU availability. */
int cam_fx_faceswap_models_present(cam_fx_t *fx);

/* Enables/disables face swap. The swap pipeline is built lazily on the first
 * enable and building is a no-op while unavailable. Explicitly disabling a
 * runtime-Failed pipeline retires it after installing segmentation-only, so
 * the next enable builds a fresh strict CUDA session. Healthy pipelines stay
 * cached across ordinary off/on toggles. */
void cam_fx_faceswap_set_enabled(cam_fx_t *fx, int enabled);

/* Sets or replaces the source face from an image file (jpg/png). Replacement
 * is fail-closed: every nonempty request clears the pending and live identity
 * before loading, and an unsuccessful replacement leaves no active source.
 * An empty path clears and succeeds without requiring models; NULL is invalid.
 * Returns 0 on success, -1 on failure (no usable face, unreadable file, models
 * missing). */
int cam_fx_faceswap_set_source(cam_fx_t *fx, const char *image_path);

/* Live swap params (see fx::FaceSwapParams). preserve_mouth is 0-100
 * (0 = off), converted to the 0..1 mouth-restore strength internally.
 * detect_every_frame: zero selects detection every two processed frames
 * (balanced default); nonzero selects detection every processed frame.
 * The AI disclosure badge is a filter-side post-composite overlay
 * (spec §9), not a pipeline param. */
void cam_fx_faceswap_set_params(cam_fx_t *fx, float intensity, float sharpness, int preserve_mouth,
				int detect_every_frame);

/* Fetches the latest full-route processed frame. Returns 1 if a frame exists,
 * 0 otherwise. This compatibility wrapper does not expose AI provenance; use
 * cam_fx_try_get_result for CAM_FX_RESULT_AI. On success *bgra points to an
 * internal w*h*4 uint8 buffer valid until the next result/mask/frame retrieval
 * call, and *seq is the result sequence number (shared with the mask). */
int cam_fx_try_get_frame(cam_fx_t *fx, const uint8_t **bgra, int *w,
			 int *h, uint64_t *seq);

/* Starts the 2-stage face-swap download (inswapper_128_fp16, then
 * w600k_r50); stages whose file already exists are skipped (an existing
 * fp32 inswapper_128 also satisfies the first stage). Returns 0 on
 * start or when nothing is needed, -1 if busy/invalid. */
int cam_fx_start_faceswap_download(cam_fx_t *fx);

/* Face-swap download status: current stage id via id_buf ("" when the
 * chain is idle), state string via state_buf (one of idle/downloading/
 * verifying/extracting/done/error), progress 0..1 or -1. Advances the
 * stage chain. Returns 0. */
int cam_fx_faceswap_download_state(cam_fx_t *fx, char *id_buf, int id_len,
				   char *state_buf, int state_len,
				   double *progress);

#ifdef __cplusplus
}
#endif
