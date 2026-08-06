#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cam_fx cam_fx_t;

/* Creates the segmentation engine (loads model, starts worker thread).
 * lite_path / standard_path: bundled model paths. quality_path: cache
 * path of the runtime-downloaded quality model (may not exist yet;
 * Quality tier construction is deferred until the file appears).
 * Returns NULL on failure. */
cam_fx_t *cam_fx_create(const char *lite_path, const char *standard_path,
			const char *quality_path, int threads);

void cam_fx_destroy(cam_fx_t *fx);

/* Submits a packed BGRA frame (any size; copied internally). Never blocks. */
void cam_fx_submit(cam_fx_t *fx, const uint8_t *bgra, int w, int h,
		   int linesize);

/* Submits a packed BGRA frame at the target's full resolution for the
 * face-swap dataflow: the worker runs swap -> segmentation on it and
 * publishes frame+mask. Same non-blocking semantics as cam_fx_submit.
 * Use exactly one of the two per render tick (full-res when face swap
 * is on, the small stage otherwise — never both). */
void cam_fx_submit_full(cam_fx_t *fx, const uint8_t *bgra, int w, int h,
			int linesize);

/* Fetches the latest mask. Returns 1 if a mask exists, 0 otherwise.
 * On success *px points to an internal w*h uint8 buffer valid until the
 * next call, and *seq is the mask sequence number (increments per mask). */
int cam_fx_try_get_mask(cam_fx_t *fx, const uint8_t **px, int *w, int *h,
			uint64_t *seq);

/* 1 if a mask was published within max_age_ms, else 0. */
int cam_fx_is_fresh(cam_fx_t *fx, uint64_t max_age_ms);

/* 1 if at least one mask has been published since start, else 0. */
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

/* Masks published per second (last completed 1s window). */
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

/* 1 when both face-swap models are in the models cache AND CUDA
 * acceleration is usable, else 0. */
int cam_fx_faceswap_available(cam_fx_t *fx);

/* Why face swap is unavailable: "models not downloaded" /
 * "no GPU acceleration", or "" when available. Returns 1 when
 * something is missing, 0 when available. */
int cam_fx_faceswap_missing(cam_fx_t *fx, char *buf, int buf_len);

/* Enables/disables face swap. The swap pipeline is built lazily on
 * the first enable and building is a no-op while unavailable. */
void cam_fx_faceswap_set_enabled(cam_fx_t *fx, int enabled);

/* Sets the source face from an image file (jpg/png). Returns 0 on
 * success, -1 on failure (no usable face, unreadable file, models
 * missing). */
int cam_fx_faceswap_set_source(cam_fx_t *fx, const char *image_path);

/* Live swap params (see fx::FaceSwapParams). */
void cam_fx_faceswap_set_params(cam_fx_t *fx, float intensity,
				float sharpness, int preserve_mouth,
				int watermark);

/* Fetches the latest processed (swapped) frame. Returns 1 if a frame
 * exists, 0 otherwise. On success *bgra points to an internal w*h*4
 * uint8 buffer valid until the next call, and *seq is the result
 * sequence number (shared with the mask seq). */
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
