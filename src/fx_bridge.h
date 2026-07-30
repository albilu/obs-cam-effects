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

/* Fetches the latest mask. Returns 1 if a mask exists, 0 otherwise.
 * On success *px points to an internal w*h uint8 buffer valid until the
 * next call, and *seq is the mask sequence number (increments per mask). */
int cam_fx_try_get_mask(cam_fx_t *fx, const uint8_t **px, int *w, int *h,
			uint64_t *seq);

/* 1 if a mask was published within max_age_ms, else 0. */
int cam_fx_is_fresh(cam_fx_t *fx, uint64_t max_age_ms);

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

/* License/notice text for a manifest entry id (empty if unknown).
 * Returns 0 on success, -1 if the id is unknown. */
int cam_fx_notice(cam_fx_t *fx, const char *id, char *buf, int buf_len);

/* Masks published per second (last completed 1s window). */
uint64_t cam_fx_fps(cam_fx_t *fx);

/* Active processing backend name ("CPU" or "CUDA") for status display.
 * Returns 0 on success, -1 on NULL fx. */
int cam_fx_backend(cam_fx_t *fx, char *buf, int buf_len);

#ifdef __cplusplus
}
#endif
