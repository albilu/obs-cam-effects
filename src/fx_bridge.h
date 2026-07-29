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
