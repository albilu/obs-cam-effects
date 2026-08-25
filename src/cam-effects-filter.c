#include "cam-effects-filter.h"

#include "fx_bridge.h"

#include <obs-module.h>
#include <graphics/image-file.h>

#include <stdio.h>
#include <stdlib.h>

#define SETTING_MODE "mode"
#define SETTING_IMAGE_PATH "image_path"
#define SETTING_BLUR_STRENGTH "blur_strength"
#define SETTING_GREENSCREEN_COLOR "greenscreen_color"
#define SETTING_FAILURE "failure_mode"
#define SETTING_STATUS "status"
#define SETTING_TIER "tier"
#define SETTING_MASK_THRESHOLD "mask_threshold"
#define SETTING_MASK_CONTOUR "mask_contour"
#define SETTING_MASK_FEATHER "mask_feather"
#define SETTING_MASK_TEMPORAL "mask_temporal"
#define SETTING_FACE_SWAP "face_swap"
#define SETTING_FACE_IMAGE "face_image"
#define SETTING_SWAP_INTENSITY "swap_intensity"
#define SETTING_SWAP_SHARPNESS "swap_sharpness"
#define SETTING_SWAP_DETECT_EVERY_FRAME "swap_detect_every_frame"
#define SETTING_SWAP_PRESERVE_MOUTH "swap_preserve_mouth"
#define SETTING_SWAP_WATERMARK "swap_watermark"

#define STAGE_SIZE 192
#define MASK_STALE_MS 1000
/* Face-swap staging runs at the target's base size, capped at 1080p:
 * beyond that the full-res stage/download/submit cost is not real-time
 * viable, so face swap is skipped (background-only behavior). */
#define FULL_STAGE_MAX_W 1920
#define FULL_STAGE_MAX_H 1080

#include <stdatomic.h>

enum cam_mode { MODE_OFF, MODE_TRANSPARENT, MODE_IMAGE, MODE_BLUR, MODE_GREEN_SCREEN };
enum cam_failure { FAILURE_PASSTHROUGH, FAILURE_FREEZE };

static int parse_mode(const char *s)
{
	if (strcmp(s, "transparent") == 0)
		return MODE_TRANSPARENT;
	if (strcmp(s, "image") == 0)
		return MODE_IMAGE;
	if (strcmp(s, "blur") == 0)
		return MODE_BLUR;
	if (strcmp(s, "green_screen") == 0)
		return MODE_GREEN_SCREEN;
	return MODE_OFF;
}

static int parse_failure(const char *s)
{
	if (strcmp(s, "freeze") == 0)
		return FAILURE_FREEZE;
	return FAILURE_PASSTHROUGH;
}

/* 0=auto, 1=lite, 2=standard, 3=quality. */
static int parse_tier(const char *s)
{
	if (strcmp(s, "lite") == 0)
		return 1;
	if (strcmp(s, "standard") == 0)
		return 2;
	if (strcmp(s, "quality") == 0)
		return 3;
	return 0;
}

struct cam_effects_filter {
	obs_source_t *source;

	gs_texrender_t *stage_render;  /* STAGE_SIZE x STAGE_SIZE */
	gs_stagesurf_t *stage_surface; /* STAGE_SIZE x STAGE_SIZE BGRA */
	gs_texrender_t *full_render;   /* base-size staging (face swap) */
	gs_stagesurf_t *full_surface;  /* base-size BGRA (face swap) */
	uint32_t full_w;               /* current full staging width */
	uint32_t full_h;               /* current full staging height */
	gs_texture_t *frame_tex;       /* latest full-route frame */
	uint32_t frame_tex_w;
	uint32_t frame_tex_h;
	uint64_t frame_seq;         /* last uploaded swapped-frame seq */
	atomic_bool full_oversize;  /* target exceeds the 1080p cap */
	bool full_oversize_logged;  /* cap warning already logged */
	gs_texrender_t *out_render; /* committed frame-size composite + freeze */
	gs_texrender_t *out_work;   /* uncommitted frame-size composite */
	bool out_has_ai;            /* committed out_render contains AI-modified pixels */
	gs_texture_t *mask_tex;     /* STAGE_SIZE x STAGE_SIZE R8 */
	gs_effect_t *effect;        /* mask_composite.effect */
	gs_texrender_t *blur_a;     /* half-res ping */
	gs_texrender_t *blur_b;     /* half-res pong */
	gs_effect_t *blur_effect;   /* kawase_blur.effect */
	gs_texture_t *wm_tex;       /* lazy AI badge (graphics thread) */
	uint32_t wm_w;              /* badge width */
	uint32_t wm_h;              /* badge height */

	cam_fx_t *fx;

	atomic_int mode_id;           /* enum cam_mode */
	atomic_int failure_id;        /* enum cam_failure */
	atomic_int greenscreen_color; /* 0xAABBGGRR (OBS color property) */
	int blur_strength;
	int tier; /* 0=auto, 1=lite, 2=standard, 3=quality */
	char status[768];
	char *image_path;
	gs_image_file_t bg_image;
	bool bg_loaded;

	atomic_int face_swap;     /* 0/1 */
	atomic_int watermark_on;  /* 0/1: AI disclosure badge overlay */
	char *face_image_path;    /* current setting */
	char *face_image_applied; /* last path applied to the bridge */
};

/* Forward declarations (used before definition below). */
static void cam_effects_update(void *data, obs_data_t *settings);

static const char *cam_effects_get_name(void *type_data)
{
	UNUSED_PARAMETER(type_data);
	return "Camera Effects";
}

static void cam_effects_destroy_graphics(struct cam_effects_filter *filter)
{
	obs_enter_graphics();
	gs_texrender_destroy(filter->stage_render);
	gs_stagesurface_destroy(filter->stage_surface);
	gs_texrender_destroy(filter->full_render);
	gs_stagesurface_destroy(filter->full_surface);
	gs_texture_destroy(filter->frame_tex);
	gs_texrender_destroy(filter->out_render);
	gs_texrender_destroy(filter->out_work);
	gs_texrender_destroy(filter->blur_a);
	gs_texrender_destroy(filter->blur_b);
	gs_texture_destroy(filter->mask_tex);
	gs_effect_destroy(filter->effect);
	gs_effect_destroy(filter->blur_effect);
	gs_texture_destroy(filter->wm_tex);
	gs_image_file_free(&filter->bg_image);
	obs_leave_graphics();
}

static void cam_effects_load_effect(struct cam_effects_filter *filter)
{
	char *path = obs_module_file("effects/mask_composite.effect");
	obs_enter_graphics();
	if (path)
		filter->effect = gs_effect_create_from_file(path, NULL);
	obs_leave_graphics();
	bfree(path);

	char *blur_path = obs_module_file("effects/kawase_blur.effect");
	obs_enter_graphics();
	if (blur_path)
		filter->blur_effect = gs_effect_create_from_file(blur_path, NULL);
	obs_leave_graphics();
	bfree(blur_path);

	obs_enter_graphics();
	filter->mask_tex = gs_texture_create(STAGE_SIZE, STAGE_SIZE, GS_R8, 1, NULL, GS_DYNAMIC);
	obs_leave_graphics();
}

static void *cam_effects_create(obs_data_t *settings, obs_source_t *source)
{
	struct cam_effects_filter *filter = bzalloc(sizeof(struct cam_effects_filter));
	filter->source = source;
	filter->out_has_ai = false;
	atomic_init(&filter->full_oversize, false);

	obs_enter_graphics();
	filter->stage_render = gs_texrender_create(GS_BGRA, GS_ZS_NONE);
	filter->stage_surface = gs_stagesurface_create(STAGE_SIZE, STAGE_SIZE, GS_BGRA);
	filter->out_render = gs_texrender_create(GS_BGRA, GS_ZS_NONE);
	filter->out_work = gs_texrender_create(GS_BGRA, GS_ZS_NONE);
	filter->blur_a = gs_texrender_create(GS_BGRA, GS_ZS_NONE);
	filter->blur_b = gs_texrender_create(GS_BGRA, GS_ZS_NONE);
	obs_leave_graphics();

	cam_effects_load_effect(filter);
	cam_effects_update(filter, settings);
	return filter;
}

static void cam_effects_destroy(void *data)
{
	struct cam_effects_filter *filter = data;
	if (filter->fx)
		cam_fx_destroy(filter->fx);
	cam_effects_destroy_graphics(filter);
	bfree(filter->image_path);
	bfree(filter->face_image_path);
	bfree(filter->face_image_applied);
	bfree(filter);
}

/* Composes the status line shown in the properties dialog. Called from
 * update(), from properties() itself (OBS calls get_properties every
 * time the dialog opens) and from the button callbacks; the dialog
 * picks the text up when (re)opened — OBS properties are static while
 * open. */
static void cam_effects_compose_status(struct cam_effects_filter *filter)
{
	if (!filter->fx) {
		snprintf(filter->status, sizeof(filter->status), "Engine not started (select a background mode).");
		return;
	}

	/* 64: the provider-done state carries the restart hint
	 * ("done — restart OBS to enable GPU acceleration"). */
	char dl_state[64] = {0};
	double dl_prog = -1.0;
	cam_fx_download_state(filter->fx, dl_state, sizeof(dl_state), &dl_prog);
	char dl_text[192];
	if (strcmp(dl_state, "downloading") == 0 && dl_prog >= 0.0)
		snprintf(dl_text, sizeof(dl_text), "downloading %.0f%%", dl_prog * 100.0);
	else if (strcmp(dl_state, "error") == 0) {
		char dl_err[128] = {0};
		cam_fx_download_error(filter->fx, dl_err, sizeof(dl_err));
		snprintf(dl_text, sizeof(dl_text), "error: %s", dl_err[0] ? dl_err : "unknown");
	} else
		snprintf(dl_text, sizeof(dl_text), "%s", dl_state[0] ? dl_state : "idle");

	char backend[16] = "CPU";
	cam_fx_backend(filter->fx, backend, sizeof(backend));
	char fs_backend[16] = "disabled";
	cam_fx_faceswap_backend(filter->fx, fs_backend, sizeof(fs_backend));

	/* Fresh Worker results show result FPS. Without freshness, preserve
	 * the existing valid-mask history distinction for warm/stale text. */
	char fps_text[32];
	if (cam_fx_is_fresh(filter->fx, 2000))
		snprintf(fps_text, sizeof(fps_text), "%llu fps", (unsigned long long)cam_fx_fps(filter->fx));
	else if (cam_fx_has_mask(filter->fx))
		snprintf(fps_text, sizeof(fps_text), "—");
	else
		snprintf(fps_text, sizeof(fps_text), "warming up…");

	/* Face-swap state: download stage/progress while the chain is
	 * active, else availability and on/off. Unavailability never
	 * affects the background modes above. */
	char fs_id[64] = {0};
	char fs_state[32] = {0};
	double fs_prog = -1.0;
	cam_fx_faceswap_download_state(filter->fx, fs_id, sizeof(fs_id), fs_state, sizeof(fs_state), &fs_prog);
	bool fs_busy = fs_id[0] != '\0' && (strcmp(fs_state, "downloading") == 0 ||
					    strcmp(fs_state, "verifying") == 0 || strcmp(fs_state, "extracting") == 0);
	char fs_text[192];
	if (fs_busy && fs_prog >= 0.0)
		snprintf(fs_text, sizeof(fs_text), "Face swap: downloading %s %.0f%%", fs_id, fs_prog * 100.0);
	else if (fs_busy)
		snprintf(fs_text, sizeof(fs_text), "Face swap: %s %s", fs_state, fs_id);
	else if (fs_id[0] != '\0' && strcmp(fs_state, "error") == 0) {
		char fs_err[96] = {0};
		cam_fx_download_error(filter->fx, fs_err, sizeof(fs_err));
		snprintf(fs_text, sizeof(fs_text), "Face swap: download error: %s", fs_err[0] ? fs_err : "unknown");
	} else if (cam_fx_faceswap_available(filter->fx)) {
		bool face_swap = atomic_load_explicit(&filter->face_swap, memory_order_relaxed) != 0;
		if (!face_swap)
			snprintf(fs_text, sizeof(fs_text), "Face swap: off");
		else if (!cam_fx_faceswap_enabled(filter->fx))
			snprintf(fs_text, sizeof(fs_text), "Face swap: failed to initialize");
		else
			snprintf(fs_text, sizeof(fs_text), "Face swap: on (%s)%s", fs_backend,
				 atomic_load_explicit(&filter->full_oversize, memory_order_relaxed)
					 ? " — source too large, background only"
					 : "");
	} else {
		char reason[96] = {0};
		cam_fx_faceswap_missing(filter->fx, reason, sizeof(reason));
		snprintf(fs_text, sizeof(fs_text), "Face swap: unavailable (%s)", reason[0] ? reason : "unknown");
	}

	snprintf(filter->status, sizeof(filter->status), "Quality model: %s | Download: %s | Backend: %s | %s | %s",
		 cam_fx_quality_available(filter->fx) ? "downloaded" : "not downloaded", dl_text, backend, fps_text,
		 fs_text);
}

static bool cam_effects_download_rvm_clicked(obs_properties_t *props, obs_property_t *property, void *data)
{
	UNUSED_PARAMETER(props);
	UNUSED_PARAMETER(property);
	struct cam_effects_filter *filter = data;
	if (filter && filter->fx) {
		cam_fx_start_download(filter->fx, "rvm_mobilenetv3_fp32");
		cam_effects_compose_status(filter);
	}
	return true; /* refresh properties (status shows new state) */
}

static bool cam_effects_download_cuda_clicked(obs_properties_t *props, obs_property_t *property, void *data)
{
	UNUSED_PARAMETER(props);
	UNUSED_PARAMETER(property);
	struct cam_effects_filter *filter = data;
	if (filter && filter->fx) {
		cam_fx_start_download(filter->fx, "ort_cuda_ep_1.28.0");
		cam_effects_compose_status(filter);
	}
	return true;
}

static bool cam_effects_download_faceswap_clicked(obs_properties_t *props, obs_property_t *property, void *data)
{
	UNUSED_PARAMETER(props);
	UNUSED_PARAMETER(property);
	struct cam_effects_filter *filter = data;
	if (filter && filter->fx) {
		cam_fx_start_faceswap_download(filter->fx);
		cam_effects_compose_status(filter);
	}
	return true;
}

static void cam_effects_update(void *data, obs_data_t *settings)
{
	struct cam_effects_filter *filter = data;

	char *mode = bstrdup(obs_data_get_string(settings, SETTING_MODE));
	char *failure = bstrdup(obs_data_get_string(settings, SETTING_FAILURE));
	atomic_store_explicit(&filter->mode_id, parse_mode(mode), memory_order_relaxed);
	atomic_store_explicit(&filter->failure_id, parse_failure(failure), memory_order_relaxed);
	atomic_store_explicit(&filter->greenscreen_color, (int)obs_data_get_int(settings, SETTING_GREENSCREEN_COLOR),
			      memory_order_relaxed);
	bfree(mode);
	bfree(failure);

	bfree(filter->image_path);
	filter->image_path = bstrdup(obs_data_get_string(settings, SETTING_IMAGE_PATH));
	filter->blur_strength = (int)obs_data_get_int(settings, SETTING_BLUR_STRENGTH);
	filter->tier = parse_tier(obs_data_get_string(settings, SETTING_TIER));
	float mask_threshold = (float)obs_data_get_double(settings, SETTING_MASK_THRESHOLD);
	float mask_contour = (float)obs_data_get_double(settings, SETTING_MASK_CONTOUR);
	float mask_feather = (float)obs_data_get_double(settings, SETTING_MASK_FEATHER);
	float mask_temporal = (float)obs_data_get_double(settings, SETTING_MASK_TEMPORAL);

	bool face_swap = obs_data_get_bool(settings, SETTING_FACE_SWAP);
	atomic_store_explicit(&filter->face_swap, face_swap ? 1 : 0, memory_order_relaxed);
	bfree(filter->face_image_path);
	filter->face_image_path = bstrdup(obs_data_get_string(settings, SETTING_FACE_IMAGE));
	float swap_intensity = (float)obs_data_get_double(settings, SETTING_SWAP_INTENSITY);
	float swap_sharpness = (float)obs_data_get_double(settings, SETTING_SWAP_SHARPNESS);
	bool swap_detect_every_frame = obs_data_get_bool(settings, SETTING_SWAP_DETECT_EVERY_FRAME);
	int swap_preserve_mouth = (int)obs_data_get_int(settings, SETTING_SWAP_PRESERVE_MOUTH);
	bool swap_watermark = obs_data_get_bool(settings, SETTING_SWAP_WATERMARK);
	atomic_store_explicit(&filter->watermark_on, swap_watermark ? 1 : 0, memory_order_relaxed);

	/* Free any previous background image (destroys its texture, so
	 * inside the graphics lock). */
	obs_enter_graphics();
	gs_image_file_free(&filter->bg_image);
	filter->bg_loaded = false;
	obs_leave_graphics();

	/* (Re)load the background image if the path changed and mode
	 * needs it. Decode (disk I/O) outside the graphics lock; only
	 * the texture upload runs inside. */
	if (atomic_load_explicit(&filter->mode_id, memory_order_relaxed) == MODE_IMAGE &&
	    filter->image_path[0] != '\0') {
		gs_image_file_init(&filter->bg_image, filter->image_path);
		obs_enter_graphics();
		gs_image_file_init_texture(&filter->bg_image);
		filter->bg_loaded = filter->bg_image.texture != NULL;
		obs_leave_graphics();
	}

	/* Create the inference engine lazily on first non-off mode (or
	 * when face swap is on: it shares the same engine). cam_fx_create
	 * touches no graphics resources (ORT sessions + worker thread
	 * only) and can take seconds (model load, first CUDA EP init) —
	 * keep it out of the graphics lock so the render thread is not
	 * stalled; the lock only publishes filter->fx to the render
	 * thread. */
	if (!filter->fx && (atomic_load_explicit(&filter->mode_id, memory_order_relaxed) != MODE_OFF || face_swap)) {
		char *lite = obs_module_file("models/selfie_segmentation.onnx");
		char *standard = obs_module_file("models/pphumanseg_fp32.onnx");
		const char *home = getenv("HOME");
		char quality[1024];
		snprintf(quality, sizeof(quality),
			 "%s/.config/obs-cam-effects/models/"
			 "rvm_mobilenetv3_fp32.onnx",
			 home ? home : ".");
		cam_fx_t *fx = NULL;
		if (lite && standard)
			fx = cam_fx_create(lite, standard, quality, 2);
		obs_enter_graphics();
		filter->fx = fx;
		obs_leave_graphics();
		bfree(lite);
		bfree(standard);
	}

	/* Apply tier + mask params on every update: cheap and idempotent
	 * (the bridge rebuilds the pipeline only when the effective tier
	 * changes). */
	if (filter->fx) {
		cam_fx_set_tier(filter->fx, filter->tier);
		cam_fx_set_mask_params(filter->fx, mask_threshold, mask_contour, mask_feather, mask_temporal);
		/* fs + background off skips the per-frame segmentation
		 * run in the worker (the composite draws the swapped
		 * frame directly). */
		cam_fx_set_background_active(
			filter->fx, atomic_load_explicit(&filter->mode_id, memory_order_relaxed) != MODE_OFF ? 1 : 0);

		/* Face swap: params every call (cheap); the source
		 * embedding only when the path changed (it runs
		 * detect+ArcFace on this thread); enable last so a
		 * lazy build picks up params + pending source. */
		cam_fx_faceswap_set_params(filter->fx, swap_intensity, swap_sharpness, swap_preserve_mouth,
					   swap_detect_every_frame ? 1 : 0);
		bool src_changed = (filter->face_image_applied == NULL) != (filter->face_image_path == NULL) ||
				   (filter->face_image_applied && filter->face_image_path &&
				    strcmp(filter->face_image_applied, filter->face_image_path) != 0);
		if (src_changed) {
			int rc = cam_fx_faceswap_set_source(filter->fx, filter->face_image_path);
			/* A failed replacement clears the bridge source. Invalidate
			 * the cache so selecting the previous path restores it. */
			if (rc == 0) {
				bfree(filter->face_image_applied);
				filter->face_image_applied = bstrdup(filter->face_image_path);
			} else {
				bfree(filter->face_image_applied);
				filter->face_image_applied = NULL;
			}
		}
		cam_fx_faceswap_set_enabled(filter->fx, face_swap ? 1 : 0);
	}
	cam_effects_compose_status(filter);
}

static void cam_effects_get_defaults(obs_data_t *settings)
{
	obs_data_set_default_string(settings, SETTING_MODE, "transparent");
	obs_data_set_default_string(settings, SETTING_FAILURE, "passthrough");
	obs_data_set_default_int(settings, SETTING_BLUR_STRENGTH, 2);
	obs_data_set_default_int(settings, SETTING_GREENSCREEN_COLOR, 0xFF00FF00);
	obs_data_set_default_string(settings, SETTING_TIER, "auto");
	obs_data_set_default_double(settings, SETTING_MASK_THRESHOLD, 0.0);
	obs_data_set_default_double(settings, SETTING_MASK_CONTOUR, 0.0);
	obs_data_set_default_double(settings, SETTING_MASK_FEATHER, 0.0);
	obs_data_set_default_double(settings, SETTING_MASK_TEMPORAL, 0.6);
	obs_data_set_default_bool(settings, SETTING_FACE_SWAP, false);
	obs_data_set_default_string(settings, SETTING_FACE_IMAGE, "");
	obs_data_set_default_double(settings, SETTING_SWAP_INTENSITY, 1.0);
	obs_data_set_default_double(settings, SETTING_SWAP_SHARPNESS, 0.0);
	obs_data_set_default_bool(settings, SETTING_SWAP_DETECT_EVERY_FRAME, false);
	obs_data_set_default_int(settings, SETTING_SWAP_PRESERVE_MOUTH, 0);
	obs_data_set_default_bool(settings, SETTING_SWAP_WATERMARK, true);
}

static obs_properties_t *cam_effects_properties(void *data)
{
	struct cam_effects_filter *filter = data;
	obs_properties_t *props = obs_properties_create();

	obs_property_t *mode = obs_properties_add_list(props, SETTING_MODE, "Background", OBS_COMBO_TYPE_LIST,
						       OBS_COMBO_FORMAT_STRING);
	obs_property_list_add_string(mode, "Off", "off");
	obs_property_list_add_string(mode, "Transparent", "transparent");
	obs_property_list_add_string(mode, "Replace with image", "image");
	obs_property_list_add_string(mode, "Blur", "blur");
	obs_property_list_add_string(mode, "Green Screen", "green_screen");

	obs_properties_add_color(props, SETTING_GREENSCREEN_COLOR, "Green screen color");

	obs_properties_add_path(props, SETTING_IMAGE_PATH, "Background image", OBS_PATH_FILE,
				"Images (*.png *.jpg *.jpeg *.bmp)", NULL);
	obs_properties_add_int_slider(props, SETTING_BLUR_STRENGTH, "Blur strength", 1, 7, 1);

	obs_property_t *tier = obs_properties_add_list(props, SETTING_TIER, "Segmentation model", OBS_COMBO_TYPE_LIST,
						       OBS_COMBO_FORMAT_STRING);
	obs_property_list_add_string(tier, "Auto (recommended)", "auto");
	obs_property_list_add_string(tier, "Lite (MediaPipe - fastest)", "lite");
	obs_property_list_add_string(tier, "Standard (PP-HumanSeg - balanced)", "standard");
	obs_property_list_add_string(tier, "Quality (RVM - best edges)", "quality");

	obs_properties_add_float_slider(props, SETTING_MASK_THRESHOLD, "Mask threshold", 0.0, 1.0, 0.01);
	obs_properties_add_float_slider(props, SETTING_MASK_CONTOUR, "Mask contour cleanup", 0.0, 0.5, 0.01);
	obs_properties_add_float_slider(props, SETTING_MASK_FEATHER, "Mask feather", 0.0, 8.0, 0.5);
	obs_properties_add_float_slider(props, SETTING_MASK_TEMPORAL, "Temporal smoothing", 0.0, 0.95, 0.05);

	obs_property_t *fm = obs_properties_add_list(props, SETTING_FAILURE, "On processing failure",
						     OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_STRING);
	obs_property_list_add_string(fm, "Show camera feed", "passthrough");
	obs_property_list_add_string(fm, "Freeze last processed frame", "freeze");

	char notice[512] = {0};
	if (filter && filter->fx)
		cam_fx_notice(filter->fx, "rvm_mobilenetv3_fp32", notice, sizeof(notice));
	if (notice[0] == '\0')
		snprintf(notice, sizeof(notice), "%s",
			 "Robust Video Matting (RVM) MobileNetV3, licensed "
			 "GPL-3.0-only. By downloading you accept the "
			 "license terms.");
	obs_properties_add_text(props, "rvm_notice", notice, OBS_TEXT_INFO);

	obs_property_t *rvm_btn = obs_properties_add_button2(props, "download_btn",
							     "Download Quality model (GPL-3.0, 15 MB)",
							     cam_effects_download_rvm_clicked, filter);
	if (filter && filter->fx && cam_fx_quality_available(filter->fx)) {
		obs_property_set_enabled(rvm_btn, false);
		obs_property_set_description(rvm_btn, "Quality model: downloaded ✓");
	}
	obs_property_t *cuda_btn = obs_properties_add_button2(props, "download_cuda_btn",
							      "Download GPU acceleration (MIT, ~240 MB)",
							      cam_effects_download_cuda_clicked, filter);
	if (filter && filter->fx && cam_fx_gpu_build_present(filter->fx)) {
		obs_property_set_enabled(cuda_btn, false);
		obs_property_set_description(cuda_btn, "GPU acceleration: downloaded (restart OBS to enable)");
	}

	/* --- Face swap --- */
	bool fs_available = filter && filter->fx && cam_fx_faceswap_available(filter->fx) == 1;
	obs_property_t *fs_toggle = obs_properties_add_bool(props, SETTING_FACE_SWAP, "Face swap (GPU required)");
	if (filter && filter->fx && !fs_available)
		obs_property_set_enabled(fs_toggle, false);
	obs_properties_add_path(props, SETTING_FACE_IMAGE, "Source face image", OBS_PATH_FILE,
				"Images (*.png *.jpg *.jpeg *.bmp)", NULL);
	obs_properties_add_float_slider(props, SETTING_SWAP_INTENSITY, "Swap intensity", 0.0, 1.0, 0.05);
	obs_properties_add_float_slider(props, SETTING_SWAP_SHARPNESS, "Swap sharpness", 0.0, 1.0, 0.05);
	obs_properties_add_bool(props, SETTING_SWAP_DETECT_EVERY_FRAME, "High-quality tracking (detect every frame)");
	obs_properties_add_int_slider(props, SETTING_SWAP_PRESERVE_MOUTH, "Preserve mouth region", 0, 100, 1);
	obs_properties_add_bool(props, SETTING_SWAP_WATERMARK, "AI disclosure badge (recommended)");

	char fs_notice[768];
	snprintf(fs_notice, sizeof(fs_notice), "%s",
		 "Face swap models are licensed for NON-COMMERCIAL research "
		 "use only (InsightFace). Use only faces you have "
		 "rights/consent to use. Output is AI-generated; a "
		 "disclosure badge is applied by default (EU AI Act "
		 "Art. 50).");
	if (filter && filter->fx && !fs_available) {
		char reason[128] = {0};
		cam_fx_faceswap_missing(filter->fx, reason, sizeof(reason));
		size_t len = strlen(fs_notice);
		snprintf(fs_notice + len, sizeof(fs_notice) - len, " Currently unavailable: %s.",
			 reason[0] ? reason : "unknown");
	}
	obs_properties_add_text(props, "faceswap_notice", fs_notice, OBS_TEXT_INFO);
	obs_property_t *fs_dl_btn = obs_properties_add_button2(props, "download_faceswap_btn",
							       "Download face swap models (non-commercial, ~450 MB)",
							       cam_effects_download_faceswap_clicked, filter);
	if (filter && filter->fx && cam_fx_faceswap_models_present(filter->fx)) {
		obs_property_set_enabled(fs_dl_btn, false);
		obs_property_set_description(fs_dl_btn, "Face swap models: downloaded ✓");
	}

	/* OBS calls get_properties every time the properties dialog
	 * opens: recompose the status from live bridge state so the
	 * dialog never shows a stale creation-time snapshot (fps is
	 * legitimately 0 at creation). */
	if (filter)
		cam_effects_compose_status(filter);
	const char *status = (filter && filter->status[0]) ? filter->status : "Status unavailable";
	obs_property_t *status_prop = obs_properties_add_text(props, SETTING_STATUS, status, OBS_TEXT_INFO);
	obs_property_set_description(status_prop, status);
	return props;
}

/* Render the parent source into the small staging surface and submit. */
static void cam_effects_stage(struct cam_effects_filter *filter, obs_source_t *target)
{
	uint32_t tw = obs_source_get_base_width(target);
	uint32_t th = obs_source_get_base_height(target);
	if (tw == 0 || th == 0)
		return;

	gs_texrender_reset(filter->stage_render);
	if (!gs_texrender_begin(filter->stage_render, STAGE_SIZE, STAGE_SIZE))
		return;
	struct vec4 clear;
	vec4_zero(&clear);
	gs_clear(GS_CLEAR_COLOR, &clear, 0.0f, 0);
	gs_ortho(0.0f, (float)STAGE_SIZE, 0.0f, (float)STAGE_SIZE, -100.0f, 100.0f);
	gs_blend_state_push();
	gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);
	gs_matrix_push();
	gs_matrix_scale3f((float)STAGE_SIZE / (float)tw, (float)STAGE_SIZE / (float)th, 1.0f);
	/* The target always has filters (ours), so obs_source_main_render
	 * skips obs_source_default_render and sources without their own
	 * effect loop (e.g. image sources) would draw with whatever
	 * effect is current — none here. Provide the default effect's
	 * Draw pass around the target render, like default_render. */
	gs_effect_t *def = obs_get_base_effect(OBS_EFFECT_DEFAULT);
	while (gs_effect_loop(def, "Draw"))
		obs_source_video_render(target);
	gs_matrix_pop();
	gs_blend_state_pop();
	gs_texrender_end(filter->stage_render);

	gs_stage_texture(filter->stage_surface, gs_texrender_get_texture(filter->stage_render));
	uint8_t *data = NULL;
	uint32_t linesize = 0;
	if (gs_stagesurface_map(filter->stage_surface, &data, &linesize)) {
		cam_fx_submit(filter->fx, data, STAGE_SIZE, STAGE_SIZE, (int)linesize);
		gs_stagesurface_unmap(filter->stage_surface);
	}
}

/* Render the parent source at its base size into the full-res staging
 * surface and submit it to the face-swap worker (which runs
 * swap -> segmentation on it and publishes frame+mask). This is the
 * face-swap counterpart of cam_effects_stage — when face swap is on
 * it is the ONLY submission (the 192 stage would double-submit; the
 * worker derives the mask from the swapped full-res frame itself).
 * full_render/full_surface are created lazily here; video_render runs
 * on the graphics thread. */
static void cam_effects_stage_full(struct cam_effects_filter *filter, obs_source_t *target, uint32_t w, uint32_t h)
{
	if (!filter->full_render)
		filter->full_render = gs_texrender_create(GS_BGRA, GS_ZS_NONE);
	if (!filter->full_render)
		return;
	if (!filter->full_surface || filter->full_w != w || filter->full_h != h) {
		gs_stagesurface_destroy(filter->full_surface);
		filter->full_surface = gs_stagesurface_create(w, h, GS_BGRA);
		filter->full_w = w;
		filter->full_h = h;
	}
	if (!filter->full_surface)
		return;

	gs_texrender_reset(filter->full_render);
	if (!gs_texrender_begin(filter->full_render, w, h))
		return;
	struct vec4 clear;
	vec4_zero(&clear);
	gs_clear(GS_CLEAR_COLOR, &clear, 0.0f, 0);
	gs_ortho(0.0f, (float)w, 0.0f, (float)h, -100.0f, 100.0f);
	gs_blend_state_push();
	gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);
	/* Same default-effect wrap as cam_effects_stage: sources without
	 * their own effect loop (e.g. image sources) would otherwise
	 * draw with no effect active. */
	gs_effect_t *def = obs_get_base_effect(OBS_EFFECT_DEFAULT);
	while (gs_effect_loop(def, "Draw"))
		obs_source_video_render(target);
	gs_blend_state_pop();
	gs_texrender_end(filter->full_render);

	gs_stage_texture(filter->full_surface, gs_texrender_get_texture(filter->full_render));
	uint8_t *data = NULL;
	uint32_t linesize = 0;
	if (gs_stagesurface_map(filter->full_surface, &data, &linesize)) {
		cam_fx_submit_full(filter->fx, data, (int)w, (int)h, (int)linesize);
		gs_stagesurface_unmap(filter->full_surface);
	}
}

/* Upload the latest full-route frame to frame_tex (created/recreated on
 * size change; uploaded only when the sequence advanced). Returns
 * frame_tex, or NULL when unavailable. The bridge buffer is packed
 * BGRA (linesize == w*4) and stays valid until the next result
 * retrieval call. */
static gs_texture_t *cam_effects_frame_tex(struct cam_effects_filter *filter, const uint8_t *data, int w, int h,
					   uint64_t seq)
{
	if (w <= 0 || h <= 0)
		return NULL;
	if (!filter->frame_tex || filter->frame_tex_w != (uint32_t)w || filter->frame_tex_h != (uint32_t)h) {
		gs_texture_destroy(filter->frame_tex);
		filter->frame_tex = gs_texture_create((uint32_t)w, (uint32_t)h, GS_BGRA, 1, NULL, GS_DYNAMIC);
		filter->frame_tex_w = (uint32_t)w;
		filter->frame_tex_h = (uint32_t)h;
		filter->frame_seq = 0; /* force re-upload below */
	}
	if (!filter->frame_tex)
		return NULL;
	if (seq != filter->frame_seq) {
		gs_texture_set_image(filter->frame_tex, data, (uint32_t)w * 4, false);
		filter->frame_seq = seq;
	}
	return filter->frame_tex;
}

/* Runs `passes` Kawase blur iterations at half resolution. The source
 * is src_tex when given (face swap: the full-route frame), otherwise the
 * target is rendered directly (background-only path, unchanged).
 * Returns the texture containing the blurred result, or NULL. */
static gs_texture_t *cam_effects_blur(struct cam_effects_filter *filter, obs_source_t *target, gs_texture_t *src_tex,
				      uint32_t w, uint32_t h)
{
	uint32_t bw = w / 2 > 0 ? w / 2 : 1;
	uint32_t bh = h / 2 > 0 ? h / 2 : 1;

	/* Downsample the source into blur_a using the default effect. */
	gs_texrender_reset(filter->blur_a);
	if (!gs_texrender_begin(filter->blur_a, bw, bh))
		return NULL;
	struct vec4 clear;
	vec4_zero(&clear);
	gs_clear(GS_CLEAR_COLOR, &clear, 0.0f, 0);
	gs_ortho(0.0f, (float)bw, 0.0f, (float)bh, -100.0f, 100.0f);
	gs_blend_state_push();
	gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);
	gs_effect_t *def = obs_get_base_effect(OBS_EFFECT_DEFAULT);
	if (src_tex) {
		gs_effect_set_texture(gs_effect_get_param_by_name(def, "image"), src_tex);
		while (gs_effect_loop(def, "Draw"))
			gs_draw_sprite(src_tex, 0, bw, bh);
	} else {
		gs_matrix_push();
		gs_matrix_scale3f((float)bw / (float)w, (float)bh / (float)h, 1.0f);
		/* Like the staging path: sources without their own effect
		 * loop (e.g. image sources) need the default effect active. */
		while (gs_effect_loop(def, "Draw"))
			obs_source_video_render(target);
		gs_matrix_pop();
	}
	gs_blend_state_pop();
	gs_texrender_end(filter->blur_a);

	gs_texture_t *src = gs_texrender_get_texture(filter->blur_a);
	for (int i = 0; i < filter->blur_strength; i++) {
		gs_texrender_t *dst = (i % 2 == 0) ? filter->blur_b : filter->blur_a;
		gs_texrender_reset(dst);
		if (!gs_texrender_begin(dst, bw, bh))
			return src;
		gs_effect_set_texture(gs_effect_get_param_by_name(filter->blur_effect, "image"), src);
		struct vec2 texel = {1.0f / (float)bw, 1.0f / (float)bh};
		gs_effect_set_vec2(gs_effect_get_param_by_name(filter->blur_effect, "texel"), &texel);
		gs_effect_set_float(gs_effect_get_param_by_name(filter->blur_effect, "iteration"), (float)i);
		gs_clear(GS_CLEAR_COLOR, &clear, 0.0f, 0);
		gs_ortho(0.0f, (float)bw, 0.0f, (float)bh, -100.0f, 100.0f);
		while (gs_effect_loop(filter->blur_effect, "Draw"))
			gs_draw_sprite(src, 0, bw, bh);
		gs_texrender_end(dst);
		src = gs_texrender_get_texture(dst);
	}
	return src;
}

/* Lazily create the badge texture on the graphics thread (first fs
 * render with the badge on). The pixels come from the fx core via the
 * bridge (tightly-packed RGBA, bfree'd after upload). */
static bool cam_effects_ensure_watermark(struct cam_effects_filter *filter)
{
	if (filter->wm_tex)
		return true;
	int bw = 0, bh = 0;
	uint8_t *px = cam_fx_watermark_badge_rgba(&bw, &bh);
	if (!px || bw <= 0 || bh <= 0) {
		bfree(px);
		return false;
	}
	const uint8_t *levels[1] = {px};
	filter->wm_tex = gs_texture_create((uint32_t)bw, (uint32_t)bh, GS_RGBA, 1, levels, 0);
	bfree(px);
	if (!filter->wm_tex)
		return false;
	filter->wm_w = (uint32_t)bw;
	filter->wm_h = (uint32_t)bh;
	return true;
}

static bool cam_effects_watermark_fits(struct cam_effects_filter *filter, uint32_t w, uint32_t h)
{
	if (!filter->wm_tex || filter->wm_w == 0 || filter->wm_h == 0 || filter->wm_w > w || filter->wm_h > h)
		return false;
	uint32_t margin = h / 40;
	return margin <= w - filter->wm_w && margin <= h - filter->wm_h;
}

/* Post-composite AI disclosure badge (spec §9): drawn by the FILTER on
 * top of the final output, so it survives every background mode — the
 * old worker-side stamp sat in the background region (mask ~0) of
 * bg x (1-mask) + frame x mask and was blended away. Also drawn over
 * the frozen out_render so the freeze failure mode keeps the
 * disclosure (spec §8/§9). Never drawn on the passthrough path (raw
 * feed = no AI content). */
static void cam_effects_draw_watermark(struct cam_effects_filter *filter, uint32_t w, uint32_t h)
{
	if (!cam_effects_watermark_fits(filter, w, h))
		return;
	uint32_t wm_w = filter->wm_w, wm_h = filter->wm_h;
	uint32_t margin = h / 40;
	gs_matrix_push();
	gs_matrix_translate3f((float)(w - wm_w - margin), (float)(h - wm_h - margin), 0.0f);
	gs_effect_t *def = obs_get_base_effect(OBS_EFFECT_DEFAULT);
	gs_eparam_t *image = gs_effect_get_param_by_name(def, "image");
	gs_effect_set_texture(image, filter->wm_tex);
	while (gs_effect_loop(def, "Draw"))
		gs_draw_sprite(filter->wm_tex, 0, wm_w, wm_h);
	gs_matrix_pop();
}

static bool cam_effects_commit_black(struct cam_effects_filter *filter, uint32_t w, uint32_t h)
{
	if (!filter->out_work || w == 0 || h == 0)
		return false;
	gs_texrender_reset(filter->out_work);
	if (!gs_texrender_begin(filter->out_work, w, h))
		return false;
	struct vec4 clear;
	vec4_zero(&clear);
	gs_clear(GS_CLEAR_COLOR, &clear, 0.0f, 0);
	gs_texrender_end(filter->out_work);
	gs_texrender_t *committed = filter->out_render;
	filter->out_render = filter->out_work;
	filter->out_work = committed;
	filter->out_has_ai = false;
	return true;
}

/* The only path that displays out_render. If enabled disclosure cannot be
 * shown, passthrough uses raw while freeze commits privacy-safe black. */
static void cam_effects_draw_output(struct cam_effects_filter *filter, uint32_t w, uint32_t h, bool freeze)
{
	bool badge_required = filter->out_has_ai &&
			      atomic_load_explicit(&filter->watermark_on, memory_order_relaxed) != 0;
	if (badge_required && (!cam_effects_ensure_watermark(filter) || !cam_effects_watermark_fits(filter, w, h))) {
		if (!freeze) {
			obs_source_skip_video_filter(filter->source);
			return;
		}
		if (!cam_effects_commit_black(filter, w, h))
			return;
		badge_required = false;
	}
	if (!filter->out_render)
		return;
	gs_texture_t *tex = gs_texrender_get_texture(filter->out_render);
	if (!tex)
		return;
	gs_effect_t *def = obs_get_base_effect(OBS_EFFECT_DEFAULT);
	gs_eparam_t *image = gs_effect_get_param_by_name(def, "image");
	gs_effect_set_texture(image, tex);
	while (gs_effect_loop(def, "Draw"))
		gs_draw_sprite(tex, 0, w, h);
	if (badge_required)
		cam_effects_draw_watermark(filter, w, h);
}

static void cam_effects_video_render(void *data, gs_effect_t *effect)
{
	UNUSED_PARAMETER(effect);
	struct cam_effects_filter *filter = data;
	int mode = atomic_load_explicit(&filter->mode_id, memory_order_relaxed);
	bool freeze = atomic_load_explicit(&filter->failure_id, memory_order_relaxed) == FAILURE_FREEZE;
	bool face_swap = atomic_load_explicit(&filter->face_swap, memory_order_relaxed) != 0;

	obs_source_t *target = obs_filter_get_target(filter->source);
	uint32_t w = target ? obs_source_get_base_width(target) : 0;
	uint32_t h = target ? obs_source_get_base_height(target) : 0;
	bool faceswap_failed = filter->fx && cam_fx_faceswap_failed(filter->fx) == 1;

	/* A terminal backend failure supersedes every route immediately. Never
	 * stage or reuse a prior result while waiting for ordinary staleness. */
	if (faceswap_failed) {
		if (freeze)
			cam_effects_draw_output(filter, w ? w : 1920, h ? h : 1080, freeze);
		else
			obs_source_skip_video_filter(filter->source);
		return;
	}

	if (!target || !filter->effect) {
		/* No target, or the effect failed to load: honor the
		 * failure mode. Freeze shows the last composite (black
		 * before the first), passthrough shows the raw feed. */
		if (freeze && (mode != MODE_OFF || face_swap)) {
			cam_effects_draw_output(filter, w ? w : 1920, h ? h : 1080, freeze);
			return;
		}
		obs_source_skip_video_filter(filter->source);
		return;
	}

	/* Face swap dataflow: submit ONLY the full-res frame (the worker
	 * runs swap -> segmentation on it and publishes frame+mask) and
	 * composite the processed frame it returns. Requires the setting,
	 * a successfully activated bridge pipeline and a target within the
	 * staging cap; anything else falls back to the background-only 192
	 * dataflow (byte-identical to face swap off). */
	bool oversize = face_swap && (w > FULL_STAGE_MAX_W || h > FULL_STAGE_MAX_H);
	atomic_store_explicit(&filter->full_oversize, oversize, memory_order_relaxed);
	if (oversize && !filter->full_oversize_logged) {
		filter->full_oversize_logged = true;
		blog(LOG_WARNING,
		     "obs-cam-effects: source %ux%u exceeds the %dx%d face-swap staging cap; face swap skipped (background only)",
		     w, h, FULL_STAGE_MAX_W, FULL_STAGE_MAX_H);
	}
	bool fs = face_swap && filter->fx && w > 0 && h > 0 && !oversize && cam_fx_faceswap_enabled(filter->fx) == 1;

	if (!fs && (mode == MODE_OFF || w == 0 || h == 0 || !filter->fx)) {
		if (freeze && (mode != MODE_OFF || face_swap)) {
			cam_effects_draw_output(filter, w ? w : 1920, h ? h : 1080, freeze);
			return;
		}
		obs_source_skip_video_filter(filter->source);
		return;
	}

	if (fs)
		cam_effects_stage_full(filter, target, w, h);
	else
		cam_effects_stage(filter, target);

	const uint8_t *mask = NULL;
	int mw = 0, mh = 0;
	const uint8_t *fdata = NULL;
	int fw = 0, fh = 0;
	uint64_t seq = 0;
	int result_flags = cam_fx_try_get_result(filter->fx, &mask, &mw, &mh, &fdata, &fw, &fh, &seq);
	bool have_mask = (result_flags & CAM_FX_RESULT_MASK) != 0;
	bool have_frame = (result_flags & CAM_FX_RESULT_FRAME) != 0;
	bool have_ai = (result_flags & CAM_FX_RESULT_AI) != 0;

	/* Every background composite requires a current, correctly sized
	 * mask. Every face-swap composite requires a frame matching this
	 * target exactly; face-swap-only output does not consume a mask. */
	bool valid_mask = have_mask && mw == STAGE_SIZE && mh == STAGE_SIZE;
	bool valid_frame = have_frame && fw == (int)w && fh == (int)h;
	bool ready = fs ? (mode == MODE_OFF ? valid_frame : valid_frame && valid_mask) : valid_mask && !have_frame;
	if (!ready) {
		/* Startup: nothing processed yet. */
		if (freeze) {
			cam_effects_draw_output(filter, w, h, freeze); /* black before first output */
			return;
		}
		obs_source_skip_video_filter(filter->source);
		return;
	}

	if (!cam_fx_is_fresh(filter->fx, MASK_STALE_MS)) {
		/* Inference stalled. */
		if (freeze) {
			cam_effects_draw_output(filter, w, h, freeze);
			return;
		}
		obs_source_skip_video_filter(filter->source);
		return;
	}

	gs_texture_t *frame_tex = fs ? cam_effects_frame_tex(filter, fdata, fw, fh, seq) : NULL;

	/* Face swap alone (background off): render the full-route frame into
	 * out_work and commit it only after a draw. The disclosure badge is a
	 * post-composite overlay drawn on top (spec §8/§9). */
	if (fs && mode == MODE_OFF) {
		bool wrote = false;
		if (frame_tex && filter->out_work) {
			gs_texrender_reset(filter->out_work);
			if (gs_texrender_begin(filter->out_work, w, h)) {
				struct vec4 clear;
				vec4_zero(&clear);
				gs_clear(GS_CLEAR_COLOR, &clear, 0.0f, 0);
				gs_ortho(0.0f, (float)w, 0.0f, (float)h, -100.0f, 100.0f);
				gs_blend_state_push();
				gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);
				gs_effect_t *def = obs_get_base_effect(OBS_EFFECT_DEFAULT);
				gs_effect_set_texture(gs_effect_get_param_by_name(def, "image"), frame_tex);
				while (gs_effect_loop(def, "Draw")) {
					gs_draw_sprite(frame_tex, 0, w, h);
					wrote = true;
				}
				gs_blend_state_pop();
				gs_texrender_end(filter->out_work);
			}
		}
		if (wrote) {
			gs_texrender_t *previous = filter->out_render;
			filter->out_render = filter->out_work;
			filter->out_work = previous;
			filter->out_has_ai = have_ai;
		}
		if (wrote || freeze)
			cam_effects_draw_output(filter, w, h, freeze);
		else
			obs_source_skip_video_filter(filter->source);
		return;
	}

	/* Upload mask and composite into out_work. */
	if (filter->mask_tex && mw == STAGE_SIZE && mh == STAGE_SIZE)
		gs_texture_set_image(filter->mask_tex, mask, (uint32_t)mw, false);

	const char *tech = "DrawTransparent";
	if (mode == MODE_IMAGE && filter->bg_loaded)
		tech = "DrawReplace";
	if (mode == MODE_GREEN_SCREEN)
		tech = "DrawSolidColor";

	/* OBS color properties are stored 0xAABBGGRR; vec4_from_rgba
	 * unpacks R from the low byte (the same helper OBS's bundled
	 * color-correction filter uses). */
	struct vec4 solid_color;
	vec4_from_rgba(&solid_color, (uint32_t)atomic_load_explicit(&filter->greenscreen_color, memory_order_relaxed));

	/* Blur must run before out_work begins (it renders into its
	 * own texrenders). Fall back to transparent if the kawase effect
	 * failed to load or the blur pass fails. Face swap blurs
	 * frame_tex; otherwise the target is re-rendered as before. */
	gs_texture_t *blur = NULL;
	if (mode == MODE_BLUR && filter->blur_effect && (!fs || frame_tex)) {
		blur = cam_effects_blur(filter, target, fs ? frame_tex : NULL, w, h);
		if (blur)
			tech = "DrawBlur";
	}

	bool wrote = false;
	if (filter->out_work && filter->mask_tex && (!fs || frame_tex)) {
		gs_texrender_reset(filter->out_work);
		if (gs_texrender_begin(filter->out_work, w, h)) {
			/* texrender keeps the caller's projection: set our own
		 * ortho + replace-blend + clear, like the staging path,
		 * so the composite fills the target exactly. */
			struct vec4 clear;
			vec4_zero(&clear);
			gs_clear(GS_CLEAR_COLOR, &clear, 0.0f, 0);
			gs_ortho(0.0f, (float)w, 0.0f, (float)h, -100.0f, 100.0f);
			gs_blend_state_push();
			gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);
			if (fs) {
				/* Full-route frame: drive the technique manually.
			 * process_filter_tech_end -> render_filter_tex
			 * rebinds "image" to the unswapped parent
			 * texture unconditionally (params upload at
			 * pass begin, last-set-wins), so an override
			 * after process_filter_begin would be dead
			 * code — and process_filter_begin would render
			 * the parent for nothing. ViewProj is
			 * auto-populated from the current projection
			 * (same convention as the Kawase helper). */
				gs_effect_set_texture(gs_effect_get_param_by_name(filter->effect, "image"), frame_tex);
				gs_effect_set_texture(gs_effect_get_param_by_name(filter->effect, "mask"),
						      filter->mask_tex);
				if (filter->bg_loaded)
					gs_effect_set_texture(gs_effect_get_param_by_name(filter->effect, "bg_image"),
							      filter->bg_image.texture);
				if (blur)
					gs_effect_set_texture(gs_effect_get_param_by_name(filter->effect, "blur_image"),
							      blur);
				if (mode == MODE_GREEN_SCREEN)
					gs_effect_set_vec4(gs_effect_get_param_by_name(filter->effect, "solid_color"),
							   &solid_color);
				while (gs_effect_loop(filter->effect, tech)) {
					gs_draw_sprite(frame_tex, 0, w, h);
					wrote = true;
				}
			} else if (obs_source_process_filter_begin(filter->source, GS_BGRA, OBS_NO_DIRECT_RENDERING)) {
				gs_effect_set_texture(gs_effect_get_param_by_name(filter->effect, "mask"),
						      filter->mask_tex);
				if (filter->bg_loaded)
					gs_effect_set_texture(gs_effect_get_param_by_name(filter->effect, "bg_image"),
							      filter->bg_image.texture);
				if (blur)
					gs_effect_set_texture(gs_effect_get_param_by_name(filter->effect, "blur_image"),
							      blur);
				if (mode == MODE_GREEN_SCREEN)
					gs_effect_set_vec4(gs_effect_get_param_by_name(filter->effect, "solid_color"),
							   &solid_color);
				obs_source_process_filter_tech_end(filter->source, filter->effect, w, h, tech);
				wrote = true;
			}
			gs_blend_state_pop();
			gs_texrender_end(filter->out_work);
		}
	}
	if (wrote) {
		gs_texrender_t *previous = filter->out_render;
		filter->out_render = filter->out_work;
		filter->out_work = previous;
		filter->out_has_ai = have_ai;
	}
	if (wrote || freeze)
		cam_effects_draw_output(filter, w, h, freeze);
	else
		obs_source_skip_video_filter(filter->source);
}

static struct obs_source_info cam_effects_filter_info = {
	.id = "obs_cam_effects_filter",
	.type = OBS_SOURCE_TYPE_FILTER,
	.output_flags = OBS_SOURCE_VIDEO,
	.get_name = cam_effects_get_name,
	.create = cam_effects_create,
	.destroy = cam_effects_destroy,
	.update = cam_effects_update,
	.get_defaults = cam_effects_get_defaults,
	.video_render = cam_effects_video_render,
	.get_properties = cam_effects_properties,
};

void cam_effects_register_filter(void)
{
	obs_register_source(&cam_effects_filter_info);
}
