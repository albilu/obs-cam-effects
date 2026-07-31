#include "cam-effects-filter.h"

#include "fx_bridge.h"

#include <obs-module.h>
#include <graphics/image-file.h>

#include <stdio.h>
#include <stdlib.h>

#define SETTING_MODE "mode"
#define SETTING_IMAGE_PATH "image_path"
#define SETTING_BLUR_STRENGTH "blur_strength"
#define SETTING_FAILURE "failure_mode"
#define SETTING_STATUS "status"
#define SETTING_TIER "tier"
#define SETTING_MASK_THRESHOLD "mask_threshold"
#define SETTING_MASK_CONTOUR "mask_contour"
#define SETTING_MASK_FEATHER "mask_feather"
#define SETTING_MASK_TEMPORAL "mask_temporal"

#define STAGE_SIZE 192
#define MASK_STALE_MS 1000

#include <stdatomic.h>

enum cam_mode { MODE_OFF, MODE_TRANSPARENT, MODE_IMAGE, MODE_BLUR };
enum cam_failure { FAILURE_PASSTHROUGH, FAILURE_FREEZE };

static int parse_mode(const char *s)
{
	if (strcmp(s, "transparent") == 0)
		return MODE_TRANSPARENT;
	if (strcmp(s, "image") == 0)
		return MODE_IMAGE;
	if (strcmp(s, "blur") == 0)
		return MODE_BLUR;
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

	gs_texrender_t *stage_render;   /* STAGE_SIZE x STAGE_SIZE */
	gs_stagesurf_t *stage_surface;  /* STAGE_SIZE x STAGE_SIZE BGRA */
	gs_texrender_t *out_render;     /* frame-size composite + freeze */
	gs_texture_t *mask_tex;         /* STAGE_SIZE x STAGE_SIZE R8 */
	gs_effect_t *effect;            /* mask_composite.effect */
	gs_texrender_t *blur_a;         /* half-res ping */
	gs_texrender_t *blur_b;         /* half-res pong */
	gs_effect_t *blur_effect;       /* kawase_blur.effect */

	cam_fx_t *fx;

	atomic_int mode_id;    /* enum cam_mode */
	atomic_int failure_id; /* enum cam_failure */
	int blur_strength;
	int tier;	    /* 0=auto, 1=lite, 2=standard, 3=quality */
	char status[512];
	char *image_path;
	gs_image_file_t bg_image;
	bool bg_loaded;
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
	gs_texrender_destroy(filter->out_render);
	gs_texrender_destroy(filter->blur_a);
	gs_texrender_destroy(filter->blur_b);
	gs_texture_destroy(filter->mask_tex);
	gs_effect_destroy(filter->effect);
	gs_effect_destroy(filter->blur_effect);
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
		filter->blur_effect =
			gs_effect_create_from_file(blur_path, NULL);
	obs_leave_graphics();
	bfree(blur_path);

	obs_enter_graphics();
	filter->mask_tex = gs_texture_create(STAGE_SIZE, STAGE_SIZE, GS_R8, 1,
					     NULL, GS_DYNAMIC);
	obs_leave_graphics();
}

static void *cam_effects_create(obs_data_t *settings, obs_source_t *source)
{
	struct cam_effects_filter *filter =
		bzalloc(sizeof(struct cam_effects_filter));
	filter->source = source;

	obs_enter_graphics();
	filter->stage_render = gs_texrender_create(GS_BGRA, GS_ZS_NONE);
	filter->stage_surface =
		gs_stagesurface_create(STAGE_SIZE, STAGE_SIZE, GS_BGRA);
	filter->out_render = gs_texrender_create(GS_BGRA, GS_ZS_NONE);
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
	bfree(filter);
}

/* Composes the status line shown in the properties dialog. Called from
 * update(), from properties() itself (OBS calls get_properties every
 * time the dialog opens) and from the button callbacks; the dialog
 * picks the text up when (re)opened — OBS properties are static while
 * open. */
static void cam_effects_compose_status(struct cam_effects_filter *filter)
{
	static const char *active_names[] = {
		"unknown", "Lite (MediaPipe)", "Standard (PP-HumanSeg)",
		"Quality (RVM MobileNetV3)"};
	if (!filter->fx) {
		snprintf(filter->status, sizeof(filter->status),
			 "Engine not started (select a background mode).");
		return;
	}
	int eff = cam_fx_tier_in_effect(filter->fx);
	const char *eff_name =
		(eff >= 1 && eff <= 3) ? active_names[eff] : "unknown";

	char dl_state[32] = {0};
	double dl_prog = -1.0;
	cam_fx_download_state(filter->fx, dl_state, sizeof(dl_state),
			      &dl_prog);
	char dl_text[192];
	if (strcmp(dl_state, "downloading") == 0 && dl_prog >= 0.0)
		snprintf(dl_text, sizeof(dl_text), "downloading %.0f%%",
			 dl_prog * 100.0);
	else if (strcmp(dl_state, "error") == 0) {
		char dl_err[128] = {0};
		cam_fx_download_error(filter->fx, dl_err, sizeof(dl_err));
		snprintf(dl_text, sizeof(dl_text), "error: %s",
			 dl_err[0] ? dl_err : "unknown");
	} else
		snprintf(dl_text, sizeof(dl_text), "%s",
			 dl_state[0] ? dl_state : "idle");

	char backend[16] = "CPU";
	cam_fx_backend(filter->fx, backend, sizeof(backend));

	/* fps is meaningful only while masks flow: show "warming up…"
	 * before the first mask and "—" once the stream goes stale. */
	char fps_text[32];
	if (cam_fx_is_fresh(filter->fx, 2000))
		snprintf(fps_text, sizeof(fps_text), "%llu fps",
			 (unsigned long long)cam_fx_fps(filter->fx));
	else if (cam_fx_has_mask(filter->fx))
		snprintf(fps_text, sizeof(fps_text), "—");
	else
		snprintf(fps_text, sizeof(fps_text), "warming up…");

	snprintf(filter->status, sizeof(filter->status),
		 "Active model: %s | Quality model: %s | Download: %s | "
		 "Backend: %s | %s",
		 eff_name,
		 cam_fx_quality_available(filter->fx) ? "downloaded"
						      : "not downloaded",
		 dl_text, backend, fps_text);
}

static bool cam_effects_download_rvm_clicked(obs_properties_t *props,
					     obs_property_t *property,
					     void *data)
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

static bool cam_effects_download_cuda_clicked(obs_properties_t *props,
					      obs_property_t *property,
					      void *data)
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

static void cam_effects_update(void *data, obs_data_t *settings)
{
	struct cam_effects_filter *filter = data;

	char *mode = bstrdup(obs_data_get_string(settings, SETTING_MODE));
	char *failure = bstrdup(obs_data_get_string(settings, SETTING_FAILURE));
	atomic_store_explicit(&filter->mode_id, parse_mode(mode),
			      memory_order_relaxed);
	atomic_store_explicit(&filter->failure_id, parse_failure(failure),
			      memory_order_relaxed);
	bfree(mode);
	bfree(failure);

	bfree(filter->image_path);
	filter->image_path =
		bstrdup(obs_data_get_string(settings, SETTING_IMAGE_PATH));
	filter->blur_strength =
		(int)obs_data_get_int(settings, SETTING_BLUR_STRENGTH);
	filter->tier = parse_tier(obs_data_get_string(settings, SETTING_TIER));
	float mask_threshold =
		(float)obs_data_get_double(settings, SETTING_MASK_THRESHOLD);
	float mask_contour =
		(float)obs_data_get_double(settings, SETTING_MASK_CONTOUR);
	float mask_feather =
		(float)obs_data_get_double(settings, SETTING_MASK_FEATHER);
	float mask_temporal =
		(float)obs_data_get_double(settings, SETTING_MASK_TEMPORAL);

	/* Free any previous background image (destroys its texture, so
	 * inside the graphics lock). */
	obs_enter_graphics();
	gs_image_file_free(&filter->bg_image);
	filter->bg_loaded = false;
	obs_leave_graphics();

	/* (Re)load the background image if the path changed and mode
	 * needs it. Decode (disk I/O) outside the graphics lock; only
	 * the texture upload runs inside. */
	if (atomic_load_explicit(&filter->mode_id, memory_order_relaxed) ==
		    MODE_IMAGE &&
	    filter->image_path[0] != '\0') {
		gs_image_file_init(&filter->bg_image, filter->image_path);
		obs_enter_graphics();
		gs_image_file_init_texture(&filter->bg_image);
		filter->bg_loaded = filter->bg_image.texture != NULL;
		obs_leave_graphics();
	}

	/* Create the inference engine lazily on first non-off mode.
	 * The graphics lock synchronizes the publication of filter->fx
	 * with the render thread. */
	if (!filter->fx &&
	    atomic_load_explicit(&filter->mode_id, memory_order_relaxed) !=
		    MODE_OFF) {
		char *lite = obs_module_file("models/selfie_segmentation.onnx");
		char *standard =
			obs_module_file("models/pphumanseg_fp32.onnx");
		const char *home = getenv("HOME");
		char quality[1024];
		snprintf(quality, sizeof(quality),
			 "%s/.config/obs-cam-effects/models/"
			 "rvm_mobilenetv3_fp32.onnx",
			 home ? home : ".");
		obs_enter_graphics();
		if (lite && standard)
			filter->fx = cam_fx_create(lite, standard, quality, 2);
		obs_leave_graphics();
		bfree(lite);
		bfree(standard);
	}

	/* Apply tier + mask params on every update: cheap and idempotent
	 * (the bridge rebuilds the pipeline only when the effective tier
	 * changes). */
	if (filter->fx) {
		cam_fx_set_tier(filter->fx, filter->tier);
		cam_fx_set_mask_params(filter->fx, mask_threshold,
				       mask_contour, mask_feather,
				       mask_temporal);
	}
	cam_effects_compose_status(filter);
}

static void cam_effects_get_defaults(obs_data_t *settings)
{
	obs_data_set_default_string(settings, SETTING_MODE, "transparent");
	obs_data_set_default_string(settings, SETTING_FAILURE, "passthrough");
	obs_data_set_default_int(settings, SETTING_BLUR_STRENGTH, 2);
	obs_data_set_default_string(settings, SETTING_TIER, "auto");
	obs_data_set_default_double(settings, SETTING_MASK_THRESHOLD, 0.0);
	obs_data_set_default_double(settings, SETTING_MASK_CONTOUR, 0.0);
	obs_data_set_default_double(settings, SETTING_MASK_FEATHER, 0.0);
	obs_data_set_default_double(settings, SETTING_MASK_TEMPORAL, 0.6);
}

static bool cam_effects_setting_changed(obs_properties_t *props,
					obs_property_t *property,
					obs_data_t *settings)
{
	UNUSED_PARAMETER(property);
	struct cam_effects_filter *filter = obs_properties_get_param(props);
	/* Apply immediately so the recomposed status reflects the new
	 * state (update() is idempotent). */
	if (filter)
		cam_effects_update(filter, settings);
	return true; /* rebuild the dialog -> fresh status line */
}

static obs_properties_t *cam_effects_properties(void *data)
{
	struct cam_effects_filter *filter = data;
	obs_properties_t *props = obs_properties_create();

	/* Needed by cam_effects_setting_changed() to retrieve the filter. */
	obs_properties_set_param(props, filter, NULL);

	obs_property_t *mode = obs_properties_add_list(
		props, SETTING_MODE, "Background", OBS_COMBO_TYPE_LIST,
		OBS_COMBO_FORMAT_STRING);
	obs_property_list_add_string(mode, "Off", "off");
	obs_property_list_add_string(mode, "Transparent", "transparent");
	obs_property_list_add_string(mode, "Replace with image", "image");
	obs_property_list_add_string(mode, "Blur", "blur");
	obs_property_set_modified_callback(mode, cam_effects_setting_changed);

	obs_properties_add_path(props, SETTING_IMAGE_PATH, "Background image",
				OBS_PATH_FILE,
				"Images (*.png *.jpg *.jpeg *.bmp)", NULL);
	obs_properties_add_int_slider(props, SETTING_BLUR_STRENGTH,
				      "Blur strength", 1, 4, 1);

	obs_property_t *tier = obs_properties_add_list(
		props, SETTING_TIER, "Segmentation model",
		OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_STRING);
	obs_property_list_add_string(tier, "Auto (recommended)", "auto");
	obs_property_list_add_string(tier, "Lite (fastest)", "lite");
	obs_property_list_add_string(tier, "Standard", "standard");
	obs_property_list_add_string(tier, "Quality (downloaded model)",
				     "quality");
	obs_property_set_modified_callback(tier, cam_effects_setting_changed);

	obs_properties_add_float_slider(props, SETTING_MASK_THRESHOLD,
					"Mask threshold", 0.0, 1.0, 0.01);
	obs_properties_add_float_slider(props, SETTING_MASK_CONTOUR,
					"Mask contour cleanup", 0.0, 0.5,
					0.01);
	obs_properties_add_float_slider(props, SETTING_MASK_FEATHER,
					"Mask feather", 0.0, 8.0, 0.5);
	obs_properties_add_float_slider(props, SETTING_MASK_TEMPORAL,
					"Temporal smoothing", 0.0, 0.95,
					0.05);

	obs_property_t *fm = obs_properties_add_list(
		props, SETTING_FAILURE, "On processing failure",
		OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_STRING);
	obs_property_list_add_string(fm, "Show camera feed", "passthrough");
	obs_property_list_add_string(fm, "Freeze last processed frame",
				     "freeze");

	char notice[512] = {0};
	if (filter && filter->fx)
		cam_fx_notice(filter->fx, "rvm_mobilenetv3_fp32", notice,
			      sizeof(notice));
	if (notice[0] == '\0')
		snprintf(notice, sizeof(notice), "%s",
			 "Robust Video Matting (RVM) MobileNetV3, licensed "
			 "GPL-3.0-only. By downloading you accept the "
			 "license terms.");
	obs_properties_add_text(props, "rvm_notice", notice, OBS_TEXT_INFO);

	obs_properties_add_button(props, "download_btn",
				  "Download Quality model (GPL-3.0, 15 MB)",
				  cam_effects_download_rvm_clicked);
	obs_properties_add_button(
		props, "download_cuda_btn",
		"Download GPU acceleration (MIT, ~240 MB)",
		cam_effects_download_cuda_clicked);

	/* OBS calls get_properties every time the properties dialog
	 * opens: recompose the status from live bridge state so the
	 * dialog never shows a stale creation-time snapshot (fps is
	 * legitimately 0 at creation). */
	if (filter)
		cam_effects_compose_status(filter);
	const char *status = (filter && filter->status[0])
				     ? filter->status
				     : "Status unavailable";
	obs_property_t *status_prop = obs_properties_add_text(
		props, SETTING_STATUS, status, OBS_TEXT_INFO);
	obs_property_set_description(status_prop, status);
	return props;
}

/* Render the parent source into the small staging surface and submit. */
static void cam_effects_stage(struct cam_effects_filter *filter,
			      obs_source_t *target)
{
	uint32_t tw = obs_source_get_base_width(target);
	uint32_t th = obs_source_get_base_height(target);
	if (tw == 0 || th == 0)
		return;

	gs_texrender_reset(filter->stage_render);
	if (!gs_texrender_begin(filter->stage_render, STAGE_SIZE,
				STAGE_SIZE))
		return;
	struct vec4 clear;
	vec4_zero(&clear);
	gs_clear(GS_CLEAR_COLOR, &clear, 0.0f, 0);
	gs_ortho(0.0f, (float)STAGE_SIZE, 0.0f, (float)STAGE_SIZE, -100.0f,
		 100.0f);
	gs_blend_state_push();
	gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);
	gs_matrix_push();
	gs_matrix_scale3f((float)STAGE_SIZE / (float)tw,
			  (float)STAGE_SIZE / (float)th, 1.0f);
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

	gs_stage_texture(filter->stage_surface,
			 gs_texrender_get_texture(filter->stage_render));
	uint8_t *data = NULL;
	uint32_t linesize = 0;
	if (gs_stagesurface_map(filter->stage_surface, &data, &linesize)) {
		cam_fx_submit(filter->fx, data, STAGE_SIZE, STAGE_SIZE,
			      (int)linesize);
		gs_stagesurface_unmap(filter->stage_surface);
	}
}

/* Draw the contents of out_render to screen. */
static void cam_effects_draw_out(struct cam_effects_filter *filter,
				 uint32_t w, uint32_t h)
{
	gs_texture_t *tex = gs_texrender_get_texture(filter->out_render);
	if (!tex)
		return;
	gs_effect_t *def = obs_get_base_effect(OBS_EFFECT_DEFAULT);
	gs_eparam_t *image = gs_effect_get_param_by_name(def, "image");
	gs_effect_set_texture(image, tex);
	while (gs_effect_loop(def, "Draw"))
		gs_draw_sprite(tex, 0, w, h);
}

/* Runs `passes` Kawase blur iterations on target at half resolution.
 * Returns the texture containing the blurred result, or NULL. */
static gs_texture_t *cam_effects_blur(struct cam_effects_filter *filter,
				      obs_source_t *target, uint32_t w,
				      uint32_t h)
{
	uint32_t bw = w / 2 > 0 ? w / 2 : 1;
	uint32_t bh = h / 2 > 0 ? h / 2 : 1;

	/* Downsample target into blur_a using the default effect. */
	gs_texrender_reset(filter->blur_a);
	if (!gs_texrender_begin(filter->blur_a, bw, bh))
		return NULL;
	struct vec4 clear;
	vec4_zero(&clear);
	gs_clear(GS_CLEAR_COLOR, &clear, 0.0f, 0);
	gs_ortho(0.0f, (float)bw, 0.0f, (float)bh, -100.0f, 100.0f);
	gs_blend_state_push();
	gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);
	gs_matrix_push();
	gs_matrix_scale3f((float)bw / (float)w, (float)bh / (float)h, 1.0f);
	/* Like the staging path: sources without their own effect loop
	 * (e.g. image sources) need the default effect active. */
	gs_effect_t *def = obs_get_base_effect(OBS_EFFECT_DEFAULT);
	while (gs_effect_loop(def, "Draw"))
		obs_source_video_render(target);
	gs_matrix_pop();
	gs_blend_state_pop();
	gs_texrender_end(filter->blur_a);

	gs_texture_t *src = gs_texrender_get_texture(filter->blur_a);
	for (int i = 0; i < filter->blur_strength; i++) {
		gs_texrender_t *dst = (i % 2 == 0) ? filter->blur_b
						   : filter->blur_a;
		gs_texrender_reset(dst);
		if (!gs_texrender_begin(dst, bw, bh))
			return src;
		gs_effect_set_texture(
			gs_effect_get_param_by_name(filter->blur_effect,
						    "image"),
			src);
		struct vec2 texel = {1.0f / (float)bw, 1.0f / (float)bh};
		gs_effect_set_vec2(
			gs_effect_get_param_by_name(filter->blur_effect,
						    "texel"),
			&texel);
		gs_effect_set_float(
			gs_effect_get_param_by_name(filter->blur_effect,
						    "iteration"),
			(float)i);
		gs_clear(GS_CLEAR_COLOR, &clear, 0.0f, 0);
		gs_ortho(0.0f, (float)bw, 0.0f, (float)bh, -100.0f, 100.0f);
		while (gs_effect_loop(filter->blur_effect, "Draw"))
			gs_draw_sprite(src, 0, bw, bh);
		gs_texrender_end(dst);
		src = gs_texrender_get_texture(dst);
	}
	return src;
}

static void cam_effects_video_render(void *data, gs_effect_t *effect)
{
	UNUSED_PARAMETER(effect);
	struct cam_effects_filter *filter = data;
	int mode = atomic_load_explicit(&filter->mode_id, memory_order_relaxed);
	bool freeze = atomic_load_explicit(&filter->failure_id,
					   memory_order_relaxed) ==
		      FAILURE_FREEZE;

	obs_source_t *target = obs_filter_get_target(filter->source);
	uint32_t w = target ? obs_source_get_base_width(target) : 0;
	uint32_t h = target ? obs_source_get_base_height(target) : 0;

	if (!target || !filter->effect) {
		/* No target, or the effect failed to load: honor the
		 * failure mode. Freeze shows the last composite (black
		 * before the first), passthrough shows the raw feed. */
		if (freeze && mode != MODE_OFF) {
			cam_effects_draw_out(filter, w ? w : 1920,
					     h ? h : 1080);
			return;
		}
		obs_source_skip_video_filter(filter->source);
		return;
	}

	if (mode == MODE_OFF || w == 0 || h == 0 || !filter->fx) {
		if (freeze && mode != MODE_OFF) {
			cam_effects_draw_out(filter, w ? w : 1920,
					     h ? h : 1080);
			return;
		}
		obs_source_skip_video_filter(filter->source);
		return;
	}

	cam_effects_stage(filter, target);

	const uint8_t *mask = NULL;
	int mw = 0, mh = 0;
	uint64_t seq = 0;
	bool have_mask =
		cam_fx_try_get_mask(filter->fx, &mask, &mw, &mh, &seq) == 1;

	if (!have_mask) {
		/* Startup: no mask ever processed yet. */
		if (freeze) {
			cam_effects_draw_out(filter, w, h); /* black */
			return;
		}
		obs_source_skip_video_filter(filter->source);
		return;
	}

	if (!cam_fx_is_fresh(filter->fx, MASK_STALE_MS)) {
		/* Inference stalled. */
		if (freeze) {
			cam_effects_draw_out(filter, w, h);
			return;
		}
		obs_source_skip_video_filter(filter->source);
		return;
	}

	/* Upload mask and composite into out_render. */
	if (mw == STAGE_SIZE && mh == STAGE_SIZE)
		gs_texture_set_image(filter->mask_tex, mask, (uint32_t)mw,
				     false);

	const char *tech = "DrawTransparent";
	if (mode == MODE_IMAGE && filter->bg_loaded)
		tech = "DrawReplace";

	/* Blur must run before out_render begins (it renders the target
	 * into its own texrenders). Fall back to transparent if the
	 * kawase effect failed to load or the blur pass fails. */
	gs_texture_t *blur = NULL;
	if (mode == MODE_BLUR && filter->blur_effect) {
		blur = cam_effects_blur(filter, target, w, h);
		if (blur)
			tech = "DrawBlur";
	}

	gs_texrender_reset(filter->out_render);
	if (gs_texrender_begin(filter->out_render, w, h)) {
		/* texrender keeps the caller's projection: set our own
		 * ortho + replace-blend + clear, like the staging path,
		 * so the composite fills the target exactly. */
		struct vec4 clear;
		vec4_zero(&clear);
		gs_clear(GS_CLEAR_COLOR, &clear, 0.0f, 0);
		gs_ortho(0.0f, (float)w, 0.0f, (float)h, -100.0f, 100.0f);
		gs_blend_state_push();
		gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);
		if (obs_source_process_filter_begin(filter->source, GS_BGRA,
						    OBS_NO_DIRECT_RENDERING)) {
			gs_effect_set_texture(
				gs_effect_get_param_by_name(filter->effect,
							    "mask"),
				filter->mask_tex);
			if (filter->bg_loaded)
				gs_effect_set_texture(
					gs_effect_get_param_by_name(
						filter->effect, "bg_image"),
					filter->bg_image.texture);
			if (blur)
				gs_effect_set_texture(
					gs_effect_get_param_by_name(
						filter->effect, "blur_image"),
					blur);
			obs_source_process_filter_tech_end(filter->source,
							   filter->effect, w,
							   h, tech);
		}
		gs_blend_state_pop();
		gs_texrender_end(filter->out_render);
	}
	cam_effects_draw_out(filter, w, h);
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
