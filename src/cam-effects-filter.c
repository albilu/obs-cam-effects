#include "cam-effects-filter.h"

#include "fx_bridge.h"

#include <obs-module.h>
#include <graphics/image-file.h>

#define SETTING_MODE "mode"
#define SETTING_IMAGE_PATH "image_path"
#define SETTING_BLUR_STRENGTH "blur_strength"
#define SETTING_FAILURE "failure_mode"
#define SETTING_STATUS "status"

#define STAGE_SIZE 192
#define MASK_STALE_MS 1000

struct cam_effects_filter {
	obs_source_t *source;

	gs_texrender_t *stage_render;   /* STAGE_SIZE x STAGE_SIZE */
	gs_stagesurf_t *stage_surface;  /* STAGE_SIZE x STAGE_SIZE BGRA */
	gs_texrender_t *out_render;     /* frame-size composite + freeze */
	gs_texture_t *mask_tex;         /* STAGE_SIZE x STAGE_SIZE R8 */
	gs_effect_t *effect;            /* mask_composite.effect */

	cam_fx_t *fx;
	uint64_t mask_seq;

	char *mode;         /* off | transparent | image | blur */
	char *failure_mode; /* passthrough | freeze */
	int blur_strength;
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
	gs_texture_destroy(filter->mask_tex);
	gs_effect_destroy(filter->effect);
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
	bfree(filter->mode);
	bfree(filter->failure_mode);
	bfree(filter->image_path);
	bfree(filter);
}

static void cam_effects_update(void *data, obs_data_t *settings)
{
	struct cam_effects_filter *filter = data;

	bfree(filter->mode);
	bfree(filter->failure_mode);
	bfree(filter->image_path);
	filter->mode = bstrdup(obs_data_get_string(settings, SETTING_MODE));
	filter->failure_mode =
		bstrdup(obs_data_get_string(settings, SETTING_FAILURE));
	filter->image_path =
		bstrdup(obs_data_get_string(settings, SETTING_IMAGE_PATH));
	filter->blur_strength =
		(int)obs_data_get_int(settings, SETTING_BLUR_STRENGTH);

	/* (Re)load the background image if the path changed and mode
	 * needs it. */
	obs_enter_graphics();
	gs_image_file_free(&filter->bg_image);
	filter->bg_loaded = false;
	if (strcmp(filter->mode, "image") == 0 &&
	    filter->image_path[0] != '\0') {
		gs_image_file_init(&filter->bg_image, filter->image_path);
		gs_image_file_init_texture(&filter->bg_image);
		filter->bg_loaded = filter->bg_image.texture != NULL;
	}
	obs_leave_graphics();

	/* Create the inference engine lazily on first non-off mode. */
	if (!filter->fx && strcmp(filter->mode, "off") != 0) {
		char *model = obs_module_file("models/pphumanseg_fp32.onnx");
		if (model)
			filter->fx = cam_fx_create(model, 2);
		bfree(model);
	}
}

static void cam_effects_get_defaults(obs_data_t *settings)
{
	obs_data_set_default_string(settings, SETTING_MODE, "transparent");
	obs_data_set_default_string(settings, SETTING_FAILURE, "passthrough");
	obs_data_set_default_int(settings, SETTING_BLUR_STRENGTH, 2);
}

static obs_properties_t *cam_effects_properties(void *data)
{
	UNUSED_PARAMETER(data);
	obs_properties_t *props = obs_properties_create();

	obs_property_t *mode = obs_properties_add_list(
		props, SETTING_MODE, "Background", OBS_COMBO_TYPE_LIST,
		OBS_COMBO_FORMAT_STRING);
	obs_property_list_add_string(mode, "Off", "off");
	obs_property_list_add_string(mode, "Transparent", "transparent");
	obs_property_list_add_string(mode, "Replace with image", "image");
	obs_property_list_add_string(mode, "Blur", "blur");

	obs_properties_add_path(props, SETTING_IMAGE_PATH, "Background image",
				OBS_PATH_FILE,
				"Images (*.png *.jpg *.jpeg *.bmp)", NULL);
	obs_properties_add_int_slider(props, SETTING_BLUR_STRENGTH,
				      "Blur strength", 1, 4, 1);

	obs_property_t *fm = obs_properties_add_list(
		props, SETTING_FAILURE, "On processing failure",
		OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_STRING);
	obs_property_list_add_string(fm, "Show camera feed", "passthrough");
	obs_property_list_add_string(fm, "Freeze last processed frame",
				     "freeze");

	obs_properties_add_text(props, SETTING_STATUS, "Status",
				OBS_TEXT_INFO);
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

static void cam_effects_video_render(void *data, gs_effect_t *effect)
{
	UNUSED_PARAMETER(effect);
	struct cam_effects_filter *filter = data;
	obs_source_t *target = obs_filter_get_target(filter->source);
	if (!target || !filter->effect) {
		obs_source_skip_video_filter(filter->source);
		return;
	}

	bool mode_off = strcmp(filter->mode, "off") == 0;
	bool freeze = strcmp(filter->failure_mode, "freeze") == 0;

	uint32_t w = obs_source_get_base_width(target);
	uint32_t h = obs_source_get_base_height(target);
	if (mode_off || w == 0 || h == 0 || !filter->fx) {
		if (freeze && !mode_off) {
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
	gs_texture_set_image(filter->mask_tex, mask, (uint32_t)mw, false);

	const char *tech = "DrawTransparent";
	if (strcmp(filter->mode, "image") == 0 && filter->bg_loaded)
		tech = "DrawReplace";
	else if (strcmp(filter->mode, "blur") == 0)
		tech = "DrawTransparent"; /* blur arrives in Task 8 */

	gs_texrender_reset(filter->out_render);
	if (gs_texrender_begin(filter->out_render, w, h)) {
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
			obs_source_process_filter_tech_end(filter->source,
							   filter->effect, w,
							   h, tech);
		}
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
