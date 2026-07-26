#include "cam-effects-filter.h"

#include <obs-module.h>

struct cam_effects_filter {
	obs_source_t *source;
};

static const char *cam_effects_get_name(void *type_data)
{
	UNUSED_PARAMETER(type_data);
	return "Camera Effects";
}

static void *cam_effects_create(obs_data_t *settings, obs_source_t *source)
{
	UNUSED_PARAMETER(settings);
	struct cam_effects_filter *filter =
		bzalloc(sizeof(struct cam_effects_filter));
	filter->source = source;
	return filter;
}

static void cam_effects_destroy(void *data)
{
	struct cam_effects_filter *filter = data;
	bfree(filter);
}

static void cam_effects_video_render(void *data, gs_effect_t *effect)
{
	UNUSED_PARAMETER(effect);
	struct cam_effects_filter *filter = data;

	/* Plan 1: passthrough only. Real compositing arrives in Plan 2. */
	obs_source_skip_video_filter(filter->source);
}

static obs_properties_t *cam_effects_properties(void *data)
{
	UNUSED_PARAMETER(data);
	return obs_properties_create();
}

static struct obs_source_info cam_effects_filter_info = {
	.id = "obs_cam_effects_filter",
	.type = OBS_SOURCE_TYPE_FILTER,
	.output_flags = OBS_SOURCE_VIDEO,
	.get_name = cam_effects_get_name,
	.create = cam_effects_create,
	.destroy = cam_effects_destroy,
	.video_render = cam_effects_video_render,
	.get_properties = cam_effects_properties,
};

void cam_effects_register_filter(void)
{
	obs_register_source(&cam_effects_filter_info);
}
