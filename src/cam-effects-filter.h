#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/* Registers the "Camera Effects" video filter with libobs.
   Call once from obs_module_load(). */
void cam_effects_register_filter(void);

#ifdef __cplusplus
}
#endif
