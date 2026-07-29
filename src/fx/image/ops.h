#pragma once

#include "fx/types.h"

#include <vector>

namespace fx {

/* Grayscale [0,1] from a packed BGRA frame (Rec. 601 luma). */
std::vector<float> grayFromBgra(const Frame &f);

/* In-place exponential moving average: cur = beta*prev + (1-beta)*cur. */
void emaMask(Mask &cur, const Mask &prev, float beta);

/* Guided filter (He et al. 2013): edge-preserving smoothing of src using
 * guide. Both single-channel [0,1], size w*h. r = box radius, eps =
 * regularization. Returns the filtered image. */
std::vector<float> guidedFilter(const std::vector<float> &guide,
				const std::vector<float> &src, int w, int h,
				int r, float eps);

} // namespace fx
