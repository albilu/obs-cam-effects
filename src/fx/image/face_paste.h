#pragma once

#include "fx/image/align.h"
#include "fx/types.h"

#include <cstdint>
#include <vector>

namespace fx {

/* Paste a crop-sized BGR face through an equally sized uint8 alpha mask.
 * frame must be positive with exactly width*height*4 BGRA bytes; cropSize
 * must be positive, with exactly cropSize*cropSize*3 face bytes and
 * cropSize*cropSize mask bytes. Invalid buffers/dimensions, non-finite
 * affines, and affines with |determinant| < 1e-12 are no-ops. Intensity is
 * not clamped: values < 1 use the legacy second blend, while values >= 1
 * keep the full first blend. Frame alpha and pixels outside the mapped crop
 * are unchanged. */
void pasteFaceCrop(Frame &frame, const std::vector<uint8_t> &face, const std::vector<uint8_t> &mask, int cropSize,
		   const Affine23 &frameToCrop, float intensity);

} // namespace fx
