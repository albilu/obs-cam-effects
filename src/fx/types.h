#pragma once

#include <cstdint>
#include <vector>

namespace fx {

/* A packed BGRA frame (linesize == width*4 after copy). */
struct Frame {
	int width = 0;
	int height = 0;
	std::vector<uint8_t> bgra;
	/* Worker routing metadata only; the packed pixel payload is unchanged. */
	bool bypassFaceSwap = false;
};

/* A single-channel person mask, values in [0,1]. */
struct Mask {
	int width = 0;
	int height = 0;
	std::vector<float> px;
};

} // namespace fx
