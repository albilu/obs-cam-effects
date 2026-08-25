#pragma once

#include "fx/image/align.h"
#include "fx/types.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace fx::test {

inline void pasteFaceCropReference(Frame &frame, const std::vector<uint8_t> &face, const std::vector<uint8_t> &mask,
				   int cropSize, const Affine23 &frameToCrop, float intensity)
{
	const int w = frame.width;
	const int h = frame.height;
	const size_t n = (size_t)w * (size_t)h;
	const Affine23 cropToFrame = invertAffine(frameToCrop);
	std::vector<uint8_t> faceFull(n * 3);
	std::vector<uint8_t> maskFull(n);
	warpAffineBilinear(face.data(), cropSize, cropSize, 3, cropToFrame, faceFull.data(), w, h);
	warpAffineBilinear(mask.data(), cropSize, cropSize, 1, cropToFrame, maskFull.data(), w, h);

	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			const float u = frameToCrop.m[0] * (float)x + frameToCrop.m[1] * (float)y + frameToCrop.m[2];
			const float v = frameToCrop.m[3] * (float)x + frameToCrop.m[4] * (float)y + frameToCrop.m[5];
			if (u < 0.0f || u >= (float)cropSize || v < 0.0f || v >= (float)cropSize)
				maskFull[(size_t)y * (size_t)w + (size_t)x] = 0;
		}
	}

	std::vector<uint8_t> original;
	if (intensity < 1.0f)
		original = frame.bgra;
	for (size_t i = 0; i < n; i++) {
		const float alpha = (float)maskFull[i] / 255.0f;
		if (alpha <= 0.0f)
			continue;
		uint8_t *dst = frame.bgra.data() + i * 4;
		const uint8_t *src = faceFull.data() + i * 3;
		for (int c = 0; c < 3; c++) {
			const float value = (float)src[c] * alpha + (float)dst[c] * (1.0f - alpha);
			dst[c] = (uint8_t)std::clamp((int)std::lround(value), 0, 255);
		}
	}

	if (intensity < 1.0f) {
		const float keep = intensity;
		const float restore = 1.0f - keep;
		for (size_t i = 0; i < n; i++) {
			if (maskFull[i] == 0)
				continue;
			uint8_t *dst = frame.bgra.data() + i * 4;
			const uint8_t *src = original.data() + i * 4;
			for (int c = 0; c < 3; c++) {
				const float value = (float)dst[c] * keep + (float)src[c] * restore;
				dst[c] = (uint8_t)std::clamp((int)std::lround(value), 0, 255);
			}
		}
	}
}

} // namespace fx::test
