#include "fx/models/mediapipe_lite.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace fx {

MediaPipeLite::MediaPipeLite(const std::string &modelPath, int threads,
			     const std::string &providersDir)
	: model_(modelPath, threads, providersDir),
	  tensor_(3 * kSize * kSize)
{
	const auto &shape = model_.input().shape;
	if (shape.size() != 4 || shape[1] != kSize || shape[2] != kSize ||
	    shape[3] != 3)
		throw std::runtime_error(
			"fx: unexpected MediaPipe input shape (want NHWC 256)");
}

Mask MediaPipeLite::infer(const Frame &frame)
{
	const int sw = frame.width, sh = frame.height;
	constexpr int dw = kSize, dh = kSize;

	/* Bilinear-resize BGRA -> 256x256, /255, NHWC layout. */
	for (int y = 0; y < dh; y++) {
		float fy = (y + 0.5f) * sh / (float)dh - 0.5f;
		int y0 = std::clamp((int)std::floor(fy), 0, sh - 1);
		int y1 = std::min(y0 + 1, sh - 1);
		float wy = std::clamp(fy - (float)y0, 0.0f, 1.0f);
		for (int x = 0; x < dw; x++) {
			float sx = (x + 0.5f) * sw / (float)dw - 0.5f;
			int x0 = std::clamp((int)std::floor(sx), 0, sw - 1);
			int x1 = std::min(x0 + 1, sw - 1);
			float wx = std::clamp(sx - (float)x0, 0.0f, 1.0f);
			const uint8_t *p00 =
				frame.bgra.data() + (y0 * sw + x0) * 4;
			const uint8_t *p01 =
				frame.bgra.data() + (y0 * sw + x1) * 4;
			const uint8_t *p10 =
				frame.bgra.data() + (y1 * sw + x0) * 4;
			const uint8_t *p11 =
				frame.bgra.data() + (y1 * sw + x1) * 4;
			float *dst = tensor_.data() + (y * dw + x) * 3;
			for (int c = 0; c < 3; c++) {
				float v = (p00[c] * (1 - wx) + p01[c] * wx) *
						  (1 - wy) +
					  (p10[c] * (1 - wx) + p11[c] * wx) * wy;
				dst[c] = v / 255.0f;
			}
		}
	}

	std::vector<float> out = model_.run(tensor_); // [1,256,256,1]

	/* Downscale 256x256 -> 192x192 (bilinear on the mask). */
	Mask m;
	m.width = 192;
	m.height = 192;
	m.px.resize(192 * 192);
	for (int y = 0; y < 192; y++) {
		float fy = (y + 0.5f) * 256.0f / 192.0f - 0.5f;
		int y0 = std::clamp((int)std::floor(fy), 0, 255);
		int y1 = std::min(y0 + 1, 255);
		float wy = std::clamp(fy - (float)y0, 0.0f, 1.0f);
		for (int x = 0; x < 192; x++) {
			float sx = (x + 0.5f) * 256.0f / 192.0f - 0.5f;
			int x0 = std::clamp((int)std::floor(sx), 0, 255);
			int x1 = std::min(x0 + 1, 255);
			float wx = std::clamp(sx - (float)x0, 0.0f, 1.0f);
			float v = (out[y0 * 256 + x0] * (1 - wx) +
				   out[y0 * 256 + x1] * wx) *
					  (1 - wy) +
				  (out[y1 * 256 + x0] * (1 - wx) +
				   out[y1 * 256 + x1] * wx) *
					  wy;
			m.px[y * 192 + x] = std::clamp(v, 0.0f, 1.0f);
		}
	}
	return m;
}

} // namespace fx
