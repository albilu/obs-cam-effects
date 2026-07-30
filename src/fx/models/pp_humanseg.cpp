#include "fx/models/pp_humanseg.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace fx {

PPHumanSeg::PPHumanSeg(const std::string &modelPath, int threads,
		       const std::string &providersDir)
	: model_(modelPath, threads, providersDir),
	  tensor_(3 * kSize * kSize)
{
	const auto &shape = model_.input().shape;
	if (shape.size() != 4 || shape[1] != 3 || shape[2] != kSize ||
	    shape[3] != kSize)
		throw std::runtime_error("fx: unexpected PPHumanSeg input shape");
}

Mask PPHumanSeg::infer(const Frame &frame)
{
	const int sw = frame.width, sh = frame.height;
	constexpr int dw = kSize, dh = kSize;

	/* Bilinear-resize BGRA -> 192x192, normalize into NCHW float. */
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
			for (int c = 0; c < 3; c++) { // 0=B,1=G,2=R kept
				float v = (p00[c] * (1 - wx) + p01[c] * wx) *
						  (1 - wy) +
					  (p10[c] * (1 - wx) + p11[c] * wx) * wy;
				tensor_[c * dh * dw + y * dw + x] =
					(v / 256.0f - 0.5f) / 0.5f;
			}
		}
	}

	std::vector<float> out = model_.run(tensor_); // [1,192,192,2]

	Mask m;
	m.width = dw;
	m.height = dh;
	m.px.resize(dw * dh);
	float lo = 1e30f, hi = -1e30f;
	for (int i = 0; i < dw * dh; i++) {
		float person = out[i * 2 + 1];
		m.px[i] = person;
		lo = std::min(lo, person);
		hi = std::max(hi, person);
	}
	float range = hi - lo;
	if (range > 1e-6f) {
		for (float &v : m.px)
			v = (v - lo) / range;
	} else {
		std::fill(m.px.begin(), m.px.end(), 0.0f);
	}
	return m;
}

} // namespace fx
