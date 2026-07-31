#include "fx/models/rvm.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace fx {

Rvm::Rvm(const std::string &modelPath, int threads,
	 const std::string &providersDir)
	: model_(modelPath, threads, ""), tensor_(3 * kSize * kSize)
{
	/* RVM is pinned to CPU regardless of providersDir: the ORT 1.28
	 * CUDA EP segfaults (not an exception — uncatchable) executing
	 * this model's dynamic 6-in/6-out graph on a Blackwell (sm_120)
	 * GPU. Verified 2026-07-30 on RTX 5070 / driver 580.126.09 /
	 * cuDNN 9.24: PP-HumanSeg and MediaPipe run fine on the same
	 * CUDA EP, RVM crashes inside libonnxruntime during Run().
	 * CPU is ~17ms/frame (57fps) — plenty. Re-test before re-enabling
	 * (drop the "" override and re-run the RVM bench). */
	(void)providersDir;
	if (model_.inputCount() != 6 || model_.outputCount() != 6)
		throw std::runtime_error("fx: unexpected RVM IO count");

	/* Recurrent state shapes: r_i channels 16/20/40/64, spatial
	 * kSize / 2^i. Verified by introspection at 192x192. */
	static const int channels[4] = {16, 20, 40, 64};
	for (int i = 0; i < 4; i++) {
		int s = kSize >> (i + 1);
		stateShapes_[i] = {1, channels[i], s, s};
		states_[i].assign((size_t)channels[i] * s * s, 0.0f);
	}
}

void Rvm::resetState()
{
	for (auto &s : states_)
		std::fill(s.begin(), s.end(), 0.0f);
}

Mask Rvm::infer(const Frame &frame)
{
	const int sw = frame.width, sh = frame.height;
	constexpr int dw = kSize, dh = kSize;

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
			for (int c = 0; c < 3; c++) {
				float v = (p00[c] * (1 - wx) + p01[c] * wx) *
						  (1 - wy) +
					  (p10[c] * (1 - wx) + p11[c] * wx) * wy;
				tensor_[c * dh * dw + y * dw + x] = v / 255.0f;
			}
		}
	}

	/* IO order (verified): inputs src, r1i..r4i, downsample_ratio;
	 * outputs fgr, pha, r1o..r4o. */
	std::vector<std::vector<float>> inputs;
	inputs.reserve(6);
	inputs.push_back(tensor_);
	for (int i = 0; i < 4; i++)
		inputs.push_back(states_[i]);
	inputs.push_back({1.0f}); // downsample_ratio

	std::vector<std::vector<int64_t>> shapes = {
		{1, 3, kSize, kSize}, stateShapes_[0], stateShapes_[1],
		stateShapes_[2], stateShapes_[3], {1}};
	auto outs = model_.runWithShapes(inputs, shapes);
	states_[0] = outs[2];
	states_[1] = outs[3];
	states_[2] = outs[4];
	states_[3] = outs[5];

	const std::vector<float> &pha = outs[1];
	Mask m;
	m.width = dw;
	m.height = dh;
	m.px.assign(pha.begin(), pha.end());
	return m;
}

} // namespace fx
