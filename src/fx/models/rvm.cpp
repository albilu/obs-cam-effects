#include "fx/models/rvm.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

/* ============================ KNOWN BUGS ============================
 *
 * BUG-1 (Critical, worked around): ORT 1.28 CUDA EP hard-segfaults
 *   executing this model on Blackwell (sm_120) GPUs.
 *     - Symptom: uncatchable SIGSEGV inside libonnxruntime during
 *       session Run(). NOT a C++ exception, so OrtModel::tryRun's lazy
 *       CPU fallback CANNOT cover it — the whole OBS process dies.
 *     - Verified: 2026-07-30, RTX 5070 (sm_120), driver 580.126.09,
 *       cuDNN 9.24, onnxruntime-linux-x64-gpu_cuda13-1.28.0.tgz.
 *       PP-HumanSeg and MediaPipe run FINE on the same EP (simple
 *       static-IO graphs); RVM's dynamic 6-in/6-out graph crashes.
 *     - Root cause (upstream): prebuilt ORT CUDA packages ship
 *       incomplete sm_120 kernel coverage — microsoft/onnxruntime
 *       issue #26177. Community fix is building ORT from source with
 *       CMAKE_CUDA_ARCHITECTURES=120 (confirmed working on RTX 5070 Ti
 *       with ORT 1.24.2). NOT system-specific; no prebuilt ORT version
 *       (older or newer) is known to work for this graph on sm_120.
 *     - Workaround: ctor pins the model to CPU (~17ms/frame, 57fps).
 *     - Re-enable procedure: after an ORT/cuDNN bump, drop the ""
 *       providersDir override in the ctor and re-run the RVM CUDA
 *       bench (100 frames at 192x192, watch for SIGSEGV).
 *
 * BUG-2 (Test infra): tests/test_rvm.cpp needs the RVM model file at
 *   build_x86_64/models/rvm_mobilenetv3_fp32.onnx, but that file is a
 *   RUNTIME download (not CMake-fetched) and is wiped when the build
 *   dir is deleted. The test GTEST_SKIPs gracefully when absent.
 *   Re-stage with: cp ~/.config/obs-cam-effects/models/rvm_mobilenetv3_fp32.onnx build_x86_64/models/
 *
 * BUG-3 (Perf, minor): the fgr output (foreground RGB, [1,3,192,192])
 *   is computed and copied to host every inference, then discarded —
 *   ~432KB/frame of pointless copy (OrtModel::runImpl copies ALL
 *   outputs out unconditionally). Harmless at 192x192 (~2-3% of the
 *   frame budget); fix with an output allowlist if it ever matters.
 * ================================================================== */

namespace fx {

Rvm::Rvm(const std::string &modelPath, int threads,
	 const std::string &providersDir)
	: model_(modelPath, threads, ""), tensor_(3 * kSize * kSize)
{
	/* CPU pin — see KNOWN BUGS / BUG-1 at the top of this file.
	 * providersDir is deliberately ignored until the ORT CUDA EP can
	 * execute this graph on Blackwell without segfaulting. */
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
