#pragma once

#include "fx/engine/ort_backend.h"
#include "fx/models/segmentation_model.h"

#include <array>
#include <string>
#include <vector>

namespace fx {

/* Robust Video Matting MobileNetV3 (192x192, GPL-3.0) — Quality tier.
 * Recurrent temporal memory: states r1..r4 are zero-initialized and fed
 * back frame-to-frame. Input BGR /255 NCHW; mask = pha output. */
class Rvm : public SegmentationModel {
public:
	static constexpr int kSize = 192;

	explicit Rvm(const std::string &modelPath, int threads = 2,
		     const std::string &providersDir = "");

	Mask infer(const Frame &frame) override;
	bool usesCuda() const override { return model_.usesCuda(); }

	/* Clears recurrent states (e.g. on source change). */
	void resetState();

private:
	OrtModel model_;
	std::vector<float> tensor_; // 3*192*192 scratch
	std::array<std::vector<float>, 4> states_;
	std::array<std::vector<int64_t>, 4> stateShapes_;
};

} // namespace fx
