#pragma once

#include "fx/engine/ort_backend.h"
#include "fx/models/segmentation_model.h"

#include <string>
#include <vector>

namespace fx {

/* MediaPipe Selfie Segmentation (256x256 NHWC, Apache-2.0) — Lite tier.
 * Input /255 NHWC; output single-channel person mask, downscaled to
 * 192x192 to satisfy the pipeline contract. */
class MediaPipeLite : public SegmentationModel {
public:
	static constexpr int kSize = 256;

	explicit MediaPipeLite(const std::string &modelPath, int threads = 2,
			       bool tryCuda = false);

	Mask infer(const Frame &frame) override;
	bool usesCuda() const override { return model_.usesCuda(); }

private:
	OrtModel model_;
	std::vector<float> tensor_; // 3*256*256 scratch
};

} // namespace fx
