#pragma once

#include "fx/models/segmentation_model.h"
#include "fx/types.h"

#include <memory>
#include <string>

namespace fx {

enum class SegTier { Lite, Standard, Quality };

struct MaskParams {
	float threshold = 0.0f;    // 0 = off; else binarize cutoff
	float contourFrac = 0.0f;  // drop blobs < this fraction of frame
	float featherRadius = 0.0f;// mask blur radius in px (0 = off)
	float beta = 0.6f;         // temporal EMA factor
};

/* model inference -> temporal EMA -> guided filter -> threshold ->
 * contour filter -> feather. */
class SegmentationPipeline {
public:
	/* Creates a pipeline for the given tier.
	 * modelPaths: lite / standard / quality ONNX paths (quality may be
	 * empty ONLY if tier != Quality). Throws on missing/invalid model. */
	SegmentationPipeline(SegTier tier, const std::string &litePath,
			     const std::string &standardPath,
			     const std::string &qualityPath, int threads = 2);

	std::shared_ptr<Mask> process(const Frame &frame);

	void setMaskParams(const MaskParams &p) { params_ = p; }
	MaskParams maskParams() const { return params_; }

private:
	std::unique_ptr<SegmentationModel> model_;
	MaskParams params_;
	std::shared_ptr<Mask> prev_;
};

} // namespace fx
