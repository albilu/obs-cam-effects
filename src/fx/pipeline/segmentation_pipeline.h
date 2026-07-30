#pragma once

#include "fx/models/segmentation_model.h"
#include "fx/types.h"

#include <atomic>
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
	 * empty ONLY if tier != Quality). Throws on missing/invalid model.
	 * providersDir: dir with the CUDA provider library ("" = CPU). */
	SegmentationPipeline(SegTier tier, const std::string &litePath,
			     const std::string &standardPath,
			     const std::string &qualityPath, int threads = 2,
			     const std::string &providersDir = "");

	std::shared_ptr<Mask> process(const Frame &frame);

	/* True when the active model runs on the CUDA execution provider. */
	bool usesCuda() const { return model_ && model_->usesCuda(); }

	void setMaskParams(const MaskParams &p)
	{
		params_.store(p, std::memory_order_relaxed);
	}
	MaskParams maskParams() const
	{
		return params_.load(std::memory_order_relaxed);
	}

private:
	std::unique_ptr<SegmentationModel> model_;
	/* Atomic (trivially copyable struct): written by the UI thread via
	 * the bridge, read per frame by the worker thread. */
	std::atomic<MaskParams> params_{MaskParams{}};
	std::shared_ptr<Mask> prev_;
};

} // namespace fx
