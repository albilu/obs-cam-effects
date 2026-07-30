#pragma once

#include "fx/types.h"

namespace fx {

/* Common interface for segmentation models (PP-HumanSeg, MediaPipe Lite,
 * RVM). infer() always returns a 192x192 mask (the pipeline contract). */
class SegmentationModel {
public:
	virtual ~SegmentationModel() = default;
	virtual Mask infer(const Frame &frame) = 0;

	/* True when inference runs on the CUDA execution provider
	 * (status display only). */
	virtual bool usesCuda() const { return false; }
};

} // namespace fx
