#pragma once

#include "fx/models/pp_humanseg.h"
#include "fx/types.h"

#include <memory>

namespace fx {

/* model inference -> temporal EMA -> guided-filter edge refinement. */
class SegmentationPipeline {
public:
	explicit SegmentationPipeline(const std::string &modelPath,
				      int threads = 2);

	std::shared_ptr<Mask> process(const Frame &frame);

	void setTemporalBeta(float beta) { beta_ = beta; }

private:
	PPHumanSeg model_;
	float beta_ = 0.6f;
	std::shared_ptr<Mask> prev_;
};

} // namespace fx
