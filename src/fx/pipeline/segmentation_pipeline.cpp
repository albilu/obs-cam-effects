#include "fx/pipeline/segmentation_pipeline.h"

#include "fx/image/ops.h"
#include "fx/models/mediapipe_lite.h"
#include "fx/models/pp_humanseg.h"
#include "fx/models/rvm.h"

namespace fx {

SegmentationPipeline::SegmentationPipeline(SegTier tier,
					   const std::string &litePath,
					   const std::string &standardPath,
					   const std::string &qualityPath,
					   int threads)
{
	switch (tier) {
	case SegTier::Lite:
		model_ = std::make_unique<MediaPipeLite>(litePath, threads);
		break;
	case SegTier::Standard:
		model_ = std::make_unique<PPHumanSeg>(standardPath, threads);
		break;
	case SegTier::Quality:
		model_ = std::make_unique<Rvm>(qualityPath, threads);
		break;
	}
}

std::shared_ptr<Mask> SegmentationPipeline::process(const Frame &frame)
{
	const MaskParams p = params_.load(std::memory_order_relaxed);
	auto m = std::make_shared<Mask>(model_->infer(frame));
	if (prev_ && prev_->px.size() == m->px.size())
		emaMask(*m, *prev_, p.beta);
	std::vector<float> guide = grayFromBgra(frame);
	if ((int)guide.size() == m->width * m->height)
		m->px = guidedFilter(guide, m->px, m->width, m->height, 4,
				     0.01f);
	if (p.threshold > 0.0f) {
		for (float &v : m->px)
			v = (v >= p.threshold) ? 1.0f : 0.0f;
	}
	if (p.contourFrac > 0.0f)
		contourFilter(*m, p.contourFrac);
	if (p.featherRadius > 0.0f)
		featherMask(*m, p.featherRadius);
	prev_ = m;
	return m;
}

} // namespace fx
