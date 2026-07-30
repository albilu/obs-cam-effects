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
	auto m = std::make_shared<Mask>(model_->infer(frame));
	if (prev_ && prev_->px.size() == m->px.size())
		emaMask(*m, *prev_, params_.beta);
	std::vector<float> guide = grayFromBgra(frame);
	if ((int)guide.size() == m->width * m->height)
		m->px = guidedFilter(guide, m->px, m->width, m->height, 4,
				     0.01f);
	if (params_.threshold > 0.0f) {
		for (float &v : m->px)
			v = (v >= params_.threshold) ? 1.0f : 0.0f;
	}
	if (params_.contourFrac > 0.0f)
		contourFilter(*m, params_.contourFrac);
	if (params_.featherRadius > 0.0f)
		featherMask(*m, params_.featherRadius);
	prev_ = m;
	return m;
}

} // namespace fx
