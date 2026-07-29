#include "fx/pipeline/segmentation_pipeline.h"

#include "fx/image/ops.h"

namespace fx {

SegmentationPipeline::SegmentationPipeline(const std::string &modelPath,
					   int threads)
	: model_(modelPath, threads)
{
}

std::shared_ptr<Mask> SegmentationPipeline::process(const Frame &frame)
{
	auto m = std::make_shared<Mask>(model_.infer(frame));
	if (prev_ && prev_->px.size() == m->px.size())
		emaMask(*m, *prev_, beta_);
	std::vector<float> guide = grayFromBgra(frame);
	if ((int)guide.size() == m->width * m->height)
		m->px = guidedFilter(guide, m->px, m->width, m->height, 4,
				     0.01f);
	prev_ = m;
	return m;
}

} // namespace fx
