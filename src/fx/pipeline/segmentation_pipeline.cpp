#include "fx/pipeline/segmentation_pipeline.h"

#include "fx/image/ops.h"
#include "fx/models/mediapipe_lite.h"
#include "fx/models/pp_humanseg.h"
#include "fx/models/rvm.h"

#include <algorithm>
#include <cmath>

namespace fx {

SegmentationPipeline::SegmentationPipeline(SegTier tier,
					   const std::string &litePath,
					   const std::string &standardPath,
					   const std::string &qualityPath,
					   int threads, bool tryCuda)
{
	switch (tier) {
	case SegTier::Lite:
		model_ = std::make_unique<MediaPipeLite>(litePath, threads,
							 tryCuda);
		break;
	case SegTier::Standard:
		model_ = std::make_unique<PPHumanSeg>(standardPath, threads,
						      tryCuda);
		break;
	case SegTier::Quality:
		model_ = std::make_unique<Rvm>(qualityPath, threads, tryCuda);
		break;
	}
}

namespace {

/* Bilinear resize of a single-channel [0,1] plane to dw x dh (same
 * pixel-center alignment as the models' BGRA resizes). The guided
 * filter's guide must match the mask dimensions; the face-swap path
 * feeds full-res frames while the mask stays at the model's output
 * size, so the guide is downsampled instead of skipping edge
 * refinement. */
std::vector<float> resizeGray(const std::vector<float> &src, int sw, int sh,
			      int dw, int dh)
{
	std::vector<float> out((size_t)dw * dh);
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
			float p00 = src[(size_t)y0 * sw + x0];
			float p01 = src[(size_t)y0 * sw + x1];
			float p10 = src[(size_t)y1 * sw + x0];
			float p11 = src[(size_t)y1 * sw + x1];
			out[(size_t)y * dw + x] =
				(p00 * (1 - wx) + p01 * wx) * (1 - wy) +
				(p10 * (1 - wx) + p11 * wx) * wy;
		}
	}
	return out;
}

} // namespace

std::shared_ptr<Mask> SegmentationPipeline::process(const Frame &frame)
{
	const MaskParams p = params_.load(std::memory_order_relaxed);
	auto m = std::make_shared<Mask>(model_->infer(frame));
	if (prev_ && prev_->px.size() == m->px.size())
		emaMask(*m, *prev_, p.beta);
	std::vector<float> guide = grayFromBgra(frame);
	if ((int)guide.size() != m->width * m->height)
		guide = resizeGray(guide, frame.width, frame.height, m->width,
				   m->height);
	m->px = guidedFilter(guide, m->px, m->width, m->height, 4, 0.01f);
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
