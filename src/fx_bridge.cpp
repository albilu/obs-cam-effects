#include "fx_bridge.h"

#include "fx/pipeline/segmentation_pipeline.h"
#include "fx/worker.h"

#include <cstring>
#include <memory>
#include <vector>

struct cam_fx {
	std::unique_ptr<fx::SegmentationPipeline> pipeline;
	std::unique_ptr<fx::Worker> worker;
	uint64_t seenSeq = 0;
	std::vector<uint8_t> u8;
};

extern "C" {

cam_fx_t *cam_fx_create(const char *model_path, int threads)
{
	try {
		auto fx = std::make_unique<cam_fx>();
		fx->pipeline = std::make_unique<fx::SegmentationPipeline>(
			model_path, threads);
		fx->worker = std::make_unique<fx::Worker>(
			[pipeline = fx->pipeline.get()](const fx::Frame &f) {
				return pipeline->process(f);
			});
		fx->worker->start();
		return fx.release();
	} catch (...) {
		return nullptr;
	}
}

void cam_fx_destroy(cam_fx_t *fx)
{
	delete fx;
}

void cam_fx_submit(cam_fx_t *fx, const uint8_t *bgra, int w, int h,
		   int linesize)
{
	auto frame = std::make_shared<fx::Frame>();
	frame->width = w;
	frame->height = h;
	frame->bgra.resize((size_t)w * h * 4);
	for (int y = 0; y < h; y++)
		std::memcpy(frame->bgra.data() + (size_t)y * w * 4,
			    bgra + (size_t)y * linesize, (size_t)w * 4);
	fx->worker->submit(std::move(frame));
}

int cam_fx_try_get_mask(cam_fx_t *fx, const uint8_t **px, int *w, int *h,
			uint64_t *seq)
{
	uint64_t s = 0;
	auto m = fx->worker->tryGetLatest(s);
	if (!m)
		return 0;
	if (s != fx->seenSeq) {
		fx->u8.resize(m->px.size());
		for (size_t i = 0; i < m->px.size(); i++) {
			float v = m->px[i];
			v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
			fx->u8[i] = (uint8_t)(v * 255.0f + 0.5f);
		}
		fx->seenSeq = s;
	}
	*px = fx->u8.data();
	*w = m->width;
	*h = m->height;
	*seq = s;
	return 1;
}

int cam_fx_is_fresh(cam_fx_t *fx, uint64_t max_age_ms)
{
	return fx->worker->isFresh(max_age_ms) ? 1 : 0;
}

} // extern "C"
