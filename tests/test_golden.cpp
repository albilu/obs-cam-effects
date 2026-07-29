#include <gtest/gtest.h>

#include "fx/pipeline/segmentation_pipeline.h"

#include <cmath>

/* RECORDED VALUES — from a verified-good run of this exact test on
 * ORT 1.28.0 / pphumanseg_fp32.onnx (single process() call, fresh
 * pipeline, so EMA is identity). Verified identical across two
 * consecutive runs of the same binary. */
static constexpr double GOLDEN_MEAN = 0.26034758069342356;
static constexpr double GOLDEN_VAR = 0.18238129331059641;
static constexpr uint64_t GOLDEN_CHECKSUM = 9536951122226537109ULL;

/* Golden anchor: fixed synthetic frame -> pipeline -> mask statistics.
 * The expected values were recorded from a verified-good run (see plan);
 * they catch model/ORT/preprocessing regressions, not absolute truth. */
TEST(Golden, MaskStatisticsMatchRecordedRun)
{
	fx::Frame f;
	f.width = 640;
	f.height = 360;
	f.bgra.assign(640u * 360u * 4u, 0);
	/* Deterministic pseudo-image: gradient + center ellipse. */
	for (int y = 0; y < 360; y++) {
		for (int x = 0; x < 640; x++) {
			uint8_t *p = f.bgra.data() + (y * 640 + x) * 4;
			float dx = (x - 320.0f) / 120.0f;
			float dy = (y - 200.0f) / 150.0f;
			bool inEllipse = dx * dx + dy * dy < 1.0f;
			p[0] = inEllipse ? 200 : (uint8_t)(x % 256);
			p[1] = inEllipse ? 160 : (uint8_t)(y % 256);
			p[2] = inEllipse ? 140 : 60;
			p[3] = 255;
		}
	}

	fx::SegmentationPipeline pipe(FX_MODEL_PATH, 1);
	auto m = pipe.process(f);

	double mean = 0, sq = 0;
	uint64_t checksum = 0;
	for (size_t i = 0; i < m->px.size(); i++) {
		mean += m->px[i];
		sq += m->px[i] * m->px[i];
		checksum = checksum * 1099511628211ULL +
			   (uint64_t)(m->px[i] * 255.0f);
	}
	mean /= m->px.size();
	double var = sq / m->px.size() - mean * mean;

	EXPECT_NEAR(mean, GOLDEN_MEAN, 1e-3);
	EXPECT_NEAR(var, GOLDEN_VAR, 1e-3);
	EXPECT_EQ(checksum, GOLDEN_CHECKSUM);
}
