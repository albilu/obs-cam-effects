#include <gtest/gtest.h>

#include "fx/models/rvm.h"

#include <cstdio>

static bool fileExists(const char *p)
{
	FILE *f = fopen(p, "r");
	if (f)
		fclose(f);
	return f != nullptr;
}

/* Structured fixture (gradient + center ellipse, as in the golden test).
 * A flat frame makes RVM's pha saturate to all-zero (no person present),
 * which would leave the recurrent state feedback unobservable in the
 * mask output. */
static fx::Frame makeFrame()
{
	fx::Frame f;
	f.width = 640;
	f.height = 360;
	f.bgra.assign(640u * 360u * 4u, 0);
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
	return f;
}

TEST(Rvm, InfersMaskAndFeedsBackState)
{
	if (!fileExists(FX_RVM_MODEL_PATH))
		GTEST_SKIP() << "RVM model not downloaded (runtime model)";

	fx::Rvm model(FX_RVM_MODEL_PATH, 1);
	fx::Frame f = makeFrame();

	fx::Mask m = model.infer(f);
	ASSERT_EQ(m.width, 192);
	ASSERT_EQ(m.height, 192);
	ASSERT_EQ(m.px.size(), 192u * 192u);
	for (float v : m.px) {
		ASSERT_GE(v, 0.0f);
		ASSERT_LE(v, 1.0f);
	}

	/* Second inference with fed-back state: deterministic given same
	 * input, but different from a fresh model (proves state flows). */
	fx::Mask m2 = model.infer(f);
	fx::Rvm fresh(FX_RVM_MODEL_PATH, 1);
	fx::Mask mFresh = fresh.infer(f);
	ASSERT_NE(m2.px, mFresh.px);

	/* After reset, the model reproduces the fresh result exactly. */
	model.resetState();
	fx::Mask m3 = model.infer(f);
	ASSERT_EQ(m3.px, mFresh.px);
}
