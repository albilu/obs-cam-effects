#include <gtest/gtest.h>

#include "fx/models/pp_humanseg.h"

static fx::Frame makeBlobFrame()
{
	/* 320x240 dark frame with a bright vertical blob in the center. */
	fx::Frame f;
	f.width = 320;
	f.height = 240;
	f.bgra.assign(320u * 240u * 4u, 30);
	for (int y = 40; y < 220; y++) {
		for (int x = 120; x < 200; x++) {
			uint8_t *p = f.bgra.data() + (y * 320 + x) * 4;
			p[0] = 200; // B
			p[1] = 160; // G
			p[2] = 140; // R
			p[3] = 255; // A
		}
	}
	return f;
}

TEST(PPHumanSeg, InfersValidDeterministicMask)
{
	fx::PPHumanSeg model(FX_MODEL_PATH, 1);
	fx::Frame f = makeBlobFrame();

	fx::Mask m = model.infer(f);
	ASSERT_EQ(m.width, 192);
	ASSERT_EQ(m.height, 192);
	ASSERT_EQ(m.px.size(), 192u * 192u);
	for (float v : m.px) {
		ASSERT_GE(v, 0.0f);
		ASSERT_LE(v, 1.0f);
	}

	/* Determinism: identical input -> identical output. */
	fx::Mask m2 = model.infer(f);
	ASSERT_EQ(m.px, m2.px);
}
