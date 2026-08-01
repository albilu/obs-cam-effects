#include <gtest/gtest.h>

#include "fx/models/yunet.h"

TEST(YuNet, NoCrashOnBlankFrame)
{
	fx::YuNet det(FX_YUNET_MODEL_PATH, 1);
	fx::Frame f;
	f.width = 320;
	f.height = 240;
	f.bgra.assign(320u * 240u * 4u, 128);
	auto faces = det.detect(f);
	/* Blank frame: no assertion on count (model may fire or not on
	 * noise) — the contract is: valid geometry on whatever returns. */
	for (const auto &b : faces) {
		ASSERT_GT(b.w, 0.0f);
		ASSERT_GT(b.h, 0.0f);
		ASSERT_GE(b.score, 0.6f);
	}
}
