#include <gtest/gtest.h>

#include "fx/models/mediapipe_lite.h"

TEST(MediaPipeLite, Infers192MaskThroughInterface)
{
	fx::MediaPipeLite model(FX_MEDIAPIPE_MODEL_PATH, 1);
	fx::Frame f;
	f.width = 320;
	f.height = 240;
	f.bgra.assign(320u * 240u * 4u, 128);

	fx::SegmentationModel &iface = model;
	fx::Mask m = iface.infer(f);
	ASSERT_EQ(m.width, 192);
	ASSERT_EQ(m.height, 192);
	ASSERT_EQ(m.px.size(), 192u * 192u);
	for (float v : m.px) {
		ASSERT_GE(v, 0.0f);
		ASSERT_LE(v, 1.0f);
	}
	fx::Mask m2 = iface.infer(f);
	ASSERT_EQ(m.px, m2.px); // determinism
}
