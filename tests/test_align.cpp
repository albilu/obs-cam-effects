#include <gtest/gtest.h>

#include "fx/image/align.h"

#include <cmath>

TEST(Umeyama, IdentityWhenSrcEqualsDst)
{
	fx::Landmarks5 t = fx::template128();
	fx::Affine23 m = fx::umeyama(t, t);
	ASSERT_NEAR(m.m[0], 1.0f, 1e-5f);
	ASSERT_NEAR(m.m[1], 0.0f, 1e-5f);
	ASSERT_NEAR(m.m[2], 0.0f, 1e-4f);
	ASSERT_NEAR(m.m[3], 0.0f, 1e-5f);
	ASSERT_NEAR(m.m[4], 1.0f, 1e-5f);
	ASSERT_NEAR(m.m[5], 0.0f, 1e-4f);
}

TEST(Umeyama, UniformScaleTranslate)
{
	fx::Landmarks5 src = {{{0, 0}, {10, 0}, {10, 10}, {0, 10}, {5, 5}}};
	fx::Landmarks5 dst;
	for (int i = 0; i < 5; i++)
		dst[i] = {2.0f * src[i][0] + 3.0f, 2.0f * src[i][1] + 7.0f};
	fx::Affine23 m = fx::umeyama(src, dst);
	ASSERT_NEAR(m.m[0], 2.0f, 1e-5f);
	ASSERT_NEAR(m.m[1], 0.0f, 1e-5f);
	ASSERT_NEAR(m.m[2], 3.0f, 1e-4f);
	ASSERT_NEAR(m.m[3], 0.0f, 1e-5f);
	ASSERT_NEAR(m.m[4], 2.0f, 1e-5f);
	ASSERT_NEAR(m.m[5], 7.0f, 1e-4f);
}

TEST(Affine, InvertRoundTrip)
{
	fx::Landmarks5 src = {{{12, 3}, {99, 20}, {60, 80}, {25, 90}, {70, 95}}};
	fx::Landmarks5 dst = fx::template128();
	fx::Affine23 m = fx::umeyama(src, dst);
	fx::Affine23 inv = fx::invertAffine(m);
	/* inv(m) · m(p) == p for all src points */
	for (const auto &p : src) {
		float x = m.m[0] * p[0] + m.m[1] * p[1] + m.m[2];
		float y = m.m[3] * p[0] + m.m[4] * p[1] + m.m[5];
		float ox = inv.m[0] * x + inv.m[1] * y + inv.m[2];
		float oy = inv.m[3] * x + inv.m[4] * y + inv.m[5];
		ASSERT_NEAR(ox, p[0], 1e-2f);
		ASSERT_NEAR(oy, p[1], 1e-2f);
	}
}

TEST(WarpAffine, TranslateByIntegerPixels)
{
	/* 4x4 single-channel-ish BGR image with a marked pixel at (1,1). */
	uint8_t src[4 * 4 * 3] = {0};
	src[(1 * 4 + 1) * 3 + 0] = 255;
	uint8_t dst[4 * 4 * 3] = {0};
	fx::Affine23 m; /* translate (+2,+1): dst(3,2) should be 255 */
	m.m[0] = 1; m.m[1] = 0; m.m[2] = 2;
	m.m[3] = 0; m.m[4] = 1; m.m[5] = 1;
	fx::warpAffineBilinear(src, 4, 4, 3, m, dst, 4, 4);
	ASSERT_EQ(dst[(2 * 4 + 3) * 3 + 0], 255);
	ASSERT_EQ(dst[(1 * 4 + 1) * 3 + 0], 0);
}

TEST(EllipseMask, CenterOneCornerZero)
{
	auto mask = fx::ellipseMask(128, 0.35f, 0.45f, 8);
	ASSERT_EQ(mask.size(), 128u * 128u);
	ASSERT_FLOAT_EQ(mask[(64 * 128) + 64], 1.0f);
	ASSERT_FLOAT_EQ(mask[0], 0.0f);
	ASSERT_FLOAT_EQ(mask[(127 * 128) + 127], 0.0f);
}

TEST(UnsharpMask, SharpensStep)
{
	uint8_t img[8 * 3];
	for (int i = 0; i < 8; i++) {
		uint8_t v = i < 4 ? 50 : 200;
		img[i * 3 + 0] = v;
		img[i * 3 + 1] = v;
		img[i * 3 + 2] = v;
	}
	uint8_t before[8 * 3];
	memcpy(before, img, sizeof(before));
	fx::unsharpMask(img, 8, 1, 3, 2, 1.0f);
	ASSERT_LT(img[3 * 3], before[3 * 3]); // darker side dips
	ASSERT_GT(img[4 * 3], before[4 * 3]); // bright side overshoots
}

TEST(RestoreMouthRegion, OnlyMouthRestored)
{
	const int w = 32, h = 32, ch = 3;
	std::vector<uint8_t> img(w * h * ch, 200), orig(w * h * ch, 50);
	fx::restoreMouthRegion(img.data(), orig.data(), w, h, ch, {10, 24},
			       {22, 24}, 1.0f, 2);
	ASSERT_EQ(img[(24 * w + 16) * 3], 50);	 // mouth center restored
	ASSERT_EQ(img[(4 * w + 4) * 3], 200);	 // far away untouched
}

TEST(Watermark, StampsPixels)
{
	std::vector<uint8_t> img(64 * 64 * 3, 0);
	fx::stampWatermarkAI(img.data(), 64, 64, 3);
	bool anyLit = false;
	for (uint8_t v : img)
		anyLit |= (v > 200);
	ASSERT_TRUE(anyLit);
}
