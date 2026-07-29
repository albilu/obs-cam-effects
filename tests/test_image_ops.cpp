#include <gtest/gtest.h>

#include "fx/image/ops.h"

#include <cmath>

TEST(GrayFromBgra, LumaCoefficients)
{
	fx::Frame f;
	f.width = 1;
	f.height = 1;
	f.bgra = {50, 100, 200, 255}; // B=50, G=100, R=200
	std::vector<float> g = fx::grayFromBgra(f);
	ASSERT_EQ(g.size(), 1u);
	float expect = (0.114f * 50 + 0.587f * 100 + 0.299f * 200) / 255.0f;
	ASSERT_NEAR(g[0], expect, 1e-6f);
}

TEST(EmaMask, ConvergesToNewValue)
{
	fx::Mask cur, prev;
	cur.width = prev.width = 2;
	cur.height = prev.height = 2;
	cur.px = {1.0f, 1.0f, 1.0f, 1.0f};
	prev.px = {0.0f, 0.0f, 0.0f, 0.0f};
	fx::emaMask(cur, prev, 0.6f); // 0.6*prev + 0.4*cur
	for (float v : cur.px)
		ASSERT_NEAR(v, 0.4f, 1e-6f);
}

TEST(GuidedFilter, PreservesStepEdge)
{
	/* 16x1 guide with a step at x=8; src = guide + heavy noise. */
	const int w = 16, h = 1;
	std::vector<float> guide(w * h), src(w * h);
	for (int x = 0; x < w; x++) {
		guide[x] = x < 8 ? 0.0f : 1.0f;
		src[x] = guide[x] + (x % 3 == 0 ? 0.2f : -0.1f);
	}
	std::vector<float> out = fx::guidedFilter(guide, src, w, h, 3, 1e-3f);
	/* Edge cells stay separated (edge preserved), unlike a box blur
	 * which would pull x=7 and x=8 toward ~0.5. */
	ASSERT_LT(out[7], 0.35f);
	ASSERT_GT(out[8], 0.65f);
}

TEST(GuidedFilter, SmoothsFlatGuide)
{
	/* Flat guide -> behaves like a local average (noise removed). */
	const int w = 8, h = 8;
	std::vector<float> guide(w * h, 0.5f), src(w * h);
	for (int i = 0; i < w * h; i++)
		src[i] = (i % 2 == 0) ? 0.9f : 0.1f;
	std::vector<float> out = fx::guidedFilter(guide, src, w, h, 2, 1e-3f);
	float mean = 0;
	for (float v : out)
		mean += v;
	mean /= out.size();
	ASSERT_NEAR(mean, 0.5f, 0.05f);
}
