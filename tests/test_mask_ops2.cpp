#include <gtest/gtest.h>

#include "fx/image/ops.h"

TEST(ContourFilter, RemovesSmallBlobs)
{
	/* 20x20 frame: one 8x8 blob (area 64) and one 2x2 blob (area 4). */
	fx::Mask m;
	m.width = 20;
	m.height = 20;
	m.px.assign(400, 0.0f);
	for (int y = 2; y < 10; y++)
		for (int x = 2; x < 10; x++)
			m.px[y * 20 + x] = 1.0f;
	for (int y = 14; y < 16; y++)
		for (int x = 14; x < 16; x++)
			m.px[y * 20 + x] = 1.0f;
	fx::contourFilter(m, 0.05f); // 5% of 400 = 20 px threshold
	float bigSum = 0, smallSum = 0;
	for (int y = 2; y < 10; y++)
		for (int x = 2; x < 10; x++)
			bigSum += m.px[y * 20 + x];
	for (int y = 14; y < 16; y++)
		for (int x = 14; x < 16; x++)
			smallSum += m.px[y * 20 + x];
	ASSERT_EQ(bigSum, 64.0f);
	ASSERT_EQ(smallSum, 0.0f);
}

TEST(ContourFilter, ZeroFracKeepsEverything)
{
	fx::Mask m;
	m.width = 4;
	m.height = 4;
	m.px.assign(16, 1.0f);
	fx::contourFilter(m, 0.0f);
	for (float v : m.px)
		ASSERT_EQ(v, 1.0f);
}

TEST(FeatherMask, BlursStepEdge)
{
	fx::Mask m;
	m.width = 16;
	m.height = 1;
	m.px.assign(16, 0.0f);
	for (int x = 8; x < 16; x++)
		m.px[x] = 1.0f;
	fx::featherMask(m, 2.0f);
	ASSERT_GT(m.px[7], 0.0f); // edge softened leftward
	ASSERT_LT(m.px[7], 1.0f);
	ASSERT_EQ(m.px[0], 0.0f); // far field untouched
}
