#include <gtest/gtest.h>

#include "fx/models/face_embedder.h"

#include <cstdio>

static bool fileExists(const char *p)
{
	FILE *f = fopen(p, "r");
	if (f)
		fclose(f);
	return f != nullptr;
}

TEST(FaceEmbedder, LatentIsUnitNorm)
{
	if (!fileExists(FX_ARCFACE_PATH) || !fileExists(FX_INSWAPPER_PATH))
		GTEST_SKIP() << "runtime models not downloaded";
	fx::FaceEmbedder emb(FX_ARCFACE_PATH, FX_INSWAPPER_PATH, 1);
	std::vector<uint8_t> crop(112 * 112 * 3, 128);
	auto latent = emb.embed(crop);
	ASSERT_EQ(latent.size(), 512u);
	double n = 0;
	for (float v : latent)
		n += v * v;
	ASSERT_NEAR(n, 1.0, 1e-3);
}
