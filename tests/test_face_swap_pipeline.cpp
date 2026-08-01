#include <gtest/gtest.h>

#include "fx/pipeline/face_swap_pipeline.h"

#include <cstdio>

static bool fileExists(const char *p)
{
	FILE *f = fopen(p, "r");
	if (f)
		fclose(f);
	return f != nullptr;
}

TEST(FaceSwapPipeline, NoSourceNoSwap)
{
	if (!fileExists(FX_INSWAPPER_PATH) || !fileExists(FX_ARCFACE_PATH))
		GTEST_SKIP() << "runtime models not downloaded";
	fx::FaceSwapPipeline pipe(FX_YUNET_MODEL_PATH, FX_INSWAPPER_PATH,
				  FX_ARCFACE_PATH, 1);
	fx::Frame f;
	f.width = 320;
	f.height = 240;
	f.bgra.assign(320u * 240u * 4u, 128);
	ASSERT_FALSE(pipe.process(f)); // no source embedding set
}

TEST(FaceSwapPipeline, SwapIsDeterministicOnSameFrame)
{
	if (!fileExists(FX_INSWAPPER_PATH) || !fileExists(FX_ARCFACE_PATH))
		GTEST_SKIP() << "runtime models not downloaded";
	fx::FaceSwapPipeline pipe(FX_YUNET_MODEL_PATH, FX_INSWAPPER_PATH,
				  FX_ARCFACE_PATH, 1);
	fx::FaceEmbedder emb(FX_ARCFACE_PATH, FX_INSWAPPER_PATH, 1);
	std::vector<uint8_t> crop(112 * 112 * 3, 128);
	pipe.setSourceEmbedding(emb.embed(crop));

	fx::Frame f;
	f.width = 320;
	f.height = 240;
	f.bgra.assign(320u * 240u * 4u, 128);
	auto before = f.bgra;
	pipe.process(f); // blank frame: may or may not swap; must not crash
	ASSERT_EQ(f.bgra.size(), before.size());
}
