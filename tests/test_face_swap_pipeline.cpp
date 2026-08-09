#include <gtest/gtest.h>

#include "fx/pipeline/face_swap_pipeline.h"

#include "fx/third_party/stb_image.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

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

/* Detection decimation on blank frames: with no face ever found, every
 * frame is a detect frame (no reusable box) and must return false
 * without crashing, detectEveryN=2 included. */
TEST(FaceSwapPipeline, DetectEveryNBlankFrames)
{
	if (!fileExists(FX_INSWAPPER_PATH) || !fileExists(FX_ARCFACE_PATH))
		GTEST_SKIP() << "runtime models not downloaded";
	fx::FaceSwapPipeline pipe(FX_YUNET_MODEL_PATH, FX_INSWAPPER_PATH,
				  FX_ARCFACE_PATH, 1);
	fx::FaceEmbedder emb(FX_ARCFACE_PATH, FX_INSWAPPER_PATH, 1);
	std::vector<uint8_t> crop(112 * 112 * 3, 128);
	pipe.setSourceEmbedding(emb.embed(crop));
	fx::FaceSwapParams params;
	params.detectEveryN = 2;
	pipe.setParams(params);

	fx::Frame f;
	f.width = 320;
	f.height = 240;
	f.bgra.assign(320u * 240u * 4u, 128);
	for (int i = 0; i < 8; i++)
		ASSERT_FALSE(pipe.process(f))
			<< "blank frame " << i << " must not swap";
}

/* Decimation must NOT skip the swap itself: with detectEveryN=2 the
 * second frame reuses the previous box and is still swapped. */
TEST(FaceSwapPipeline, DetectEveryNStillSwapsEveryFrame)
{
	if (!fileExists(FX_INSWAPPER_PATH) || !fileExists(FX_ARCFACE_PATH))
		GTEST_SKIP() << "runtime models not downloaded";
	int w = 0, h = 0, channels = 0;
	stbi_uc *rgb = stbi_load(FX_FIXTURE_FACE_PATH, &w, &h, &channels, 3);
	ASSERT_TRUE(rgb != nullptr) << "fixture missing: "
				    << FX_FIXTURE_FACE_PATH;
	fx::Frame f;
	f.width = w;
	f.height = h;
	f.bgra.resize((size_t)w * (size_t)h * 4);
	for (size_t i = 0, n = (size_t)w * (size_t)h; i < n; i++) {
		f.bgra[i * 4 + 0] = rgb[i * 3 + 2];
		f.bgra[i * 4 + 1] = rgb[i * 3 + 1];
		f.bgra[i * 4 + 2] = rgb[i * 3 + 0];
		f.bgra[i * 4 + 3] = 255;
	}
	stbi_image_free(rgb);

	fx::FaceSwapPipeline pipe(FX_YUNET_MODEL_PATH, FX_INSWAPPER_PATH,
				  FX_ARCFACE_PATH, 1);
	fx::FaceEmbedder emb(FX_ARCFACE_PATH, FX_INSWAPPER_PATH, 1);
	std::vector<uint8_t> crop(112 * 112 * 3, 128);
	pipe.setSourceEmbedding(emb.embed(crop));
	fx::FaceSwapParams params;
	params.detectEveryN = 2;
	pipe.setParams(params);

	/* Frame 0 detects; frames 1 and 3 reuse the box; frame 2
	 * detects again. All four must swap. */
	for (int i = 0; i < 4; i++) {
		fx::Frame work = f;
		ASSERT_TRUE(pipe.process(work))
			<< "frame " << i << " must be swapped";
	}
}

/* Regression: the paste-back blend must not touch ANY pixel outside the
 * face region (whole-frame mask bleed: the feathered ellipse does not
 * reach 0 at the crop border, and the mask warp clamps out-of-crop
 * samples to that border, smearing swap content across the frame). */
TEST(FaceSwapPipeline, NoBleedOutsideFaceRegion)
{
	if (!fileExists(FX_INSWAPPER_PATH) || !fileExists(FX_ARCFACE_PATH))
		GTEST_SKIP() << "runtime models not downloaded";

	/* Committed PD fixture (see tests/data/README.txt). */
	int w = 0, h = 0, channels = 0;
	stbi_uc *rgb = stbi_load(FX_FIXTURE_FACE_PATH, &w, &h, &channels, 3);
	ASSERT_TRUE(rgb != nullptr) << "fixture missing: "
				    << FX_FIXTURE_FACE_PATH;
	fx::Frame f;
	f.width = w;
	f.height = h;
	f.bgra.resize((size_t)w * (size_t)h * 4);
	for (size_t i = 0, n = (size_t)w * (size_t)h; i < n; i++) {
		f.bgra[i * 4 + 0] = rgb[i * 3 + 2];
		f.bgra[i * 4 + 1] = rgb[i * 3 + 1];
		f.bgra[i * 4 + 2] = rgb[i * 3 + 0];
		f.bgra[i * 4 + 3] = 255;
	}
	stbi_image_free(rgb);

	/* Largest detection box, padded 2x about its center. */
	fx::YuNet det(FX_YUNET_MODEL_PATH, 1);
	auto faces = det.detect(f);
	ASSERT_FALSE(faces.empty())
		<< "fixture must contain a detectable face";
	const fx::FaceBox *best = &faces[0];
	for (const auto &b : faces)
		if (b.w * b.h > best->w * best->h)
			best = &b;
	const float cx = best->x + best->w * 0.5f;
	const float cy = best->y + best->h * 0.5f;
	const int px0 = std::max(0, (int)std::floor(cx - best->w));
	const int py0 = std::max(0, (int)std::floor(cy - best->h));
	const int px1 = std::min(w - 1, (int)std::ceil(cx + best->w));
	const int py1 = std::min(h - 1, (int)std::ceil(cy + best->h));

	const auto orig = f.bgra;

	fx::FaceSwapPipeline pipe(FX_YUNET_MODEL_PATH, FX_INSWAPPER_PATH,
				  FX_ARCFACE_PATH, 1);
	fx::FaceEmbedder emb(FX_ARCFACE_PATH, FX_INSWAPPER_PATH, 1);
	std::vector<uint8_t> crop(112 * 112 * 3, 128);
	pipe.setSourceEmbedding(emb.embed(crop));

	ASSERT_TRUE(pipe.process(f)) << "fixture face must be swapped";

	size_t changed = 0, firstByte = 0;
	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			if (x >= px0 && x <= px1 && y >= py0 && y <= py1)
				continue;
			const size_t i = ((size_t)y * (size_t)w + x) * 4;
			if (memcmp(f.bgra.data() + i, orig.data() + i, 4) != 0) {
				if (changed == 0)
					firstByte = i;
				changed++;
			}
		}
	}
	ASSERT_EQ(changed, 0u)
		<< "outside-region pixels changed, first at byte " << firstByte;
}
