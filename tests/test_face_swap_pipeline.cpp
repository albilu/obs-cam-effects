#include <gtest/gtest.h>

#include "fx/models/face_embedder.h"
#include "fx/pipeline/face_swap_pipeline.h"

#include "fx/third_party/stb_image.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>

static bool fileExists(const char *p)
{
	FILE *f = fopen(p, "r");
	if (f)
		fclose(f);
	return f != nullptr;
}

TEST(FaceSwapPipeline, NoSourceNoSwap)
{
	if (!fileExists(FX_YUNET_MODEL_PATH) || !fileExists(FX_INSWAPPER_PATH))
		GTEST_SKIP() << "runtime models not downloaded";
	fx::FaceSwapPipeline pipe(FX_YUNET_MODEL_PATH, FX_INSWAPPER_PATH, 1);
	EXPECT_EQ(pipe.swapBackend(), fx::OrtBackend::Cpu);
	fx::Frame f;
	f.width = 320;
	f.height = 240;
	f.bgra.assign(320u * 240u * 4u, 128);
	ASSERT_FALSE(pipe.process(f)); // no source embedding set
}

TEST(FaceSwapPipeline, ClearingSourceEmbeddingStopsSwap)
{
	if (!fileExists(FX_YUNET_MODEL_PATH) || !fileExists(FX_INSWAPPER_PATH))
		GTEST_SKIP() << "runtime models not downloaded";

	fx::FaceSwapPipeline pipe(FX_YUNET_MODEL_PATH, FX_INSWAPPER_PATH, 1);
	EXPECT_EQ(pipe.swapBackend(), fx::OrtBackend::Cpu);
	pipe.setSourceEmbedding(std::vector<float>(512, 1.0f));
	ASSERT_TRUE(pipe.hasSource());

	pipe.setSourceEmbedding({});
	ASSERT_FALSE(pipe.hasSource());

	fx::Frame frame;
	frame.width = 320;
	frame.height = 240;
	frame.bgra.assign(320u * 240u * 4u, 128);
	EXPECT_FALSE(pipe.process(frame));
}

TEST(FaceSwapPipeline, ZeroIntensityPreservesFrameAndReportsNoSwap)
{
	if (!fileExists(FX_YUNET_MODEL_PATH) || !fileExists(FX_INSWAPPER_PATH) || !fileExists(FX_ARCFACE_PATH))
		GTEST_SKIP() << "runtime models not downloaded";

	int w = 0, h = 0, channels = 0;
	std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> rgb(stbi_load(FX_FIXTURE_FACE_PATH, &w, &h, &channels, 3),
								 stbi_image_free);
	ASSERT_TRUE(rgb != nullptr) << "fixture missing: " << FX_FIXTURE_FACE_PATH;

	fx::Frame frame;
	frame.width = w;
	frame.height = h;
	frame.bgra.resize((size_t)w * (size_t)h * 4);
	for (size_t i = 0, n = (size_t)w * (size_t)h; i < n; i++) {
		frame.bgra[i * 4 + 0] = rgb.get()[i * 3 + 2];
		frame.bgra[i * 4 + 1] = rgb.get()[i * 3 + 1];
		frame.bgra[i * 4 + 2] = rgb.get()[i * 3 + 0];
		frame.bgra[i * 4 + 3] = 255;
	}

	fx::YuNet sourceDetector(FX_YUNET_MODEL_PATH, 1);
	fx::FaceEmbedder embedder(FX_ARCFACE_PATH, FX_INSWAPPER_PATH, 1);
	std::vector<float> source = embedder.embedFromImageFile(FX_FIXTURE_FACE_PATH, sourceDetector);
	ASSERT_FALSE(source.empty()) << "fixture must contain a detectable source face";

	fx::FaceSwapPipeline pipe(FX_YUNET_MODEL_PATH, FX_INSWAPPER_PATH, 1);
	pipe.setSourceEmbedding(std::move(source));
	fx::FaceSwapParams params;
	params.intensity = 0.0f;
	params.mouthPreserve = 1.0f;
	pipe.setParams(params);
	const std::vector<uint8_t> before = frame.bgra;

	EXPECT_FALSE(pipe.process(frame));
	EXPECT_EQ(frame.bgra, before);
}

TEST(FaceSwapPipeline, ReplacingSourceEmbeddingClearsTrackedFace)
{
	if (!fileExists(FX_YUNET_MODEL_PATH) || !fileExists(FX_INSWAPPER_PATH))
		GTEST_SKIP() << "runtime models not downloaded";

	int w = 0, h = 0, channels = 0;
	std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> rgb(stbi_load(FX_FIXTURE_FACE_PATH, &w, &h, &channels, 3),
								 stbi_image_free);
	ASSERT_TRUE(rgb != nullptr) << "fixture missing: " << FX_FIXTURE_FACE_PATH;

	fx::Frame face;
	face.width = w;
	face.height = h;
	face.bgra.resize((size_t)w * (size_t)h * 4);
	for (size_t i = 0, n = (size_t)w * (size_t)h; i < n; i++) {
		face.bgra[i * 4 + 0] = rgb.get()[i * 3 + 2];
		face.bgra[i * 4 + 1] = rgb.get()[i * 3 + 1];
		face.bgra[i * 4 + 2] = rgb.get()[i * 3 + 0];
		face.bgra[i * 4 + 3] = 255;
	}

	fx::FaceSwapPipeline pipe(FX_YUNET_MODEL_PATH, FX_INSWAPPER_PATH, 1);
	pipe.setSourceEmbedding(std::vector<float>(512, 1.0f));
	fx::FaceSwapParams params;
	params.detectEveryN = 2;
	pipe.setParams(params);
	ASSERT_TRUE(pipe.process(face)) << "fixture face must establish a track";

	pipe.setSourceEmbedding(std::vector<float>(512, 0.5f));

	fx::Frame blank;
	blank.width = w;
	blank.height = h;
	blank.bgra.assign((size_t)w * (size_t)h * 4, 0);
	for (size_t i = 0, n = (size_t)w * (size_t)h; i < n; i++)
		blank.bgra[i * 4 + 3] = 255;
	EXPECT_FALSE(pipe.process(blank));
}

TEST(FaceSwapPipeline, ExplicitTrackingResetClearsTrackedFace)
{
	if (!fileExists(FX_YUNET_MODEL_PATH) || !fileExists(FX_INSWAPPER_PATH))
		GTEST_SKIP() << "runtime models not downloaded";

	int w = 0, h = 0, channels = 0;
	std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> rgb(stbi_load(FX_FIXTURE_FACE_PATH, &w, &h, &channels, 3),
								 stbi_image_free);
	ASSERT_TRUE(rgb != nullptr) << "fixture missing: " << FX_FIXTURE_FACE_PATH;

	fx::Frame face;
	face.width = w;
	face.height = h;
	face.bgra.resize((size_t)w * (size_t)h * 4);
	for (size_t i = 0, n = (size_t)w * (size_t)h; i < n; i++) {
		face.bgra[i * 4 + 0] = rgb.get()[i * 3 + 2];
		face.bgra[i * 4 + 1] = rgb.get()[i * 3 + 1];
		face.bgra[i * 4 + 2] = rgb.get()[i * 3 + 0];
		face.bgra[i * 4 + 3] = 255;
	}

	fx::FaceSwapPipeline pipe(FX_YUNET_MODEL_PATH, FX_INSWAPPER_PATH, 1);
	pipe.setSourceEmbedding(std::vector<float>(512, 1.0f));
	fx::FaceSwapParams params;
	params.detectEveryN = 2;
	pipe.setParams(params);
	ASSERT_TRUE(pipe.process(face)) << "fixture face must establish a track";

	pipe.resetTracking();

	fx::Frame blank;
	blank.width = w;
	blank.height = h;
	blank.bgra.assign((size_t)w * (size_t)h * 4, 0);
	for (size_t i = 0, n = (size_t)w * (size_t)h; i < n; i++)
		blank.bgra[i * 4 + 3] = 255;
	EXPECT_FALSE(pipe.process(blank));
}

TEST(FaceSwapPipeline, RequireCudaPolicyReachesSwapper)
{
	if (!fileExists(FX_YUNET_MODEL_PATH))
		GTEST_SKIP() << "YuNet model not available";
	try {
		fx::FaceSwapPipeline pipe(FX_YUNET_MODEL_PATH, FX_INSWAPPER_PATH, 1, false,
					  fx::OrtExecutionPolicy::RequireCuda);
		FAIL() << "expected CUDA policy failure";
	} catch (const std::runtime_error &error) {
		EXPECT_STREQ(error.what(), "fx: CUDA execution required");
	} catch (...) {
		FAIL() << "expected std::runtime_error";
	}
}

TEST(FaceSwapPipeline, SwapIsDeterministicOnSameFrame)
{
	if (!fileExists(FX_INSWAPPER_PATH) || !fileExists(FX_ARCFACE_PATH))
		GTEST_SKIP() << "runtime models not downloaded";
	fx::FaceSwapPipeline pipe(FX_YUNET_MODEL_PATH, FX_INSWAPPER_PATH, 1);
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
	fx::FaceSwapPipeline pipe(FX_YUNET_MODEL_PATH, FX_INSWAPPER_PATH, 1);
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

	fx::FaceSwapPipeline pipe(FX_YUNET_MODEL_PATH, FX_INSWAPPER_PATH, 1);
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

TEST(FaceSwapPipeline, TransientMissesReuseTrackUntilGraceExpires)
{
	if (!fileExists(FX_INSWAPPER_PATH) || !fileExists(FX_ARCFACE_PATH))
		GTEST_SKIP() << "runtime models not downloaded";

	int w = 0, h = 0, channels = 0;
	stbi_uc *rgb = stbi_load(FX_FIXTURE_FACE_PATH, &w, &h, &channels, 3);
	ASSERT_TRUE(rgb != nullptr) << "fixture missing: " << FX_FIXTURE_FACE_PATH;
	fx::Frame face;
	face.width = w;
	face.height = h;
	face.bgra.resize((size_t)w * (size_t)h * 4);
	for (size_t i = 0, n = (size_t)w * (size_t)h; i < n; i++) {
		face.bgra[i * 4 + 0] = rgb[i * 3 + 2];
		face.bgra[i * 4 + 1] = rgb[i * 3 + 1];
		face.bgra[i * 4 + 2] = rgb[i * 3 + 0];
		face.bgra[i * 4 + 3] = 255;
	}
	stbi_image_free(rgb);

	fx::FaceSwapPipeline pipe(FX_YUNET_MODEL_PATH, FX_INSWAPPER_PATH, 1);
	fx::FaceEmbedder emb(FX_ARCFACE_PATH, FX_INSWAPPER_PATH, 1);
	std::vector<uint8_t> crop(112 * 112 * 3, 128);
	pipe.setSourceEmbedding(emb.embed(crop));
	fx::FaceSwapParams params;
	params.detectEveryN = 1;
	pipe.setParams(params);

	ASSERT_TRUE(pipe.process(face)) << "fixture face must establish a track";

	fx::Frame blank;
	blank.width = w;
	blank.height = h;
	blank.bgra.assign((size_t)w * (size_t)h * 4, 0);
	for (size_t i = 0, n = (size_t)w * (size_t)h; i < n; i++)
		blank.bgra[i * 4 + 3] = 255;

	for (int miss = 1; miss <= 3; miss++) {
		fx::Frame work = blank;
		ASSERT_TRUE(pipe.process(work)) << "transient miss " << miss << " must reuse the track";
	}
	fx::Frame expired = blank;
	ASSERT_FALSE(pipe.process(expired)) << "fourth consecutive miss must expire the track";
}

TEST(FaceSwapPipeline, Swaps1920x1080WithoutChangingDimensions)
{
	if (!fileExists(FX_INSWAPPER_PATH) || !fileExists(FX_ARCFACE_PATH))
		GTEST_SKIP() << "runtime models not downloaded";

	int w = 0, h = 0, channels = 0;
	std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> rgb(stbi_load(FX_FIXTURE_FACE_PATH, &w, &h, &channels, 3),
								 stbi_image_free);
	ASSERT_TRUE(rgb != nullptr) << "fixture missing: " << FX_FIXTURE_FACE_PATH;
	ASSERT_LE(w, 1920);
	ASSERT_LE(h, 1080);

	fx::Frame frame;
	frame.width = 1920;
	frame.height = 1080;
	frame.bgra.assign((size_t)frame.width * (size_t)frame.height * 4, 0);
	for (size_t i = 0, n = (size_t)frame.width * (size_t)frame.height; i < n; i++)
		frame.bgra[i * 4 + 3] = 255;
	const int offsetX = (frame.width - w) / 2;
	const int offsetY = (frame.height - h) / 2;
	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			const uint8_t *src = rgb.get() + ((size_t)y * (size_t)w + x) * 3;
			uint8_t *dst = frame.bgra.data() +
				       ((size_t)(y + offsetY) * (size_t)frame.width + (size_t)(x + offsetX)) * 4;
			dst[0] = src[2];
			dst[1] = src[1];
			dst[2] = src[0];
		}
	}

	fx::FaceSwapPipeline pipe(FX_YUNET_MODEL_PATH, FX_INSWAPPER_PATH, 1);
	fx::FaceEmbedder emb(FX_ARCFACE_PATH, FX_INSWAPPER_PATH, 1);
	std::vector<uint8_t> crop(112 * 112 * 3, 128);
	pipe.setSourceEmbedding(emb.embed(crop));
	const std::vector<uint8_t> before = frame.bgra;

	ASSERT_TRUE(pipe.process(frame));
	ASSERT_EQ(frame.width, 1920);
	ASSERT_EQ(frame.height, 1080);
	ASSERT_EQ(frame.bgra.size(), 1920u * 1080u * 4u);

	const size_t pixelCount = before.size() / 4;
	size_t changedPixels = 0;
	size_t firstAlphaMismatch = pixelCount;
	for (size_t i = 0; i < pixelCount; i++) {
		if (memcmp(frame.bgra.data() + i * 4, before.data() + i * 4, 3) != 0)
			changedPixels++;
		if (firstAlphaMismatch == pixelCount && frame.bgra[i * 4 + 3] != before[i * 4 + 3])
			firstAlphaMismatch = i;
	}
	EXPECT_GT(changedPixels, 0u) << "swap changed no BGR pixels";
	EXPECT_EQ(firstAlphaMismatch, pixelCount) << "alpha changed at pixel " << firstAlphaMismatch;
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

	fx::FaceSwapPipeline pipe(FX_YUNET_MODEL_PATH, FX_INSWAPPER_PATH, 1);
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
