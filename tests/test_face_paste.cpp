#include <gtest/gtest.h>

#include "face_paste_reference.h"
#include "fx/image/face_paste.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

constexpr int kCropSize = 128;

fx::Frame makeFrame(int width, int height)
{
	fx::Frame frame;
	frame.width = width;
	frame.height = height;
	frame.bgra.resize((size_t)width * (size_t)height * 4);
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			uint8_t *pixel = frame.bgra.data() + ((size_t)y * (size_t)width + (size_t)x) * 4;
			pixel[0] = (uint8_t)((x * 29 + y * 17 + 11) & 255);
			pixel[1] = (uint8_t)((x * 7 + y * 43 + 73) & 255);
			pixel[2] = (uint8_t)((x * 53 + y * 3 + 149) & 255);
			pixel[3] = (uint8_t)((x * 13 + y * 19 + 191) & 255);
		}
	}
	return frame;
}

std::vector<uint8_t> makeFace()
{
	std::vector<uint8_t> face((size_t)kCropSize * kCropSize * 3);
	for (int y = 0; y < kCropSize; y++) {
		for (int x = 0; x < kCropSize; x++) {
			uint8_t *pixel = face.data() + ((size_t)y * kCropSize + (size_t)x) * 3;
			pixel[0] = ((x ^ y) & 1) != 0 ? 255 : 0;
			pixel[1] = (uint8_t)((x * 97 + y * 61 + 37) & 255);
			pixel[2] = (uint8_t)((x * 11 + y * 173 + x * y) & 255);
		}
	}
	return face;
}

std::vector<uint8_t> makeMask()
{
	std::vector<uint8_t> mask((size_t)kCropSize * kCropSize);
	for (int y = 0; y < kCropSize; y++) {
		for (int x = 0; x < kCropSize; x++) {
			mask[(size_t)y * kCropSize + (size_t)x] =
				((x + y) & 1) != 0 ? 255 : (uint8_t)(23 + ((x * 31 + y * 47) % 181));
		}
	}
	return mask;
}

fx::Affine23 makeFrameToCrop(float centerX, float centerY, float scale, float radians)
{
	const float cosine = std::cos(radians);
	const float sine = std::sin(radians);
	const float half = (float)kCropSize * 0.5f;
	fx::Affine23 cropToFrame = {{scale * cosine, -scale * sine, 0.0f, scale * sine, scale * cosine, 0.0f}};
	cropToFrame.m[2] = centerX - (cropToFrame.m[0] * half + cropToFrame.m[1] * half);
	cropToFrame.m[5] = centerY - (cropToFrame.m[3] * half + cropToFrame.m[4] * half);
	return fx::invertAffine(cropToFrame);
}

void expectBytesEqual(const std::vector<uint8_t> &actual, const std::vector<uint8_t> &expected)
{
	ASSERT_EQ(actual.size(), expected.size());
	const auto mismatch = std::mismatch(actual.begin(), actual.end(), expected.begin());
	if (mismatch.first != actual.end()) {
		const size_t index = (size_t)(mismatch.first - actual.begin());
		FAIL() << "first mismatch at byte " << index << ": actual=" << (int)*mismatch.first
		       << ", expected=" << (int)*mismatch.second;
	}
}

void expectMatchesReference(int width, int height, const fx::Affine23 &frameToCrop, float intensity)
{
	const std::vector<uint8_t> face = makeFace();
	const std::vector<uint8_t> mask = makeMask();
	fx::Frame actual = makeFrame(width, height);
	fx::Frame expected = actual;
	fx::test::pasteFaceCropReference(expected, face, mask, kCropSize, frameToCrop, intensity);
	fx::pasteFaceCrop(actual, face, mask, kCropSize, frameToCrop, intensity);
	expectBytesEqual(actual.bgra, expected.bgra);
}

void expectTransformNoOp(const fx::Affine23 &frameToCrop)
{
	const std::vector<uint8_t> face = makeFace();
	const std::vector<uint8_t> mask = makeMask();
	fx::Frame frame = makeFrame(64, 48);
	const std::vector<uint8_t> original = frame.bgra;
	fx::pasteFaceCrop(frame, face, mask, kCropSize, frameToCrop, 1.0f);
	expectBytesEqual(frame.bgra, original);
}

} // namespace

TEST(FacePaste, MatchesReferenceAt640x480AcrossIntensities)
{
	const fx::Affine23 transform = makeFrameToCrop(301.25f, 247.75f, 1.37f, 0.0f);
	for (float intensity : std::array<float, 3>{0.0f, 0.5f, 1.0f}) {
		SCOPED_TRACE(intensity);
		expectMatchesReference(640, 480, transform, intensity);
	}
}

TEST(FacePaste, MatchesReferenceAt1920x1080AcrossIntensities)
{
	const fx::Affine23 transform = makeFrameToCrop(947.375f, 557.625f, 1.73f, 0.0f);
	for (float intensity : std::array<float, 3>{0.0f, 0.5f, 1.0f}) {
		SCOPED_TRACE(intensity);
		expectMatchesReference(1920, 1080, transform, intensity);
	}
}

TEST(FacePaste, MatchesReferenceWhenCropCrossesLeftEdge)
{
	expectMatchesReference(640, 480, makeFrameToCrop(23.25f, 239.5f, 1.29f, 0.0f), 1.0f);
}

TEST(FacePaste, MatchesReferenceWhenCropCrossesRightEdge)
{
	expectMatchesReference(640, 480, makeFrameToCrop(616.75f, 239.5f, 1.29f, 0.0f), 1.0f);
}

TEST(FacePaste, MatchesReferenceWhenCropCrossesTopEdge)
{
	expectMatchesReference(640, 480, makeFrameToCrop(319.5f, 21.75f, 1.29f, 0.0f), 1.0f);
}

TEST(FacePaste, MatchesReferenceWhenCropCrossesBottomEdge)
{
	expectMatchesReference(640, 480, makeFrameToCrop(319.5f, 458.25f, 1.29f, 0.0f), 1.0f);
}

TEST(FacePaste, MatchesReferenceForRotatedScaledTransformAtHalfIntensity)
{
	expectMatchesReference(640, 480, makeFrameToCrop(318.375f, 237.625f, 1.83f, 0.37f), 0.5f);
}

TEST(FacePaste, PreservesUnclampedNegativeIntensitySemantics)
{
	expectMatchesReference(640, 480, makeFrameToCrop(287.125f, 263.875f, 1.41f, -0.19f), -0.25f);
}

TEST(FacePaste, LeavesOutsidePixelsByteIdentical)
{
	const std::vector<uint8_t> face = makeFace();
	const std::vector<uint8_t> mask = makeMask();
	fx::Frame frame = makeFrame(640, 480);
	const std::vector<uint8_t> original = frame.bgra;
	const fx::Affine23 transform = makeFrameToCrop(321.25f, 241.75f, 1.47f, 0.23f);
	fx::pasteFaceCrop(frame, face, mask, kCropSize, transform, 1.0f);

	for (int y = 0; y < frame.height; y++) {
		for (int x = 0; x < frame.width; x++) {
			const float u = transform.m[0] * (float)x + transform.m[1] * (float)y + transform.m[2];
			const float v = transform.m[3] * (float)x + transform.m[4] * (float)y + transform.m[5];
			if (u >= 0.0f && u < (float)kCropSize && v >= 0.0f && v < (float)kCropSize)
				continue;
			const size_t offset = ((size_t)y * (size_t)frame.width + (size_t)x) * 4;
			for (int c = 0; c < 4; c++)
				ASSERT_EQ(frame.bgra[offset + (size_t)c], original[offset + (size_t)c]);
		}
	}
}

TEST(FacePaste, InvalidDimensionsAndBuffersAreNoOps)
{
	const std::vector<uint8_t> validFace = makeFace();
	const std::vector<uint8_t> validMask = makeMask();
	const fx::Affine23 transform = makeFrameToCrop(16.0f, 12.0f, 1.0f, 0.0f);

	auto expectNoOp = [&](fx::Frame frame, std::vector<uint8_t> face, std::vector<uint8_t> mask, int cropSize) {
		const fx::Frame before = frame;
		fx::pasteFaceCrop(frame, face, mask, cropSize, transform, 0.5f);
		EXPECT_EQ(frame.width, before.width);
		EXPECT_EQ(frame.height, before.height);
		expectBytesEqual(frame.bgra, before.bgra);
	};

	fx::Frame validFrame = makeFrame(32, 24);
	fx::Frame zeroWidth = validFrame;
	zeroWidth.width = 0;
	expectNoOp(zeroWidth, validFace, validMask, kCropSize);
	expectNoOp(validFrame, validFace, validMask, 0);

	fx::Frame shortFrame = validFrame;
	shortFrame.bgra.pop_back();
	expectNoOp(shortFrame, validFace, validMask, kCropSize);
	std::vector<uint8_t> shortFace = validFace;
	shortFace.pop_back();
	expectNoOp(validFrame, shortFace, validMask, kCropSize);
	std::vector<uint8_t> longMask = validMask;
	longMask.push_back(0);
	expectNoOp(validFrame, validFace, longMask, kCropSize);
}

TEST(FacePaste, ExactlySingularZeroAffineIsNoOp)
{
	const fx::Affine23 transform = {{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}};
	expectTransformNoOp(transform);
}

TEST(FacePaste, NearSingularAffineBelowInvertThresholdIsNoOp)
{
	const fx::Affine23 transform = {{1.0e-7f, 0.0f, 0.0f, 0.0f, 1.0e-6f, 0.0f}};
	expectTransformNoOp(transform);
}

TEST(FacePaste, AffineContainingNaNIsNoOp)
{
	const float nan = std::numeric_limits<float>::quiet_NaN();
	const fx::Affine23 transform = {{1.0f, 0.0f, nan, 0.0f, 1.0f, 0.0f}};
	expectTransformNoOp(transform);
}

TEST(FacePaste, AffineContainingInfinityIsNoOp)
{
	const float infinity = std::numeric_limits<float>::infinity();
	for (float translation : std::array<float, 2>{-infinity, infinity}) {
		SCOPED_TRACE(translation);
		const fx::Affine23 transform = {{1.0f, 0.0f, translation, 0.0f, 1.0f, 0.0f}};
		expectTransformNoOp(transform);
	}
}

TEST(FacePaste, HugeFiniteOffscreenTranslationsAreNoOps)
{
	for (float translation : std::array<float, 2>{-3.0e9f, 3.0e9f}) {
		SCOPED_TRACE(translation);
		const fx::Affine23 horizontal = {{1.0f, 0.0f, translation, 0.0f, 1.0f, 0.0f}};
		expectTransformNoOp(horizontal);
		const fx::Affine23 vertical = {{1.0f, 0.0f, 0.0f, 0.0f, 1.0f, translation}};
		expectTransformNoOp(vertical);
	}
}
