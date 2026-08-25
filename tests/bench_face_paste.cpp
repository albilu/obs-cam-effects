#include "face_paste_reference.h"
#include "fx/image/align.h"
#include "fx/image/face_paste.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

constexpr int kCropSize = 128;
constexpr int kWarmupIterations = 5;
constexpr int kMeasuredIterations = 31;

std::atomic<uint64_t> checksumSink{0};

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
			pixel[0] = (uint8_t)((x * 19 + y * 37 + 17) & 255);
			pixel[1] = (uint8_t)((x * 71 + y * 13 + 89) & 255);
			pixel[2] = (uint8_t)((x * 43 + y * 97 + x * y) & 255);
		}
	}
	return face;
}

std::vector<uint8_t> makeMask()
{
	const std::vector<float> ellipse = fx::ellipseMask(kCropSize, 0.35f, 0.45f, 12);
	std::vector<uint8_t> mask(ellipse.size());
	for (size_t i = 0; i < ellipse.size(); i++)
		mask[i] = (uint8_t)std::lround(ellipse[i] * 255.0f);
	return mask;
}

fx::Affine23 makeCenteredTransform(int width, int height)
{
	const float left = (float)(width - kCropSize) * 0.5f;
	const float top = (float)(height - kCropSize) * 0.5f;
	return {{1.0f, 0.0f, -left, 0.0f, 1.0f, -top}};
}

void consumeResult(const fx::Frame &frame)
{
	uint64_t checksum = 1469598103934665603ULL;
	const int left = (frame.width - kCropSize) / 2;
	const int top = (frame.height - kCropSize) / 2;
	for (int y = 0; y < kCropSize; y += 16) {
		for (int x = 0; x < kCropSize; x += 16) {
			const uint8_t *pixel =
				frame.bgra.data() + ((size_t)(top + y) * (size_t)frame.width + (size_t)(left + x)) * 4;
			for (int channel = 0; channel < 4; channel++) {
				checksum ^= pixel[channel];
				checksum *= 1099511628211ULL;
			}
		}
	}
	checksumSink.fetch_add(checksum, std::memory_order_relaxed);
}

template<typename Paste>
double medianMilliseconds(const fx::Frame &source, const std::vector<uint8_t> &face, const std::vector<uint8_t> &mask,
			  const fx::Affine23 &frameToCrop, Paste paste)
{
	std::array<double, kMeasuredIterations> samples{};
	for (int iteration = -kWarmupIterations; iteration < kMeasuredIterations; iteration++) {
		fx::Frame frame = source;
		const auto start = std::chrono::steady_clock::now();
		paste(frame, face, mask, kCropSize, frameToCrop, 1.0f);
		const auto end = std::chrono::steady_clock::now();
		consumeResult(frame);
		if (iteration >= 0)
			samples[(size_t)iteration] = std::chrono::duration<double, std::milli>(end - start).count();
	}

	std::sort(samples.begin(), samples.end());
	return samples[samples.size() / 2];
}

void runResolution(int width, int height)
{
	const fx::Frame source = makeFrame(width, height);
	const std::vector<uint8_t> face = makeFace();
	const std::vector<uint8_t> mask = makeMask();
	const fx::Affine23 transform = makeCenteredTransform(width, height);
	const double full = medianMilliseconds(source, face, mask, transform, fx::test::pasteFaceCropReference);
	const double bounded = medianMilliseconds(source, face, mask, transform, fx::pasteFaceCrop);
	const uint64_t scratchBytes = (uint64_t)width * (uint64_t)height * (3 + 1);

	std::cout << width << 'x' << height << " full=" << std::fixed << std::setprecision(3) << full
		  << " ms bounded=" << bounded << " ms speedup=" << full / bounded
		  << "x old_full_frame_scratch=" << scratchBytes << " bytes\n";
}

} // namespace

int main()
{
	runResolution(640, 480);
	runResolution(1920, 1080);
	return 0;
}
