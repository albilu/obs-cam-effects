#include "fx/image/face_paste.h"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace fx {

namespace {

bool isFinite(const Affine23 &matrix)
{
	for (float coefficient : matrix.m) {
		if (!std::isfinite(coefficient))
			return false;
	}
	return true;
}

bool sampleAxis(float coordinate, int size, int &first, int &second, float &fraction)
{
	if (!std::isfinite(coordinate))
		return false;
	if (coordinate < 0.0f) {
		first = 0;
		second = 0;
		fraction = 0.0f;
		return true;
	}
	if (coordinate >= (float)(size - 1)) {
		first = size - 1;
		second = size - 1;
		fraction = 0.0f;
		return true;
	}

	first = (int)std::floor(coordinate);
	second = first + 1;
	fraction = coordinate - (float)first;
	return true;
}

uint8_t sampleBilinear(const uint8_t *src, int size, int channels, int channel, int xa, int xb, int ya, int yb,
		       float fx, float fy)
{
	const float v00 = src[((size_t)ya * (size_t)size + (size_t)xa) * (size_t)channels + (size_t)channel];
	const float v01 = src[((size_t)ya * (size_t)size + (size_t)xb) * (size_t)channels + (size_t)channel];
	const float v10 = src[((size_t)yb * (size_t)size + (size_t)xa) * (size_t)channels + (size_t)channel];
	const float v11 = src[((size_t)yb * (size_t)size + (size_t)xb) * (size_t)channels + (size_t)channel];
	const float value = (v00 * (1.0f - fx) + v01 * fx) * (1.0f - fy) + (v10 * (1.0f - fx) + v11 * fx) * fy;
	return (uint8_t)std::clamp((int)std::lround(value), 0, 255);
}

} // namespace

void pasteFaceCrop(Frame &frame, const std::vector<uint8_t> &face, const std::vector<uint8_t> &mask, int cropSize,
		   const Affine23 &frameToCrop, float intensity)
{
	const int width = frame.width;
	const int height = frame.height;
	if (width <= 0 || height <= 0 || cropSize <= 0)
		return;

	const size_t framePixels = (size_t)width * (size_t)height;
	const size_t cropPixels = (size_t)cropSize * (size_t)cropSize;
	if (frame.bgra.size() != framePixels * 4 || face.size() != cropPixels * 3 || mask.size() != cropPixels)
		return;

	if (!isFinite(frameToCrop))
		return;
	const double determinant = (double)frameToCrop.m[0] * (double)frameToCrop.m[4] -
				   (double)frameToCrop.m[1] * (double)frameToCrop.m[3];
	if (!std::isfinite(determinant) || std::fabs(determinant) < 1e-12)
		return;

	const Affine23 cropToFrame = invertAffine(frameToCrop);
	if (!isFinite(cropToFrame))
		return;
	const Affine23 sampleTransform = invertAffine(cropToFrame);
	if (!isFinite(sampleTransform))
		return;

	const float cropExtent = (float)cropSize;
	const double cornerX[4] = {0.0, (double)cropSize, 0.0, (double)cropSize};
	const double cornerY[4] = {0.0, 0.0, (double)cropSize, (double)cropSize};
	double minX = (double)cropToFrame.m[2];
	double maxX = minX;
	double minY = (double)cropToFrame.m[5];
	double maxY = minY;
	for (int i = 1; i < 4; i++) {
		const double x = (double)cropToFrame.m[0] * cornerX[i] + (double)cropToFrame.m[1] * cornerY[i] +
				 (double)cropToFrame.m[2];
		const double y = (double)cropToFrame.m[3] * cornerX[i] + (double)cropToFrame.m[4] * cornerY[i] +
				 (double)cropToFrame.m[5];
		minX = std::min(minX, x);
		maxX = std::max(maxX, x);
		minY = std::min(minY, y);
		maxY = std::max(maxY, y);
	}
	if (!std::isfinite(minX) || !std::isfinite(maxX) || !std::isfinite(minY) || !std::isfinite(maxY))
		return;

	const double xBeginValue = std::max(0.0, std::floor(minX) - 1.0);
	const double xEndValue = std::min((double)(width - 1), std::ceil(maxX) + 1.0);
	const double yBeginValue = std::max(0.0, std::floor(minY) - 1.0);
	const double yEndValue = std::min((double)(height - 1), std::ceil(maxY) + 1.0);
	if (xBeginValue > xEndValue || yBeginValue > yEndValue)
		return;
	const int xBegin = (int)xBeginValue;
	const int xEnd = (int)xEndValue;
	const int yBegin = (int)yBeginValue;
	const int yEnd = (int)yEndValue;

	for (int y = yBegin; y <= yEnd; y++) {
		for (int x = xBegin; x <= xEnd; x++) {
			const float u = frameToCrop.m[0] * (float)x + frameToCrop.m[1] * (float)y + frameToCrop.m[2];
			const float v = frameToCrop.m[3] * (float)x + frameToCrop.m[4] * (float)y + frameToCrop.m[5];
			if (!std::isfinite(u) || !std::isfinite(v) || u < 0.0f || u >= cropExtent || v < 0.0f ||
			    v >= cropExtent)
				continue;

			const float sx = sampleTransform.m[0] * (float)x + sampleTransform.m[1] * (float)y +
					 sampleTransform.m[2];
			const float sy = sampleTransform.m[3] * (float)x + sampleTransform.m[4] * (float)y +
					 sampleTransform.m[5];
			int xa, xb, ya, yb;
			float fractionX, fractionY;
			if (!sampleAxis(sx, cropSize, xa, xb, fractionX) ||
			    !sampleAxis(sy, cropSize, ya, yb, fractionY))
				continue;
			const uint8_t maskValue =
				sampleBilinear(mask.data(), cropSize, 1, 0, xa, xb, ya, yb, fractionX, fractionY);
			if (maskValue == 0)
				continue;

			const float alpha = (float)maskValue / 255.0f;
			uint8_t *dst = frame.bgra.data() + ((size_t)y * (size_t)width + (size_t)x) * 4;
			for (int channel = 0; channel < 3; channel++) {
				const uint8_t original = dst[channel];
				const uint8_t sampled = sampleBilinear(face.data(), cropSize, 3, channel, xa, xb, ya,
								       yb, fractionX, fractionY);
				const float firstValue = (float)sampled * alpha + (float)original * (1.0f - alpha);
				const uint8_t firstBlend = (uint8_t)std::clamp((int)std::lround(firstValue), 0, 255);
				if (intensity < 1.0f) {
					const float value =
						(float)firstBlend * intensity + (float)original * (1.0f - intensity);
					dst[channel] = (uint8_t)std::clamp((int)std::lround(value), 0, 255);
				} else {
					dst[channel] = firstBlend;
				}
			}
		}
	}
}

} // namespace fx
