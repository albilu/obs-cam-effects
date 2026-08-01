#include "fx/models/face_embedder.h"

#include "fx/image/align.h"

#include <cmath>
#include <cstring>
#include <stdexcept>

#define STB_IMAGE_IMPLEMENTATION
#include "fx/third_party/stb_image.h"

namespace fx {

namespace {

constexpr int kSize = 112;
constexpr int kDim = 512;

void l2Normalize(std::vector<float> &v)
{
	double sum = 0.0;
	for (float x : v)
		sum += (double)x * (double)x;
	const float inv = sum > 0.0 ? (float)(1.0 / std::sqrt(sum)) : 0.0f;
	for (float &x : v)
		x *= inv;
}

} // namespace

FaceEmbedder::FaceEmbedder(const std::string &arcfacePath,
			   const std::string &inswapperPathForEmap, int threads)
	: arcface_(arcfacePath, threads),
	  emap_(onnxLastInitializerFloats(inswapperPathForEmap, kDim * kDim))
{
	/* Batch dim is dynamic (-1); only the fixed dims are checked. */
	const auto &in = arcface_.input().shape;
	if (in.size() != 4 || in[1] != 3 || in[2] != kSize || in[3] != kSize)
		throw std::runtime_error("fx: unexpected ArcFace input shape");
	const auto &out = arcface_.output().shape;
	if (out.size() != 2 || out[1] != kDim)
		throw std::runtime_error("fx: unexpected ArcFace output shape");
}

std::vector<float> FaceEmbedder::embed(const std::vector<uint8_t> &bgrCrop112)
{
	if (bgrCrop112.size() != (size_t)kSize * kSize * 3)
		throw std::runtime_error("fx: ArcFace crop must be 112x112x3 BGR");

	/* NCHW float, RGB order (swapRB), (x - 127.5) / 127.5. */
	std::vector<float> tensor(3 * kSize * kSize);
	for (int y = 0; y < kSize; y++) {
		for (int x = 0; x < kSize; x++) {
			const uint8_t *p =
				bgrCrop112.data() + (y * kSize + x) * 3;
			for (int c = 0; c < 3; c++)
				tensor[c * kSize * kSize + y * kSize + x] =
					((float)p[2 - c] - 127.5f) / 127.5f;
		}
	}

	std::vector<float> emb = arcface_
					 .runWithShapes({tensor},
							{{1, 3, kSize, kSize}})
					 .at(0);
	if (emb.size() != (size_t)kDim)
		throw std::runtime_error("fx: unexpected ArcFace output size");
	l2Normalize(emb);

	/* inswapper's emap projection, row-major:
	 * latent[j] = sum_i emb[i] * emap[i*512 + j] */
	std::vector<float> latent(kDim, 0.0f);
	for (int i = 0; i < kDim; i++) {
		const float e = emb[i];
		const float *row = emap_.data() + (size_t)i * kDim;
		for (int j = 0; j < kDim; j++)
			latent[j] += e * row[j];
	}
	l2Normalize(latent);
	return latent;
}

std::vector<float> FaceEmbedder::embedFromImageFile(const char *path,
						    YuNet &det)
{
	int w = 0, h = 0, channels = 0;
	stbi_uc *rgb = stbi_load(path, &w, &h, &channels, 3);
	if (!rgb)
		return {};

	/* stb gives RGB; YuNet + the warp path work in BGRA. */
	Frame frame;
	frame.width = w;
	frame.height = h;
	frame.bgra.resize((size_t)w * (size_t)h * 4);
	for (size_t i = 0, n = (size_t)w * (size_t)h; i < n; i++) {
		frame.bgra[i * 4 + 0] = rgb[i * 3 + 2];
		frame.bgra[i * 4 + 1] = rgb[i * 3 + 1];
		frame.bgra[i * 4 + 2] = rgb[i * 3 + 0];
		frame.bgra[i * 4 + 3] = 255;
	}
	stbi_image_free(rgb);

	auto faces = det.detect(frame);
	if (faces.empty())
		return {};

	const FaceBox *best = &faces[0];
	for (const FaceBox &f : faces)
		if (f.w * f.h > best->w * best->h)
			best = &f;

	const Affine23 m = umeyama(best->landmarks, template112());
	std::vector<uint8_t> cropBgra(kSize * kSize * 4);
	warpAffineBilinear(frame.bgra.data(), w, h, 4, m, cropBgra.data(),
			   kSize, kSize);

	std::vector<uint8_t> crop(kSize * kSize * 3);
	for (int i = 0; i < kSize * kSize; i++)
		memcpy(crop.data() + (size_t)i * 3,
		       cropBgra.data() + (size_t)i * 4, 3);
	return embed(crop);
}

} // namespace fx
