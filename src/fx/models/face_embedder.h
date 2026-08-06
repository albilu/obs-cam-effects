#pragma once

#include "fx/models/onnx_init.h"
#include "fx/models/yunet.h"
#include "fx/types.h"

#include <string>
#include <vector>

namespace fx {

/* Source-identity embedder: ArcFace w600k_r50 (112x112 aligned,
 * RGB, (x-127.5)/127.5) + inswapper's emap projection. */
class FaceEmbedder {
public:
	FaceEmbedder(const std::string &arcfacePath,
		     const std::string &inswapperPathForEmap, int threads = 2,
		     bool tryCuda = false);

	/* Embedding for one aligned 112x112 BGR crop (from YuNet
	 * landmarks via template112 umeyama warp). Returns the 512-d
	 * inswapper-ready latent: L2norm(L2norm(arcface) @ emap). */
	std::vector<float> embed(const std::vector<uint8_t> &bgrCrop112);

	/* One-call helper: decode an image file (jpg/png) with stb_image,
	 * detect the largest face, warp to 112, embed. Empty on failure. */
	std::vector<float> embedFromImageFile(const char *path, YuNet &det);

private:
	OrtModel arcface_;
	std::vector<float> emap_; // 512*512
};

} // namespace fx
