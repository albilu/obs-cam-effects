#pragma once

#include "fx/models/face_embedder.h"
#include "fx/models/yunet.h"
#include "fx/types.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace fx {

struct FaceSwapParams {
	float intensity = 1.0f;   // 0..1 swap opacity (amendment 9)
	float sharpness = 0.0f;   // 0..1 unsharp amount (amendment 9)
	bool preserveMouth = false; // geometric mouth restore (amendment 9)
	bool watermark = true;      // AI disclosure badge (spec §9)
	float bboxEma = 0.7f;       // detection smoothing
	int detectEveryN = 2;       // YuNet every Nth frame (<=1: every frame)
};

/* YuNet detect (bbox EMA) -> umeyama align -> inswapper -> paste-back
 * (feathered ellipse from the swap affine, DLC anti-wobble pattern) ->
 * optional mouth restore / intensity / watermark. Owns its models. */
class FaceSwapPipeline {
public:
	FaceSwapPipeline(const std::string &yunetPath,
			 const std::string &inswapperPath,
			 const std::string &arcfacePath, int threads = 2,
			 bool tryCuda = false);

	void setSourceEmbedding(std::vector<float> latent);
	bool hasSource() const { return !sourceLatent_.empty(); }
	void setParams(const FaceSwapParams &p) { params_ = p; }

	/* Swaps the largest detected face of `frame` in place (BGRA).
	 * Returns true if a swap was applied. */
	bool process(Frame &frame);

private:
	YuNet detector_;
	OrtModel swapper_;
	FaceEmbedder embedder_;
	std::vector<float> sourceLatent_;
	FaceSwapParams params_;

	/* Temporal state */
	bool havePrevBox_ = false;
	FaceBox prevBox_{};
	uint64_t frameCount_ = 0;   // detection decimation counter
	std::vector<uint8_t> aimg_;       // 128x128x3 aligned crop
	std::vector<uint8_t> fake128_;    // swap output crop
	std::vector<uint8_t> origFrame_;  // for mouth restore
};

} // namespace fx
