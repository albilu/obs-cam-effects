#pragma once

#include "fx/engine/ort_backend.h"
#include "fx/models/yunet.h"
#include "fx/pipeline/face_tracker.h"
#include "fx/types.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace fx {

struct FaceSwapParams {
	float intensity = 1.0f;     // 0..1 swap opacity (amendment 9)
	float sharpness = 0.0f;     // 0..1 unsharp amount (amendment 9)
	float mouthPreserve = 0.0f; // 0..1 mouth-restore strength (amendment 9)
	/* Defaults mirror deep-live-cam: raw per-detection landmarks
	 * (0.7 EMA made the aligned crop lag ~2 detections behind the
	 * face and ghost during motion) and detection on every frame. */
	float bboxEma = 0.0f; // detection smoothing (0 = raw landmarks)
	int detectEveryN = 1; // YuNet every Nth frame (<=1: every frame)
};

/* YuNet detect (bbox EMA) -> umeyama align -> inswapper -> paste-back
 * (feathered ellipse from the swap affine, DLC anti-wobble pattern) ->
 * optional mouth restore / intensity. Owns its models. The AI
 * disclosure badge (spec §9) is a filter-side post-composite overlay,
 * not stamped here. */
class FaceSwapPipeline {
public:
	FaceSwapPipeline(const std::string &yunetPath, const std::string &inswapperPath, int threads = 2,
			 bool tryCuda = false, OrtExecutionPolicy swapPolicy = OrtExecutionPolicy::AllowCpuFallback);

	void setSourceEmbedding(std::vector<float> latent);
	void resetTracking();
	bool hasSource() const { return !sourceLatent_.empty(); }
	OrtBackend swapBackend() const noexcept { return swapper_.backend(); }
	bool hasFailedBackend() const noexcept
	{
		return detector_.backend() == OrtBackend::Failed || swapper_.backend() == OrtBackend::Failed;
	}
	void setParams(const FaceSwapParams &p) { params_ = p; }

	/* Swaps the largest detected or tracked face of `frame` in place (BGRA).
	 * Returns true if a swap was applied. */
	bool process(Frame &frame);

private:
	YuNet detector_;
	OrtModel swapper_;
	std::vector<float> sourceLatent_;
	FaceSwapParams params_;

	FaceTracker tracker_;
	std::vector<uint8_t> aimg_;      // 128x128x3 aligned crop
	std::vector<uint8_t> fake128_;   // swap output crop
	std::vector<uint8_t> mask128_;   // fixed paste-back mask
	std::vector<uint8_t> origFrame_; // for mouth restore
};

} // namespace fx
