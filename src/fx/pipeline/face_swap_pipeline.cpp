#include "fx/pipeline/face_swap_pipeline.h"

#include "fx/image/align.h"
#include "fx/image/face_paste.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace fx {

namespace {

constexpr int kCrop = 128;  // inswapper aligned crop size
constexpr int kLatent = 512; // inswapper source latent dims

} // namespace

FaceSwapPipeline::FaceSwapPipeline(const std::string &yunetPath, const std::string &inswapperPath, int threads,
				   bool tryCuda, OrtExecutionPolicy swapPolicy)
	: detector_(yunetPath, threads, tryCuda),
	  swapper_(inswapperPath, threads, tryCuda, swapPolicy)
{
	/* Verified inswapper_128 IO: target [1,3,128,128], source [1,512],
	 * output [1,3,128,128]. */
	if (swapper_.inputCount() != 2 || swapper_.outputCount() != 1)
		throw std::runtime_error("fx: unexpected inswapper IO count");
	const auto &in0 = swapper_.input(0).shape;
	const auto &in1 = swapper_.input(1).shape;
	const auto &out0 = swapper_.output(0).shape;
	if (in0.size() != 4 || in0[1] != 3 || in0[2] != kCrop ||
	    in0[3] != kCrop || in1.size() != 2 || in1[1] != kLatent ||
	    out0.size() != 4 || out0[1] != 3 || out0[2] != kCrop ||
	    out0[3] != kCrop)
		throw std::runtime_error("fx: unexpected inswapper IO shape");

	const std::vector<float> mask = ellipseMask(kCrop, 0.35f, 0.45f, 12);
	mask128_.resize(mask.size());
	for (size_t i = 0; i < mask.size(); i++)
		mask128_[i] = (uint8_t)std::lround(mask[i] * 255.0f);
}

void FaceSwapPipeline::setSourceEmbedding(std::vector<float> latent)
{
	sourceLatent_ = std::move(latent);
	resetTracking();
}

void FaceSwapPipeline::resetTracking()
{
	tracker_.reset();
}

bool FaceSwapPipeline::process(Frame &frame)
{
	if (swapBackend() == OrtBackend::Failed)
		throw std::runtime_error("fx: CUDA execution failed");
	if (!hasSource())
		return false;
	const int w = frame.width, h = frame.height;
	if (w <= 0 || h <= 0 || frame.bgra.size() != (size_t)w * h * 4)
		return false;

	if (tracker_.shouldDetect(w, h, params_.detectEveryN)) {
		auto faces = detector_.detect(frame);
		if (!tracker_.observe(faces, params_.bboxEma))
			return false;
	}
	if (!tracker_.hasBox())
		return false;
	const FaceBox box = tracker_.box();

	/* Forward affine frame -> 128-crop (warpAffineBilinear inverts it
	 * internally, so crop(p) = frame(M^-1 * p)). */
	const Affine23 m = umeyama(box.landmarks, template128());

	aimg_.resize((size_t)kCrop * kCrop * 3);
	{
		std::vector<uint8_t> aimg4((size_t)kCrop * kCrop * 4);
		warpAffineBilinear(frame.bgra.data(), w, h, 4, m, aimg4.data(),
				   kCrop, kCrop);
		for (int i = 0; i < kCrop * kCrop; i++)
			memcpy(aimg_.data() + (size_t)i * 3,
			       aimg4.data() + (size_t)i * 4, 3);
	}

	/* inswapper: target = crop RGB /255 in [0,1] NCHW, source = latent. */
	std::vector<float> tensor((size_t)3 * kCrop * kCrop);
	for (int y = 0; y < kCrop; y++) {
		for (int x = 0; x < kCrop; x++) {
			const uint8_t *p =
				aimg_.data() + ((size_t)y * kCrop + x) * 3;
			for (int c = 0; c < 3; c++)
				tensor[(size_t)c * kCrop * kCrop +
				       (size_t)y * kCrop + x] =
					(float)p[2 - c] / 255.0f;
		}
	}
	const std::vector<float> out =
		swapper_
			.runWithShapes({tensor, sourceLatent_},
				       {{1, 3, kCrop, kCrop}, {1, kLatent}})
			.at(0);
	if (out.size() != (size_t)3 * kCrop * kCrop)
		throw std::runtime_error("fx: unexpected inswapper output size");

	/* Output [0,1] NCHW RGB -> x255, channel flip into 128x128x3 BGR. */
	fake128_.resize((size_t)kCrop * kCrop * 3);
	for (int y = 0; y < kCrop; y++) {
		for (int x = 0; x < kCrop; x++) {
			uint8_t *d =
				fake128_.data() + ((size_t)y * kCrop + x) * 3;
			for (int c = 0; c < 3; c++) {
				const float v =
					out[(size_t)(2 - c) * kCrop * kCrop +
					    (size_t)y * kCrop + x] *
					255.0f;
				d[c] = (uint8_t)std::clamp((int)std::lround(v),
							   0, 255);
			}
		}
	}

	if (params_.sharpness > 0.0f)
		unsharpMask(fake128_.data(), kCrop, kCrop, 3, 2,
			    params_.sharpness);

	const bool preserveMouth = params_.mouthPreserve > 0.0f;
	if (preserveMouth)
		origFrame_ = frame.bgra;

	pasteFaceCrop(frame, fake128_, mask128_, kCrop, m, params_.intensity);
	if (preserveMouth)
		restoreMouthRegion(frame.bgra.data(), origFrame_.data(), w, h, 4, box.landmarks[3], box.landmarks[4],
				   1.0f, 6, params_.mouthPreserve);

	/* Exact zero restores the original bytes. Every other value, including
	 * non-UI negative and non-finite values, can modify the paste result. */
	return params_.intensity != 0.0f;
}

} // namespace fx
