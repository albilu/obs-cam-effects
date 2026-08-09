#include "fx/pipeline/face_swap_pipeline.h"

#include "fx/image/align.h"

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

FaceSwapPipeline::FaceSwapPipeline(const std::string &yunetPath,
				   const std::string &inswapperPath,
				   const std::string &arcfacePath, int threads,
				   bool tryCuda)
	: detector_(yunetPath, threads, tryCuda),
	  swapper_(inswapperPath, threads, tryCuda),
	  embedder_(arcfacePath, inswapperPath, threads, tryCuda)
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
}

void FaceSwapPipeline::setSourceEmbedding(std::vector<float> latent)
{
	sourceLatent_ = std::move(latent);
}

bool FaceSwapPipeline::process(Frame &frame)
{
	if (!hasSource())
		return false;
	const int w = frame.width, h = frame.height;
	if (w <= 0 || h <= 0 || frame.bgra.size() != (size_t)w * h * 4)
		return false;

	/* Detection decimation: YuNet runs on every detectEveryN-th
	 * frame (and whenever there is no usable previous box); skipped
	 * frames reuse prevBox_, which the EMA below already stabilizes.
	 * Align/swap/paste-back still run EVERY frame from the latest
	 * box. A no-face detect frame invalidates the previous box. */
	const int everyN = params_.detectEveryN;
	const bool doDetect = everyN <= 1 ||
			      (frameCount_ % (uint64_t)everyN) == 0 ||
			      !havePrevBox_;
	frameCount_++;
	if (doDetect) {
		auto faces = detector_.detect(frame);
		if (faces.empty()) {
			havePrevBox_ = false;
			return false;
		}
		const FaceBox *best = &faces[0];
		for (const FaceBox &f : faces)
			if (f.w * f.h > best->w * best->h)
				best = &f;

		/* Temporal smoothing: smoothed = ema*prev + (1-ema)*current,
		 * per box component and per landmark coordinate. The
		 * SMOOTHED geometry drives everything below (anti-
		 * jitter). */
		FaceBox box = *best;
		if (havePrevBox_) {
			const float ema = params_.bboxEma, cur = 1.0f - ema;
			box.x = ema * prevBox_.x + cur * box.x;
			box.y = ema * prevBox_.y + cur * box.y;
			box.w = ema * prevBox_.w + cur * box.w;
			box.h = ema * prevBox_.h + cur * box.h;
			for (int i = 0; i < 5; i++)
				for (int k = 0; k < 2; k++)
					box.landmarks[i][k] =
						ema * prevBox_.landmarks[i][k] +
						cur * box.landmarks[i][k];
		}
		prevBox_ = box;
		havePrevBox_ = true;
	}
	const FaceBox box = prevBox_;

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

	/* Original needed by intensity < 1 and/or mouth restore. */
	const bool needOrig = params_.intensity < 1.0f || params_.mouthPreserve > 0.0f;
	if (needOrig)
		origFrame_ = frame.bgra;

	/* Paste-back. warpAffineBilinear(dst, src, forwardM) samples
	 * dst(p) = src(forwardM^-1 * p). We need frame(p) = crop(M * p)
	 * (each frame pixel samples where IT lands in crop space), i.e.
	 * (argument)^-1 == M, so the "forward" argument must be invM. */
	const Affine23 invM = invertAffine(m);
	std::vector<uint8_t> fakeFull((size_t)w * h * 3);
	warpAffineBilinear(fake128_.data(), kCrop, kCrop, 3, invM,
			   fakeFull.data(), w, h);

	/* DLC anti-wobble: FIXED feathered ellipse in 128-crop space (NOT
	 * landmark-derived), warped to frame space with the same invM
	 * transform (uint8 plane; the warp is channel-count generic). */
	std::vector<uint8_t> maskFull((size_t)w * h);
	{
		const std::vector<float> m128 =
			ellipseMask(kCrop, 0.35f, 0.45f, 12);
		std::vector<uint8_t> m128u((size_t)kCrop * kCrop);
		for (size_t i = 0; i < m128.size(); i++)
			m128u[i] = (uint8_t)std::lround(m128[i] * 255.0f);
		warpAffineBilinear(m128u.data(), kCrop, kCrop, 1, invM,
				   maskFull.data(), w, h);
	}

	/* In-crop gate (whole-frame mask-bleed fix): the ellipse mask does
	 * NOT reach 0 at the crop border (the post-feather blur leaves up
	 * to ~0.18 there) and warpAffineBilinear CLAMPS out-of-crop
	 * samples to that border — so every frame pixel whose crop
	 * coordinate u,v = M*p leaves [0,kCrop)^2 would otherwise inherit
	 * a nonzero alpha and smear clamped edge content across the whole
	 * frame. The ellipse is only meaningful inside the crop; force
	 * alpha 0 outside it so those pixels stay byte-identical. */
	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			const float u =
				m.m[0] * (float)x + m.m[1] * (float)y + m.m[2];
			const float v =
				m.m[3] * (float)x + m.m[4] * (float)y + m.m[5];
			if (u < 0.0f || u >= (float)kCrop || v < 0.0f ||
			    v >= (float)kCrop)
				maskFull[(size_t)y * (size_t)w + x] = 0;
		}
	}

	const size_t n = (size_t)w * h;
	for (size_t i = 0; i < n; i++) {
		const float a = (float)maskFull[i] / 255.0f;
		if (a <= 0.0f)
			continue;
		uint8_t *d = frame.bgra.data() + i * 4;
		const uint8_t *s = fakeFull.data() + i * 3;
		for (int c = 0; c < 3; c++) {
			const float v = (float)s[c] * a + (float)d[c] * (1.0f - a);
			d[c] = (uint8_t)std::clamp((int)std::lround(v), 0, 255);
		}
	}

	if (params_.intensity < 1.0f) {
		const float keep = params_.intensity, orig = 1.0f - keep;
		for (size_t i = 0; i < n; i++) {
			if (maskFull[i] == 0)
				continue; /* within the ellipse region */
			uint8_t *d = frame.bgra.data() + i * 4;
			const uint8_t *o = origFrame_.data() + i * 4;
			for (int c = 0; c < 3; c++) {
				const float v = (float)d[c] * keep +
						(float)o[c] * orig;
				d[c] = (uint8_t)std::clamp((int)std::lround(v),
							   0, 255);
			}
		}
	}

	if (params_.mouthPreserve > 0.0f)
		restoreMouthRegion(frame.bgra.data(), origFrame_.data(), w, h, 4, box.landmarks[3], box.landmarks[4],
				   1.0f, 6, params_.mouthPreserve);

	if (params_.watermark)
		stampWatermarkAI(frame.bgra.data(), w, h, 4);

	return true;
}

} // namespace fx
