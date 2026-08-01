#include "fx/models/yunet.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace fx {

namespace {

constexpr int kSize = 640;
constexpr int kStrideCount = 3;
constexpr int kStrides[kStrideCount] = {8, 16, 32};

float clamp01(float v)
{
	return std::clamp(v, 0.0f, 1.0f);
}

float boxIou(const FaceBox &a, const FaceBox &b)
{
	float x1 = std::max(a.x, b.x);
	float y1 = std::max(a.y, b.y);
	float x2 = std::min(a.x + a.w, b.x + b.w);
	float y2 = std::min(a.y + a.h, b.y + b.h);
	float inter = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
	float uni = a.w * a.h + b.w * b.h - inter;
	return uni > 0.0f ? inter / uni : 0.0f;
}

} // namespace

YuNet::YuNet(const std::string &modelPath, int threads)
	: model_(modelPath, threads), tensor_(3 * kSize * kSize)
{
	const auto &shape = model_.input().shape;
	if (shape.size() != 4 || shape[1] != 3 || shape[2] != kSize ||
	    shape[3] != kSize)
		throw std::runtime_error("fx: unexpected YuNet input shape");
	if (model_.outputCount() != 12)
		throw std::runtime_error("fx: unexpected YuNet output count");

	/* Outputs are matched by NAME at detect time (positional order is
	 * not trusted); verify all 12 expected tensors exist now, with
	 * shapes consistent with the stride geometry. */
	for (int s = 0; s < kStrideCount; s++) {
		const int cells = (kSize / kStrides[s]) * (kSize / kStrides[s]);
		const char *kind[4] = {"cls", "obj", "bbox", "kps"};
		const int64_t depth[4] = {1, 1, 4, 10};
		for (int k = 0; k < 4; k++) {
			std::string want = std::string(kind[k]) + "_" +
					   std::to_string(kStrides[s]);
			bool found = false;
			for (size_t i = 0; i < model_.outputCount(); i++) {
				const auto &out = model_.output(i);
				if (out.name != want)
					continue;
				found = true;
				if (out.shape.size() != 3 ||
				    out.shape[1] != cells ||
				    out.shape[2] != depth[k])
					throw std::runtime_error(
						"fx: unexpected YuNet output shape");
			}
			if (!found)
				throw std::runtime_error(
					"fx: missing YuNet output " + want);
		}
	}
}

std::vector<FaceBox> YuNet::detect(const Frame &frame, float scoreThresh)
{
	const int sw = frame.width, sh = frame.height;
	constexpr int dw = kSize, dh = kSize;

	/* Bilinear-resize BGRA -> 640x640, NCHW float, BGR order, raw
	 * 0-255 (YuNet takes NO normalization). */
	for (int y = 0; y < dh; y++) {
		float fy = (y + 0.5f) * sh / (float)dh - 0.5f;
		int y0 = std::clamp((int)std::floor(fy), 0, sh - 1);
		int y1 = std::min(y0 + 1, sh - 1);
		float wy = std::clamp(fy - (float)y0, 0.0f, 1.0f);
		for (int x = 0; x < dw; x++) {
			float sx = (x + 0.5f) * sw / (float)dw - 0.5f;
			int x0 = std::clamp((int)std::floor(sx), 0, sw - 1);
			int x1 = std::min(x0 + 1, sw - 1);
			float wx = std::clamp(sx - (float)x0, 0.0f, 1.0f);
			const uint8_t *p00 =
				frame.bgra.data() + (y0 * sw + x0) * 4;
			const uint8_t *p01 =
				frame.bgra.data() + (y0 * sw + x1) * 4;
			const uint8_t *p10 =
				frame.bgra.data() + (y1 * sw + x0) * 4;
			const uint8_t *p11 =
				frame.bgra.data() + (y1 * sw + x1) * 4;
			for (int c = 0; c < 3; c++) { // 0=B,1=G,2=R kept
				float v = (p00[c] * (1 - wx) + p01[c] * wx) *
						  (1 - wy) +
					  (p10[c] * (1 - wx) + p11[c] * wx) * wy;
				tensor_[c * dh * dw + y * dw + x] = v;
			}
		}
	}

	auto outs = model_.runWithShapes({tensor_}, {{1, 3, kSize, kSize}});

	/* Map outputs by name: cls/obj/bbox/kps per stride level. */
	const float *cls[kStrideCount], *obj[kStrideCount], *bbox[kStrideCount],
		*kps[kStrideCount];
	for (int s = 0; s < kStrideCount; s++) {
		const std::string suffix = "_" + std::to_string(kStrides[s]);
		for (size_t i = 0; i < model_.outputCount(); i++) {
			const std::string &n = model_.output(i).name;
			if (n == "cls" + suffix)
				cls[s] = outs[i].data();
			else if (n == "obj" + suffix)
				obj[s] = outs[i].data();
			else if (n == "bbox" + suffix)
				bbox[s] = outs[i].data();
			else if (n == "kps" + suffix)
				kps[s] = outs[i].data();
		}
	}

	/* Decode (opencv_zoo face_detect.cpp formulas, spike-verified):
	 * per cell (r,c) at stride s —
	 *   score = sqrt(clamp(cls,0,1) * clamp(obj,0,1))
	 *   cx = (c + bbox[0]) * s,  cy = (r + bbox[1]) * s
	 *   w  = exp(bbox[2]) * s,   h  = exp(bbox[3]) * s
	 *   landmark n: x = (kps[2n] + c) * s, y = (kps[2n+1] + r) * s */
	std::vector<FaceBox> cand;
	for (int s = 0; s < kStrideCount; s++) {
		const int stride = kStrides[s];
		const int grid = kSize / stride;
		for (int r = 0; r < grid; r++) {
			for (int c = 0; c < grid; c++) {
				const int idx = r * grid + c;
				float score = std::sqrt(
					clamp01(cls[s][idx]) *
					clamp01(obj[s][idx]));
				if (score < scoreThresh)
					continue;
				const float *bb = bbox[s] + idx * 4;
				const float *kp = kps[s] + idx * 10;
				FaceBox b;
				b.w = std::exp(bb[2]) * (float)stride;
				b.h = std::exp(bb[3]) * (float)stride;
				b.x = (c + bb[0]) * (float)stride - b.w * 0.5f;
				b.y = (r + bb[1]) * (float)stride - b.h * 0.5f;
				for (int n = 0; n < 5; n++) {
					b.landmarks[n] = {
						(kp[2 * n] + c) * (float)stride,
						(kp[2 * n + 1] + r) *
							(float)stride};
				}
				b.score = score;
				cand.push_back(b);
			}
		}
	}

	/* Greedy IoU-NMS 0.3: best score first, suppress overlaps. */
	std::sort(cand.begin(), cand.end(),
		  [](const FaceBox &a, const FaceBox &b) {
			  return a.score > b.score;
		  });
	std::vector<FaceBox> kept;
	kept.reserve(cand.size());
	for (const FaceBox &b : cand) {
		bool suppressed = false;
		for (const FaceBox &k : kept) {
			if (boxIou(b, k) > 0.3f) {
				suppressed = true;
				break;
			}
		}
		if (!suppressed)
			kept.push_back(b);
	}

	/* Map 640-space coordinates back to frame pixels. */
	const float scaleX = (float)sw / (float)kSize;
	const float scaleY = (float)sh / (float)kSize;
	for (FaceBox &b : kept) {
		b.x *= scaleX;
		b.w *= scaleX;
		b.y *= scaleY;
		b.h *= scaleY;
		for (Point2 &p : b.landmarks) {
			p[0] *= scaleX;
			p[1] *= scaleY;
		}
	}
	return kept;
}

} // namespace fx
