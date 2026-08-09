#pragma once

#include "fx/types.h"

#include <array>
#include <vector>

namespace fx {

using Point2 = std::array<float, 2>;
using Landmarks5 = std::array<Point2, 5>;

/* 2x3 affine matrix, row-major [a11 a12 tx; a21 a22 ty]. */
struct Affine23 {
	float m[6];
};

/* The two verified alignment templates (insightface face_align.py). */
Landmarks5 template112();
Landmarks5 template128(); // template112 with x += 8

/* Umeyama similarity transform src -> dst (least squares, 5 points).
 * Returns the FORWARD affine (cv2.warpAffine convention: pass forward,
 * the warp inverts internally). */
Affine23 umeyama(const Landmarks5 &src, const Landmarks5 &dst);

Affine23 invertAffine(const Affine23 &m);

/* Bilinear warp of an interleaved-uint8 BGR(A) image.
 * dst(x,y) = src(M⁻¹ · (x,y)) — caller passes the FORWARD transform.
 * Channels: 3 (BGR) or 4 (BGRA). Out-of-bounds samples are clamped to
 * the edge pixel. */
void warpAffineBilinear(const uint8_t *src, int sw, int sh, int channels,
			const Affine23 &forwardM, uint8_t *dst, int dw,
			int dh);

/* Feathered elliptical alpha mask centered in a crop of size s×s
 * (DLC-style anti-wobble paste mask), radii relative to s, feather in px. */
std::vector<float> ellipseMask(int s, float rx, float ry, int feather);

/* unsharp mask on an interleaved uint8 image: out = src + amount·(src − blur). */
void unsharpMask(uint8_t *img, int w, int h, int channels, int radius,
		 float amount);

/* Composite `origMouth` (same geometry) back over `img` inside an
 * ellipse centered between mouth landmarks, feathered. `strength`
 * (0..1) scales the ellipse alpha (0 = no restore, 1 = full). */
void restoreMouthRegion(uint8_t *img, const uint8_t *orig, int w, int h, int channels, Point2 mouthL, Point2 mouthR,
			float widthScale, int feather, float strength);

/* Stamp a small "AI" disclosure badge (white on black rounded box) at
 * the bottom-right corner, ~2.5% of frame width. No-op if too small. */
void stampWatermarkAI(uint8_t *img, int w, int h, int channels);

} // namespace fx
