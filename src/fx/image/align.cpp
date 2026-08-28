#include "fx/image/align.h"

#include <algorithm>
#include <cmath>

namespace fx {

Landmarks5 template112()
{
	return {{{38.2946f, 51.6963f},
		 {73.5318f, 51.5014f},
		 {56.0252f, 71.7366f},
		 {41.5493f, 92.3655f},
		 {70.7299f, 92.2041f}}};
}

Landmarks5 template128()
{
	Landmarks5 t = template112();
	for (auto &p : t)
		p[0] += 8.0f;
	return t;
}

Affine23 umeyama(const Landmarks5 &src, const Landmarks5 &dst)
{
	double srcMx = 0, srcMy = 0, dstMx = 0, dstMy = 0;
	for (int i = 0; i < 5; i++) {
		srcMx += src[i][0];
		srcMy += src[i][1];
		dstMx += dst[i][0];
		dstMy += dst[i][1];
	}
	srcMx /= 5.0;
	srcMy /= 5.0;
	dstMx /= 5.0;
	dstMy /= 5.0;

	double norm = 0, sa = 0, sb = 0;
	for (int i = 0; i < 5; i++) {
		double x = src[i][0] - srcMx, y = src[i][1] - srcMy;
		double u = dst[i][0] - dstMx, v = dst[i][1] - dstMy;
		norm += x * x + y * y;
		sa += x * u + y * v;
		sb += x * v - y * u;
	}

	Affine23 m;
	if (norm < 1e-12) { /* degenerate point cloud: identity */
		m.m[0] = 1;
		m.m[1] = 0;
		m.m[2] = 0;
		m.m[3] = 0;
		m.m[4] = 1;
		m.m[5] = 0;
		return m;
	}
	double a = sa / norm, b = sb / norm;
	m.m[0] = (float)a;
	m.m[1] = (float)-b;
	m.m[2] = (float)(dstMx - (a * srcMx - b * srcMy));
	m.m[3] = (float)b;
	m.m[4] = (float)a;
	m.m[5] = (float)(dstMy - (b * srcMx + a * srcMy));
	return m;
}

Affine23 invertAffine(const Affine23 &m)
{
	Affine23 r;
	double a = m.m[0], b = m.m[1], c = m.m[3], d = m.m[4];
	double det = a * d - b * c;
	if (std::fabs(det) < 1e-12) {
		r.m[0] = 1;
		r.m[1] = 0;
		r.m[2] = 0;
		r.m[3] = 0;
		r.m[4] = 1;
		r.m[5] = 0;
		return r;
	}
	double ia = d / det, ib = -b / det, ic = -c / det, id = a / det;
	double tx = m.m[2], ty = m.m[5];
	r.m[0] = (float)ia;
	r.m[1] = (float)ib;
	r.m[2] = (float)-(ia * tx + ib * ty);
	r.m[3] = (float)ic;
	r.m[4] = (float)id;
	r.m[5] = (float)-(ic * tx + id * ty);
	return r;
}

void warpAffineBilinear(const uint8_t *src, int sw, int sh, int channels,
			const Affine23 &forwardM, uint8_t *dst, int dw, int dh)
{
	Affine23 inv = invertAffine(forwardM);
	for (int y = 0; y < dh; y++) {
		for (int x = 0; x < dw; x++) {
			float sx = inv.m[0] * x + inv.m[1] * y + inv.m[2];
			float sy = inv.m[3] * x + inv.m[4] * y + inv.m[5];
			int x0 = (int)std::floor(sx), y0 = (int)std::floor(sy);
			float fx = sx - (float)x0, fy = sy - (float)y0;
			int xa = std::clamp(x0, 0, sw - 1);
			int xb = std::clamp(x0 + 1, 0, sw - 1);
			int ya = std::clamp(y0, 0, sh - 1);
			int yb = std::clamp(y0 + 1, 0, sh - 1);
			uint8_t *d = dst + (y * dw + x) * channels;
			for (int c = 0; c < channels; c++) {
				float v00 = src[(ya * sw + xa) * channels + c];
				float v01 = src[(ya * sw + xb) * channels + c];
				float v10 = src[(yb * sw + xa) * channels + c];
				float v11 = src[(yb * sw + xb) * channels + c];
				float v = (v00 * (1.0f - fx) + v01 * fx) *
						  (1.0f - fy) +
					  (v10 * (1.0f - fx) + v11 * fx) * fy;
				d[c] = (uint8_t)std::clamp((int)std::lround(v), 0,
							   255);
			}
		}
	}
}

namespace {

/* Feathered ellipse alpha at offset (dx,dy) from the center, radii a/b
 * in px. 1 inside, 0 outside, linear falloff across `feather` px at the
 * boundary (signed distance via the gradient of the normalized eq). */
float ellipseAlpha(float dx, float dy, float a, float b, int feather)
{
	float nx = dx / a, ny = dy / b;
	float t = nx * nx + ny * ny;
	if (feather <= 0)
		return t <= 1.0f ? 1.0f : 0.0f;
	float gx = 2.0f * dx / (a * a), gy = 2.0f * dy / (b * b);
	float g = std::sqrt(gx * gx + gy * gy);
	float dist = g > 1e-12f ? (t - 1.0f) / g : -std::min(a, b);
	return std::clamp(0.5f - dist / (float)feather, 0.0f, 1.0f);
}

/* Light box blur (clamped window) over a single-channel float image. */
void boxBlurFloat(std::vector<float> &img, int w, int h, int r)
{
	std::vector<float> src = img;
	for (int y = 0; y < h; y++) {
		int y0 = std::max(0, y - r), y1 = std::min(h - 1, y + r);
		for (int x = 0; x < w; x++) {
			int x0 = std::max(0, x - r), x1 =
				std::min(w - 1, x + r);
			float sum = 0;
			for (int yy = y0; yy <= y1; yy++)
				for (int xx = x0; xx <= x1; xx++)
					sum += src[yy * w + xx];
			img[y * w + x] =
				sum / (float)((y1 - y0 + 1) * (x1 - x0 + 1));
		}
	}
}

/* Separable Gaussian blur over a single-channel float image, matching
 * cv2.GaussianBlur(k, k, sigma) (normalized kernel, BORDER_REFLECT_101
 * edges). Used for the DLC-parity paste-mask feather. */
void gaussianBlurFloat(std::vector<float> &img, int w, int h, int radius, float sigma)
{
	if (radius <= 0 || sigma <= 0.0f || w <= 0 || h <= 0)
		return;
	const int k = 2 * radius + 1;
	std::vector<float> kern((size_t)k);
	double sum = 0.0;
	for (int i = -radius; i <= radius; i++) {
		const float v = (float)std::exp(-0.5 * (double)(i * i) / ((double)sigma * (double)sigma));
		kern[(size_t)(i + radius)] = v;
		sum += v;
	}
	for (float &v : kern)
		v = (float)(v / sum);
	auto reflect101 = [](int i, int n) {
		if (n == 1)
			return 0;
		while (i < 0 || i >= n) {
			if (i < 0)
				i = -i;
			if (i >= n)
				i = 2 * (n - 1) - i;
		}
		return i;
	};
	std::vector<float> tmp((size_t)w * (size_t)h);
	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			float acc = 0.0f;
			for (int t = -radius; t <= radius; t++)
				acc += kern[(size_t)(t + radius)] *
				       img[(size_t)y * (size_t)w + (size_t)reflect101(x + t, w)];
			tmp[(size_t)y * (size_t)w + (size_t)x] = acc;
		}
	}
	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			float acc = 0.0f;
			for (int t = -radius; t <= radius; t++)
				acc += kern[(size_t)(t + radius)] *
				       tmp[(size_t)reflect101(y + t, h) * (size_t)w + (size_t)x];
			img[(size_t)y * (size_t)w + (size_t)x] = acc;
		}
	}
}

} // namespace

std::vector<float> ellipseMask(int s, float rx, float ry, int feather)
{
	std::vector<float> m((size_t)s * (size_t)s);
	float c = (float)s * 0.5f;
	float a = rx * (float)s, b = ry * (float)s;
	for (int y = 0; y < s; y++)
		for (int x = 0; x < s; x++)
			m[y * s + x] = ellipseAlpha((float)x - c, (float)y - c, a,
						    b, feather);
	if (feather > 0)
		boxBlurFloat(m, s, s, std::max(1, feather / 4));
	return m;
}

std::vector<float> softEllipseMask(int s, float axesRatio, int gaussianRadius, float gaussianSigma)
{
	std::vector<float> m((size_t)s * (size_t)s, 0.0f);
	if (s <= 0 || axesRatio <= 0.0f)
		return m;
	const float c = 0.5f * (float)s;
	const float a = axesRatio * (float)s;
	for (int y = 0; y < s; y++) {
		for (int x = 0; x < s; x++) {
			const float nx = ((float)x - c) / a;
			const float ny = ((float)y - c) / a;
			m[(size_t)y * (size_t)s + (size_t)x] =
				(nx * nx + ny * ny) <= 1.0f ? 1.0f : 0.0f;
		}
	}
	gaussianBlurFloat(m, s, s, gaussianRadius, gaussianSigma);
	return m;
}

void unsharpMask(uint8_t *img, int w, int h, int channels, int radius,
		 float amount)
{
	if (radius <= 0 || amount == 0.0f)
		return;
	std::vector<uint8_t> src(img, img + (size_t)w * h * channels);
	for (int y = 0; y < h; y++) {
		int y0 = std::max(0, y - radius),
		    y1 = std::min(h - 1, y + radius);
		for (int x = 0; x < w; x++) {
			int x0 = std::max(0, x - radius),
			    x1 = std::min(w - 1, x + radius);
			int count = (y1 - y0 + 1) * (x1 - x0 + 1);
			uint8_t *d = img + (y * w + x) * channels;
			for (int c = 0; c < channels; c++) {
				int sum = 0;
				for (int yy = y0; yy <= y1; yy++)
					for (int xx = x0; xx <= x1; xx++)
						sum += src[(yy * w + xx) *
								   channels +
							   c];
				float blur = (float)sum / (float)count;
				float sv = src[(y * w + x) * channels + c];
				float v = sv + amount * (sv - blur);
				d[c] = (uint8_t)std::clamp((int)std::lround(v),
							   0, 255);
			}
		}
	}
}

void restoreMouthRegion(uint8_t *img, const uint8_t *orig, int w, int h, int channels, Point2 mouthL, Point2 mouthR,
			float widthScale, int feather, float strength)
{
	float cx = (mouthL[0] + mouthR[0]) * 0.5f;
	float cy = (mouthL[1] + mouthR[1]) * 0.5f;
	float dx = mouthR[0] - mouthL[0], dy = mouthR[1] - mouthL[1];
	float a = std::sqrt(dx * dx + dy * dy) * widthScale * 0.75f;
	float b = a * 0.5f;
	if (a < 1e-6f)
		return;
	float grow = (float)std::max(feather, 0);
	int x0 = std::max(0, (int)std::floor(cx - a - grow));
	int x1 = std::min(w - 1, (int)std::ceil(cx + a + grow));
	int y0 = std::max(0, (int)std::floor(cy - b - grow));
	int y1 = std::min(h - 1, (int)std::ceil(cy + b + grow));
	for (int y = y0; y <= y1; y++) {
		for (int x = x0; x <= x1; x++) {
			float alpha = ellipseAlpha((float)x - cx, (float)y - cy,
						   a, b, feather);
			alpha *= strength;
			if (alpha <= 0.0f)
				continue;
			uint8_t *d = img + (y * w + x) * channels;
			const uint8_t *o = orig + (y * w + x) * channels;
			for (int c = 0; c < channels; c++) {
				float v = o[c] * alpha +
					  d[c] * (1.0f - alpha);
				d[c] = (uint8_t)std::clamp((int)std::lround(v),
							   0, 255);
			}
		}
	}
}

namespace {

/* 5x7 glyph bitmaps, bit 4 (0x10) = leftmost column. */
const uint8_t GLYPH_A[7] = {0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11};
const uint8_t GLYPH_I[7] = {0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x1F};

/* Alpha-blend color `fg` over one pixel; color channels only (alpha
 * channel of BGRA is left untouched). */
void blendPx(uint8_t *img, int idx, int channels, uint8_t fg, float alpha)
{
	int n = std::min(channels, 3);
	for (int c = 0; c < n; c++) {
		float v = img[idx + c] * (1.0f - alpha) + fg * alpha;
		img[idx + c] =
			(uint8_t)std::clamp((int)std::lround(v), 0, 255);
	}
}

} // namespace

void stampWatermarkAI(uint8_t *img, int w, int h, int channels)
{
	int scale = std::max(3, h / 270); /* >=3 so the badge stays readable on small feeds (e.g. 640x480); ~2.5% of frame width at 1080p */
	int margin = std::max(1, h / 40);
	int glyphW = 5 * scale, glyphH = 7 * scale, gap = scale;
	int pad = scale;
	int boxW = 2 * glyphW + gap + 2 * pad;
	int boxH = glyphH + 2 * pad;
	int bx = w - margin - boxW, by = h - margin - boxH;
	if (bx < 0 || by < 0)
		return; /* does not fit */

	const float kAlpha = 0.8f;
	/* dark box (corners knocked out for a rounded look) */
	for (int y = 0; y < boxH; y++) {
		for (int x = 0; x < boxW; x++) {
			bool corner = (x == 0 || x == boxW - 1) &&
				      (y == 0 || y == boxH - 1);
			if (corner)
				continue;
			blendPx(img, ((by + y) * w + (bx + x)) * channels,
				channels, 0, kAlpha);
		}
	}
	/* white glyphs */
	const uint8_t *glyphs[2] = {GLYPH_A, GLYPH_I};
	for (int gi = 0; gi < 2; gi++) {
		int ox = bx + pad + gi * (glyphW + gap);
		for (int row = 0; row < 7; row++) {
			for (int col = 0; col < 5; col++) {
				if (!(glyphs[gi][row] & (0x10 >> col)))
					continue;
				for (int sy = 0; sy < scale; sy++)
					for (int sx = 0; sx < scale; sx++)
						blendPx(img,
							((by + pad +
							  row * scale + sy) *
								 w +
							 (ox + col * scale +
							  sx)) *
								channels,
							channels, 255,
							kAlpha);
			}
		}
	}
}

std::vector<uint8_t> renderWatermarkBadgeRGBA(int &outW, int &outH)
{
	/* Fixed scale 4 (52x36 box): the overlay texture has no frame
	 * size to scale from; this matches stampWatermarkAI at 1080p and
	 * stays readable down to 640x480. */
	const int scale = 4;
	const int glyphW = 5 * scale, glyphH = 7 * scale, gap = scale;
	const int pad = scale;
	const int boxW = 2 * glyphW + gap + 2 * pad;
	const int boxH = glyphH + 2 * pad;
	outW = boxW;
	outH = boxH;

	std::vector<uint8_t> buf((size_t)boxW * (size_t)boxH * 4, 0);
	/* Semi-transparent dark box (same 0.8 alpha as the stamp),
	 * corners knocked out for a rounded look. */
	const uint8_t boxA = (uint8_t)std::lround(0.8f * 255.0f);
	for (int y = 0; y < boxH; y++) {
		for (int x = 0; x < boxW; x++) {
			bool corner = (x == 0 || x == boxW - 1) &&
				      (y == 0 || y == boxH - 1);
			if (corner)
				continue;
			buf[((size_t)y * (size_t)boxW + (size_t)x) * 4 + 3] =
				boxA;
		}
	}
	/* Opaque white glyphs. */
	const uint8_t *glyphs[2] = {GLYPH_A, GLYPH_I};
	for (int gi = 0; gi < 2; gi++) {
		int ox = pad + gi * (glyphW + gap);
		for (int row = 0; row < 7; row++) {
			for (int col = 0; col < 5; col++) {
				if (!(glyphs[gi][row] & (0x10 >> col)))
					continue;
				for (int sy = 0; sy < scale; sy++)
					for (int sx = 0; sx < scale; sx++) {
						size_t i = ((size_t)(pad + row * scale + sy) * (size_t)boxW +
							    (size_t)(ox + col * scale + sx)) *
							   4;
						buf[i + 0] = 255;
						buf[i + 1] = 255;
						buf[i + 2] = 255;
						buf[i + 3] = 255;
					}
			}
		}
	}
	return buf;
}

} // namespace fx
