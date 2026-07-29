#include "fx/image/ops.h"

#include <algorithm>

namespace fx {

std::vector<float> grayFromBgra(const Frame &f)
{
	std::vector<float> g(f.width * f.height);
	for (int i = 0; i < f.width * f.height; i++) {
		const uint8_t *p = f.bgra.data() + i * 4;
		g[i] = (0.114f * p[0] + 0.587f * p[1] + 0.299f * p[2]) / 255.0f;
	}
	return g;
}

void emaMask(Mask &cur, const Mask &prev, float beta)
{
	if (cur.px.size() != prev.px.size())
		return;
	for (size_t i = 0; i < cur.px.size(); i++)
		cur.px[i] = beta * prev.px[i] + (1.0f - beta) * cur.px[i];
}

namespace {

/* Integral image (w+1)x(h+1), double precision accumulation. */
std::vector<double> integral(const std::vector<float> &img, int w, int h)
{
	std::vector<double> ii((w + 1) * (h + 1), 0.0);
	for (int y = 0; y < h; y++) {
		double rowSum = 0;
		for (int x = 0; x < w; x++) {
			rowSum += img[y * w + x];
			ii[(y + 1) * (w + 1) + (x + 1)] =
				ii[y * (w + 1) + (x + 1)] + rowSum;
		}
	}
	return ii;
}

/* Box blur via integral image, clamped window, radius r. */
std::vector<double> boxBlur(const std::vector<float> &img, int w, int h,
			    int r)
{
	std::vector<double> ii = integral(img, w, h);
	std::vector<double> out(w * h);
	for (int y = 0; y < h; y++) {
		int y0 = std::max(0, y - r), y1 = std::min(h - 1, y + r);
		for (int x = 0; x < w; x++) {
			int x0 = std::max(0, x - r), x1 =
				std::min(w - 1, x + r);
			double sum = ii[(y1 + 1) * (w + 1) + (x1 + 1)] -
				     ii[y0 * (w + 1) + (x1 + 1)] -
				     ii[(y1 + 1) * (w + 1) + x0] +
				     ii[y0 * (w + 1) + x0];
			out[y * w + x] =
				sum / ((double)(y1 - y0 + 1) * (x1 - x0 + 1));
		}
	}
	return out;
}

std::vector<float> mul(const std::vector<float> &a,
		       const std::vector<float> &b)
{
	std::vector<float> out(a.size());
	for (size_t i = 0; i < a.size(); i++)
		out[i] = a[i] * b[i];
	return out;
}

} // namespace

std::vector<float> guidedFilter(const std::vector<float> &guide,
				const std::vector<float> &src, int w, int h,
				int r, float eps)
{
	std::vector<double> meanI = boxBlur(guide, w, h, r);
	std::vector<double> meanP = boxBlur(src, w, h, r);
	std::vector<double> corrI = boxBlur(mul(guide, guide), w, h, r);
	std::vector<double> corrIp = boxBlur(mul(guide, src), w, h, r);

	std::vector<float> a(w * h), b(w * h);
	for (int i = 0; i < w * h; i++) {
		double varI = corrI[i] - meanI[i] * meanI[i];
		double covIp = corrIp[i] - meanI[i] * meanP[i];
		a[i] = (float)(covIp / (varI + eps));
		b[i] = (float)(meanP[i] - a[i] * meanI[i]);
	}

	std::vector<double> meanA = boxBlur(a, w, h, r);
	std::vector<double> meanB = boxBlur(b, w, h, r);
	std::vector<float> q(w * h);
	for (int i = 0; i < w * h; i++)
		q[i] = (float)(meanA[i] * guide[i] + meanB[i]);
	return q;
}

} // namespace fx
