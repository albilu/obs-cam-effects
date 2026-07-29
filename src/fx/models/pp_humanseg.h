#pragma once

#include "fx/engine/ort_backend.h"
#include "fx/types.h"

#include <string>
#include <vector>

namespace fx {

/* PP-HumanSeg v2 lite (192x192, Apache-2.0).
 * Preprocessing matches obs-backgroundremoval: BGR order kept,
 * (v/256 - 0.5)/0.5, NCHW. Postprocess: output channel 1 = person,
 * min-max normalized to [0,1]. */
class PPHumanSeg {
public:
	static constexpr int kSize = 192;

	explicit PPHumanSeg(const std::string &modelPath, int threads = 2);

	Mask infer(const Frame &frame);

private:
	OrtModel model_;
	std::vector<float> tensor_; // 3*192*192 scratch
};

} // namespace fx
