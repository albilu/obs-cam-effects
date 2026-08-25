#pragma once

#include "fx/engine/ort_backend.h"
#include "fx/image/align.h"
#include "fx/types.h"

#include <string>
#include <vector>

namespace fx {

struct FaceBox {
	float x, y, w, h;     // top-left + size, in input-frame pixels
	Landmarks5 landmarks; // re, le, nose, rcm, lcm
	float score;
};

/* YuNet 2023mar face detector (640x640, BGR, no normalization). */
class YuNet {
public:
	explicit YuNet(const std::string &modelPath, int threads = 2, bool tryCuda = false);

	/* Detects faces in a BGRA frame of any size (resized to 640x640
	 * internally; coordinates mapped back to frame pixels). Returns
	 * boxes sorted by score desc, NMS 0.3 applied. */
	std::vector<FaceBox> detect(const Frame &frame, float scoreThresh = 0.6f);
	OrtBackend backend() const noexcept { return model_.backend(); }

private:
	OrtModel model_;
	std::vector<float> tensor_; // 3*640*640 scratch
};

} // namespace fx
