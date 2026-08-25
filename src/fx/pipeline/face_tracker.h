#pragma once

#include "fx/models/yunet.h"

#include <vector>

namespace fx {

class FaceTracker {
public:
	void reset();
	bool shouldDetect(int width, int height, int everyN);
	bool observe(const std::vector<FaceBox> &faces, float bboxEma);
	bool hasBox() const;
	const FaceBox &box() const;
	int consecutiveMisses() const;

private:
	void clearTrack();
	static constexpr int kMissGrace = 3;
	bool haveBox_ = false;
	FaceBox box_{};
	int consecutiveMisses_ = 0;
	int skippedSinceDetection_ = 0;
	int frameWidth_ = 0;
	int frameHeight_ = 0;
};

} // namespace fx
