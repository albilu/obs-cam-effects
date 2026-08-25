#include "fx/pipeline/face_tracker.h"

namespace fx {

void FaceTracker::reset()
{
	clearTrack();
	frameWidth_ = 0;
	frameHeight_ = 0;
}

bool FaceTracker::shouldDetect(int width, int height, int everyN)
{
	if (width != frameWidth_ || height != frameHeight_) {
		frameWidth_ = width;
		frameHeight_ = height;
		clearTrack();
		return true;
	}

	if (!haveBox_ || consecutiveMisses_ > 0 || everyN <= 1)
		return true;

	if (skippedSinceDetection_ < everyN - 1) {
		skippedSinceDetection_++;
		return false;
	}

	return true;
}

bool FaceTracker::observe(const std::vector<FaceBox> &faces, float bboxEma)
{
	if (faces.empty()) {
		if (!haveBox_)
			return false;

		consecutiveMisses_++;
		if (consecutiveMisses_ <= kMissGrace)
			return true;

		clearTrack();
		return false;
	}

	const FaceBox *largest = &faces.front();
	for (const FaceBox &face : faces) {
		if (face.w * face.h > largest->w * largest->h)
			largest = &face;
	}

	FaceBox next = *largest;
	if (haveBox_) {
		const float currentWeight = 1.0f - bboxEma;
		next.x = bboxEma * box_.x + currentWeight * next.x;
		next.y = bboxEma * box_.y + currentWeight * next.y;
		next.w = bboxEma * box_.w + currentWeight * next.w;
		next.h = bboxEma * box_.h + currentWeight * next.h;
		for (int i = 0; i < 5; i++) {
			for (int coordinate = 0; coordinate < 2; coordinate++) {
				next.landmarks[i][coordinate] = bboxEma * box_.landmarks[i][coordinate] +
								currentWeight * next.landmarks[i][coordinate];
			}
		}
	}

	box_ = next;
	haveBox_ = true;
	consecutiveMisses_ = 0;
	skippedSinceDetection_ = 0;
	return true;
}

bool FaceTracker::hasBox() const
{
	return haveBox_;
}

const FaceBox &FaceTracker::box() const
{
	return box_;
}

int FaceTracker::consecutiveMisses() const
{
	return consecutiveMisses_;
}

void FaceTracker::clearTrack()
{
	haveBox_ = false;
	box_ = {};
	consecutiveMisses_ = 0;
	skippedSinceDetection_ = 0;
}

} // namespace fx
