#include <gtest/gtest.h>

#include "fx/pipeline/face_tracker.h"

namespace {

fx::FaceBox makeFace(float x, float y, float w, float h, float landmarkBase = 0.0f, float score = 0.9f)
{
	fx::FaceBox face{};
	face.x = x;
	face.y = y;
	face.w = w;
	face.h = h;
	for (int i = 0; i < 5; i++) {
		face.landmarks[i][0] = landmarkBase + (float)i * 2.0f;
		face.landmarks[i][1] = landmarkBase + (float)i * 2.0f + 1.0f;
	}
	face.score = score;
	return face;
}

} // namespace

TEST(FaceTracker, BalancedCadenceDetectsFirstSkipsOneDetectsNext)
{
	fx::FaceTracker tracker;

	EXPECT_TRUE(tracker.shouldDetect(640, 480, 2));
	ASSERT_TRUE(tracker.observe({makeFace(10, 20, 100, 100)}, 0.5f));
	EXPECT_FALSE(tracker.shouldDetect(640, 480, 2));
	EXPECT_TRUE(tracker.shouldDetect(640, 480, 2));
}

TEST(FaceTracker, ResetClearsTrackAndForcesDetectionAtSameResolution)
{
	fx::FaceTracker tracker;

	ASSERT_TRUE(tracker.shouldDetect(640, 480, 2));
	ASSERT_TRUE(tracker.observe({makeFace(10, 20, 100, 100)}, 0.5f));
	ASSERT_TRUE(tracker.hasBox());

	tracker.reset();

	EXPECT_FALSE(tracker.hasBox());
	EXPECT_EQ(tracker.consecutiveMisses(), 0);
	EXPECT_TRUE(tracker.shouldDetect(640, 480, 2));
}

TEST(FaceTracker, QualityCadenceDetectsEveryFrame)
{
	fx::FaceTracker tracker;

	EXPECT_TRUE(tracker.shouldDetect(640, 480, 1));
	ASSERT_TRUE(tracker.observe({makeFace(10, 20, 100, 100)}, 0.5f));
	EXPECT_TRUE(tracker.shouldDetect(640, 480, 1));
	EXPECT_TRUE(tracker.shouldDetect(640, 480, 1));
}

TEST(FaceTracker, CadenceThreeSkipsTwoFramesBeforeDetection)
{
	fx::FaceTracker tracker;

	EXPECT_TRUE(tracker.shouldDetect(640, 480, 3));
	ASSERT_TRUE(tracker.observe({makeFace(10, 20, 100, 100)}, 0.5f));
	EXPECT_FALSE(tracker.shouldDetect(640, 480, 3));
	EXPECT_FALSE(tracker.shouldDetect(640, 480, 3));
	EXPECT_TRUE(tracker.shouldDetect(640, 480, 3));
}

TEST(FaceTracker, ThreeMissesReuseBoxFourthClearsAndNoBoxDetectsEveryFrame)
{
	fx::FaceTracker tracker;
	ASSERT_TRUE(tracker.shouldDetect(640, 480, 2));
	ASSERT_TRUE(tracker.observe({makeFace(10, 20, 100, 100)}, 0.5f));

	for (int miss = 1; miss <= 3; miss++) {
		EXPECT_TRUE(tracker.observe({}, 0.5f));
		EXPECT_TRUE(tracker.hasBox());
		EXPECT_EQ(tracker.consecutiveMisses(), miss);
		EXPECT_TRUE(tracker.shouldDetect(640, 480, 2));
	}

	EXPECT_FALSE(tracker.observe({}, 0.5f));
	EXPECT_FALSE(tracker.hasBox());
	EXPECT_TRUE(tracker.shouldDetect(640, 480, 2));
	EXPECT_TRUE(tracker.shouldDetect(640, 480, 2));
}

TEST(FaceTracker, RecoveryHitResetsMissesAndBalancedCadence)
{
	fx::FaceTracker tracker;
	ASSERT_TRUE(tracker.shouldDetect(640, 480, 2));
	ASSERT_TRUE(tracker.observe({makeFace(10, 20, 100, 100)}, 0.5f));
	EXPECT_FALSE(tracker.shouldDetect(640, 480, 2));
	EXPECT_TRUE(tracker.shouldDetect(640, 480, 2));

	ASSERT_TRUE(tracker.observe({}, 0.5f));
	EXPECT_TRUE(tracker.shouldDetect(640, 480, 2));
	ASSERT_TRUE(tracker.observe({makeFace(20, 30, 110, 110)}, 0.5f));
	EXPECT_EQ(tracker.consecutiveMisses(), 0);
	EXPECT_FALSE(tracker.shouldDetect(640, 480, 2));
}

TEST(FaceTracker, ResolutionChangeClearsCoordinatesAndForcesDetection)
{
	fx::FaceTracker tracker;
	ASSERT_TRUE(tracker.shouldDetect(640, 480, 2));
	ASSERT_TRUE(tracker.observe({makeFace(10, 20, 100, 100, 30)}, 0.5f));

	EXPECT_TRUE(tracker.shouldDetect(1280, 720, 2));
	EXPECT_FALSE(tracker.hasBox());
	EXPECT_EQ(tracker.consecutiveMisses(), 0);
	EXPECT_FLOAT_EQ(tracker.box().x, 0.0f);
	EXPECT_FLOAT_EQ(tracker.box().y, 0.0f);
	EXPECT_FLOAT_EQ(tracker.box().w, 0.0f);
	EXPECT_FLOAT_EQ(tracker.box().h, 0.0f);
	for (int i = 0; i < 5; i++) {
		EXPECT_FLOAT_EQ(tracker.box().landmarks[i][0], 0.0f);
		EXPECT_FLOAT_EQ(tracker.box().landmarks[i][1], 0.0f);
	}
}

TEST(FaceTracker, LargestFaceIsSelectedAndExistingEmaIsPreserved)
{
	fx::FaceTracker tracker;
	const fx::FaceBox previous = makeFace(10, 20, 100, 100, 10, 0.8f);
	const fx::FaceBox smaller = makeFace(200, 300, 50, 200, 100, 0.7f);
	const fx::FaceBox largest = makeFace(30, 40, 120, 120, 30, 0.95f);
	ASSERT_TRUE(tracker.observe({previous}, 0.75f));

	ASSERT_TRUE(tracker.observe({smaller, largest}, 0.75f));
	ASSERT_TRUE(tracker.hasBox());
	EXPECT_FLOAT_EQ(tracker.box().x, 15.0f);
	EXPECT_FLOAT_EQ(tracker.box().y, 25.0f);
	EXPECT_FLOAT_EQ(tracker.box().w, 105.0f);
	EXPECT_FLOAT_EQ(tracker.box().h, 105.0f);
	for (int i = 0; i < 5; i++) {
		EXPECT_FLOAT_EQ(tracker.box().landmarks[i][0], 15.0f + (float)i * 2.0f);
		EXPECT_FLOAT_EQ(tracker.box().landmarks[i][1], 16.0f + (float)i * 2.0f);
	}
	EXPECT_FLOAT_EQ(tracker.box().score, 0.95f);
}
