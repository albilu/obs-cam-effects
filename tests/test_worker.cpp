#include <gtest/gtest.h>

#include "fx/pipeline/segmentation_pipeline.h"
#include "fx/worker.h"

#include <chrono>
#include <latch>
#include <thread>

using namespace std::chrono_literals;

static std::shared_ptr<fx::Frame> makeFrame(uint8_t v)
{
	auto f = std::make_shared<fx::Frame>();
	f->width = 4;
	f->height = 4;
	f->bgra.assign(4u * 4u * 4u, v);
	return f;
}

TEST(Worker, ProcessesAndPublishesLatest)
{
	auto slow = [](const fx::Frame &) {
		auto m = std::make_shared<fx::Mask>();
		m->width = 2;
		m->height = 2;
		m->px = {0.5f, 0.5f, 0.5f, 0.5f};
		return m;
	};
	fx::Worker w(slow);
	w.start();
	w.submit(makeFrame(1));
	uint64_t seq = 0;
	for (int i = 0; i < 100 && seq == 0; i++) {
		std::this_thread::sleep_for(5ms);
		w.tryGetLatest(seq);
	}
	w.stop();
	ASSERT_EQ(seq, 1u);
	ASSERT_TRUE(w.isFresh(5000));
}

TEST(Worker, LatestWinsDropsStaleFrames)
{
	/* Processor blocks on a latch; while blocked, submit 3 frames.
	 * After release, the NEXT processed frame must be the newest. */
	std::latch gate(2);
	std::atomic<int> processed{0};
	std::atomic<uint8_t> lastSeen{0};
	auto blocking = [&](const fx::Frame &f) {
		if (processed.fetch_add(1) == 0)
			gate.count_down(); // first call: hold until released
		lastSeen.store(f.bgra[0]);
		auto m = std::make_shared<fx::Mask>();
		m->width = 1;
		m->height = 1;
		m->px = {1.0f};
		return m;
	};
	fx::Worker w(blocking);
	w.start();
	w.submit(makeFrame(10));
	std::this_thread::sleep_for(20ms); // first frame picked up
	w.submit(makeFrame(20));
	w.submit(makeFrame(30)); // replaces 20
	gate.count_down();       // release the gate
	for (int i = 0; i < 100 && processed.load() < 2; i++)
		std::this_thread::sleep_for(5ms);
	w.stop();
	ASSERT_EQ(lastSeen.load(), 30);
}

TEST(Worker, StopIsIdempotent)
{
	auto fast = [](const fx::Frame &) {
		return std::make_shared<fx::Mask>();
	};
	fx::Worker w(fast);
	w.start();
	w.stop();
	w.stop(); // must not hang or crash
	SUCCEED();
}

TEST(SegmentationPipeline, EndToEndWithRealModel)
{
	fx::SegmentationPipeline pipe(FX_MODEL_PATH, 1);
	fx::Frame f;
	f.width = 320;
	f.height = 240;
	f.bgra.assign(320u * 240u * 4u, 128);
	auto m = pipe.process(f);
	ASSERT_EQ(m->width, 192);
	ASSERT_EQ(m->height, 192);
	ASSERT_EQ(m->px.size(), 192u * 192u);
}
