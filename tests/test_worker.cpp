#include <gtest/gtest.h>

#include "fx/pipeline/segmentation_pipeline.h"
#include "fx/worker.h"

#include <atomic>
#include <chrono>
#include <latch>
#include <stdexcept>
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
		fx::WorkerResult r;
		auto m = std::make_shared<fx::Mask>();
		m->width = 2;
		m->height = 2;
		m->px = {0.5f, 0.5f, 0.5f, 0.5f};
		r.mask = m;
		return r;
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
	/* The first processor call blocks inside the latch. While it is
	 * blocked, submit frames 20 and 30: 30 must replace 20. After
	 * release, exactly one more frame is processed and it is 30 —
	 * a FIFO queue would process 20 next (processed would reach 3). */
	std::latch gate(2);
	std::atomic<int> processed{0};
	std::atomic<uint8_t> lastSeen{0};
	auto blocking = [&](const fx::Frame &f) {
		int n = processed.fetch_add(1) + 1;
		lastSeen.store(f.bgra[0]);
		if (n == 1)
			gate.arrive_and_wait();
		fx::WorkerResult r;
		auto m = std::make_shared<fx::Mask>();
		m->width = 1;
		m->height = 1;
		m->px = {1.0f};
		r.mask = m;
		return r;
	};
	fx::Worker w(blocking);
	w.start();
	w.submit(makeFrame(10));
	for (int i = 0; i < 100 && processed.load() < 1; i++)
		std::this_thread::sleep_for(5ms); // wait until worker is blocked in the latch
	ASSERT_EQ(processed.load(), 1);
	w.submit(makeFrame(20));
	w.submit(makeFrame(30)); // replaces 20 in the pending slot
	gate.count_down();       // release the first call
	for (int i = 0; i < 100 && processed.load() < 2; i++)
		std::this_thread::sleep_for(5ms);
	w.stop();
	ASSERT_EQ(processed.load(), 2);
	ASSERT_EQ(lastSeen.load(), 30);
}

TEST(Worker, StopIsIdempotent)
{
	auto fast = [](const fx::Frame &) {
		fx::WorkerResult r;
		r.mask = std::make_shared<fx::Mask>();
		return r;
	};
	fx::Worker w(fast);
	w.start();
	w.stop();
	w.stop(); // must not hang or crash
	SUCCEED();
}

TEST(Worker, ProcessorThrowDoesNotCrash)
{
	std::atomic<int> calls{0};
	auto throwing = [&](const fx::Frame &) -> fx::WorkerResult {
		calls.fetch_add(1);
		throw std::runtime_error("inference failed");
	};
	fx::Worker w(throwing);
	w.start();
	w.submit(makeFrame(1));
	std::this_thread::sleep_for(50ms);
	uint64_t seq = 42;
	auto m = w.tryGetLatestMask(seq); // nothing published
	EXPECT_EQ(seq, 0u);
	EXPECT_FALSE(w.isFresh(1000));
	w.stop();
	EXPECT_GE(calls.load(), 1);
	SUCCEED(); // reaching here means no std::terminate
}

TEST(Worker, SetProcessorDropsInFlightResultBeforePublishingReplacement)
{
	std::latch firstProcessorACallEntered(1);
	std::latch releaseFirstProcessorACall(1);
	std::latch secondProcessorACallEntered(1);
	std::latch releaseSecondProcessorACall(1);
	std::latch processorBEntered(1);
	std::latch releaseProcessorB(1);
	std::atomic<int> processorACalls{0};
	auto processorA = [&](const fx::Frame &) {
		const int call = processorACalls.fetch_add(1) + 1;
		if (call == 1) {
			firstProcessorACallEntered.count_down();
			releaseFirstProcessorACall.wait();
		} else if (call == 2) {
			secondProcessorACallEntered.count_down();
			releaseSecondProcessorACall.wait();
		}
		fx::WorkerResult r;
		auto m = std::make_shared<fx::Mask>();
		m->width = 1;
		m->height = 1;
		m->px = {(float)call};
		r.mask = m;
		return r;
	};
	auto processorB = [&](const fx::Frame &) {
		processorBEntered.count_down();
		releaseProcessorB.wait();
		fx::WorkerResult r;
		auto m = std::make_shared<fx::Mask>();
		m->width = 1;
		m->height = 1;
		m->px = {2.0f};
		r.mask = m;
		return r;
	};
	fx::Worker w(processorA);
	w.start();
	w.submit(makeFrame(1));
	firstProcessorACallEntered.wait();
	w.submit(makeFrame(2));
	releaseFirstProcessorACall.count_down();
	secondProcessorACallEntered.wait();

	uint64_t seq = 0;
	auto m = w.tryGetLatestMask(seq);
	EXPECT_EQ(seq, 1u);
	EXPECT_NE(m, nullptr);
	if (m) {
		EXPECT_FLOAT_EQ(m->px[0], 1.0f);
	}
	EXPECT_TRUE(w.isFresh(5000));

	w.setProcessor(processorB);
	m = w.tryGetLatestMask(seq);
	EXPECT_EQ(seq, 1u);
	EXPECT_EQ(m, nullptr);
	EXPECT_FALSE(w.isFresh(5000));

	w.submit(makeFrame(3));
	releaseSecondProcessorACall.count_down();
	processorBEntered.wait();
	m = w.tryGetLatestMask(seq);
	EXPECT_EQ(seq, 1u);
	EXPECT_EQ(m, nullptr);
	EXPECT_FALSE(w.isFresh(5000));

	releaseProcessorB.count_down();
	w.stop();
	m = w.tryGetLatestMask(seq);
	ASSERT_EQ(seq, 2u);
	ASSERT_NE(m, nullptr);
	ASSERT_FLOAT_EQ(m->px[0], 2.0f);
}

TEST(Worker, InvalidateClearsLatestAndDropsInFlightResult)
{
	std::latch firstCallEntered(1);
	std::latch releaseFirstCall(1);
	std::latch secondCallEntered(1);
	std::latch releaseSecondCall(1);
	std::latch thirdCallEntered(1);
	std::latch releaseThirdCall(1);
	std::atomic<int> calls{0};
	auto processor = [&](const fx::Frame &frame) {
		const int call = calls.fetch_add(1) + 1;
		if (call == 1) {
			firstCallEntered.count_down();
			releaseFirstCall.wait();
		} else if (call == 2) {
			secondCallEntered.count_down();
			releaseSecondCall.wait();
		} else if (call == 3) {
			thirdCallEntered.count_down();
			releaseThirdCall.wait();
		}
		fx::WorkerResult r;
		auto m = std::make_shared<fx::Mask>();
		m->width = 1;
		m->height = 1;
		m->px = {(float)frame.bgra[0]};
		r.mask = m;
		return r;
	};

	fx::Worker w(processor);
	w.start();
	w.submit(makeFrame(10));
	firstCallEntered.wait();
	w.submit(makeFrame(20));
	releaseFirstCall.count_down();
	secondCallEntered.wait();

	uint64_t seq = 0;
	auto m = w.tryGetLatestMask(seq);
	EXPECT_EQ(seq, 1u);
	EXPECT_NE(m, nullptr);
	if (m) {
		EXPECT_FLOAT_EQ(m->px[0], 10.0f);
	}
	EXPECT_TRUE(w.isFresh(5000));

	w.invalidate();
	m = w.tryGetLatestMask(seq);
	EXPECT_EQ(seq, 1u);
	EXPECT_EQ(m, nullptr);
	EXPECT_FALSE(w.isFresh(5000));

	w.submit(makeFrame(30));
	releaseSecondCall.count_down();
	thirdCallEntered.wait();
	m = w.tryGetLatestMask(seq);
	EXPECT_EQ(seq, 1u);
	EXPECT_EQ(m, nullptr);
	EXPECT_FALSE(w.isFresh(5000));

	releaseThirdCall.count_down();
	w.stop();
	m = w.tryGetLatestMask(seq);
	ASSERT_EQ(seq, 2u);
	ASSERT_NE(m, nullptr);
	EXPECT_FLOAT_EQ(m->px[0], 30.0f);
	EXPECT_TRUE(w.isFresh(5000));
}

TEST(Worker, PublishesFrameAndMaskBundle)
{
	auto proc = [](const fx::Frame &) {
		fx::WorkerResult r;
		auto m = std::make_shared<fx::Mask>();
		m->width = 1;
		m->height = 1;
		m->px = {0.5f};
		r.mask = m;
		auto f = std::make_shared<fx::Frame>();
		f->width = 2;
		f->height = 2;
		f->bgra.assign(16, 7);
		r.frame = f;
		r.aiModified = true;
		return r;
	};
	fx::Worker w(proc);
	w.start();
	auto f = std::make_shared<fx::Frame>();
	f->width = 4;
	f->height = 4;
	f->bgra.assign(64, 1);
	w.submit(f);
	uint64_t seq = 0;
	fx::WorkerResult r;
	for (int i = 0; i < 100 && seq == 0; i++) {
		std::this_thread::sleep_for(std::chrono::milliseconds(5));
		r = w.tryGetLatest(seq);
	}
	w.stop();
	ASSERT_EQ(seq, 1u);
	ASSERT_TRUE(r.mask != nullptr);
	ASSERT_TRUE(r.frame != nullptr);
	ASSERT_TRUE(r.aiModified);
	ASSERT_EQ(r.frame->width, 2);
	ASSERT_EQ(r.frame->bgra[0], 7);
}

TEST(SegmentationPipeline, EndToEndWithRealModel)
{
	fx::SegmentationPipeline pipe(fx::SegTier::Standard, "",
				      FX_MODEL_PATH, "", 1);
	fx::Frame f;
	f.width = 320;
	f.height = 240;
	f.bgra.assign(320u * 240u * 4u, 128);
	auto m = pipe.process(f);
	ASSERT_EQ(m->width, 192);
	ASSERT_EQ(m->height, 192);
	ASSERT_EQ(m->px.size(), 192u * 192u);
}
