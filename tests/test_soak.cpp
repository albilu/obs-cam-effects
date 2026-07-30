#include <gtest/gtest.h>

#include "fx/pipeline/segmentation_pipeline.h"
#include "fx/worker.h"

#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>

/* Soak: 600 frames through the worker, submitted slightly faster than
 * the pipeline can process, so the worker stays saturated and the
 * latest-wins drop policy must engage (processed < submitted).
 * Asserts sustained throughput > 30 fps. */
TEST(Soak, ThroughputAndDropPolicy)
{
	fx::SegmentationPipeline pipe(fx::SegTier::Standard, "",
				      FX_MODEL_PATH, "", 2);
	std::atomic<uint64_t> processed{0};
	fx::Worker w([&](const fx::Frame &f) {
		auto m = pipe.process(f);
		processed.fetch_add(1);
		return m;
	});
	w.start();

	auto t0 = std::chrono::steady_clock::now();
	const int kFrames = 600;
	for (int i = 0; i < kFrames; i++) {
		auto f = std::make_shared<fx::Frame>();
		f->width = 192;
		f->height = 192;
		f->bgra.assign(192u * 192u * 4u, (uint8_t)(i & 0xFF));
		w.submit(std::move(f));
		/* Throttle submission (~1000 fps): an unthrottled burst
		 * finishes in ~30 ms and latest-wins collapses all 600
		 * frames into ~2 processed, leaving the worker idle for
		 * the rest of the measurement window. */
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}
	w.stop();
	auto t1 = std::chrono::steady_clock::now();

	double ms =
		std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0)
			.count();
	uint64_t done = processed.load();
	double fps = done / (ms / 1000.0);
	std::cout << "soak: " << done << "/" << kFrames << " frames in " << ms
		  << " ms = " << fps << " fps" << std::endl;
	EXPECT_GT(fps, 30.0);
	/* Saturation => drops must have occurred. */
	EXPECT_LT(done, (uint64_t)kFrames);
}
