#pragma once

#include "fx/types.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>

namespace fx {

/* Payload published by the worker: a mask and/or processed frame. AI
 * provenance is meaningful only when frame is non-null. */
struct WorkerResult {
	std::shared_ptr<const Mask> mask;
	std::shared_ptr<const Frame> frame; // may be null (background-only)
	bool aiModified = false;
};

/* Single-worker latest-wins frame processor.
 * submit() never blocks: a still-pending frame is replaced by the newest.
 * tryGetLatest() returns the newest published result (empty before first). */
class Worker {
public:
	using Processor = std::function<WorkerResult(const Frame &)>;

	explicit Worker(Processor processor);
	~Worker();

	Worker(const Worker &) = delete;
	Worker &operator=(const Worker &) = delete;

	void start();
	void stop(); // idempotent; joins

	/* Replaces the processor (model hot-swap) and invalidates pending,
	 * in-flight, and published results from the previous processor. */
	void setProcessor(Processor processor);
	/* Clears pending and published results and rejects any currently
	 * in-flight result without replacing the processor. */
	void invalidate();

	void submit(std::shared_ptr<Frame> frame);

	/* Latest published result and its sequence (0 = none yet). */
	WorkerResult tryGetLatest(uint64_t &seqOut) const;

	/* Convenience: just the mask of the latest published result. */
	std::shared_ptr<const Mask> tryGetLatestMask(uint64_t &seqOut) const;

	/* True if a result was published within maxAgeMs. */
	bool isFresh(uint64_t maxAgeMs) const;

private:
	void loop();

	Processor processor_;
	std::thread thread_;
	std::atomic<bool> running_{false};

	mutable std::mutex inM_;
	std::condition_variable inCv_;
	std::shared_ptr<Frame> pending_;
	uint64_t epoch_ = 0;

	mutable std::mutex outM_;
	WorkerResult latest_;
	uint64_t seq_ = 0;
	std::atomic<int64_t> lastPublishMs_{0};
};

} // namespace fx
