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

/* Single-worker latest-wins frame processor.
 * submit() never blocks: a still-pending frame is replaced by the newest.
 * tryGetLatest() returns the newest published mask (nullptr before first). */
class Worker {
public:
	using Processor =
		std::function<std::shared_ptr<Mask>(const Frame &)>;

	explicit Worker(Processor processor);
	~Worker();

	Worker(const Worker &) = delete;
	Worker &operator=(const Worker &) = delete;

	void start();
	void stop(); // idempotent; joins

	/* Replaces the processor (model hot-swap). Clears any pending frame
	 * so no frame is processed by a stale/torn pipeline. */
	void setProcessor(Processor processor);

	void submit(std::shared_ptr<Frame> frame);

	/* Latest published mask and its sequence (0 = none yet). */
	std::shared_ptr<const Mask> tryGetLatest(uint64_t &seqOut) const;

	/* True if a mask was published within maxAgeMs. */
	bool isFresh(uint64_t maxAgeMs) const;

private:
	void loop();

	Processor processor_;
	std::thread thread_;
	std::atomic<bool> running_{false};

	mutable std::mutex inM_;
	std::condition_variable inCv_;
	std::shared_ptr<Frame> pending_;

	mutable std::mutex outM_;
	std::shared_ptr<const Mask> latest_;
	uint64_t seq_ = 0;
	std::atomic<int64_t> lastPublishMs_{0};
};

} // namespace fx
