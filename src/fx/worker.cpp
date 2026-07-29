#include "fx/worker.h"

#include <chrono>

namespace fx {

static int64_t nowMs()
{
	return std::chrono::duration_cast<std::chrono::milliseconds>(
		       std::chrono::steady_clock::now().time_since_epoch())
		.count();
}

Worker::Worker(Processor processor) : processor_(std::move(processor)) {}

Worker::~Worker()
{
	stop();
}

void Worker::start()
{
	bool expected = false;
	if (!running_.compare_exchange_strong(expected, true))
		return;
	thread_ = std::thread([this] { loop(); });
}

void Worker::stop()
{
	bool expected = true;
	if (!running_.compare_exchange_strong(expected, false))
		return;
	{
		std::lock_guard<std::mutex> lk(inM_);
		inCv_.notify_all();
	}
	if (thread_.joinable())
		thread_.join();
}

void Worker::submit(std::shared_ptr<Frame> frame)
{
	{
		std::lock_guard<std::mutex> lk(inM_);
		pending_ = std::move(frame); // latest-wins: replaces stale
	}
	inCv_.notify_one();
}

std::shared_ptr<const Mask> Worker::tryGetLatest(uint64_t &seqOut) const
{
	std::lock_guard<std::mutex> lk(outM_);
	seqOut = seq_;
	return latest_;
}

bool Worker::isFresh(uint64_t maxAgeMs) const
{
	int64_t last = lastPublishMs_.load();
	return last != 0 && (nowMs() - last) <= (int64_t)maxAgeMs;
}

void Worker::loop()
{
	while (running_.load()) {
		std::shared_ptr<Frame> frame;
		{
			std::unique_lock<std::mutex> lk(inM_);
			inCv_.wait(lk, [this] {
				return pending_ != nullptr || !running_.load();
			});
			if (!running_.load())
				break;
			frame = std::move(pending_);
			pending_.reset();
		}
		std::shared_ptr<Mask> result = processor_(*frame);
		{
			std::lock_guard<std::mutex> lk(outM_);
			latest_ = std::move(result);
			seq_++;
			lastPublishMs_.store(nowMs());
		}
	}
}

} // namespace fx
