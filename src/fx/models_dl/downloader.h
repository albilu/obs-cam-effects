#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace fx::models_dl {

struct DownloadRequest {
	std::string url;           // http(s):// or file:// (tests)
	std::string destPath;      // final file path
	std::string sha256;        // expected hex digest (lowercase)
	uint64_t expectedSize = 0; // bytes; 0 = unknown (progress then indeterminate)
	/* Optional: treat destPath as a tgz and extract these member paths
	 * (relative to archive root) into extractDestDir, then delete the tgz. */
	std::vector<std::string> extractMembers;
	std::string extractDestDir;
	/* Passed to tar as --strip-components=N when > 0: drops the N
	 * leading path components of extracted members (e.g. 2 makes the
	 * ORT provider libs land flat in extractDestDir). */
	int stripComponents = 0;
	/* When true, each extracted member is moved into extractDestDir
	 * with an individual atomic rename() (existing files with the same
	 * name are replaced, unrelated files are preserved). When false,
	 * the whole extractDestDir is swapped with the staging dir. Use
	 * overlay mode when the destination holds other live content
	 * (e.g. the plugin bin dir): the swap would wipe it, and
	 * overwriting a mapped .so in place can SIGBUS the running
	 * process — rename() swaps the dirent, the old inode stays
	 * mapped. */
	bool extractOverlay = false;
};

enum class State { Idle, Downloading, Verifying, Extracting, Done, Error };

const char *stateName(State s);

/* Single-shot background downloader using system binaries
 * (curl, sha256sum, tar). Thread-safe; progress polled from the
 * growing .part file size (no stderr parsing). */
class Downloader {
public:
	Downloader() = default;
	~Downloader();

	Downloader(const Downloader &) = delete;
	Downloader &operator=(const Downloader &) = delete;

	/* Starts the download. Throws std::runtime_error if already busy. */
	void start(const DownloadRequest &req);
	void cancel();

	State state() const { return state_.load(); }
	double progress() const; // 0..1, or -1 if expectedSize unknown
	std::string error() const;

private:
	void run(DownloadRequest req);

	std::thread thread_;
	std::atomic<State> state_{State::Idle};
	std::atomic<bool> cancel_{false};
	std::atomic<uint64_t> expectedSize_{0};
	std::string destPath_;
	std::string error_;
	mutable std::mutex errorM_;
};

} // namespace fx::models_dl
