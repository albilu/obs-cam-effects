#include "fx/models_dl/downloader.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

namespace fx::models_dl {

const char *stateName(State s)
{
	switch (s) {
	case State::Idle:
		return "idle";
	case State::Downloading:
		return "downloading";
	case State::Verifying:
		return "verifying";
	case State::Extracting:
		return "extracting";
	case State::Done:
		return "done";
	case State::Error:
		return "error";
	}
	return "unknown";
}

Downloader::~Downloader()
{
	cancel();
	if (thread_.joinable())
		thread_.join();
}

void Downloader::start(const DownloadRequest &req)
{
	State expected = State::Idle;
	if (!state_.compare_exchange_strong(expected, State::Downloading) &&
	    expected != State::Done && expected != State::Error) {
		throw std::runtime_error("fx-dl: downloader busy");
	}
	cancel_.store(false);
	error_.clear();
	expectedSize_.store(req.expectedSize);
	destPath_ = req.destPath;
	thread_ = std::thread([this, req] { run(req); });
}

void Downloader::cancel()
{
	cancel_.store(true);
}

std::string Downloader::error() const
{
	std::lock_guard<std::mutex> lk(errorM_);
	return error_;
}

double Downloader::progress() const
{
	uint64_t total = expectedSize_.load();
	if (total == 0)
		return -1.0;
	struct stat st;
	std::string part = destPath_ + ".part";
	if (stat(part.c_str(), &st) != 0)
		return 0.0;
	double p = (double)st.st_size / (double)total;
	return p > 1.0 ? 1.0 : p;
}

namespace {

int runCmd(const std::string &cmd)
{
	int rc = std::system(cmd.c_str());
	if (rc == -1)
		return -1;
	if (WIFEXITED(rc))
		return WEXITSTATUS(rc);
	return -1;
}

std::string shellQuote(const std::string &s)
{
	std::string q = "'";
	for (char c : s)
		q += (c == '\'') ? "'\\''" : std::string(1, c);
	return q + "'";
}

} // namespace

void Downloader::run(DownloadRequest req)
{
	auto fail = [&](const std::string &msg) {
		{
			std::lock_guard<std::mutex> lk(errorM_);
			error_ = msg;
		}
		state_.store(State::Error);
	};

	std::string part = req.destPath + ".part";
	std::string cmd = "curl -fSL -C - -o " + shellQuote(part) + " " +
			  shellQuote(req.url) + " 2>/dev/null";
	int rc = runCmd(cmd);
	if (cancel_.load()) {
		state_.store(State::Idle);
		return;
	}
	if (rc != 0) {
		/* 33 = resume not supported: retry from scratch. */
		if (rc == 33) {
			std::remove(part.c_str());
			rc = runCmd(cmd);
		}
		if (rc != 0) {
			fail("curl failed with exit code " + std::to_string(rc));
			return;
		}
	}

	state_.store(State::Verifying);
	std::string verify = "sha256sum " + shellQuote(part);
	FILE *p = popen(verify.c_str(), "r");
	char buf[65] = {0};
	if (!p || fgets(buf, sizeof(buf), p) == nullptr) {
		if (p)
			pclose(p);
		fail("sha256sum failed");
		return;
	}
	pclose(p);
	std::string got(buf, strnlen(buf, 64));
	if (got != req.sha256) {
		std::remove(part.c_str());
		fail("hash mismatch: got " + got);
		return;
	}

	if (!req.extractMembers.empty()) {
		state_.store(State::Extracting);
		std::string mk = "mkdir -p " + shellQuote(req.extractDestDir);
		runCmd(mk);
		std::string tar = "tar xzf " + shellQuote(part) + " -C " +
				  shellQuote(req.extractDestDir);
		for (const auto &m : req.extractMembers)
			tar += " " + shellQuote(m);
		rc = runCmd(tar);
		if (rc != 0) {
			fail("tar extract failed with exit code " +
			     std::to_string(rc));
			return;
		}
		std::remove(part.c_str());
	} else {
		std::string tmpFinal = req.destPath + ".verify";
		std::rename(part.c_str(), tmpFinal.c_str());
		if (std::rename(tmpFinal.c_str(), req.destPath.c_str()) != 0) {
			fail("atomic rename failed");
			return;
		}
	}
	state_.store(State::Done);
}

} // namespace fx::models_dl
