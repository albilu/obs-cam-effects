#include <gtest/gtest.h>

#include "fx/models_dl/downloader.h"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <thread>

using namespace std::chrono_literals;
using fx::models_dl::Downloader;
using fx::models_dl::State;

namespace {

std::string tmpPath(const char *name)
{
	return std::string("/tmp/opencode/fx-dl-test-") + name;
}

std::string writeFixture(const std::string &path, const std::string &content)
{
	std::ofstream f(path, std::ios::binary);
	f << content;
	return path;
}

std::string sha256Of(const std::string &path)
{
	std::string cmd = "sha256sum " + path;
	FILE *p = popen(cmd.c_str(), "r");
	char buf[65] = {0};
	if (p) {
		if (fgets(buf, sizeof(buf), p) == nullptr)
			buf[0] = 0;
		pclose(p);
	}
	return std::string(buf, strnlen(buf, 64));
}

bool waitDone(Downloader &d, int ms)
{
	for (int i = 0; i < ms / 10; i++) {
		auto s = d.state();
		if (s == State::Done || s == State::Error)
			return true;
		std::this_thread::sleep_for(10ms);
	}
	return false;
}

} // namespace

TEST(Downloader, DownloadsAndVerifiesFileUrl)
{
	std::system("mkdir -p /tmp/opencode");
	std::string src = writeFixture(tmpPath("src.bin"), "hello fx downloader\n");
	std::string dst = tmpPath("dst.bin");
	std::remove(dst.c_str());
	std::remove((dst + ".part").c_str());

	Downloader d;
	fx::models_dl::DownloadRequest req;
	req.url = "file://" + src;
	req.destPath = dst;
	req.sha256 = sha256Of(src);
	req.expectedSize = 20;
	d.start(req);

	ASSERT_TRUE(waitDone(d, 10000));
	ASSERT_EQ(d.state(), State::Done) << d.error();
	std::ifstream f(dst);
	std::string content((std::istreambuf_iterator<char>(f)),
			    std::istreambuf_iterator<char>());
	ASSERT_EQ(content, "hello fx downloader\n");
}

TEST(Downloader, HashMismatchIsError)
{
	std::system("mkdir -p /tmp/opencode");
	std::string src = writeFixture(tmpPath("src2.bin"), "bad hash test\n");
	std::string dst = tmpPath("dst2.bin");
	std::remove(dst.c_str());
	std::remove((dst + ".part").c_str());

	Downloader d;
	fx::models_dl::DownloadRequest req;
	req.url = "file://" + src;
	req.destPath = dst;
	req.sha256 =
		"0000000000000000000000000000000000000000000000000000000000000000";
	d.start(req);

	ASSERT_TRUE(waitDone(d, 10000));
	ASSERT_EQ(d.state(), State::Error);
	ASSERT_FALSE(d.error().empty());
	std::ifstream part(dst + ".part");
	ASSERT_FALSE(part.is_open());
}

TEST(Downloader, ExtractsTgzMembers)
{
	std::system("mkdir -p /tmp/opencode");
	/* Build a tiny tgz fixture: root/pkg/file.txt = "payload". */
	std::string root = tmpPath("tgzsrc");
	std::string mk = "mkdir -p " + root + "/pkg && printf payload > " +
			 root + "/pkg/file.txt && tar czf " + tmpPath("f.tgz") +
			 " -C " + root + " .";
	ASSERT_EQ(std::system(mk.c_str()), 0);
	std::string dst = tmpPath("f2.tgz");
	std::remove(dst.c_str());
	std::remove((dst + ".part").c_str());
	std::string outDir = tmpPath("tgzout");
	std::string mkOut = "mkdir -p " + outDir;
	ASSERT_EQ(std::system(mkOut.c_str()), 0);

	Downloader d;
	fx::models_dl::DownloadRequest req;
	req.url = "file://" + tmpPath("f.tgz");
	req.destPath = dst;
	req.sha256 = sha256Of(tmpPath("f.tgz"));
	req.extractMembers = {"./pkg/file.txt"};
	req.extractDestDir = outDir;
	d.start(req);

	ASSERT_TRUE(waitDone(d, 10000));
	ASSERT_EQ(d.state(), State::Done) << d.error();
	std::ifstream f(outDir + "/pkg/file.txt");
	std::string content((std::istreambuf_iterator<char>(f)),
			    std::istreambuf_iterator<char>());
	ASSERT_EQ(content, "payload");
}

TEST(Downloader, StripComponentsLandsFlat)
{
	std::system("mkdir -p /tmp/opencode");
	/* Same fixture as ExtractsTgzMembers: root/pkg/file.txt. */
	std::string root = tmpPath("tgzsrc");
	std::string dst = tmpPath("f3.tgz");
	std::remove(dst.c_str());
	std::remove((dst + ".part").c_str());
	std::string outDir = tmpPath("tgzflat");
	std::string mkOut = "rm -rf " + outDir + " && mkdir -p " + outDir;
	ASSERT_EQ(std::system(mkOut.c_str()), 0);

	Downloader d;
	fx::models_dl::DownloadRequest req;
	req.url = "file://" + tmpPath("f.tgz");
	req.destPath = dst;
	req.sha256 = sha256Of(tmpPath("f.tgz"));
	req.extractMembers = {"./pkg/file.txt"};
	req.extractDestDir = outDir;
	/* GNU tar counts the leading "./" as a component: 2 strips it
	 * plus "pkg", so file.txt lands flat (same value the bridge
	 * passes for the CUDA provider download). */
	req.stripComponents = 2;
	d.start(req);

	ASSERT_TRUE(waitDone(d, 10000));
	ASSERT_EQ(d.state(), State::Done) << d.error();
	std::ifstream f(outDir + "/file.txt");
	std::string content((std::istreambuf_iterator<char>(f)),
			    std::istreambuf_iterator<char>());
	ASSERT_EQ(content, "payload");
	std::ifstream nested(outDir + "/pkg/file.txt");
	ASSERT_FALSE(nested.is_open());
}

TEST(Downloader, RestartAfterDoneWorks)
{
	std::system("mkdir -p /tmp/opencode");
	std::string src1 = writeFixture(tmpPath("r1.bin"), "first\n");
	std::string dst1 = tmpPath("rd1.bin");
	std::remove(dst1.c_str());
	std::remove((dst1 + ".part").c_str());

	Downloader d;
	fx::models_dl::DownloadRequest req;
	req.url = "file://" + src1;
	req.destPath = dst1;
	req.sha256 = sha256Of(src1);
	d.start(req);

	ASSERT_TRUE(waitDone(d, 10000));
	ASSERT_EQ(d.state(), State::Done) << d.error();

	std::string src2 = writeFixture(tmpPath("r2.bin"), "second fixture\n");
	std::string dst2 = tmpPath("rd2.bin");
	std::remove(dst2.c_str());
	std::remove((dst2 + ".part").c_str());
	req.url = "file://" + src2;
	req.destPath = dst2;
	req.sha256 = sha256Of(src2);
	d.start(req);

	ASSERT_TRUE(waitDone(d, 10000));
	ASSERT_EQ(d.state(), State::Done) << d.error();
	std::ifstream f(dst2);
	std::string content((std::istreambuf_iterator<char>(f)),
			    std::istreambuf_iterator<char>());
	ASSERT_EQ(content, "second fixture\n");
}

TEST(Downloader, StateNamesAreStable)
{
	ASSERT_STREQ(fx::models_dl::stateName(State::Done), "done");
	ASSERT_STREQ(fx::models_dl::stateName(State::Error), "error");
	ASSERT_STREQ(fx::models_dl::stateName(State::Downloading),
		     "downloading");
}
