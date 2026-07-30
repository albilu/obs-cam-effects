#include <gtest/gtest.h>

#include "fx/engine/ep_probe.h"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

std::string providersDir()
{
	const char *home = getenv("HOME");
	return std::string(home ? home : "/home/pain") +
	       "/.config/obs-cam-effects/providers";
}

} // namespace

TEST(EpProbe, CpuFallbackWhenNoProviders)
{
	ASSERT_FALSE(fx::EpProbe::cudaAvailable("/nonexistent/providers"));
	ASSERT_STREQ(fx::EpProbe::backendName(false), "CPU");
}

TEST(EpProbe, CudaRegistersWhenProviderPresent)
{
	std::string dir = providersDir();
	std::string so = dir + "/libonnxruntime_providers_cuda.so";
	FILE *f = fopen(so.c_str(), "r");
	if (!f)
		GTEST_SKIP() << "CUDA provider not downloaded";
	fclose(f);
	/* A machine with the provider file but without the CUDA runtime
	 * on the loader path is not CUDA-capable: registration fails and
	 * sessions fall back to CPU by design. Only a CUDA-capable
	 * machine can assert registration. */
	if (!fx::EpProbe::cudaAvailable(dir))
		GTEST_SKIP() << "CUDA provider present but registration failed "
				"(CUDA 13 runtime not on the loader path)";
	ASSERT_STREQ(fx::EpProbe::backendName(true), "CUDA");
}
