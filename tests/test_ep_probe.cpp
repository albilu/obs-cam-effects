#include <gtest/gtest.h>

#include "fx/engine/ep_probe.h"

#include <dlfcn.h>

TEST(EpProbe, CpuFallbackConsistent)
{
	/* On the bundled CPU build the classic symbol is absent; on a
	 * GPU-build install it is present. Either way the probe must
	 * agree with dlsym. */
	bool expected = dlsym(RTLD_DEFAULT,
			      "OrtSessionOptionsAppendExecutionProvider_CUDA") != nullptr;
	ASSERT_EQ(fx::EpProbe::cudaAvailable(), expected);
	ASSERT_STREQ(fx::EpProbe::backendName(expected),
		     expected ? "CUDA" : "CPU");
}
