#include <gtest/gtest.h>

#include "fx/models/onnx_init.h"

#include <cmath>
#include <cstdio>

static bool fileExists(const char *p)
{
	FILE *f = fopen(p, "r");
	if (f)
		fclose(f);
	return f != nullptr;
}

TEST(OnnxInit, ExtractsEmap)
{
	const char *path = FX_INSWAPPER_PATH;
	if (!fileExists(path))
		GTEST_SKIP() << "inswapper_128.onnx not downloaded";
	auto emap = fx::onnxLastInitializerFloats(path, 512 * 512);
	ASSERT_EQ(emap.size(), 512u * 512u);
	ASSERT_NEAR(emap[0], 0.12484695f, 1e-6f);
	ASSERT_NEAR(emap[1], -0.00845782f, 1e-6f);
	ASSERT_NEAR(emap[2], 0.08038428f, 1e-6f);
	ASSERT_NEAR(emap[3], -0.1220004f, 1e-6f);
	size_t last = 512u * 512u - 4;
	ASSERT_NEAR(emap[last + 0], -0.20361629f, 1e-6f);
	ASSERT_NEAR(emap[last + 1], -0.33891863f, 1e-6f);
	ASSERT_NEAR(emap[last + 2], 0.29195625f, 1e-6f);
	ASSERT_NEAR(emap[last + 3], -0.08580378f, 1e-6f);
	double sumAbs = 0;
	for (float v : emap)
		sumAbs += std::fabs(v);
	ASSERT_NEAR(sumAbs, 35887.31640625, 1.0);
}
