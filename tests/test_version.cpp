#include <gtest/gtest.h>

#include "fx/version.h"

TEST(FxVersion, ReturnsCurrentVersion)
{
	EXPECT_STREQ(fx::version(), "0.1.0");
}
