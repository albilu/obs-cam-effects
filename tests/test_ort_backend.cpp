#include <gtest/gtest.h>

#include "fx/engine/ort_backend.h"

TEST(OrtModel, ThrowsOnMissingFile)
{
	EXPECT_THROW(fx::OrtModel model("/nonexistent/model.onnx", 1),
		     std::exception);
}
