#include <gtest/gtest.h>

#include "fx/engine/ort_backend.h"

TEST(OrtModel, ThrowsOnMissingFile)
{
	EXPECT_THROW(fx::OrtModel model("/nonexistent/model.onnx", 1),
		     std::exception);
}

TEST(OrtModel, ExposesMultiIoInterface)
{
	fx::OrtModel model(FX_MODEL_PATH, 1);
	ASSERT_EQ(model.inputCount(), 1u);
	ASSERT_EQ(model.outputCount(), 1u);
	ASSERT_EQ(model.input(0).shape.size(), 4u);
	ASSERT_EQ(model.input(0).shape[1], 3);	 // NCHW
	ASSERT_EQ(model.input(0).shape[2], 192);
	ASSERT_EQ(model.output(0).shape[3], 2);	 // 2 classes
}
