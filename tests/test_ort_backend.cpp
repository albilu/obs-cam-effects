#include <gtest/gtest.h>

#include "fx/engine/ort_backend.h"

TEST(OrtBackendState, AllowCpuFallbackRetriesProviderCodesWithoutPublishingCpu)
{
	fx::OrtBackendState state(fx::OrtExecutionPolicy::AllowCpuFallback);
	EXPECT_EQ(state.backend(), fx::OrtBackend::Cpu);

	state.setBackend(fx::OrtBackend::Cuda);
	for (OrtErrorCode code : {ORT_FAIL, ORT_ENGINE_ERROR, ORT_RUNTIME_EXCEPTION, ORT_EP_FAIL, ORT_DEVICE_RESET}) {
		EXPECT_EQ(state.onRunFailure(code), fx::OrtRunFailureAction::RetryCpu);
		EXPECT_EQ(state.backend(), fx::OrtBackend::Cuda);
	}

	state.markCpuFallbackReady();
	EXPECT_EQ(state.backend(), fx::OrtBackend::Cpu);
}

TEST(OrtBackendState, RequireCudaProviderFailuresPoisonBackendWithoutRetry)
{
	fx::OrtBackendState state(fx::OrtExecutionPolicy::RequireCuda);
	for (OrtErrorCode code : {ORT_EP_FAIL, ORT_DEVICE_RESET}) {
		state.setBackend(fx::OrtBackend::Cuda);
		EXPECT_EQ(state.onRunFailure(code), fx::OrtRunFailureAction::Rethrow);
		EXPECT_EQ(state.backend(), fx::OrtBackend::Failed);
	}
}

TEST(OrtBackendState, InvalidArgumentRethrowsWithoutPoisoningRequiredCuda)
{
	fx::OrtBackendState state(fx::OrtExecutionPolicy::RequireCuda);
	state.setBackend(fx::OrtBackend::Cuda);

	EXPECT_EQ(state.onRunFailure(ORT_INVALID_ARGUMENT), fx::OrtRunFailureAction::Rethrow);
	EXPECT_EQ(state.backend(), fx::OrtBackend::Cuda);
}

TEST(OrtBackendState, CpuReplacementFailurePublishesFailed)
{
	fx::OrtBackendState state(fx::OrtExecutionPolicy::AllowCpuFallback);
	state.setBackend(fx::OrtBackend::Cuda);
	ASSERT_EQ(state.onRunFailure(ORT_RUNTIME_EXCEPTION), fx::OrtRunFailureAction::RetryCpu);

	state.markCpuFallbackFailed();
	EXPECT_EQ(state.backend(), fx::OrtBackend::Failed);
}

TEST(OrtModel, ThrowsOnMissingFile)
{
	EXPECT_THROW(fx::OrtModel model("/nonexistent/model.onnx", 1), std::exception);
}

TEST(OrtModel, RequireCudaWithoutCudaRequestThrowsDeterministically)
{
	try {
		fx::OrtModel model(FX_MODEL_PATH, 1, false, fx::OrtExecutionPolicy::RequireCuda);
		FAIL() << "expected CUDA policy failure";
	} catch (const std::runtime_error &error) {
		EXPECT_STREQ(error.what(), "fx: CUDA execution required");
	} catch (...) {
		FAIL() << "expected std::runtime_error";
	}
}

TEST(OrtModel, ExposesMultiIoInterface)
{
	fx::OrtModel model(FX_MODEL_PATH, 1);
	ASSERT_EQ(model.inputCount(), 1u);
	ASSERT_EQ(model.outputCount(), 1u);
	ASSERT_EQ(model.input(0).shape.size(), 4u);
	ASSERT_EQ(model.input(0).shape[1], 3); // NCHW
	ASSERT_EQ(model.input(0).shape[2], 192);
	ASSERT_EQ(model.output(0).shape[3], 2); // 2 classes
}
