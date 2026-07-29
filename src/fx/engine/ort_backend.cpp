#include "fx/engine/ort_backend.h"

#include <stdexcept>

namespace fx {

OrtModel::OrtModel(const std::string &modelPath, int intraOpThreads)
	: env_(ORT_LOGGING_LEVEL_WARNING, "fx"), session_(nullptr)
{
	Ort::SessionOptions opts;
	opts.SetIntraOpNumThreads(intraOpThreads);
	opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
	session_ = Ort::Session(env_, modelPath.c_str(), opts);

	if (session_.GetInputCount() != 1 || session_.GetOutputCount() != 1)
		throw std::runtime_error(
			"fx: expected single-input single-output model");

	Ort::AllocatorWithDefaultOptions alloc;
	auto inName = session_.GetInputNameAllocated(0, alloc);
	auto outName = session_.GetOutputNameAllocated(0, alloc);
	input_.name = inName.get();
	output_.name = outName.get();
	input_.shape = session_.GetInputTypeInfo(0)
			       .GetTensorTypeAndShapeInfo()
			       .GetShape();
	output_.shape = session_.GetOutputTypeInfo(0)
				.GetTensorTypeAndShapeInfo()
				.GetShape();
}

std::vector<float> OrtModel::run(const std::vector<float> &inputData)
{
	auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,
					      OrtMemTypeDefault);
	Ort::Value in = Ort::Value::CreateTensor<float>(
		mem, const_cast<float *>(inputData.data()), inputData.size(),
		input_.shape.data(), input_.shape.size());
	const char *inNames[] = {input_.name.c_str()};
	const char *outNames[] = {output_.name.c_str()};
	auto outs = session_.Run(Ort::RunOptions{nullptr}, inNames, &in, 1,
				 outNames, 1);
	float *data = outs[0].GetTensorMutableData<float>();
	size_t count = outs[0].GetTensorTypeAndShapeInfo().GetElementCount();
	return std::vector<float>(data, data + count);
}

} // namespace fx
