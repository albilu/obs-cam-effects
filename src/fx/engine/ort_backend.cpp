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

	if (session_.GetInputCount() == 0 || session_.GetOutputCount() == 0)
		throw std::runtime_error("fx: model with no inputs/outputs");

	Ort::AllocatorWithDefaultOptions alloc;
	for (size_t i = 0; i < session_.GetInputCount(); i++) {
		auto name = session_.GetInputNameAllocated(i, alloc);
		TensorDesc d;
		d.name = name.get();
		d.shape = session_.GetInputTypeInfo(i)
				  .GetTensorTypeAndShapeInfo()
				  .GetShape();
		inputs_.push_back(std::move(d));
	}
	for (size_t i = 0; i < session_.GetOutputCount(); i++) {
		auto name = session_.GetOutputNameAllocated(i, alloc);
		TensorDesc d;
		d.name = name.get();
		d.shape = session_.GetOutputTypeInfo(i)
				  .GetTensorTypeAndShapeInfo()
				  .GetShape();
		outputs_.push_back(std::move(d));
	}
}

[[maybe_unused]] static size_t elementCount(const std::vector<int64_t> &shape)
{
	size_t n = 1;
	for (int64_t d : shape)
		n *= (d > 0) ? (size_t)d : 1;
	return n;
}

std::vector<std::vector<float>>
OrtModel::run(const std::vector<std::vector<float>> &inputData)
{
	if (inputData.size() != inputs_.size())
		throw std::runtime_error("fx: input tensor count mismatch");

	auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,
					      OrtMemTypeDefault);
	std::vector<Ort::Value> inTensors;
	std::vector<const char *> inNames;
	inTensors.reserve(inputs_.size());
	for (size_t i = 0; i < inputs_.size(); i++) {
		inTensors.push_back(Ort::Value::CreateTensor<float>(
			mem, const_cast<float *>(inputData[i].data()),
			inputData[i].size(), inputs_[i].shape.data(),
			inputs_[i].shape.size()));
		inNames.push_back(inputs_[i].name.c_str());
	}
	std::vector<const char *> outNames;
	outNames.reserve(outputs_.size());
	for (const auto &o : outputs_)
		outNames.push_back(o.name.c_str());

	auto outs = session_.Run(Ort::RunOptions{nullptr}, inNames.data(),
				 inTensors.data(), inTensors.size(),
				 outNames.data(), outNames.size());

	std::vector<std::vector<float>> result;
	result.reserve(outs.size());
	for (auto &o : outs) {
		float *data = o.GetTensorMutableData<float>();
		size_t count = o.GetTensorTypeAndShapeInfo().GetElementCount();
		result.emplace_back(data, data + count);
	}
	return result;
}

std::vector<float> OrtModel::run(const std::vector<float> &inputData)
{
	auto out = run(std::vector<std::vector<float>>{inputData});
	return std::move(out.at(0));
}

} // namespace fx
