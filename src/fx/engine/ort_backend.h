#pragma once

#include <onnxruntime_cxx_api.h>

#include <string>
#include <vector>

namespace fx {

/* Wrapper around an ONNX Runtime session with dynamic IO discovery.
 * Supports multi-input multi-output models (RVM: 6 in / 6 out);
 * single-IO models use the convenience accessors. */
class OrtModel {
public:
	struct TensorDesc {
		std::string name;
		std::vector<int64_t> shape;
	};

	explicit OrtModel(const std::string &modelPath, int intraOpThreads = 2);

	size_t inputCount() const { return inputs_.size(); }
	size_t outputCount() const { return outputs_.size(); }
	const TensorDesc &input(size_t i) const { return inputs_.at(i); }
	const TensorDesc &output(size_t i) const { return outputs_.at(i); }

	/* Backward-compatible single-IO accessors. */
	const TensorDesc &input() const { return inputs_.at(0); }
	const TensorDesc &output() const { return outputs_.at(0); }

	/* Single-IO run (existing behavior). */
	std::vector<float> run(const std::vector<float> &inputData);

	/* Multi-IO run: one flat float tensor per declared input.
	 * Returns one flat float tensor per declared output (in order). */
	std::vector<std::vector<float>>
	run(const std::vector<std::vector<float>> &inputData);

private:
	Ort::Env env_;
	Ort::Session session_;
	std::vector<TensorDesc> inputs_;
	std::vector<TensorDesc> outputs_;
};

} // namespace fx
