#pragma once

#include <onnxruntime_cxx_api.h>

#include <string>
#include <vector>

namespace fx {

/* Thin wrapper around a single-input single-output ONNX Runtime CPU
 * session with dynamic IO name/shape discovery. */
class OrtModel {
public:
	struct TensorDesc {
		std::string name;
		std::vector<int64_t> shape;
	};

	explicit OrtModel(const std::string &modelPath, int intraOpThreads = 2);

	const TensorDesc &input() const { return input_; }
	const TensorDesc &output() const { return output_; }

	/* Runs the model on a float tensor matching input().shape.
	 * Returns the output tensor values. */
	std::vector<float> run(const std::vector<float> &inputData);

private:
	Ort::Env env_;
	Ort::Session session_;
	TensorDesc input_;
	TensorDesc output_;
};

} // namespace fx
