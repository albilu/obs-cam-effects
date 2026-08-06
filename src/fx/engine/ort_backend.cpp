#include "fx/engine/ort_backend.h"

#include "fx/engine/ep_probe.h"

#include <dlfcn.h>
#include <stdexcept>

namespace fx::engine {

Ort::Env &sharedEnv()
{
	static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "fx");
	return env;
}

} // namespace fx::engine

namespace fx {

namespace {

Ort::SessionOptions makeSessionOptions(int intraOpThreads, bool cuda)
{
	Ort::SessionOptions opts;
	opts.SetIntraOpNumThreads(intraOpThreads);
	opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
	if (cuda) {
		/* Classic CUDA EP API: present only in the full GPU ORT
		 * build, resolved at runtime so the plugin still loads
		 * with the bundled CPU build (same SONAME). */
		using AppendCudaFn = OrtStatusPtr (*)(OrtSessionOptions *, int);
		static AppendCudaFn appendCuda = reinterpret_cast<AppendCudaFn>(
			dlsym(RTLD_DEFAULT,
			      "OrtSessionOptionsAppendExecutionProvider_CUDA"));
		if (!appendCuda)
			throw std::runtime_error(
				"fx: classic CUDA API unavailable");
		OrtStatusPtr st = appendCuda(opts, 0);
		if (st) {
			Ort::GetApi().ReleaseStatus(st);
			throw std::runtime_error("fx: CUDA append failed");
		}
	}
	return opts;
}

} // namespace

OrtModel::OrtModel(const std::string &modelPath, int intraOpThreads,
		   bool tryCuda)
	: session_(nullptr), modelPath_(modelPath),
	  intraOpThreads_(intraOpThreads)
{
	if (tryCuda && EpProbe::cudaAvailable()) {
		try {
			session_ = Ort::Session(
				engine::sharedEnv(), modelPath.c_str(),
				makeSessionOptions(intraOpThreads, true));
			cuda_ = true;
		} catch (...) {
			/* CUDA append or session init failed: CPU-only
			 * fallback (fx is OBS-free, nothing to log to). */
			cuda_ = false;
		}
	}
	if (!cuda_)
		session_ = Ort::Session(
			engine::sharedEnv(), modelPath.c_str(),
			makeSessionOptions(intraOpThreads, false));

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
OrtModel::runImpl(const std::vector<std::vector<float>> &inputData,
		  const std::vector<std::vector<int64_t>> *overrides)
{
	if (inputData.size() != inputs_.size())
		throw std::runtime_error("fx: input tensor count mismatch");

	auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,
					      OrtMemTypeDefault);
	std::vector<Ort::Value> inTensors;
	std::vector<const char *> inNames;
	inTensors.reserve(inputs_.size());
	for (size_t i = 0; i < inputs_.size(); i++) {
		const std::vector<int64_t> &shape =
			(overrides && !(*overrides)[i].empty()) ? (*overrides)[i]
							      : inputs_[i].shape;
		inTensors.push_back(Ort::Value::CreateTensor<float>(
			mem, const_cast<float *>(inputData[i].data()),
			inputData[i].size(), shape.data(), shape.size()));
		inNames.push_back(inputs_[i].name.c_str());
	}
	std::vector<const char *> outNames;
	outNames.reserve(outputs_.size());
	for (const auto &o : outputs_)
		outNames.push_back(o.name.c_str());

	auto outs = tryRun(inTensors, inNames, outNames, mem);

	std::vector<std::vector<float>> result;
	result.reserve(outs.size());
	for (auto &o : outs) {
		float *data = o.GetTensorMutableData<float>();
		size_t count = o.GetTensorTypeAndShapeInfo().GetElementCount();
		result.emplace_back(data, data + count);
	}
	return result;
}

std::vector<Ort::Value>
OrtModel::tryRun(std::vector<Ort::Value> &inTensors,
		 std::vector<const char *> &inNames,
		 std::vector<const char *> &outNames,
		 Ort::MemoryInfo &cpuMem)
{
	/* IoBinding with outputs bound to CPU memory: ORT copies results
	 * back from the device (required for the CUDA EP — raw Run()
	 * returns device-resident tensors whose pointers must not be
	 * dereferenced on the host). Zero-copy for CPU sessions. */
	auto runBound = [&]() {
		Ort::IoBinding binding(session_);
		for (size_t i = 0; i < inTensors.size(); i++)
			binding.BindInput(inNames[i], inTensors[i]);
		for (size_t i = 0; i < outNames.size(); i++)
			binding.BindOutput(outNames[i], cpuMem);
		binding.SynchronizeInputs();
		session_.Run(Ort::RunOptions{nullptr}, binding);
		binding.SynchronizeOutputs();
		return binding.GetOutputValues();
	};

	try {
		return runBound();
	} catch (...) {
		if (!cuda_)
			throw;
		/* The CUDA EP loaded but fails at run time (e.g. broken
		 * cuDNN/driver on the host): degrade permanently to a CPU
		 * session instead of failing every frame. */
		cuda_ = false;
		session_ = Ort::Session(engine::sharedEnv(),
					modelPath_.c_str(),
					makeSessionOptions(intraOpThreads_,
							   false));
		return runBound();
	}
}

std::vector<std::vector<float>>
OrtModel::run(const std::vector<std::vector<float>> &inputData)
{
	return runImpl(inputData, nullptr);
}

std::vector<std::vector<float>> OrtModel::runWithShapes(
	const std::vector<std::vector<float>> &inputData,
	const std::vector<std::vector<int64_t>> &overrides)
{
	if (overrides.size() != inputs_.size())
		throw std::runtime_error("fx: shape override count mismatch");
	return runImpl(inputData, &overrides);
}

std::vector<float> OrtModel::run(const std::vector<float> &inputData)
{
	auto out = run(std::vector<std::vector<float>>{inputData});
	return std::move(out.at(0));
}

} // namespace fx
