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

Ort::SessionOptions makeSessionOptions(int intraOpThreads, bool cuda, OrtExecutionPolicy policy)
{
	Ort::SessionOptions opts;
	opts.SetIntraOpNumThreads(intraOpThreads);
	opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
	if (cuda) {
		/* Strict sessions must not let ORT silently assign nodes to CPU. */
		if (policy == OrtExecutionPolicy::RequireCuda)
			opts.AddConfigEntry("session.disable_cpu_ep_fallback", "1");

		/* The classic append API logs through ORT's DefaultLogger,
		 * which exists only once an Env has been created. Force ours
		 * first: in the OrtModel ctor the Env and the options are
		 * built in the same expression, whose evaluation order is
		 * unspecified — without this the FIRST CUDA session in a
		 * process fails with "Attempt to use DefaultLogger but
		 * none has been registered" and is treated as a CUDA
		 * initialization failure. */
		(void)engine::sharedEnv();
		/* Classic CUDA EP API: present only in the full GPU ORT
		 * build, resolved at runtime so the plugin still loads
		 * with the bundled CPU build (same SONAME). */
		using AppendCudaFn = OrtStatusPtr (*)(OrtSessionOptions *, int);
		static AppendCudaFn appendCuda = reinterpret_cast<AppendCudaFn>(
			dlsym(RTLD_DEFAULT, "OrtSessionOptionsAppendExecutionProvider_CUDA"));
		if (!appendCuda)
			throw std::runtime_error("fx: classic CUDA API unavailable");
		OrtStatusPtr st = appendCuda(opts, 0);
		Ort::ThrowOnError(st);
	}
	return opts;
}

bool isProviderFailure(OrtErrorCode code) noexcept
{
	switch (code) {
	case ORT_FAIL:
	case ORT_ENGINE_ERROR:
	case ORT_RUNTIME_EXCEPTION:
	case ORT_EP_FAIL:
	case ORT_DEVICE_RESET:
		return true;
	default:
		return false;
	}
}

} // namespace

OrtBackendState::OrtBackendState(OrtExecutionPolicy policy) noexcept : policy_(policy) {}

OrtBackend OrtBackendState::backend() const noexcept
{
	return backend_.load();
}

void OrtBackendState::setBackend(OrtBackend backend) noexcept
{
	backend_.store(backend);
}

OrtRunFailureAction OrtBackendState::onRunFailure(OrtErrorCode code) noexcept
{
	if (backend() != OrtBackend::Cuda || !isProviderFailure(code))
		return OrtRunFailureAction::Rethrow;
	if (policy_ == OrtExecutionPolicy::AllowCpuFallback)
		return OrtRunFailureAction::RetryCpu;
	setBackend(OrtBackend::Failed);
	return OrtRunFailureAction::Rethrow;
}

void OrtBackendState::markCpuFallbackReady() noexcept
{
	setBackend(OrtBackend::Cpu);
}

void OrtBackendState::markCpuFallbackFailed() noexcept
{
	setBackend(OrtBackend::Failed);
}

OrtModel::OrtModel(const std::string &modelPath, int intraOpThreads, bool tryCuda, OrtExecutionPolicy policy)
	: backendState_(policy),
	  session_(nullptr),
	  modelPath_(modelPath),
	  intraOpThreads_(intraOpThreads)
{
	const bool cudaAvailable = tryCuda && EpProbe::cudaAvailable();
	if (policy == OrtExecutionPolicy::RequireCuda && !cudaAvailable)
		throw std::runtime_error("fx: CUDA execution required");

	if (cudaAvailable) {
		try {
			session_ = Ort::Session(engine::sharedEnv(), modelPath.c_str(),
						makeSessionOptions(intraOpThreads, true, policy));
			backendState_.setBackend(OrtBackend::Cuda);
		} catch (const Ort::Exception &error) {
			if (policy == OrtExecutionPolicy::RequireCuda || !isProviderFailure(error.GetOrtErrorCode()))
				throw;
		}
	}
	if (backend() != OrtBackend::Cuda) {
		session_ = Ort::Session(engine::sharedEnv(), modelPath.c_str(),
					makeSessionOptions(intraOpThreads, false, policy));
		backendState_.setBackend(OrtBackend::Cpu);
	}

	if (session_.GetInputCount() == 0 || session_.GetOutputCount() == 0)
		throw std::runtime_error("fx: model with no inputs/outputs");

	Ort::AllocatorWithDefaultOptions alloc;
	for (size_t i = 0; i < session_.GetInputCount(); i++) {
		auto name = session_.GetInputNameAllocated(i, alloc);
		TensorDesc d;
		d.name = name.get();
		d.shape = session_.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
		inputs_.push_back(std::move(d));
	}
	for (size_t i = 0; i < session_.GetOutputCount(); i++) {
		auto name = session_.GetOutputNameAllocated(i, alloc);
		TensorDesc d;
		d.name = name.get();
		d.shape = session_.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
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

std::vector<std::vector<float>> OrtModel::runImpl(const std::vector<std::vector<float>> &inputData,
						  const std::vector<std::vector<int64_t>> *overrides)
{
	if (backend() == OrtBackend::Failed)
		throw std::runtime_error("fx: CUDA execution failed");
	if (inputData.size() != inputs_.size())
		throw std::runtime_error("fx: input tensor count mismatch");

	auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	std::vector<Ort::Value> inTensors;
	std::vector<const char *> inNames;
	inTensors.reserve(inputs_.size());
	for (size_t i = 0; i < inputs_.size(); i++) {
		const std::vector<int64_t> &shape = (overrides && !(*overrides)[i].empty()) ? (*overrides)[i]
											    : inputs_[i].shape;
		inTensors.push_back(Ort::Value::CreateTensor<float>(mem, const_cast<float *>(inputData[i].data()),
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

std::vector<Ort::Value> OrtModel::tryRun(std::vector<Ort::Value> &inTensors, std::vector<const char *> &inNames,
					 std::vector<const char *> &outNames, Ort::MemoryInfo &cpuMem)
{
	if (backend() == OrtBackend::Failed)
		throw std::runtime_error("fx: CUDA execution failed");

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
	} catch (const Ort::Exception &error) {
		if (backendState_.onRunFailure(error.GetOrtErrorCode()) != OrtRunFailureAction::RetryCpu)
			throw;
		/* Keep Cuda published until a replacement session is installed. */
		try {
			Ort::Session cpuSession(engine::sharedEnv(), modelPath_.c_str(),
						makeSessionOptions(intraOpThreads_, false,
								   OrtExecutionPolicy::AllowCpuFallback));
			session_ = std::move(cpuSession);
		} catch (...) {
			backendState_.markCpuFallbackFailed();
			throw;
		}
		backendState_.markCpuFallbackReady();
		return runBound();
	}
}

std::vector<std::vector<float>> OrtModel::run(const std::vector<std::vector<float>> &inputData)
{
	return runImpl(inputData, nullptr);
}

std::vector<std::vector<float>> OrtModel::runWithShapes(const std::vector<std::vector<float>> &inputData,
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
