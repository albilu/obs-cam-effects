#pragma once

#include <onnxruntime_cxx_api.h>

#include <atomic>

#include <string>
#include <vector>

namespace fx::engine {

/* Process-wide shared ONNX Runtime environment. ORT documents Env as
 * shareable across sessions and threads; one env per process is the
 * recommended usage.
 *
 * Lifetime: function-local static, destroyed at process exit. Every
 * Ort::Session created on it must be destroyed BEFORE that. This holds
 * for normal fx usage (pipelines die with the OBS filter / at test
 * end) and for any static whose construction completed after the first
 * sharedEnv() call (reverse destruction order). A leaked OrtModel that
 * is destroyed after main() would touch a dead env — don't do that. */
Ort::Env &sharedEnv();

} // namespace fx::engine

namespace fx {

enum class OrtExecutionPolicy { AllowCpuFallback, RequireCuda };
enum class OrtBackend { Cpu, Cuda, Failed };
enum class OrtRunFailureAction { Rethrow, RetryCpu };

/* Atomic status for one inference worker and cross-thread UI reads. This
 * does not make OrtModel::run safe for concurrent inference calls. */
class OrtBackendState {
public:
	explicit OrtBackendState(OrtExecutionPolicy policy) noexcept;

	OrtBackend backend() const noexcept;
	void setBackend(OrtBackend backend) noexcept;
	OrtRunFailureAction onRunFailure(OrtErrorCode code) noexcept;
	void markCpuFallbackReady() noexcept;
	void markCpuFallbackFailed() noexcept;

private:
	OrtExecutionPolicy policy_;
	std::atomic<OrtBackend> backend_{OrtBackend::Cpu};
};

/* Wrapper around an ONNX Runtime session with dynamic IO discovery.
 * Supports multi-input multi-output models (RVM: 6 in / 6 out);
 * single-IO models use the convenience accessors.
 * All sessions are created on fx::engine::sharedEnv(). */
class OrtModel {
public:
	struct TensorDesc {
		std::string name;
		std::vector<int64_t> shape;
	};

	/* tryCuda requests the CUDA provider when its classic append symbol is
	 * available. The default policy preserves provider-failure CPU fallback;
	 * RequireCuda rejects unavailable or failed CUDA execution instead. */
	explicit OrtModel(const std::string &modelPath, int intraOpThreads = 2, bool tryCuda = false,
			  OrtExecutionPolicy policy = OrtExecutionPolicy::AllowCpuFallback);

	size_t inputCount() const { return inputs_.size(); }
	size_t outputCount() const { return outputs_.size(); }
	const TensorDesc &input(size_t i) const { return inputs_.at(i); }
	const TensorDesc &output(size_t i) const { return outputs_.at(i); }

	/* Backward-compatible single-IO accessors. */
	const TensorDesc &input() const { return inputs_.at(0); }
	const TensorDesc &output() const { return outputs_.at(0); }

	OrtBackend backend() const noexcept { return backendState_.backend(); }
	/* True when this session runs on the CUDA execution provider. */
	bool usesCuda() const noexcept { return backend() == OrtBackend::Cuda; }

	/* Single-IO run (existing behavior). */
	std::vector<float> run(const std::vector<float> &inputData);

	/* Multi-IO run: one flat float tensor per declared input.
	 * Returns one flat float tensor per declared output (in order). */
	std::vector<std::vector<float>>
	run(const std::vector<std::vector<float>> &inputData);

	/* Multi-IO run with per-input shape override (for models with
	 * dynamic dims like RVM's src input). overrides[i] empty = keep
	 * declared shape. */
	std::vector<std::vector<float>>
	runWithShapes(const std::vector<std::vector<float>> &inputData,
		      const std::vector<std::vector<int64_t>> &overrides);

private:
	std::vector<std::vector<float>>
	runImpl(const std::vector<std::vector<float>> &inputData,
		const std::vector<std::vector<int64_t>> *overrides);
	/* Runs the session via IoBinding (outputs bound to CPU memory, so
	 * device tensors are copied back by ORT). Policy decides whether an
	 * eligible CUDA provider failure retries once on a fresh CPU session. */
	std::vector<Ort::Value> tryRun(std::vector<Ort::Value> &inTensors,
				       std::vector<const char *> &inNames,
				       std::vector<const char *> &outNames,
				       Ort::MemoryInfo &cpuMem);
	OrtBackendState backendState_;
	Ort::Session session_;
	std::string modelPath_;
	int intraOpThreads_ = 2;
	std::vector<TensorDesc> inputs_;
	std::vector<TensorDesc> outputs_;
};

} // namespace fx
