#pragma once

namespace fx {

/* Execution-provider probe (classic CUDA API path).
 * The bundled CPU ORT build has no CUDA compiled in; when the full GPU
 * build is installed next to the plugin (same SONAME), the classic
 * OrtSessionOptionsAppendExecutionProvider_CUDA symbol is resolvable at
 * runtime. CPU is always the guaranteed fallback. */
class EpProbe {
public:
	/* True when the classic CUDA append symbol resolves (i.e. we're
	 * running on the GPU ORT build) — cheap, cached. */
	static bool cudaAvailable();

	/* Human-readable backend name for status display. */
	static const char *backendName(bool cuda);
};

} // namespace fx
