#pragma once

#include <string>

namespace fx {

/* Execution-provider probe. CUDA is used when its provider library has
 * been downloaded to providersDir (via the model manager) AND
 * registration succeeds. CPU is always the guaranteed fallback. */
class EpProbe {
public:
	/* Registers the CUDA EP library on fx::engine::sharedEnv() if the
	 * file exists. Returns true if CUDA is usable for new sessions.
	 * Safe to call multiple times (registration is call_once-guarded;
	 * the result latches after the first attempt). */
	static bool cudaAvailable(const std::string &providersDir);

	/* Human-readable backend name for status display. */
	static const char *backendName(bool cuda);
};

} // namespace fx
