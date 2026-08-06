#include "fx/engine/ep_probe.h"

#include <dlfcn.h>

namespace fx {

bool EpProbe::cudaAvailable()
{
	/* Function-local static: resolved once, thread-safe. */
	static bool ok =
		dlsym(RTLD_DEFAULT,
		      "OrtSessionOptionsAppendExecutionProvider_CUDA") != nullptr;
	return ok;
}

const char *EpProbe::backendName(bool cuda)
{
	return cuda ? "CUDA" : "CPU";
}

} // namespace fx
