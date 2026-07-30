#include "fx/engine/ep_probe.h"

#include "fx/engine/ort_backend.h"

#include <mutex>
#include <sys/stat.h>

namespace fx {

namespace {

bool fileExists(const std::string &p)
{
	struct stat st;
	return stat(p.c_str(), &st) == 0;
}

} // namespace

bool EpProbe::cudaAvailable(const std::string &providersDir)
{
	static std::once_flag once;
	static bool ok = false;
	const std::string so =
		providersDir + "/libonnxruntime_providers_cuda.so";
	if (!fileExists(so))
		return false;
	std::call_once(once, [&] {
		try {
			engine::sharedEnv().RegisterExecutionProviderLibrary(
				"CUDA", so.c_str());
			for (const auto &d :
			     engine::sharedEnv().GetEpDevices()) {
				const char *name = d.EpName();
				if (name &&
				    std::string(name).find("CUDA") !=
					    std::string::npos) {
					ok = true;
					break;
				}
			}
		} catch (...) {
			ok = false;
		}
	});
	return ok;
}

const char *EpProbe::backendName(bool cuda)
{
	return cuda ? "CUDA" : "CPU";
}

} // namespace fx
