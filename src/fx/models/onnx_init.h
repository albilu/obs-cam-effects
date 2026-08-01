#pragma once

#include <string>
#include <vector>

namespace fx {

/* Extracts the LAST initializer's float32 data from an .onnx file
 * (used for inswapper's embedded `emap` projection matrix, 512x512).
 * Minimal protobuf wire-format walk: no protobuf library required.
 * Throws std::runtime_error on malformed input or missing initializer. */
std::vector<float> onnxLastInitializerFloats(const std::string &modelPath,
					     int64_t expectedCount);
}
