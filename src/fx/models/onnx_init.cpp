#include "fx/models/onnx_init.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace fx {
namespace {

/* Bounds-checked protobuf wire-format cursor.
 * Only varint (0) and length-delimited (2) are interpreted;
 * 32/64-bit fixed (5/1) are skipped by size, groups are rejected. */
struct Cursor {
	const uint8_t *p;
	const uint8_t *end;

	uint64_t varint()
	{
		uint64_t v = 0;
		for (int shift = 0; shift < 64; shift += 7) {
			if (p >= end)
				throw std::runtime_error(
					"fx: onnx truncated varint");
			uint8_t b = *p++;
			v |= (uint64_t)(b & 0x7f) << shift;
			if (!(b & 0x80))
				return v;
		}
		throw std::runtime_error("fx: onnx varint too long");
	}

	/* Reads a field key: returns field number, sets wt to wire type. */
	uint32_t key(int &wt)
	{
		uint64_t k = varint();
		wt = (int)(k & 7);
		return (uint32_t)(k >> 3);
	}

	Cursor bytes(uint64_t len)
	{
		if (len > (uint64_t)(end - p))
			throw std::runtime_error(
				"fx: onnx field overruns buffer");
		Cursor c{p, p + len};
		p += len;
		return c;
	}

	void skip(int wt)
	{
		switch (wt) {
		case 0:
			(void)varint();
			break;
		case 1:
			(void)bytes(8);
			break;
		case 2:
			(void)bytes(varint());
			break;
		case 5:
			(void)bytes(4);
			break;
		default:
			throw std::runtime_error(
				"fx: onnx unsupported wire type");
		}
	}
};

} // namespace

std::vector<float> onnxLastInitializerFloats(const std::string &modelPath,
					     int64_t expectedCount)
{
	std::ifstream f(modelPath, std::ios::binary | std::ios::ate);
	if (!f)
		throw std::runtime_error("fx: cannot open " + modelPath);
	std::streamoff size = f.tellg();
	if (size <= 0)
		throw std::runtime_error("fx: empty onnx file " + modelPath);
	std::vector<uint8_t> buf((size_t)size);
	f.seekg(0);
	if (!f.read((char *)buf.data(), size))
		throw std::runtime_error("fx: cannot read " + modelPath);

	/* ModelProto: graph is field 7 (length-delimited). */
	Cursor root{buf.data(), buf.data() + buf.size()};
	Cursor graph{nullptr, nullptr};
	while (root.p < root.end) {
		int wt;
		uint32_t field = root.key(wt);
		if (field == 7 && wt == 2)
			graph = root.bytes(root.varint());
		else
			root.skip(wt);
	}
	if (!graph.p)
		throw std::runtime_error("fx: onnx has no graph");

	/* GraphProto: initializer is field 5 (repeated); keep the LAST. */
	Cursor last{nullptr, nullptr};
	while (graph.p < graph.end) {
		int wt;
		uint32_t field = graph.key(wt);
		if (field == 5 && wt == 2)
			last = graph.bytes(graph.varint());
		else
			graph.skip(wt);
	}
	if (!last.p)
		throw std::runtime_error("fx: onnx graph has no initializer");

	/* TensorProto: dims=1 (packed or not), data_type=2,
	 * float_data=4 (packed fixed32), raw_data=9. */
	int64_t dimsProduct = 1;
	bool sawDims = false;
	int64_t dataType = 0;
	Cursor raw{nullptr, nullptr};
	Cursor fdata{nullptr, nullptr};
	while (last.p < last.end) {
		int wt;
		uint32_t field = last.key(wt);
		if (field == 1 && wt == 0) {
			dimsProduct *= (int64_t)last.varint();
			sawDims = true;
		} else if (field == 1 && wt == 2) {
			Cursor packed = last.bytes(last.varint());
			while (packed.p < packed.end) {
				dimsProduct *= (int64_t)packed.varint();
				sawDims = true;
			}
		} else if (field == 2 && wt == 0) {
			dataType = (int64_t)last.varint();
		} else if (field == 4 && wt == 2) {
			fdata = last.bytes(last.varint());
		} else if (field == 9 && wt == 2) {
			raw = last.bytes(last.varint());
		} else {
			last.skip(wt);
		}
	}

	if (!sawDims || dimsProduct != expectedCount)
		throw std::runtime_error(
			"fx: onnx initializer dims mismatch");
	if (dataType != 1) /* TensorProto::FLOAT */
		throw std::runtime_error(
			"fx: onnx initializer is not float32");
	/* raw_data wins; packed float_data otherwise (inswapper's emap). */
	Cursor blob = (raw.p && raw.end != raw.p) ? raw : fdata;
	if (!blob.p || blob.end == blob.p)
		throw std::runtime_error(
			"fx: onnx initializer has no float data");
	if ((int64_t)(blob.end - blob.p) != expectedCount * 4)
		throw std::runtime_error(
			"fx: onnx initializer data size mismatch");

	std::vector<float> out((size_t)expectedCount);
	std::memcpy(out.data(), blob.p, (size_t)expectedCount * 4);
	return out;
}

} // namespace fx
