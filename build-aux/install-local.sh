#!/usr/bin/env bash
# Installs the plugin into the per-user OBS plugin directory for local testing.
# Primary path: cmake --install (exercises the CMake install rules, which
# include the GPU-build preservation guard). Fallback: manual copy, used only
# when cmake --install is unavailable or fails.
set -euo pipefail

BUILD_DIR="${1:-build_x86_64}"
DEST="$HOME/.config/obs-studio/plugins"

cmake --install "$BUILD_DIR" --prefix "$DEST" 2>/dev/null || {
	echo "cmake --install failed; falling back to manual copy" >&2
	PLUGIN_DIR="$DEST/obs-cam-effects"
	mkdir -p "$PLUGIN_DIR/bin/64bit" "$PLUGIN_DIR/data/effects" \
		 "$PLUGIN_DIR/data/locale" "$PLUGIN_DIR/data/models"
	cp "$BUILD_DIR/obs-cam-effects.so" "$PLUGIN_DIR/bin/64bit/"
	# GPU build preservation (mirrors the CMake install-rule guard and
	# the original install-local.sh). The GPU build's libonnxruntime.so
	# exports OrtSessionOptionsAppendExecutionProvider_CUDA; the bundled
	# CPU build does not. Keep the GPU lib instead of clobbering it.
	# (grep WITHOUT -q: it must drain nm's output, else pipefail + SIGPIPE
	# makes the probe misfire.)
	if ! nm -D "$PLUGIN_DIR/bin/64bit/libonnxruntime.so.1.28.0" 2>/dev/null \
			| grep OrtSessionOptionsAppendExecutionProvider_CUDA > /dev/null; then
		cp "$BUILD_DIR/onnxruntime/lib/libonnxruntime.so.1.28.0" \
			"$PLUGIN_DIR/bin/64bit/"
	fi
	ln -sf libonnxruntime.so.1.28.0 \
		"$PLUGIN_DIR/bin/64bit/libonnxruntime.so.1"
	cp data/effects/*.effect "$PLUGIN_DIR/data/effects/"
	cp data/locale/en-US.ini "$PLUGIN_DIR/data/locale/"
	cp data/models/manifest.json "$PLUGIN_DIR/data/models/"
	cp "$BUILD_DIR/models/"*.onnx "$BUILD_DIR/models/"*.license \
		"$PLUGIN_DIR/data/models/" 2>/dev/null || true
}

# cmake --install also lays down the template's system-layout files
# (lib/x86_64-linux-gnu/obs-plugins/ + share/obs/obs-plugins/) alongside
# the per-user obs-cam-effects/ dir. These are harmless but clutter the
# dev plugin dir; the per-user layout under obs-cam-effects/ is all OBS
# loads from ~/.config, so drop the system-layout copies.
rm -rf "$DEST/lib" "$DEST/share"

echo "Installed to $DEST/obs-cam-effects"
