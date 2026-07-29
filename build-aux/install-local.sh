#!/usr/bin/env bash
# Installs the plugin into the per-user OBS plugin directory for local testing.
set -euo pipefail

BUILD_DIR="${1:-build_x86_64}"
DEST="$HOME/.config/obs-studio/plugins/obs-cam-effects"

mkdir -p "$DEST/bin/64bit" "$DEST/data/locale" "$DEST/data/models" "$DEST/data/effects"
cp "$BUILD_DIR/obs-cam-effects.so" "$DEST/bin/64bit/"
cp "$BUILD_DIR/onnxruntime/lib/libonnxruntime.so.1.28.0" "$DEST/bin/64bit/"
ln -sf libonnxruntime.so.1.28.0 "$DEST/bin/64bit/libonnxruntime.so.1"
cp data/locale/en-US.ini "$DEST/data/locale/"
cp "$BUILD_DIR/models/pphumanseg_fp32.onnx" "$DEST/data/models/"
cp "$BUILD_DIR/models/pphumanseg_fp32.onnx.license" "$DEST/data/models/"
cp data/effects/*.effect "$DEST/data/effects/"
echo "Installed to $DEST"
