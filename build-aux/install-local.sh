#!/usr/bin/env bash
# Installs the plugin into the per-user OBS plugin directory for local testing.
set -euo pipefail

BUILD_DIR="${1:-build_x86_64}"
DEST="$HOME/.config/obs-studio/plugins/obs-cam-effects"

mkdir -p "$DEST/bin/64bit" "$DEST/data/locale" "$DEST/data/models" "$DEST/data/effects"
cp "$BUILD_DIR/obs-cam-effects.so" "$DEST/bin/64bit/"
# Do not clobber a GPU ORT build previously installed by the plugin's
# CUDA download flow: the GPU build exports the classic CUDA append
# symbol, the bundled CPU build does not (same probe as EpProbe).
# (grep WITHOUT -q: it must drain nm's output, else pipefail + SIGPIPE
# makes the probe misfire.)
if ! nm -D "$DEST/bin/64bit/libonnxruntime.so.1.28.0" 2>/dev/null | grep OrtSessionOptionsAppendExecutionProvider_CUDA > /dev/null; then
  cp "$BUILD_DIR/onnxruntime/lib/libonnxruntime.so.1.28.0" "$DEST/bin/64bit/"
else
  echo "Keeping existing GPU ORT build in $DEST/bin/64bit"
fi
ln -sf libonnxruntime.so.1.28.0 "$DEST/bin/64bit/libonnxruntime.so.1"
cp data/locale/en-US.ini "$DEST/data/locale/"
cp data/models/manifest.json "$DEST/data/models/"
cp "$BUILD_DIR/models/pphumanseg_fp32.onnx" "$DEST/data/models/"
cp "$BUILD_DIR/models/pphumanseg_fp32.onnx.license" "$DEST/data/models/"
cp "$BUILD_DIR/models/selfie_segmentation.onnx" "$DEST/data/models/"
cp "$BUILD_DIR/models/selfie_segmentation.onnx.license" "$DEST/data/models/"
cp "$BUILD_DIR/models/face_detection_yunet_2023mar.onnx" "$DEST/data/models/"
cp data/effects/*.effect "$DEST/data/effects/"
echo "Installed to $DEST"
