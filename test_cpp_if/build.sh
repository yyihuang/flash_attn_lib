#!/bin/bash
BUILD_DIR="/home/yingyih/workspace/flash_attention_lib/test_cpp_if/build"

# Clean previous build artifacts
rm -rf "$BUILD_DIR" && mkdir -p "$BUILD_DIR" && cd "$BUILD_DIR"

# Configure with aggressive caching
cmake .. \
  -GNinja \
  -DCMAKE_PREFIX_PATH="$CONDA_PREFIX;/home/yingyih/workspace/libtorch" \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CUDA_FLAGS="-maxrregcount=96 --ptxas-options=-O1" \
  -DCMAKE_JOB_POOLS="compile_pool=16;link_pool=2"

# Build with resource monitoring
ninja -j16 -k0 2>&1 | tee build.log &
PID=$!

# Monitor resources while building
while ps -p $PID > /dev/null; do
  echo "=== $(date) ==="
  free -h
  nvidia-smi --query-gpu=utilization.gpu --format=csv
  top -bn1 -p $PID
  sleep 30
done