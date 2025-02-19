# Dynamically set MKL paths based on the active Conda environment
export MKLROOT=$(python -c "import sys; print(sys.prefix)")
export LD_LIBRARY_PATH=$MKLROOT/lib:$LD_LIBRARY_PATH
export CMAKE_PREFIX_PATH="$MKLROOT;$CONDA_PREFIX;/home/yingyih/workspace/libtorch"

# Confirm environment variables (for debugging)
echo "MKLROOT: $MKLROOT"
echo "CMAKE_PREFIX_PATH: $CMAKE_PREFIX_PATH"

# Run CMake and compile
cmake .. -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH"
make -j$(nproc)  # Use all available CPU cores for faster compilation