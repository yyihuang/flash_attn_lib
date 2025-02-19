# Build Everything
```
cd test_cpp_if
mkdir build && cd build

# update the path to be your local libtorch path
# cmake .. -DCMAKE_PREFIX_PATH="/home/yingyih/workspace/libtorch" (stale)
export CMAKE_PREFIX_PATH="$CONDA_PREFIX;/home/yingyih/workspace/libtorch"
cmake .. -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH"
make -j
```
# Run the Test with Shared Library
```
export LD_LIBRARY_PATH=test_cpp_if/build/lib:$LD_LIBRARY_PATH
./tests/test_flash_api
```