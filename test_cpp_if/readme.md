# Build Everything
You might want to compile with your local Python & NVCC & Torch suppport.
```
cd test_cpp_if
mkdir build && cd build

# update the path to be your local libtorch path
# cmake .. -DCMAKE_PREFIX_PATH="/home/yingyih/workspace/libtorch" (stale)
export CMAKE_PREFIX_PATH="$CONDA_PREFIX;/home/yingyih/workspace/libtorch"
cmake .. -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH"
make -j
```
[**Recommended**] You might want to build in the container environment from the root directory (provided one for py3.9, cuda12.4, gcc11).
```
cd ..
docker build --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) -t flash_container .
docker run --gpus all -it --rm --shm-size=8g \
    -v /home/yingyih/workspace:/workspace \
    flash_container

cd flash_attention_lib/test_cpp_if
rm -rf build
mkdir build && cd build
cmake ..
make -j
```

TODO: building nvcc OOM issues

# Run the Test with Shared Library
```
export LD_LIBRARY_PATH=test_cpp_if/build/lib:$LD_LIBRARY_PATH
./tests/test_flash_api
```