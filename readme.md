# Flash Attention Library Minimal Demo

The demo is to 
1. Demonstrate how we link the pre-compiled flash-attn wheel with your code on the provided interface (we use `mha_fwd` and `mha_bwd` for demo). The steps are under `Link with Pre-compiled Wheel` section.

2. Provide the local compilation configurations in case you need to build flash-attn locally as a 3rd-party dependency from source code. To reduce compilation overhead, we recommend limiting template instantiation number (in other words, building less kernel files under `3rd_party/flash-attention/csrc/flash_attn`). We will show how we made this under `Build Locally and Link` section.

## Link with Pre-compiled Wheel

The exported flash-attn API should be declared in `3rd_party/flash_api.h`. The signature should be **consistent** with the one in source code. 

[Recommended practice of file organization]
You could try to have you own APIs declared in `csrc/flash_attn.h`, and implement them with the pre-compiled flash-attn functions by `#include "flash_api.h"` in `csrc/flash_attn.cpp`.

In this demo, we make a minimal test under `test`, for `mha_fwd` and `mha_bwd` interface in flash-attn.

[Optional] I worked on catalyst-0-15 with docker image flash-attn-build. 

We use this flash-attn release:

https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.0.post2/flash_attn-2.7.0.post2+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl 

Here we provide a building environment in docker. You can build this `flash-attn-build` docker we provide at root directory. To use another version of flash-attn wheel, you can update the url in Dockerfile (the environment should match the version you download).

Build the docker with image at this project root. You could take a look at all the dependencies in it.
```
docker build -t test .
```

Run it in docker. (replace `/home/yingyih/workspace` with your local workspace to map as needed)
```
docker run --gpus all --rm -it -v /home/yingyih/workspace:/workspace --user $(id -u):$(id -g) test
```

Try to build in docker at root dir.
```
mkdir -p build && cd build
cmake ..
make -j
```

Run the test:
```
export LD_LIBRARY_PATH=/usr/local/miniconda/envs/py310/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/miniconda/envs/py310/lib/python3.10/site-packages:$LD_LIBRARY_PATH
./test_mha_fwd_bwd
```
Find the flash-attn localtion:
```
find /usr/local/miniconda/envs/py310 -name "flash_attn*.so"
# /usr/local/miniconda/envs/py310/lib/python3.10/site-packages/flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so

find /usr/local -name "flash_attn*.so"
# /usr/local/lib/python3.10/site-packages/flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so
```
Referenced Practice:  https://github.com/zhihu/ZhiLight/tree/main/3rd/flash_decoding

Here is the file structure.

- Flash-attn Lib Interface

We put the required functions' declarations from flash-attn wheel under `3rd_party/flash-attn/flash_api.h`. (Ignore `3rd_party/flash-attention`. This is for local compilation of flash-attn). You can write additional declarations here for your source code.

- Integrated Function Interface

We recommend to put the your_src-compatible (that is, Flexllm-compatible code) interface declaraiton under `csrc/flash_attn.h` and implementation under `csrc/flash_attn.cpp` (where we invoked the flash-attn lib API).



## Build Locally and Link
You can reuse the docker from last section for this test. The working directory is `test_cpp_if`, where we have `libs` to build source code from flash-attn and `tests` to run tests of flash-attn funtions. We also have some files under `3rd_party/flexflow-flash-attention` (refer to the readme under this directory).

**To limit the number of template instantiation** (fp16 && hdim=128) to achieve fast compilation, we should:

* Sepecify the .cu files in Cmake files 

Example: in `test_cpp_if/libs/CMakeLists.txt`, we have
```
set(CUDA_SOURCES
    ${CMAKE_SOURCE_DIR}/../3rd_party/flash-attention/csrc/flash_attn/src/flash_fwd_hdim128_fp16_sm80.cu
    ${CMAKE_SOURCE_DIR}/../3rd_party/flash-attention/csrc/flash_attn/src/flash_fwd_split_hdim128_fp16_sm80.cu
    ${CMAKE_SOURCE_DIR}/../3rd_party/flash-attention/csrc/flash_attn/src/flash_bwd_hdim128_fp16_sm80.cu
    ${CMAKE_SOURCE_DIR}/../3rd_party/flash-attention/csrc/flash_attn/src/flash_fwd_hdim128_fp16_causal_sm80.cu
    ${CMAKE_SOURCE_DIR}/../3rd_party/flash-attention/csrc/flash_attn/src/flash_fwd_split_hdim128_fp16_causal_sm80.cu
    ${CMAKE_SOURCE_DIR}/../3rd_party/flash-attention/csrc/flash_attn/src/flash_bwd_hdim128_fp16_causal_sm80.cu
)
```
to compile attention only with hdim=128 and datatype=fp16.

* Follow the instruction in `3rd_party/flexflow-flash-attention/readme.md`. 

Replace `3rd_party/flash-attention/csrc/flash_attn/src/static_switch.h` with `3rd_party/flexflow-flash-attention/static_switch.h`. 

In this file, we have
```
#define FP16_SWITCH(COND, ...)               \
  [&] {                                      \
    using elem_type = cutlass::half_t;       \
    return __VA_ARGS__();                    \
  }()
```
FP16 templates get instantiated.

```
#define HEADDIM_SWITCH(HEADDIM, ...)   \
  [&] {                                \
    constexpr static int kHeadDim = 128; \
    return __VA_ARGS__();              \
  }()
```
Only hdim=128 templates get instantiated.

Put interface declarations in `3rd_party/flexflow-flash-attention/flexflow_flash_api.h`. We put `mha_fwd` and `mha_bwd` for this demo.

* Compile it under `test_cpp_if/build` and run the test

```
cmake ..
make -j
./test_flash_api
```


## Submodules in this Demo

This repo tries to encapsulate the flash attention official implementation (under `3rd_party` dir) as a library module. This library module could be kept updated as an individual git submodule.

### Flash Attention Submodules
This official implementation (https://github.com/Dao-AILab/flash-attention.git) is maintained by git submodules in this library. Here we will review how we perform updates and syncs with this submodule.

#### Cloning this Repo with Flash-attn Submodules
To clone this repo:
```
git clone --recurse-submodules git@github.com:yyihuang/flash_attn_lib.git
```

In case you forgot to clone with `--resurse-submodules`:
```
git submodule update --init --recursive
```

#### Use a Commit ID with the Flash-attn Submodules
If the official flash-attention repo gets updates, you can pull the latest changes:
```
cd 3rd_party/attention
git pull origin main  # Update the submodule
cd ../..
git add 3rd_party/attention
git commit -m "Updated attention submodule"
git push origin main
```
If you wanna fix the flash-attn implementation on a specific commit id.
- Move the Submodule to a Specific Commit

```
cd 3rd_party/flash-attention
git fetch 
git checkout <commit-id>
```
- Update the Outer Repository (this library) to Track This Commit
```
cd ../..
git add 3rd_party/flash-attention
git commit -m "Rollback flash-attention submodule to <commit-id>"

git push origin main
```
- Ensure Others Get the Correct Submodule Version
```
# If someone clones your repo later, they should run:
git submodule update --init --recursive

# If they already have the repo but need to get the correct submodule commit:
git submodule update --recursive --remote
```

### Flash-attn Dependency
We import cutlass as dependecies for building this library. It is maintained as a submodule under  `3rd_party` from `git@github.com:NVIDIA/cutlass.git`.




