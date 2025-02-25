# Flash Attention Library Minimal Demo
## Link with Pre-compiled Wheel

I worked on catalyst-0-15 with this docker image.

We use this flash-attn release:

https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.0.post2/flash_attn-2.7.0.post2+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl instead of local building for now. 

For linking with extracted interface, we must ensure consistent dependencies between flash-attn wheel and local environment. Here we provide a comptiable building environment in docker.

Build the docker with image at at root. You could examine all the dependencies in it.
```
docker build -t flash-attn-build .
```

Run it in docker. (update local workspace to map as needed)
```
docker run --gpus all --rm -it -v /home/yingyih/workspace:/workspace --user $(id -u):$(id -g) flash-attn-build
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
./test_flash_attn
```

## Usgae: Exposed Interface
Referenced Practice:  https://github.com/zhihu/ZhiLight/tree/main/3rd/flash_decoding

- Flash-attn Lib Interface

We put the required interfaces from flash-attn wheel under `3rd_party/flash-attn/flash_api.h`. (Ignore `3rd_party/flash-attention`. This is for local compilation of flash-attn).

- Integrated Function Interface

Put the FlexLLM-compatible interface declaraiton under `csrc/flash_attn.h` and implementation under `csrc/flash_attn.cpp` (where we invoked the flash-attn lib API).





------------------------- stale----------------------------

This repo tries to encapsulate the flash attention official implementation (under `3rd_party` dir) as a library module. This library module could be kept updated as an individual git submodule.

## Flash Attention Submodules
This official implementation (https://github.com/Dao-AILab/flash-attention.git) is maintained by git submodules in this library. Here we will review how we perform updates and syncs with this submodule.

## Cloning this Repo with Flash-attn Submodules
To clone this repo:
```
git clone --recurse-submodules git@github.com:yyihuang/flash_attn_lib.git
```

In case you forgot to clone with `--resurse-submodules`:
```
git submodule update --init --recursive
```

## Use a Commit ID with the Flash-attn Submodules
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

# Flash-attn Dependency
We import cutlass as dependecies for building this library. It is maintained as a submodule under  `3rd_party` from `git@github.com:NVIDIA/cutlass.git`.




