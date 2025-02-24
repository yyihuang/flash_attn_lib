# Flash Attention Library Mini

We use this flash-attn release:

https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.

Build the docker at root
```
docker build -t flash-attn-build .
```

Run it in docker:
```
docker run --gpus all --rm -it -v /home/yingyih/workspace:/workspace --user $(id -u):$(id -g) flash-attn-build
```

Try to build in docker:
```
mkdir -p build && cd build
cmake ..
make -j
```





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




