# Examples on Fwd/Bwd Functions
Uses mha_fwd & mha_bwd to verify functionality.
Compiles the test program and links with flash_attn_2_cuda.so.

Compile it as a shared library + test:
```
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```