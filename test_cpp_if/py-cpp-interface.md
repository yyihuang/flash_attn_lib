# Notes on Python-C++ Interface
In `3rd_party/flash-attention/flash_attn/flash_attn_interface.py`, the actual flash attention operator is defined as `@_torch_custom_op_wrapper("flash_attn::_flash_attn_forward", mutates_args=(), device_types="cuda")` and  `@_torch_custom_op_wrapper("flash_attn::_flash_attn_backward", mutates_args=("dq", "dk", "dv"), device_types="cuda")`.

## From Python to C++ Interface
It depends on the imported `flash_attn_2_cuda`, where we could get all the required source code in `3rd_party/flash-attention/setup.py`.
```
ext_modules.append(
        CUDAExtension(
            name="flash_attn_2_cuda",
            sources=[
                "csrc/flash_attn/flash_api.cpp",
                "csrc/flash_attn/src/flash_fwd_hdim32_fp16_sm80.cu",
                ...
                "csrc/flash_attn/src/flash_fwd_split_hdim256_bf16_causal_sm80.cu",
            ],
            extra_compile_args={
                ...
            },
            include_dirs=[
                ...
            ],
        )
    )
```

We can get the mapping of Python and C++ interface in `3rd_party/flash-attention/csrc/flash_attn/flash_api.cpp`, by:
```
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashAttention";
    m.def("fwd", &FLASH_NAMESPACE::mha_fwd, "Forward pass");
    m.def("varlen_fwd", &FLASH_NAMESPACE::mha_varlen_fwd, "Forward pass (variable length)");
    m.def("bwd", &FLASH_NAMESPACE::mha_bwd, "Backward pass");
    m.def("varlen_bwd", &FLASH_NAMESPACE::mha_varlen_bwd, "Backward pass (variable length)");
    m.def("fwd_kvcache", &FLASH_NAMESPACE::mha_fwd_kvcache, "Forward pass, with KV-cache");
}
```

## What to Include in Our Library
Here we will try to provide a library similar to `flash_attn_2_cuda` and give examples on how to use it. Currenlty we will focus on the "fwd" and "bwd" interface.
```
m.def("fwd", &FLASH_NAMESPACE::mha_fwd, "Forward pass");
m.def("bwd", &FLASH_NAMESPACE::mha_bwd, "Backward pass");
```



