# Flexflow-flash-attention

This directory contains some files to replace source file under 3rd_party/flash-attention/csrc/flash_attn/src, for lower-cost and custom compilation of locally-built flash-attn wheel.

## Files
flexflow_flash_api.h (flash-attn fwd/bwd interface exposed to flexflow)

static_switch.h (--> flash-attention/csrc/flash_attn/src/static_switch.h)

flash_api.cpp (--> flash-attention/csrc/flash_attn/flash_api.cpp)



