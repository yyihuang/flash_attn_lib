cmake_minimum_required(VERSION 3.18)
project(flash_attn_2_cuda LANGUAGES CXX CUDA)

# Parallel compilation settings (at top before targets)
set(CMAKE_JOB_POOLS compile_pool=16 link_pool=4)
set(CUDA_PARALLEL_COMPILE_JOBS 16)
set(CUDA_PARALLEL_LINK_JOBS 4)
set(CUDA_SEPARABLE_COMPILATION ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set_source_files_properties(${CUDA_SOURCES} PROPERTIES LANGUAGE CUDA)

# ------------------------------------------------------------
# 1. Find Python (Interpreter + Development headers)
# ------------------------------------------------------------
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(Torch REQUIRED)
find_package(CUDA REQUIRED)

# ------------------------------------------------------------
# 2. Define the .cu files
# (we only compile the needed .cu files)
# ------------------------------------------------------------
set(CUDA_SOURCES
    ${CMAKE_SOURCE_DIR}/../3rd_party/flash-attention/csrc/flash_attn/src/flash_fwd_hdim32_fp16_sm80.cu
    ${CMAKE_SOURCE_DIR}/../3rd_party/flash-attention/csrc/flash_attn/src/flash_fwd_split_hdim32_fp16_sm80.cu
    ${CMAKE_SOURCE_DIR}/../3rd_party/flash-attention/csrc/flash_attn/src/flash_bwd_hdim32_fp16_sm80.cu
    ${CMAKE_SOURCE_DIR}/../3rd_party/flash-attention/csrc/flash_attn/src/flash_fwd_hdim32_fp16_causal_sm80.cu
    ${CMAKE_SOURCE_DIR}/../3rd_party/flash-attention/csrc/flash_attn/src/flash_fwd_split_hdim32_fp16_causal_sm80.cu
    ${CMAKE_SOURCE_DIR}/../3rd_party/flash-attention/csrc/flash_attn/src/flash_bwd_hdim32_fp16_causal_sm80.cu
)

# set(CPP_SOURCE ${CMAKE_SOURCE_DIR}/../3rd_party/flash-attention/csrc/flash_attn/flash_api.cpp)
set(CPP_SOURCE ${CMAKE_SOURCE_DIR}/../3rd_party/flexflow-flash-attention/flash_api.cpp)

# ------------------------------------------------------------
# 3. Create the shared library target
# ------------------------------------------------------------
add_library(flash_attn_2_cuda SHARED ${CPP_SOURCE} ${CUDA_SOURCES})

# ------------------------------------------------------------
# 4. Include directories
# ------------------------------------------------------------
target_include_directories(flash_attn_2_cuda PRIVATE
    ${CMAKE_SOURCE_DIR}/../3rd_party/flash-attention/csrc/flash_attn
    ${CMAKE_SOURCE_DIR}/../3rd_party/flash-attention/csrc/flash_attn/src
    ${CMAKE_SOURCE_DIR}/../3rd_party/flash-attention/csrc/cutlass/include
    ${TORCH_INCLUDE_DIRS}
    ${Python_INCLUDE_DIRS}
)

# ------------------------------------------------------------
# 5. Compiler flags
# ------------------------------------------------------------
target_compile_options(flash_attn_2_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -std=c++17>
    $<$<COMPILE_LANGUAGE:CUDA>:
    -O3
    -std=c++17
    -U__CUDA_NO_HALF_OPERATORS__
    -U__CUDA_NO_HALF_CONVERSIONS__
    -U__CUDA_NO_HALF2_OPERATORS__
    -U__CUDA_NO_BFLOAT16_CONVERSIONS__
    --expt-relaxed-constexpr
    --expt-extended-lambda
    --use_fast_math
    --ptxas-options=-v
    --ptxas-options=-O1
    -maxrregcount=128
    -Xcompiler=-Wno-unknown-pragmas
    -keep
    -keep-device-functions
    -Xcompiler -fno-var-tracking  # Reduce GCC memory usage
    # -Xcompiler=-save-temps 
    >
)

# ------------------------------------------------------------
# 6. Linker settings
# ------------------------------------------------------------
target_link_libraries(flash_attn_2_cuda PRIVATE
    ${TORCH_LIBRARIES}
    CUDA::cudart
    ${Python_LIBRARIES}
)

set_property(TARGET flash_attn_2_cuda PROPERTY JOB_POOL_COMPILE compile_pool)
set_property(TARGET flash_attn_2_cuda PROPERTY JOB_POOL_LINK link_pool)
set_target_properties(flash_attn_2_cuda PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ------------------------------------------------------------
# 7. Install step
# ------------------------------------------------------------
install(TARGETS flash_attn_2_cuda DESTINATION ${CMAKE_BINARY_DIR}/lib)