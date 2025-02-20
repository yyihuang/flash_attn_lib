cmake_minimum_required(VERSION 3.18)
project(flash_attn_2_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ------------------------------------------------------------
# 1. Find Python (Interpreter + Development headers)
#    This ensures Python_INCLUDE_DIRS / Python_LIBRARIES exist.
# ------------------------------------------------------------
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# If you haven't already found PyTorch and CUDA:
find_package(Torch REQUIRED)
find_package(CUDA REQUIRED)

# ------------------------------------------------------------
# 2. Define the .cu files you want to compile
#    If you want them all, you can either list them individually
#    or use file(GLOB_RECURSE ...).
# ------------------------------------------------------------
file(GLOB_RECURSE CUDA_SOURCES
    ${CMAKE_SOURCE_DIR}/../3rd_party/flash-attention/csrc/flash_attn/src/*.cu
)

set(CPP_SOURCE ${CMAKE_SOURCE_DIR}/../3rd_party/flash-attention/csrc/flash_attn/flash_api.cpp)

# Enable CUDA separable compilation to allow parallel builds
set(CUDA_SEPARABLE_COMPILATION ON)

# Allow up to 16 parallel CUDA compile jobs (adjust based on RAM)
set(CMAKE_CUDA_DEVICE_LINK_LIMIT 16)

# ------------------------------------------------------------
# 3. Create the shared library target
# ------------------------------------------------------------
add_library(flash_attn_2_cuda SHARED ${CPP_SOURCE} ${CUDA_SOURCES})

# ------------------------------------------------------------
# 4. Include directories for:
#    - flash-attn
#    - cutlass
#    - PyTorch
#    - Python dev headers
# ------------------------------------------------------------
target_include_directories(flash_attn_2_cuda PRIVATE
    ${CMAKE_SOURCE_DIR}/../3rd_party/flash-attention/csrc/flash_attn
    ${CMAKE_SOURCE_DIR}/../3rd_party/flash-attention/csrc/flash_attn/src
    ${CMAKE_SOURCE_DIR}/../3rd_party/flash-attention/csrc/cutlass/include
    ${TORCH_INCLUDE_DIRS}
    ${Python_INCLUDE_DIRS}   # <-- CRITICAL for <Python.h>
)

# ------------------------------------------------------------
# 5. Compiler flags for both C++ and CUDA
# ------------------------------------------------------------
target_compile_options(flash_attn_2_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -std=c++17>
    $<$<COMPILE_LANGUAGE:CUDA>:
    -O3
    -std=c++17
    -arch=sm_80
    -U__CUDA_NO_HALF_OPERATORS__
    -U__CUDA_NO_HALF_CONVERSIONS__
    -U__CUDA_NO_HALF2_OPERATORS__
    -U__CUDA_NO_BFLOAT16_CONVERSIONS__
    --expt-relaxed-constexpr
    --expt-extended-lambda
    --use_fast_math
    -v # NVCC compiler driver verbose
    --ptxas-options=-v
    # --ptxas-options=-O2 
    # --ptxas-options=-warn-lmem-usage  # Check shared memory usage
    # -maxrregcount=128          # Limit register usage
    # -Xcompiler -fPIC           # Position-independent code
    # -Xcompiler -v  # Host compiler (gcc/clang) verbose
    # -Xcompiler -ftime-report  # GCC timing report
    >
)

# ------------------------------------------------------------
# 6. Link to the Torch libraries, CUDA runtime, and Python
# ------------------------------------------------------------
target_link_libraries(flash_attn_2_cuda PRIVATE
    ${TORCH_LIBRARIES}
    CUDA::cudart
    ${Python_LIBRARIES}   # <-- If needed to link Python symbols
)

set_target_properties(flash_attn_2_cuda PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
# set_property(TARGET flash_attn_2_cuda PROPERTY JOB_POOL_COMPILE compile_pool)
# set_property(TARGET flash_attn_2_cuda PROPERTY JOB_POOL_LINK link_pool)
# set(CMAKE_JOB_POOL_COMPILE compile_pool JOB_POOLS compile_pool=1)
# set(CMAKE_JOB_POOL_LINK link_pool JOB_POOLS link_pool=1)

# ------------------------------------------------------------
# 7. Install step (optional)
# ------------------------------------------------------------
install(TARGETS flash_attn_2_cuda DESTINATION ${CMAKE_BINARY_DIR}/lib)