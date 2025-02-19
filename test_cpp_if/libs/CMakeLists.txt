cmake_minimum_required(VERSION 3.18)
project(flash_attn_2_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find required packages
# set(Torch_DIR "/home/yingyih/workspace/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)

find_package(CUDA REQUIRED)

# Define source files
file(GLOB_RECURSE CUDA_SOURCES
    ${CMAKE_SOURCE_DIR}/../3rd_party/flash-attention/csrc/flash_attn/src/*.cu
)

set(CPP_SOURCE ${CMAKE_SOURCE_DIR}/../3rd_party/flash-attention/csrc/flash_attn/flash_api.cpp)

# Create shared library
add_library(flash_attn_2_cuda SHARED ${CPP_SOURCE} ${CUDA_SOURCES})

target_include_directories(flash_attn_2_cuda PRIVATE
    ${CMAKE_SOURCE_DIR}/../3rd_party/flash-attention/csrc/flash_attn
    ${CMAKE_SOURCE_DIR}/../3rd_party/flash-attention/csrc/flash_attn/src
    ${CMAKE_SOURCE_DIR}/../3rd_party/flash-attention/csrc/cutlass/include
    ${TORCH_INCLUDE_DIRS}
)

# Compilation flags
target_compile_options(flash_attn_2_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -std=c++17>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math>
)

target_link_libraries(flash_attn_2_cuda PRIVATE ${TORCH_LIBRARIES} CUDA::cudart)

set_target_properties(flash_attn_2_cuda PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

# Install the shared library to build directory
install(TARGETS flash_attn_2_cuda DESTINATION ${CMAKE_BINARY_DIR}/lib)