cmake_minimum_required(VERSION 3.18)
project(test_flash_attn LANGUAGES CXX CUDA)

# Enable verbose output
set(CMAKE_VERBOSE_MAKEFILE ON)

# Set C++ standard to C++17 (needed for PyTorch and CUTLASS)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --------------------------------------------------------
# 1. Detect GPU + NVCC to set 'compute_capability'
#    (Based on your snippet, but done at the top-level so
#     all subdirectories can use the same setting)
# --------------------------------------------------------
find_program(GPU_dev_info nvidia-smi)
if(NOT GPU_dev_info)
    message(WARNING "nvidia-smi not found. Assuming compute_capability=80. \
                     If you have an older GPU, please set it manually.")
    set(GPU_dev_version "")
else()
    execute_process(COMMAND ${GPU_dev_info} OUTPUT_VARIABLE GPU_dev_version)
endif()

string(REGEX MATCH "H100" GPU_H100 ${GPU_dev_version})

find_program(NVCC nvcc)
if(NOT NVCC)
    message(FATAL_ERROR "NVCC not found. Please make sure CUDA Toolkit is installed.")
endif()

execute_process(COMMAND ${NVCC} --version OUTPUT_VARIABLE NVCC_VERSION_OUTPUT)
string(REGEX MATCH "([0-9]+\\.[0-9]+)" NVCC_VERSION ${NVCC_VERSION_OUTPUT})
if(NOT NVCC_VERSION)
    message(FATAL_ERROR "Failed to determine NVCC version from nvcc --version.")
endif()

# --------------------------------------------------------
# 2. Decide which compute capability to use
# --------------------------------------------------------
# Default to '80' (Ampere) if not recognized as H100
set(compute_capability "80")
if(GPU_H100 AND (NVCC_VERSION GREATER_EQUAL 12))
    set(compute_capability "90a")
    add_definitions(-DHOPPER)
endif()

message(STATUS "Detected compute_capability=${compute_capability}")

# Make it globally visible to subdirectories
# So that 'libs/CMakeLists.txt' and 'tests/CMakeLists.txt' see it
set(CMAKE_CUDA_ARCHITECTURES "${compute_capability}" CACHE INTERNAL "")

# --------------------------------------------------------
# 3. Find PyTorch, Python, etc. if needed at the top-level
#    (Adjust or remove if you do this in subdirs)
# --------------------------------------------------------
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(Torch REQUIRED)

message(STATUS "Python executable: ${Python_EXECUTABLE}")
message(STATUS "Torch libraries: ${TORCH_LIBRARIES}")

# --------------------------------------------------------
# 4. Add the subdirectories
# --------------------------------------------------------
add_subdirectory(libs)
add_subdirectory(tests)