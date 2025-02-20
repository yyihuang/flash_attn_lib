cmake_minimum_required(VERSION 3.18)
project(test_flash_attn LANGUAGES CXX CUDA)

# Enable verbose output
set(CMAKE_VERBOSE_MAKEFILE ON)

# Set C++ standard to C++17 (needed for PyTorch and CUTLASS)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Ensure CUDA Toolkit is found (avoids CMP0146 warning)
find_package(CUDAToolkit REQUIRED)

# Find Python
find_package(Python COMPONENTS Interpreter Development REQUIRED)
include_directories(${Python_INCLUDE_DIRS})

# Find MKL dynamically using the Conda environment
# find_package(MKL REQUIRED PATHS $ENV{CONDA_PREFIX} NO_DEFAULT_PATH)

# Find NVTX3 (for better compatibility)
# find_package(nvtx3 QUIET)

# Define CUTLASS path and include directories
set(CUTLASS_PATH ${CMAKE_SOURCE_DIR}/../3rd_party/cutlass)
include_directories(${CUTLASS_PATH}/include)

# Find PyTorch (Make sure it is detected properly)
if (NOT Torch_DIR)
    set(Torch_DIR ${CMAKE_PREFIX_PATH})
endif()
find_package(Torch REQUIRED)

# Print debug information
message(STATUS "Python executable: ${Python_EXECUTABLE}")
message(STATUS "Python include dir: ${Python_INCLUDE_DIRS}")
message(STATUS "Python library: ${Python_LIBRARIES}")
message(STATUS "Torch directory: ${Torch_DIR}")

# Avoid runtime path issues
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Add subdirectories for library and tests
add_subdirectory(libs)
add_subdirectory(tests)