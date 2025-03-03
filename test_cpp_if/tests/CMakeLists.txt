cmake_minimum_required(VERSION 3.18)
project(test_flash_attn LANGUAGES CXX)

# Find Dependencies
find_package(Python COMPONENTS Development REQUIRED)
find_package(Torch REQUIRED)

# Create executable
add_executable(test_flash_api test.cpp)

# Link test executable with the shared library
target_link_libraries(test_flash_api PRIVATE
    flash_attn_2_cuda
    ${TORCH_LIBRARIES}
    ${Python_LIBRARIES}
)

# Include directories
target_include_directories(test_flash_api PRIVATE
    ${CMAKE_SOURCE_DIR}/../3rd_party/flexflow-flash-attention
    ${CMAKE_SOURCE_DIR}/../3rd_party/flash-attention/csrc/flash_attn
    ${TORCH_INCLUDE_DIRS}
    ${Python_INCLUDE_DIRS}  # Critical for Python.h
)

# Ensure C++17 standard
set_target_properties(test_flash_api PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    BUILD_RPATH ${CMAKE_BINARY_DIR}/lib
)

install(TARGETS test_flash_api DESTINATION ${CMAKE_BINARY_DIR}/tests)