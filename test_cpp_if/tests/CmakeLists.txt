cmake_minimum_required(VERSION 3.18)
project(test_flash_attn LANGUAGES CXX)

# Find PyTorch
find_package(Torch REQUIRED)

# Define test executable
add_executable(test_flash_api test.cpp)

# Link against the shared library
target_link_libraries(test_flash_api PRIVATE
    ${CMAKE_BINARY_DIR}/lib/flash_attn_2_cuda.so
    ${TORCH_LIBRARIES}
)

target_include_directories(test_flash_api PRIVATE
    ${CMAKE_SOURCE_DIR}/../3rd_party/flash-attention/csrc/flash_attn
    ${TORCH_INCLUDE_DIRS}
)

# Set RPATH for dynamic library loading
set_target_properties(test_flash_api PROPERTIES
    BUILD_RPATH ${CMAKE_BINARY_DIR}/lib
)

# Install the test binary to build directory
install(TARGETS test_flash_api DESTINATION ${CMAKE_BINARY_DIR}/tests)