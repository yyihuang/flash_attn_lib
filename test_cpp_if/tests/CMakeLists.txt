cmake_minimum_required(VERSION 3.18)
project(test_flash_attn LANGUAGES CXX)

# Find PyTorch
find_package(Torch REQUIRED)

# Define the test executable
add_executable(test_flash_api test.cpp)

# Link by target name instead of the raw .so file path
target_link_libraries(test_flash_api PRIVATE
    flash_attn_2_cuda
    ${TORCH_LIBRARIES}
)

# Include directories (adjust paths if necessary)
target_include_directories(test_flash_api PRIVATE
    ${CMAKE_SOURCE_DIR}/../3rd_party/flash-attention/csrc/flash_attn
    ${TORCH_INCLUDE_DIRS}
)

# Set RPATH so the test binary can find the shared library at runtime
set_target_properties(test_flash_api PROPERTIES
    BUILD_RPATH ${CMAKE_BINARY_DIR}/lib
)

# Install the test binary to the build directory (optional)
install(TARGETS test_flash_api DESTINATION ${CMAKE_BINARY_DIR}/tests)
