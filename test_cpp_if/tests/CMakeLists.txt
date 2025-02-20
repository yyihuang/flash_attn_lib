cmake_minimum_required(VERSION 3.18)
project(test_flash_attn LANGUAGES CXX)

# Find PyTorch
find_package(Torch REQUIRED)

# Example test
add_executable(test_flash_api test.cpp)

# Link by target name rather than .so path
target_link_libraries(test_flash_api PRIVATE
    flash_attn_2_cuda
    ${TORCH_LIBRARIES}
)

target_include_directories(test_flash_api PRIVATE
    # Adjust paths for your include directories
    ${CMAKE_SOURCE_DIR}/../3rd_party/flash-attention/csrc/flash_attn
    ${TORCH_INCLUDE_DIRS}
)

# So the test executable can find flash_attn_2_cuda.so
set_target_properties(test_flash_api PROPERTIES
    BUILD_RPATH ${CMAKE_BINARY_DIR}/lib
)

install(TARGETS test_flash_api DESTINATION ${CMAKE_BINARY_DIR}/tests)