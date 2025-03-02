cmake_minimum_required(VERSION 3.20)
project(FlashAttnTest)

enable_language(CXX)
enable_language(CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
add_compile_options(-D_GLIBCXX_USE_CXX11_ABI=0)
add_compile_definitions(TORCH_USE_CUDA_DSA=1)

# Find Python
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Find Python & Torch
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Include Torch
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE Torch_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(Torch REQUIRED PATHS ${Torch_DIR})
message("Torch_DIR: ${Torch_DIR}")
message("TORCH_INCLUDE_DIRS: ${TORCH_INCLUDE_DIRS}")
message("TORCH_LIBRARIES: ${TORCH_LIBRARIES}")

include_directories(${Python3_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS})

# Define test_mha_fwd_bwd executable BEFORE linking
add_executable(test_mha_fwd_bwd test/test_mha_fwd_bwd.cpp)

# Define test_run_mha_fwd executable BEFORE linking
add_executable(test_run_mha_fwd test/test_run_mha_fwd.cpp)

# Define test_run_mha_bwd executable BEFORE linking
add_executable(test_run_mha_bwd test/test_run_mha_bwd.cpp)

# Locate flash-attn shared library
file(GLOB FLASH_ATTN_LIB "${Python_SITELIB}/flash_attn*.so")
get_filename_component(FLASH_ATTN_SO ${FLASH_ATTN_LIB} NAME)
message("FLASH_ATTN_SO: ${FLASH_ATTN_SO}")

# Link directories
target_include_directories(test_mha_fwd_bwd
    PUBLIC "3rd_party/flash-attn/"
    PUBLIC ${Python3_INCLUDE_DIRS}
    PUBLIC ${TORCH_INCLUDE_DIRS}
)

target_include_directories(test_run_mha_fwd
    PUBLIC "3rd_party/flash-attn/"
    PUBLIC ${Python3_INCLUDE_DIRS}
    PUBLIC ${TORCH_INCLUDE_DIRS}
)

target_include_directories(test_run_mha_bwd
    PUBLIC "3rd_party/flash-attn/"
    PUBLIC ${Python3_INCLUDE_DIRS}
    PUBLIC ${TORCH_INCLUDE_DIRS}
)

target_link_directories(test_mha_fwd_bwd
    PUBLIC ${Python_SITELIB}/torch/lib
    PUBLIC ${Python_SITELIB}
)

target_link_directories(test_run_mha_fwd
    PUBLIC ${Python_SITELIB}/torch/lib
    PUBLIC ${Python_SITELIB}
)

target_link_directories(test_run_mha_bwd
    PUBLIC ${Python_SITELIB}/torch/lib
    PUBLIC ${Python_SITELIB}
)

# Link the required libraries for both executables
foreach(TEST_EXECUTABLE test_mha_fwd_bwd test_run_mha_fwd test_run_mha_bwd)
    target_link_libraries(${TEST_EXECUTABLE} PRIVATE
        ${TORCH_LIBRARIES}  # Link against PyTorch
        ${Python3_LIBRARIES}  # Link against Python
        "-Wl,-Bsymbolic -Wl,-Bsymbolic-functions"
        ":${FLASH_ATTN_SO}"
        dl # Link against libdl
    )

    # Ensure runtime can find the shared library
    set_target_properties(${TEST_EXECUTABLE} PROPERTIES
        BUILD_RPATH "${Python_SITELIB}"
        INSTALL_RPATH "${Python_SITELIB}"
    )
endforeach()