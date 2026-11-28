cmake_minimum_required(VERSION 3.24)
project(bitnet_compiler LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(Python_EXECUTABLE "${CMAKE_SOURCE_DIR}/.venv/bin/python" CACHE FILEPATH "")
find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path)"
  OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)
list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX}")
find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib" REQUIRED)

Python_add_library(bitnet_ops MODULE WITH_SOABI cpp/src/packer.cpp)
target_include_directories(bitnet_ops PRIVATE cpp/include)
target_compile_definitions(bitnet_ops PRIVATE TORCH_EXTENSION_NAME=bitnet_ops)
target_compile_options(bitnet_ops PRIVATE ${TORCH_CXX_FLAGS})
target_link_libraries(bitnet_ops PRIVATE ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})
set_target_properties(bitnet_ops PROPERTIES BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib")

# --- OpenMP ---
# Apple clang has no bundled OpenMP runtime; point FindOpenMP at Homebrew's libomp.
if(APPLE AND NOT DEFINED OpenMP_ROOT)
  execute_process(COMMAND brew --prefix libomp
    OUTPUT_VARIABLE OpenMP_ROOT OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
endif()
find_package(OpenMP REQUIRED COMPONENTS CXX)

add_executable(parallelization cpp/src/parallelization.cpp)
target_include_directories(parallelization PRIVATE cpp/include)
target_link_libraries(parallelization PRIVATE OpenMP::OpenMP_CXX)
