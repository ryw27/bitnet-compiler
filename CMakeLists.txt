cmake_minimum_required(VERSION 3.24)
project(bitnet_compiler LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(PYTHON "${CMAKE_SOURCE_DIR}/.venv/bin/python" CACHE FILEPATH "")
execute_process(
  COMMAND "${PYTHON}" -c "import torch; print(torch.utils.cmake_prefix_path)"
  OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)
list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX}")
find_package(Torch REQUIRED)

add_library(bitnet_ops SHARED cpp/src/packer.cpp)
target_include_directories(bitnet_ops PUBLIC cpp/include)
target_compile_options(bitnet_ops PRIVATE ${TORCH_CXX_FLAGS})
target_link_libraries(bitnet_ops PRIVATE ${TORCH_LIBRARIES})
set_target_properties(bitnet_ops PROPERTIES BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib")
