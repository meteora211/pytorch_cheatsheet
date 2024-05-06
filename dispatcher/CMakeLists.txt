cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(dispatcher)

set(USE_CUDA OFF)
set(BUILD_CAFFE2 OFF)

execute_process(
  COMMAND
    python3 -c
    "import torch; from pathlib import Path; print(Path(torch.__file__).parent, end='')"
  OUTPUT_VARIABLE TORCH_PATH)
list(APPEND CMAKE_PREFIX_PATH ${TORCH_PATH})
if(NOT TORCH_PATH)
  message(
    FATAL_ERROR
      "Torch is not installed"
  )
endif()

find_package(Torch REQUIRED)

add_library(dispatcher SHARED op.cpp)
target_compile_features(dispatcher PRIVATE cxx_std_14)
target_link_libraries(dispatcher "${TORCH_LIBRARIES}")
