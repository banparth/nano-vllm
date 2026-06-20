cmake_minimum_required(VERSION 3.26 FATAL_ERROR)
project(nanovllm_C LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# Emit compile_commands.json so clangd / IDEs can resolve includes and symbols.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 90)  # H100 (sm_90)
endif()

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(CUDAToolkit REQUIRED)

# Locate libtorch's CMake config via the active interpreter. This is robust when
# building with --no-build-isolation, where torch lives in the current venv.
execute_process(
  COMMAND ${Python_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
  OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
  OUTPUT_STRIP_TRAILING_WHITESPACE)
list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX}")

# PyTorch selects CUDA gencode from TORCH_CUDA_ARCH_LIST (it ignores
# CMAKE_CUDA_ARCHITECTURES), so target the H100 (sm_90) unless told otherwise.
if(NOT DEFINED ENV{TORCH_CUDA_ARCH_LIST})
  set(ENV{TORCH_CUDA_ARCH_LIST} "9.0")
endif()
find_package(Torch REQUIRED)

# Match libtorch's C++ ABI (TORCH_CXX_FLAGS carries -D_GLIBCXX_USE_CXX11_ABI=...).
separate_arguments(_TORCH_CXX_FLAGS UNIX_COMMAND "${TORCH_CXX_FLAGS}")
add_compile_options(${_TORCH_CXX_FLAGS})

# Auto-discover all kernel sources under csrc/ so adding a new .cu/.cpp needs no
# CMake edit. CONFIGURE_DEPENDS re-globs at build time (works with Ninja).
file(GLOB_RECURSE NANOVLLM_KERNEL_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/*.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/*.cpp")

# csrc/ is on the include path so sources can use "ops.h", "core/registration.h".
Python_add_library(_C MODULE WITH_SOABI ${NANOVLLM_KERNEL_SOURCES})
target_include_directories(_C PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/csrc)
target_link_libraries(_C PRIVATE ${TORCH_LIBRARIES})
target_compile_options(_C PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)

# Place the compiled module inside the nanovllm package (-> nanovllm/_C*.so).
install(TARGETS _C LIBRARY DESTINATION nanovllm)
