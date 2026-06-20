#pragma once
// clangd-only shim. clangd parses .cu files as C++ (see .clangd) so that libtorch
// headers resolve correctly and go-to-definition works. nvcc never sees this file
// (guarded by __CUDACC__); it only provides no-op definitions for the CUDA
// keywords/builtins used in kernel code so the host-side parse stays clean.
#ifndef __CUDACC__

#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
#define __launch_bounds__(...)
#define __shared__
#define __constant__

// uint3/dim3 are completed by CUDA's <vector_types.h>; forward-declare them so we
// can declare the kernel builtins below without redefining the types.
struct uint3;
struct dim3;
extern uint3 threadIdx;
extern uint3 blockIdx;
extern dim3 blockDim;
extern dim3 gridDim;

#endif  // __CUDACC__
