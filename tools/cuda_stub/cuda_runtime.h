// cuda_runtime.h — a MINIMAL stand-in used ONLY to syntax/type-check src/gpu/*.cu on machines
// without the CUDA toolkit (tools/check_cuda_syntax.sh, clang's CUDA front end, host + device
// passes, -nocudainc). It declares just what the backend uses. It is never used for building or
// running anything: real builds use NVIDIA's header (scripts/gpu_check.sh).
#pragma once
#include <cstddef>

#define __global__ __attribute__((global))
#define __device__ __attribute__((device))
#define __host__ __attribute__((host))
#define __shared__ __attribute__((shared))
#define __forceinline__ __inline__ __attribute__((always_inline))
#include <__clang_cuda_builtin_vars.h>
struct dim3 {
  unsigned x, y, z;
  __host__ __device__ dim3(unsigned vx = 1, unsigned vy = 1, unsigned vz = 1) : x(vx), y(vy), z(vz) {}
};

typedef int cudaError_t;
enum { cudaSuccess = 0, cudaErrorMemoryAllocation = 2 };
enum cudaMemcpyKind { cudaMemcpyHostToDevice = 1, cudaMemcpyDeviceToHost = 2, cudaMemcpyDeviceToDevice = 3 };
struct cudaDeviceProp { char name[256]; };
typedef struct CUstream_st* cudaStream_t;

__host__ cudaError_t cudaMalloc(void** p, std::size_t n);
template <class T> __host__ cudaError_t cudaMalloc(T** p, std::size_t n) { return cudaMalloc(reinterpret_cast<void**>(p), n); }
__host__ cudaError_t cudaFree(void* p);
__host__ cudaError_t cudaMemcpy(void* d, const void* s, std::size_t n, cudaMemcpyKind k);
__host__ cudaError_t cudaMemcpyAsync(void* d, const void* s, std::size_t n, cudaMemcpyKind k, cudaStream_t st = 0);
__host__ cudaError_t cudaGetLastError();
__host__ cudaError_t cudaDeviceSynchronize();
__host__ cudaError_t cudaGetDeviceCount(int* n);
__host__ cudaError_t cudaGetDeviceProperties(cudaDeviceProp* p, int dev);
__host__ const char* cudaGetErrorString(cudaError_t e);
// kernel launch configuration hooks used by clang's <<<>>> lowering
extern "C" __host__ int __cudaPushCallConfiguration(dim3 g, dim3 b, std::size_t shmem = 0, void* stream = 0);
extern "C" __host__ cudaError_t cudaLaunchKernel(const void*, dim3, dim3, void**, std::size_t, cudaStream_t);
__host__ cudaError_t cudaConfigureCall(dim3 g, dim3 b, std::size_t shmem = 0, cudaStream_t stream = 0);

__device__ double __shfl_down_sync(unsigned mask, double v, unsigned delta, int width = 32);
__device__ float __shfl_down_sync(unsigned mask, float v, unsigned delta, int width = 32);
__device__ double fmax(double a, double b);
__device__ double fmin(double a, double b);
__device__ double fabs(double a);
__device__ bool isfinite(double a);
__device__ void __syncthreads();
