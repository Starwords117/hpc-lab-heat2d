// gpu_compat.h —— 一份源码, 两种编译路径:
//   NVIDIA 学员:   nvcc -O3 -arch=native heat_naive.cu -o naive
//                  (本头文件直接展开为 <cuda_runtime.h>, 与普通 CUDA 程序无异)
//   AMD/ROCm 机器: hipcc -O3 heat_naive.cu -o naive_hip
//                  (同一份源码经少量 API 别名直接编译, 无需 hipify 工具)
// 题面默认 NVIDIA/CUDA; AMD 兼容是备用通道 (见题面 "HIP 兼容" 一节)。
#ifndef GPU_COMPAT_H
#define GPU_COMPAT_H

#ifdef __HIP_PLATFORM_AMD__

#include <hip/hip_runtime.h>

#define cudaMalloc             hipMalloc
#define cudaFree               hipFree
#define cudaMemcpy             hipMemcpy
#define cudaMemset             hipMemset
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaEvent_t            hipEvent_t
#define cudaEventCreate        hipEventCreate
#define cudaEventDestroy       hipEventDestroy
#define cudaEventRecord        hipEventRecord
#define cudaEventSynchronize   hipEventSynchronize
#define cudaEventElapsedTime   hipEventElapsedTime
#define cudaDeviceSynchronize  hipDeviceSynchronize
#define cudaGetLastError       hipGetLastError
#define cudaGetErrorString     hipGetErrorString
#define cudaError_t            hipError_t
#define cudaSuccess            hipSuccess

#else

#include <cuda_runtime.h>

#endif
#endif // GPU_COMPAT_H
