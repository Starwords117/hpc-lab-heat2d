// warmup_matadd.cu —— 热身题: C = A + B (N×N float)
//
// 你的 GPU 标称显存带宽是几百 GB/s。这个程序做一次大矩阵加法,
// 打印耗时与"有效带宽" (读 A + 读 B + 写 C, 共 3*N*N*4 字节)。
// 达到标称带宽的多少? 差距去哪了?
//
// 用法: ./warmup [N]     (默认 N=16384, 三块矩阵约占 3GB 显存;
//                         显存紧张可用 ./warmup 8192)
#include "gpu_compat.h"
#include <cstdio>
#include <cstdlib>

static void die(const char* msg) { fprintf(stderr, "error: %s\n", msg); exit(1); }

#define CUDA_CHECK(x)                                                        \
    do {                                                                     \
        cudaError_t e_ = (x);                                                \
        if (e_ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error: %s at %s:%d (%s)\n",                \
                    cudaGetErrorString(e_), __FILE__, __LINE__, #x);         \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

__global__ void matadd(const float* a, const float* b, float* c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * N) return;
    int i = idx % N;   // 行
    int j = idx / N;   // 列
    c[i * N + j] = a[i * N + j] + b[i * N + j];
}

int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 16384;
    const size_t total = (size_t)N * N;
    const size_t bytes = total * sizeof(float);
    printf("# matadd | N=%d (%.1f MB x 3) | float\n",
           N, bytes / 1048576.0);

    float *a, *b, *c;
    CUDA_CHECK(cudaMalloc(&a, bytes));
    CUDA_CHECK(cudaMalloc(&b, bytes));
    CUDA_CHECK(cudaMalloc(&c, bytes));
    cudaMemset(a, 0x3f, bytes);   // 值本身不影响带宽测试
    cudaMemset(b, 0x3e, bytes);

    const int blocks = (int)((total + 255) / 256);
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    // 先空跑一次再计时 (首次启动包含 context/初始化开销, 不计入)
    matadd<<<blocks, 256>>>(a, b, c, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int rep = 0; rep < 5; ++rep) {
        CUDA_CHECK(cudaEventRecord(t0));
        matadd<<<blocks, 256>>>(a, b, c, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        double secs = ms / 1000.0;
        double gbps = 3.0 * total * sizeof(float) / 1e9 / secs;
        printf("rep=%d secs=%.4f eff_gbps=%.1f\n", rep, secs, gbps);
    }

    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(a); cudaFree(b); cudaFree(c);
    return 0;
}
