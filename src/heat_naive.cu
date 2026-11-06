// heat_naive.cu —— 2D 热传导 Jacobi, GPU 版
//
// 背景: 某同学已经把 ./serial (heat_serial.cpp) 移植到了 GPU。
//       最终结果与 CPU 串行版一致, 但速度……先跑跑看再说。
//
// 用法: ./naive <N> <iters> [--dump file.dump]
//       建议 ./naive 8192 500 --dump naive.dump  (与题面标准算例一致)
//
// 你的任务 (题面任务 2): 在不改变物理结果的前提下让它快起来。
#include "gpu_compat.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>

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

// ---------- 以下三个函数与 heat_serial.cpp 逐行相同 ----------
static void init_field(double* T, int W, int N) {
    memset(T, 0, sizeof(double) * (size_t)W * W);
    for (int j = 0; j < W; ++j) T[0 * W + j] = 100.0;
}

static double checksum(const double* T, int W, int N) {
    double s = 0.0;
    for (int i = 1; i <= N; i += 97)
        for (int j = 1; j <= N; j += 97) s += T[i * W + j];
    return s;
}

static void report(int it, const double* T, int W, int N) {
    printf("iter=%d chksum=%.10e center=%.10e\n",
           it, checksum(T, W, N), T[(N / 2) * W + N / 2]);
}

static void dump_field(const char* path, const double* T, int W, int N, int iters) {
    FILE* f = fopen(path, "wb");
    if (!f) die("fopen dump 失败");
    const char magic[4] = { 'H', 'J', '1', 0 };
    const unsigned int dtype = 0;
    fwrite(magic, 4, 1, f);
    fwrite(&N, sizeof(unsigned int), 1, f);
    fwrite(&iters, sizeof(unsigned int), 1, f);
    fwrite(&dtype, sizeof(unsigned int), 1, f);
    for (int i = 1; i <= N; ++i)
        fwrite(&T[i * W + 1], sizeof(double), N, f);
    fclose(f);
}

// ---------- GPU 部分 ----------
// 把 global 线程号摊到 (i,j) 上, 每个线程算一个内部格点
__global__ void jacobi_step(const double* tin, double* tout, int W, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * N) return;
    int i = idx % N + 1;   // 行号
    int j = idx / N + 1;   // 列号
    tout[i * W + j] = 0.25 * (tin[(i - 1) * W + j] + tin[(i + 1) * W + j] +
                              tin[i * W + j - 1] + tin[i * W + j + 1]);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <N> <iters> [--dump file]\n", argv[0]);
        return 1;
    }
    int N = atoi(argv[1]);
    int iters = atoi(argv[2]);
    const char* dump = nullptr;
    for (int a = 3; a < argc; ++a)
        if (!strcmp(argv[a], "--dump") && a + 1 < argc) dump = argv[++a];

    const int W = N + 2;
    const size_t total = (size_t)W * W;
    const size_t bytes = total * sizeof(double);
    double* h_T  = (double*)malloc(bytes);
    double* h_Tn = (double*)malloc(bytes);
    if (!h_T || !h_Tn) die("malloc");

    init_field(h_T, W, N);
    init_field(h_Tn, W, N);
    printf("# heat2d-jacobi | N=%d iters=%d dtype=double | GPU (移植版)\n", N, iters);

    double *d_T, *d_Tn;
    CUDA_CHECK(cudaMalloc(&d_T, bytes));
    CUDA_CHECK(cudaMalloc(&d_Tn, bytes));
    // device 侧两个缓冲都要有初值: 边界圈此后不再有人写,
    // kernel 只更新内部格点 —— 第二块缓冲不初始化, 边界就是垃圾值
    CUDA_CHECK(cudaMemcpy(d_T, h_T, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Tn, h_T, bytes, cudaMemcpyHostToDevice));

    auto t0 = std::chrono::steady_clock::now();
    const int blocks = (int)((N * N + 255) / 256);
    for (int it = 1; it <= iters; ++it) {
        CUDA_CHECK(cudaMemcpy(d_T, h_T, bytes, cudaMemcpyHostToDevice));
        jacobi_step<<<blocks, 256>>>(d_T, d_Tn, W, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(h_Tn, d_Tn, bytes, cudaMemcpyDeviceToHost));
        double* tmp = h_T; h_T = h_Tn; h_Tn = tmp;
        if (it % 100 == 0 || it == iters) report(it, h_T, W, N);
    }
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    printf("FINAL secs=%.3f mlups=%.2f\n",
           secs, (double)N * N * iters / 1e6 / secs);
    if (dump) dump_field(dump, h_T, W, N, iters);
    free(h_T); free(h_Tn);
    cudaFree(d_T); cudaFree(d_Tn);
    return 0;
}
