// heat_serial.cpp —— 2D 热传导 (Jacobi 迭代, 5 点 stencil), CPU 串行参考版
//
// 角色: ① 正确性基准 —— 所有 GPU 版本的最终温度场必须与它一致;
//       ② 计时基线 —— GPU 版本要和它比, 看你快了多少。
//
// 物理设定: N×N 内部网格 (总网格 (N+2)^2, 外圈为固定边界),
//           上边界保持 100°C, 其余边界 0°C, 内部初始为 0°C。
//           每步 T_new[i][j] = (上下左右四邻的均值), 共 iters 步。
//
// 用法: ./serial <N> <iters> [--dump file.dump]
//       建议 ./serial 8192 500 --dump ref.dump  (题面标准算例, 见题面)
//
// 与 CUDA 版本刻意保持逐行相同的函数: init_field / checksum / dump_field。
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>

static void die(const char* msg) { fprintf(stderr, "error: %s\n", msg); exit(1); }

static void init_field(double* T, int W, int N) {
    memset(T, 0, sizeof(double) * (size_t)W * W);
    for (int j = 0; j < W; ++j) T[0 * W + j] = 100.0;   // 上边界 100°C
}

// 每 97 个点抽 1 个求和 —— 自检用的现场快照 (97 质数, 避免规则采样盲区)
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

// dump 只含内部 N×N 场, 双精度, 行主序; 判分用 compare.py 比对两个 dump
static void dump_field(const char* path, const double* T, int W, int N, int iters) {
    FILE* f = fopen(path, "wb");
    if (!f) die("fopen dump 失败");
    const char magic[4] = { 'H', 'J', '1', 0 };
    const unsigned int dtype = 0;   // 0 = double
    fwrite(magic, 4, 1, f);
    fwrite(&N, sizeof(unsigned int), 1, f);
    fwrite(&iters, sizeof(unsigned int), 1, f);
    fwrite(&dtype, sizeof(unsigned int), 1, f);
    for (int i = 1; i <= N; ++i)
        fwrite(&T[i * W + 1], sizeof(double), N, f);
    fclose(f);
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

    const int W = N + 2;                       // 含固定边界圈的行宽
    const size_t total = (size_t)W * W;
    double* T  = (double*)malloc(total * sizeof(double));
    double* Tn = (double*)malloc(total * sizeof(double));
    if (!T || !Tn) die("malloc");

    init_field(T, W, N);
    init_field(Tn, W, N);
    printf("# heat2d-jacobi | N=%d iters=%d dtype=double | CPU 串行参考版\n", N, iters);

    auto t0 = std::chrono::steady_clock::now();
    for (int it = 1; it <= iters; ++it) {
        for (int i = 1; i <= N; ++i)
            for (int j = 1; j <= N; ++j)
                Tn[i * W + j] = 0.25 * (T[(i - 1) * W + j] + T[(i + 1) * W + j] +
                                        T[i * W + j - 1] + T[i * W + j + 1]);
        double* tmp = T; T = Tn; Tn = tmp;
        if (it % 100 == 0 || it == iters) report(it, T, W, N);
    }
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    printf("FINAL secs=%.3f mlups=%.2f\n",
           secs, (double)N * N * iters / 1e6 / secs);
    if (dump) dump_field(dump, T, W, N, iters);
    free(T); free(Tn);
    return 0;
}
