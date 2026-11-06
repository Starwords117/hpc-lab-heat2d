# HPC GPU 优化练习 · heat2d 热传导

> 入门题：把一个"移植到 GPU 反而没快"的热传导求解器救回来（两个经典性能陷阱；全程结果要求逐点一致）。

## 背景

二维热传导（Jacobi 迭代，5 点 stencil）是 CFD 和气象模式里最典型的内核算子。你的 GPU 显存带宽是几百 GB/s，这个 kernel 每步只需要搬约 1GB 数据——理论上每步只要几毫秒。

本仓库 `src/` 提供三份程序：

- `heat_serial.cpp` —— CPU 串行版。**它是正确性基准**，也是性能比较的分母；
- `heat_naive.cu` —— 某同学已经把它"移植到了 GPU"。**结果与 CPU 版一致，但速度基本上没有意义**；
- `warmup_matadd.cu` —— 热身题，先做（见下）。

## 你的任务

### 热身（30~45 分钟，不计大分，但必须先做）

1. `make warmup && ./warmup`（默认占 3GB 显存；紧张就用 `./warmup 8192`）。程序打印 5 次耗时与**有效带宽**——查你显卡的标称带宽，算"发挥了百分之几"；
2. `ncu ./warmup` 看 Memory Workload Analysis，找到它慢的原因（代码只有 40 行，问题肉眼可读，题面不点破）；
3. 修复（改动 ≤2 行），目标：≥70% 标称带宽；
4. block 尺寸扫一遍（128/256/512/1024），报告表格。

### 主题（3~4 小时 + 报告）

算例阶梯（规模 `N=8192` 全程不许改，`iters` 分档；磁盘留 ~1.5GB 放 dump）：

| 档位 | 命令 | 用途 | CPU 串行参考耗时 |
|---|---|---|---|
| smoke | `./serial 512 200` | 冒烟/正确性 | <1 s |
| selftest | `./serial 2048 500` | 自测/对锚点 | ~1.3 s |
| medium | `./serial 8192 500` | 开发中间检查点 | ~25 s |
| **full（评分算例）** | `./serial 8192 2500` | ①②③ 与提交 | ~2~3 min |

| 步骤 | 命令（full 档） | 要记录的东西 |
|---|---|---|
| ① CPU 基线 | `./serial 8192 2500 --dump ref.dump`（跑 1 次） | secs、mlups；`ref.dump` 是正确性基准 |
| ② 移植版 | `./naive 8192 2500 --dump naive.dump`（跑 1 次即可，很慢） | secs、mlups；与 ① 比对应 PASS |
| ③ 优化版 | 你的代码，同算例 | wall / kernel 两个时间、dram_gbps、compare 结果 |

优化阶梯（逐级拿分）：

| 级别 | 目标（用数据证明你达到了） | 说明 |
|---|---|---|
| L1 | 总时间相对 ② 有**量级改善**：用 cudaEvents 做时间分解（分阶段各量多少），把证据和你的改动依据写进报告 | 预期收益 10× 以上 |
| L2 | kernel 达到**标称显存带宽 ≥70%**（按下面 dram_gbps 口径计算并展示过程）；另外做一组 block 尺寸扫描（如 8×8/16×16/32×16/32×32）放进报告——**无论有没有提升都要解释为什么** | kernel 证据用 ncu（Memory Workload Analysis / sector 效率） |
| L3 | 切到 float（流量减半），在**误差预算内**（见判据）进一步加速；必答：为什么本题 float 只给约 2×？（想想这个 kernel 被什么卡住；什么代码才会被 GeForce 的 FP64=1/32 速率卡脖子） | 报告必答 |
| L4（加分） | 自选：shared memory tiling / 一线程多点 ILP / kernel 内归约自检 / stream 重叠 / 其他 | 上不封顶，讲清楚即可 |

## 环境

- NVIDIA 显卡 ≥4GB（8GB 富余，本题显存峰值 ~1.1GB；热身 ~3GB）+ CUDA Toolkit ≥11.8（含 `nvcc` 与 `nsight-compute`）
- `-arch` 对照见 `src/Makefile` 注释；`-arch=native` 需要 CUDA ≥11.5
- AMD 显卡的同学：`make hip`（同一份源码走 ROCm/hipcc，本题验证过这条路径）
- 无第三方库依赖；判分脚本只需 python3（numpy 可选，没有也能跑）

### Windows 用户（两条路径，均可跑）

**路径一（推荐）：WSL2** —— 本题全部脚本（make / bench.sh / ncu / compare.py）原样可用：

1. Windows 装好 NVIDIA 显卡驱动；
2. 管理员 PowerShell 执行 `wsl --install -d Ubuntu`，重启进入 Ubuntu；
3. 在 Ubuntu 内按上文安装 CUDA Toolkit（**不要**在 WSL 里再装 Linux 版 NVIDIA 驱动，Windows 驱动会自动透传）；
4. 代码 clone 到 Linux 家目录（`~/`），不要放 `/mnt/c`——否则 nvcc 编译和 512MB dump 读写慢一个量级；资源管理器地址栏输 `\\wsl$` 可直接取 dump 文件。

**路径二：原生 Windows** —— 代码可编译可跑（纯标准 C/C++ + CUDA Runtime API），但没有 make 和 bench.sh。前提：装 Visual Studio 2022 Community（勾"使用 C++ 的桌面开发"）+ CUDA Toolkit，在 **x64 Native Tools Command Prompt** 里：

```bat
nvcc -O3 -arch=native warmup_matadd.cu -o warmup.exe
nvcc -O3 -arch=native heat_naive.cu -o naive.exe
cl /O2 /utf-8 heat_serial.cpp
```

- 计时：三个程序都自己打印 `FINAL secs` / `kernel_secs`，每个配置手动跑 3 次取中位数（替代 bench.sh）；
- 判分：`python3 scripts\compare.py ref.dump yours.dump --tol 1e-9`（装 Python 3 即可）；
- `ncu` Windows 原生可用（toolkit 自带），如 `ncu .\naive.exe 8192 500`；
- 源码含中文注释，个别终端可能显示乱码——不影响编译与判分（PASS/FAIL 是 ASCII）。

## 快速开始

```bash
git clone <本仓库 URL> && cd hpc-lab-heat2d/src
make                # 产出 warmup / serial / naive; 你优化后的程序是第四个

# 冒烟（秒级; 三份程序结果都应与基准一致）
./serial 512 200 --dump /tmp/ref_smoke.dump
python3 ../scripts/compare.py ../baseline/ref_smoke_512x200.dump /tmp/ref_smoke.dump
./naive 512 200 --dump /tmp/naive_smoke.dump
python3 ../scripts/compare.py ../baseline/ref_smoke_512x200.dump /tmp/naive_smoke.dump

# 评分算例 full（参考机器: CPU 串行 ~2-3 min, naive 2-5 min, 见 baseline/BASELINE.md）
# 开发时建议先用 medium (iters=500) 迭代, 最后再上 full
./serial 8192 2500 --dump ref.dump
./naive 8192 2500 --dump naive.dump
```

## 正确性判据（一票否决）

- **最终温度场逐点比对**：`python3 ../scripts/compare.py ref.dump yours.dump --tol 1e-9`；
- 双精度容差 **1e-9**（所有档位）；float 容差按档位：medium（500 步）**1e-4**、full（2500 步）**2e-4**（标定与实测误差见 BASELINE.md；误差随迭代数增长，改迭代数必须重推容差并在报告说明）；
- dump 头里记录了 N 与 iters，规模不一致会被拒；
- 自检锚点：`baseline/BASELINE.md` 的 chksum 表（双精度跨机器逐位一致，跑完先对表）。

## 计时与提交

```bash
../scripts/bench.sh "你的程序 8192 2500"   # full 档正式计时: 3 次取中位数
```

- 3 次取中位数只要求**优化终版**；① 串行基线与 ② 移植版在 full 档各跑 1 次即可（等它们 3 次太浪费）；
- 区分 **wall**（端到端，bench.sh 量的）与 **kernel**（cudaEvents，你的程序自己打印）；
- `dram_gbps = 2 × N² × sizeof(类型) × iters / kernel时间`——口径：每格点每迭代至少从显存读 1 次、写 1 次（4 次邻域读由缓存吸收），与 ncu 的 `dram__bytes` 同口径可互相核对；
- 提交：① 最终代码 ② compare.py 完整输出 ③ 时间表（naive→L1→L2→L3/L4 的 wall/kernel/dram_gbps）④ ncu 关键页截图（Speed Of Light + Memory Workload Analysis 各一）⑤ 报告 ≤2 页（热点怎么找的→每步改动与效果→L3 必答→block 扫描结果与解释）。**报告占 30%**；
- 报告请附 `nvidia-smi` 输出与编译器版本。

## 规则

- 禁止修改标准算例（N=8192、iters=500）或 dump 头信息
- 禁止把迭代循环搬到 CPU 上算完再拷回 device、伪造 dump
- 允许：任意编译 flag、任意 kernel 结构、float（过容差）、一切保持判据通过的优化
- 诚实条款：报告中每个数字都要能复现（抽查你的命令行历史）

## 许可

- 本仓库代码为教学用途自写，MIT（见 `LICENSE`）；无第三方数据与上游代码
- 基线数据来自参考机器实测，生成环境与复现方法见 `baseline/BASELINE.md`
