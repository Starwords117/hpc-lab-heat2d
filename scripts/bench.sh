#!/usr/bin/env bash
# bench.sh -- 计时脚本: 同一命令跑 3 次, 报告墙钟时间与内存, 取中位数。
#
# 用法:  bash scripts/bench.sh "<命令>"
# 例:    cd src && ../scripts/bench.sh "./serial 8192 500"
#        cd src && ../scripts/bench.sh "./naive 8192 500 --dump naive.dump"
#
# 每次运行的 stdout 存到 bench_run_N.out (第 3 次结果可直接喂给
# scripts/compare.py)。本脚本量的是端到端墙钟时间; kernel 级时间
# 用 cudaEvents (你的优化版程序自己打印)。

set -u
[ $# -ge 1 ] || { echo "用法: $0 <命令>"; exit 1; }
command="$*"

times=()
for i in 1 2 3; do
    /usr/bin/time -f "%e %M" -o bench_time.$$ $command > bench_run_$i.out 2>bench_run_$i.err
    read -r sec rss < bench_time.$$
    times+=("$sec")
    echo "run $i: ${sec} s  (max RSS ${rss} KB)"
done
rm -f bench_time.$$
printf "%s\n" "${times[@]}" | sort -n | sed -n 2p | awk '{printf "中位数: %.2f s\n", $1}'
