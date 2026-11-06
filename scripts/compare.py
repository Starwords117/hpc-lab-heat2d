#!/usr/bin/env python3
# compare.py —— 比对两个 heat dump (HJ1 格式), 报最大偏差, 判 PASS/FAIL
# 用法: python3 compare.py ref.dump gpu.dump [--tol 1e-9]
#   --tol 按最大绝对偏差判定; 相对偏差同时打印供报告引用。
#   容差以题面标注为准 (double 1e-9 / float 1e-4, 标定见草稿 §6)。
# 依赖: 有 numpy 自动走快速路径; 无 numpy 用纯标准库流式比对 (慢但可用)。
#       设 COMPARE_NO_NUMPY=1 可强制走标准库路径 (测试用)。
import struct
import sys
import argparse


def load_header(path):
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic[:3] != b"HJ1":
            sys.exit(f"{path}: 不是 HJ1 格式 (magic={magic!r})")
        n, iters, dtype = struct.unpack("<III", f.read(12))
    if dtype not in (0, 1):
        sys.exit(f"{path}: dtype 字段异常 ({dtype})")
    return n, iters, dtype


def check_compat(ref, got):
    n1, it1, _ = load_header(ref)
    n2, it2, _ = load_header(got)
    if n1 != n2:
        sys.exit(f"规模不同: {n1} vs {n2}")
    if it1 != it2:
        print(f"注意: 迭代数不同 {it1} vs {it2} —— 不具可比性!")
    return n1, it1


def compare_numpy(ref, got, n):
    import numpy as np
    dts = {0: np.float64, 1: np.float32}
    _, _, d1 = load_header(ref)
    _, _, d2 = load_header(got)
    a = np.fromfile(ref, dtype=dts[d1], count=n * n).reshape(n, n).astype(np.float64)
    b = np.fromfile(got, dtype=dts[d2], count=n * n).reshape(n, n).astype(np.float64)
    if a.size != n * n or b.size != n * n:
        sys.exit("文件截断")
    diff = np.abs(a - b)
    imax = np.unravel_index(np.argmax(diff), diff.shape)
    rel = diff / np.maximum(np.abs(a), 1e-12)
    mask = np.abs(a) > 1e-6
    return dict(max_abs=diff[imax], loc=imax, ref=a[imax], got=b[imax],
                max_rel=rel[mask].max() if mask.any() else 0.0,
                rms=float(np.sqrt((diff * diff).mean())))


def compare_stdlib(ref, got, n, chunk=1 << 20):
    import array
    fmts = {0: "d", 1: "f"}
    _, _, d1 = load_header(ref)
    _, _, d2 = load_header(got)
    total = n * n
    max_abs, loc = -1.0, (0, 0)
    max_rel, sum_sq = 0.0, 0.0
    done = 0
    with open(ref, "rb") as fa, open(got, "rb") as fb:
        fa.seek(16)
        fb.seek(16)
        while done < total:
            take = min(chunk, total - done)
            ca = array.array(fmts[d1]); ca.frombytes(fa.read(take * ca.itemsize))
            cb = array.array(fmts[d2]); cb.frombytes(fb.read(take * cb.itemsize))
            if len(ca) != take or len(cb) != take:
                sys.exit("文件截断")
            for k in range(take):
                x = float(ca[k]); y = float(cb[k])
                d = abs(x - y)
                if d > max_abs:
                    max_abs, loc = d, divmod(done + k, n)
                if abs(x) > 1e-6:
                    r = d / max(abs(x), 1e-12)
                    if r > max_rel:
                        max_rel = r
                sum_sq += d * d
            done += take
    refv, gotv = 0.0, 0.0
    with open(ref, "rb") as f:
        f.seek(16 + loc[0] * n * (8 if d1 == 0 else 4) + loc[1] * (8 if d1 == 0 else 4))
        refv = struct.unpack("<d" if d1 == 0 else "<f", f.read(8 if d1 == 0 else 4))[0]
    with open(got, "rb") as f:
        f.seek(16 + loc[0] * n * (8 if d2 == 0 else 4) + loc[1] * (8 if d2 == 0 else 4))
        gotv = struct.unpack("<d" if d2 == 0 else "<f", f.read(8 if d2 == 0 else 4))[0]
    return dict(max_abs=max_abs, loc=loc, ref=refv, got=gotv,
                max_rel=max_rel, rms=(sum_sq / total) ** 0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("got")
    ap.add_argument("--tol", type=float, default=1e-9)
    args = ap.parse_args()

    n, iters = check_compat(args.ref, args.got)
    use_np = True
    try:
        import numpy  # noqa: F401
    except ImportError:
        use_np = False
    import os
    if os.environ.get("COMPARE_NO_NUMPY"):
        use_np = False

    if use_np:
        r = compare_numpy(args.ref, args.got, n)
    else:
        r = compare_stdlib(args.ref, args.got, n)

    print(f"N={n}  迭代数 ref/got = {iters}/{iters}")
    print(f"max_abs_diff = {r['max_abs']:.3e}  @ (i={r['loc'][0]+1}, j={r['loc'][1]+1})  "
          f"ref={r['ref']:.10e} got={r['got']:.10e}")
    print(f"max_rel_diff(|ref|>1e-6) = {r['max_rel']:.3e}")
    print(f"rms_diff    = {r['rms']:.3e}")
    verdict = "PASS" if r["max_abs"] <= args.tol else "FAIL"
    print(f"判定 (tol={args.tol:g}): {verdict}")
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
