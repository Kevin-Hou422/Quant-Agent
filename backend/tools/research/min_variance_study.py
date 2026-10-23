"""
min_variance_study.py — 复审 2026-10-19 F12："协方差有效性、估计误差仍需研究评估，不等于保证最优。"

问题：AlphaCombiner 的 min_variance 用 T 天的因子组合收益估协方差（向对角收缩 0.2），闭式解后做多截断。
T 不够长时，估出来的权重是噪声 —— 它的**真实**组合方差可能还不如等权。代码现在的门槛是 T ≥ max(2n, 20)。

做法（蒙特卡洛，已知真协方差）：
  · 真协方差 Σ：n 个因子组合，日波动 0.3%–1.0%，相关性随机（含一对近似重复 ρ≈0.9、一对反向 ρ≈−0.4）；
  · 每次从 N(0, Σ) 抽 T 天 → 用与 `AlphaCombiner._min_variance_weights` 相同的估计器出权重 ŵ；
  · 评价用**真** Σ：ŵ'Σŵ 与 等权 w_eq'Σw_eq、真最优（做多约束下）w*'Σw* 比；
  · 报告：ŵ 的真方差 / 等权真方差 的中位数与"比等权差"的概率，以及权重的抽样标准差。

运行：python tools/research/min_variance_study.py [--reps 400] [--seed 7]
结论写进 docs/MIN_VARIANCE_STUDY.md；`MIN_VAR_MIN_DAYS_PER_FACTOR` 按它定。
"""

from __future__ import annotations

import argparse

import numpy as np

SHRINK = 0.2


def estimator(sample: np.ndarray, shrink: float = SHRINK) -> np.ndarray:
    """与 AlphaCombiner._min_variance_weights 相同：收缩 → 闭式 → 做多截断 → 归一（失败 / 全 0 → 等权）。"""
    n = sample.shape[1]
    s = np.cov(sample.T)
    cov = (1.0 - shrink) * s + shrink * np.diag(np.diag(s))
    raw = np.linalg.inv(cov + 1e-6 * np.eye(n)) @ np.ones(n)
    w = np.maximum(raw, 0.0)
    return w / w.sum() if w.sum() > 1e-12 else np.ones(n) / n


def true_optimum(sigma: np.ndarray, iters: int = 4000) -> np.ndarray:
    """做多约束下的真最小方差（投影梯度；n 小，足够精确）。"""
    n = sigma.shape[0]
    w = np.ones(n) / n
    step = 1.0 / (2.0 * np.linalg.eigvalsh(sigma).max())
    for _ in range(iters):
        w = w - step * 2.0 * sigma @ w
        # 投影到单纯形
        u = np.sort(w)[::-1]
        css = np.cumsum(u)
        rho = np.nonzero(u * np.arange(1, n + 1) > (css - 1))[0][-1]
        theta = (css[rho] - 1) / (rho + 1.0)
        w = np.maximum(w - theta, 0.0)
    return w


def make_sigma(n: int, rng: np.random.Generator, design: str = "hetero") -> np.ndarray:
    """
    hetero : 波动 0.3%–1.0%（各因子差异大 —— 最小方差最占便宜）
    equal  : 波动全部 0.6%、相关性弱（等权已接近最优 —— 估计误差只会添乱）
    """
    vol = rng.uniform(0.003, 0.010, n) if design != "equal" else np.full(n, 0.006)
    a = rng.normal(0, 1, (n, n))
    corr = np.corrcoef(a @ a.T + n * np.eye(n))
    corr = 0.6 * corr + 0.4 * np.eye(n) if design != "equal" else 0.3 * corr + 0.7 * np.eye(n)
    if n >= 3 and design != "equal":
        corr[0, 1] = corr[1, 0] = 0.9        # 两个近似重复的因子
        corr[0, 2] = corr[2, 0] = -0.4       # 一个反向
    vals, vecs = np.linalg.eigh(corr)        # 修正成正定
    corr = vecs @ np.diag(np.clip(vals, 1e-3, None)) @ vecs.T
    d = np.sqrt(np.diag(corr))
    corr = corr / np.outer(d, d)
    return corr * np.outer(vol, vol)


def drift(sigma: np.ndarray, rng: np.random.Generator, amount: float = 0.5) -> np.ndarray:
    """持有期的真协方差：相关结构与波动都换掉一部分（非平稳 —— 估计窗口的协方差不再是持有期的）。"""
    n = sigma.shape[0]
    other = make_sigma(n, rng)
    return (1 - amount) * sigma + amount * other


def run(reps: int = 400, seed: int = 7, ns=(2, 4, 8), ts=(20, 40, 60, 120, 252, 504), design: str = "hetero"):
    """design：hetero / equal / drift（hetero 的协方差在估计窗口与持有期之间漂移一半）。"""
    rng = np.random.default_rng(seed)
    out = []
    for n in ns:
        for t in ts:
            ratios, regrets, wsd = [], [], []
            for _ in range(reps):
                est_sigma = make_sigma(n, rng, "equal" if design == "equal" else "hetero")
                sigma = drift(est_sigma, rng) if design == "drift" else est_sigma     # 评价用持有期的真协方差
                eq = np.ones(n) / n
                opt = true_optimum(sigma)
                x = rng.multivariate_normal(np.zeros(n), est_sigma, size=t)
                w = estimator(x)
                v, v_eq, v_opt = w @ sigma @ w, eq @ sigma @ eq, opt @ sigma @ opt
                ratios.append(v / v_eq)
                regrets.append(v / v_opt - 1.0)
                wsd.append(np.abs(w - opt).sum())
            r = np.asarray(ratios)
            out.append({"n": n, "T": t, "T_per_factor": t / n,
                        "median_var_vs_equal": float(np.median(r)),
                        "p_worse_than_equal": float((r > 1.0).mean()),
                        "p_worse_by_10pct": float((r > 1.10).mean()),
                        "median_regret_vs_true_opt": float(np.median(regrets)),
                        "median_weight_error_l1": float(np.median(wsd))})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--design", default="hetero", choices=("hetero", "equal", "drift"))
    a = ap.parse_args()
    rows = run(a.reps, a.seed, design=a.design)
    print("design", a.design, "reps", a.reps, "seed", a.seed)
    print(f"{'n':>2} {'T':>4} {'T/n':>6} {'med var/EW':>10} {'P(>EW)':>7} {'P(>1.1EW)':>9} "
          f"{'med regret':>10} {'med |w-w*|':>10}")
    for r in rows:
        print(f"{r['n']:>2} {r['T']:>4} {r['T_per_factor']:>6.1f} {r['median_var_vs_equal']:>10.3f} "
              f"{r['p_worse_than_equal']:>7.2f} {r['p_worse_by_10pct']:>9.2f} "
              f"{r['median_regret_vs_true_opt']:>10.3f} {r['median_weight_error_l1']:>10.3f}")


if __name__ == "__main__":
    main()
