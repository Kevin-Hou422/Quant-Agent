"""
factor_model.py — Barra-lite 风险因子模型与风险归因（Phase R.2）

做什么
------
1. **风格暴露** `build_exposures`：从价量算每只股票在几个风格上的暴露，
   逐日做截面标准化（z-score）。
2. **因子收益** `fit_risk_model`：逐日跑一次截面回归
   `r_t = B_t · f_t + u_t`，得到因子收益 `f_t` 与特异收益 `u_t`。
3. **结构化协方差** `RiskModel.covariance()`：`Σ = B Σ_f Bᵀ + D`
   （D 为特异方差对角阵）。它比样本协方差稳健得多 —— N 只股票的样本协方差
   要估 N(N+1)/2 个数，而结构化只要 K(K+1)/2 + N 个。
4. **风险归因** `RiskModel.attribute()`：把组合方差拆成
   **因子部分**（再拆到每个因子）与**特异部分**。

范围与诚实边界（**先读这一段**）
--------------------------------
- 本模型只用**价量**可得的风格：动量、反转、波动、流动性、市场 beta，
  外加行业哑变量。**size 与 value 需要基本面**（股数、账面价值），
  在 Phase 10 之前拿不到 —— 不是忘了，是没有数据。
  因此"风险归因覆盖了主要风格"这句话在本仓当前数据下**不成立**，
  归因结果应读作"在这 5 个价量风格 + 行业上的分解"。
- 本模块**不改变任何交易行为**：它只读权重与价格，产出诊断。路线图里
  R.2 的另一半（用回归残差替代现有 rank/行业 demean 做风格中性化）会改信号，
  属于另一次变更，不在本轮。
- `beta_neutral` 的数学闭合（B6）按 PM.5 的既有决策**仍留待开启做空之后**：
  当前 long-only，net=gross，对冲不适用。

判据（为什么这些数字可信）
--------------------------
方差分解满足恒等式 `w'Σw = w'BΣ_fB'w + w'Dw`，这是可以逐位核对的 ——
`attribute()` 返回的 `factor_var + specific_var` 必须等于 `total_var`。
测试里对它做精确断言；对不上说明实现错了，而不是"近似"。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

WidePanel = Dict[str, pd.DataFrame]

#: 价量可得的风格因子。顺序固定 —— 归因结果按名字索引，换序不影响，
#: 但保持稳定便于跨轮对比。
STYLE_FACTORS = ("momentum", "reversal", "volatility", "liquidity", "beta")

#: 截面 z-score 之后的极值截断（单位：标准差）。Barra 的惯例是 ±3，
#: 目的是不让一两只异常股主导整条因子收益。
_WINSOR_Z = 3.0


# ---------------------------------------------------------------------------
# 1. 风格暴露
# ---------------------------------------------------------------------------

def _zscore_panel(df: pd.DataFrame) -> pd.DataFrame:
    """
    逐行（逐期）截面标准化 + 截断。有效样本 <2 或零离散度的行 → 全 0（无暴露），不抛。

    **整块向量化**，不是 `df.apply(..., axis=1)`：后者在 600×5 的面板上就是
    3000 次逐行 Python 调用，实测占 build_exposures 全部耗时的大头（§P：
    昂贵推导别放进逐行循环 —— 先量再改）。语义与逐行版逐位一致。
    """
    v = df.to_numpy(dtype=float)
    valid = np.isfinite(v)
    cnt = valid.sum(axis=1)

    # 手工求和而不是 np.nanmean/np.nanstd：预热期整行都是 NaN，nan 系列函数会对
    # 每一个这种行发一条 RuntimeWarning（"Mean of empty slice"）。把告警压掉是
    # 错的方向 —— 那会连真正的数值异常一起压掉；这里直接避免产生它。
    vv = np.where(valid, v, 0.0)
    mu = np.divide(vv.sum(axis=1), np.maximum(cnt, 1),
                   out=np.zeros(len(cnt)), where=cnt > 0)
    dev = np.where(valid, v - mu[:, None], 0.0)
    var = np.divide((dev ** 2).sum(axis=1), np.maximum(cnt - 1, 1),
                    out=np.zeros(len(cnt)), where=cnt > 1)
    sd = np.sqrt(var)

    usable = (cnt >= 2) & np.isfinite(sd) & (sd > 1e-12)
    denom = np.where(usable, sd, 1.0)[:, None]
    z = (v - mu[:, None]) / denom
    z = np.clip(z, -_WINSOR_Z, _WINSOR_Z)
    z[~np.isfinite(z)] = 0.0
    z[~usable, :] = 0.0
    return pd.DataFrame(z, index=df.index, columns=df.columns)


def build_exposures(
    dataset: WidePanel,
    *,
    momentum_window: int = 252,
    momentum_skip: int = 21,
    reversal_window: int = 21,
    vol_window: int = 60,
    liquidity_window: int = 60,
    beta_window: int = 120,
) -> Dict[str, pd.DataFrame]:
    """
    产出 `{因子名: (T×N) 暴露面板}`，每行已做截面 z-score + 截断。

    各风格的构造（都取**负号或正号后仍是"暴露越高越怎样"**，见注释）：
      momentum   : 过去 momentum_window 日、跳过最近 momentum_skip 日的累计收益
                   （跳过近月是为了不把短期反转混进动量，J&T 1993 的标准做法）
      reversal   : **负的**近 reversal_window 日收益 —— 暴露高 = 近期跌得多
      volatility : 近 vol_window 日已实现波动
      liquidity  : **负的** Amihud 非流动性 —— 暴露高 = 越流动
      beta       : 对等权市场组合的滚动回归斜率

    缺 `close` 直接抛 ValueError（没有价格谈不上风险模型）。
    """
    prices = dataset.get("close")
    if prices is None or prices.empty:
        raise ValueError("dataset 缺少 close，无法构造风格暴露")

    px = prices.astype(float)
    ret = px.pct_change()
    volume = dataset.get("volume")
    if volume is None:
        # 没有成交量就没有流动性因子。**不静默造一个常数** —— 那会让
        # liquidity 暴露恒为 0 并被当成"这些股票流动性都一样"。
        logger.warning("[risk_engine] dataset 缺 volume → 流动性因子将全为 0（无区分力）")
        volume = pd.DataFrame(np.nan, index=px.index, columns=px.columns)
    vol_usd = volume.astype(float) * px

    raw: Dict[str, pd.DataFrame] = {}

    # momentum：log 价差，跳过最近一个月
    logp = np.log(px.where(px > 0))
    raw["momentum"] = logp.shift(momentum_skip) - logp.shift(momentum_window)

    # reversal：近月收益取负
    raw["reversal"] = -(logp - logp.shift(reversal_window))

    # volatility：已实现波动
    raw["volatility"] = ret.rolling(vol_window, min_periods=max(5, vol_window // 4)).std()

    # liquidity：Amihud = |r| / 美元成交额，取负 → 越大越流动
    amihud = (ret.abs() / vol_usd.replace(0.0, np.nan)).rolling(
        liquidity_window, min_periods=max(5, liquidity_window // 4)).mean()
    raw["liquidity"] = -amihud

    # beta：对等权市场的滚动回归斜率 cov(r_i, r_m) / var(r_m)
    mkt = ret.mean(axis=1)
    min_p = max(20, beta_window // 4)
    cov = ret.rolling(beta_window, min_periods=min_p).cov(mkt)
    var = mkt.rolling(beta_window, min_periods=min_p).var()
    raw["beta"] = cov.div(var.replace(0.0, np.nan), axis=0)

    out = {name: _zscore_panel(df) for name, df in raw.items()}
    return {k: out[k] for k in STYLE_FACTORS}


def sector_exposures(dataset: WidePanel) -> Optional[pd.DataFrame]:
    """
    行业哑变量的**最后一期**取值（行业不随日变动，取最后一行即可）。
    返回 (N × n_sector) 0/1 矩阵；无 `sector` 字段返回 None（调用方必须处理）。
    """
    sec = dataset.get("sector")
    if sec is None or sec.empty:
        return None
    last = sec.iloc[-1]
    codes = sorted({int(c) for c in last.dropna().unique() if int(c) >= 0})
    if not codes:
        return None
    return pd.DataFrame(
        {f"sector_{c}": (last == c).astype(float) for c in codes},
        index=sec.columns,
    )


# ---------------------------------------------------------------------------
# 2. 模型拟合与归因
# ---------------------------------------------------------------------------

@dataclass
class RiskAttribution:
    """一份组合的风险分解。`factor_var + specific_var == total_var`（恒等式）。"""
    total_var:     float
    factor_var:    float
    specific_var:  float
    #: 逐因子的方差贡献（含交叉项，按 w'B 的分量分摊；合计 == factor_var）
    by_factor:     Dict[str, float] = field(default_factory=dict)
    #: 组合在每个因子上的净暴露 w'B
    exposures:     Dict[str, float] = field(default_factory=dict)
    n_assets:      int = 0
    n_factors:     int = 0
    sector_included: bool = False

    @property
    def total_vol_ann(self) -> float:
        return float(np.sqrt(max(self.total_var, 0.0)) * np.sqrt(252.0))

    @property
    def factor_share(self) -> float:
        """因子风险占比。总方差为 0 时返回 0.0（而不是 nan）。"""
        return float(self.factor_var / self.total_var) if self.total_var > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "total_vol_ann": round(self.total_vol_ann, 6),
            "factor_share": round(self.factor_share, 4),
            "total_var": self.total_var,
            "factor_var": self.factor_var,
            "specific_var": self.specific_var,
            "by_factor": {k: round(v, 10) for k, v in self.by_factor.items()},
            "exposures": {k: round(v, 6) for k, v in self.exposures.items()},
            "n_assets": self.n_assets,
            "n_factors": self.n_factors,
            "sector_included": self.sector_included,
            # size/value 需要基本面数据（Phase 10）。**必须跟着结果一起返回** ——
            # 否则"因子风险占比 62%" 会被读成"覆盖了主要风格"。
            "styles_covered": list(STYLE_FACTORS),
            "styles_missing": ["size", "value"],
        }


@dataclass
class RiskModel:
    """拟合好的风险模型。`B` 是最后一期暴露，`factor_cov`/`specific_var` 由历史估计。"""
    exposures:    pd.DataFrame          # (N × K) 最后一期暴露（含行业哑变量）
    factor_cov:   pd.DataFrame          # (K × K) 因子收益协方差（日频）
    specific_var: pd.Series             # (N,) 特异方差（日频）
    n_obs:        int                   # 参与估计的交易日数
    sector_included: bool = False

    def covariance(self) -> pd.DataFrame:
        """结构化协方差 `Σ = B Σ_f Bᵀ + D`。"""
        B = self.exposures.to_numpy(dtype=float)
        F = self.factor_cov.to_numpy(dtype=float)
        D = np.diag(self.specific_var.reindex(self.exposures.index).fillna(0.0).to_numpy())
        return pd.DataFrame(B @ F @ B.T + D,
                            index=self.exposures.index, columns=self.exposures.index)

    def attribute(self, weights: pd.Series) -> RiskAttribution:
        """
        把组合方差拆成因子/特异两部分，并分摊到各因子。

        分摊口径：因子 j 的贡献 = `x_j · (Σ_f x)_j`，其中 `x = Bᵀw`。
        这是**边际贡献**口径（含交叉项，按 Euler 分解），各项之和恒等于
        `x'Σ_f x` = 因子方差 —— 不是"各因子单独方差"的简单相加（那会漏掉
        因子间相关性，合计对不上总数）。
        """
        w = weights.reindex(self.exposures.index).fillna(0.0).astype(float)
        B = self.exposures.to_numpy(dtype=float)
        F = self.factor_cov.to_numpy(dtype=float)
        x = B.T @ w.to_numpy()                       # 组合的因子净暴露
        Fx = F @ x
        factor_var = float(x @ Fx)
        d = self.specific_var.reindex(self.exposures.index).fillna(0.0).to_numpy()
        specific_var = float(np.sum((w.to_numpy() ** 2) * d))

        by_factor = {name: float(x[i] * Fx[i])
                     for i, name in enumerate(self.exposures.columns)}
        return RiskAttribution(
            total_var=factor_var + specific_var,
            factor_var=factor_var,
            specific_var=specific_var,
            by_factor=by_factor,
            exposures={name: float(x[i]) for i, name in enumerate(self.exposures.columns)},
            n_assets=int(len(w)),
            n_factors=int(B.shape[1]),
            sector_included=self.sector_included,
        )


def fit_risk_model(
    dataset: WidePanel,
    *,
    lookback: int = 252,
    min_obs: int = 60,
    with_sector: bool = True,
    **exposure_kwargs,
) -> RiskModel:
    """
    逐日截面回归估因子收益，再由其协方差与残差方差组成风险模型。

    Parameters
    ----------
    lookback : 用最近多少个交易日估计因子协方差。
    min_obs  : 有效回归日数的下限；不足则 **抛 ValueError**。
               返回一个由 10 天数据估出来的协方差，比不返回危险得多 ——
               它看起来和正常结果一模一样。

    Raises
    ------
    ValueError : 数据不足、或所有截面都退化到无法回归。
    """
    prices = dataset.get("close")
    if prices is None or prices.empty:
        raise ValueError("dataset 缺少 close，无法拟合风险模型")

    exps = build_exposures(dataset, **exposure_kwargs)
    ret = prices.astype(float).pct_change()

    sec = sector_exposures(dataset) if with_sector else None
    if with_sector and sec is None:
        logger.warning("[risk_engine] dataset 无 sector 字段 → 行业哑变量缺席，"
                       "行业集中带来的共同风险会被算进『特异』里（低估因子风险）")

    dates = list(prices.index[-lookback:]) if lookback > 0 else list(prices.index)
    names: List[str] = list(STYLE_FACTORS) + (list(sec.columns) if sec is not None else [])

    # 设计矩阵整块预取，逐日只做切片 —— 原来每天一次 pd.concat(5 个 Series)，
    # 400 天就是 400 次重建索引，实测是这个函数的主要开销（§P）。
    tickers = list(prices.columns)
    style_arr = np.stack(
        [exps[k].reindex(index=prices.index, columns=tickers).to_numpy(dtype=float)
         for k in STYLE_FACTORS], axis=2)                       # (T, N, K_style)
    sec_arr = (sec.reindex(tickers).to_numpy(dtype=float)
               if sec is not None else np.empty((len(tickers), 0)))
    ret_arr = ret.reindex(columns=tickers).to_numpy(dtype=float)
    pos = {d: i for i, d in enumerate(prices.index)}

    f_rows: List[np.ndarray] = []
    f_dates: List[pd.Timestamp] = []
    resid_rows: List[pd.Series] = []
    for d in dates:
        i = pos.get(d)
        if i is None:
            continue
        yv_all = ret_arr[i]
        Xv_all = np.hstack([style_arr[i], sec_arr]) if sec_arr.size else style_arr[i]
        ok = np.isfinite(yv_all) & np.isfinite(Xv_all).all(axis=1)
        if int(ok.sum()) <= Xv_all.shape[1] + 1:  # 自由度不足：回归必然完美拟合
            continue
        Xv, yv = Xv_all[ok], yv_all[ok]
        try:
            beta, *_ = np.linalg.lstsq(Xv, yv, rcond=None)
        except np.linalg.LinAlgError as exc:     # 奇异：记录，不静默跳过
            logger.warning("[risk_engine] %s 截面回归失败: %s", d, exc)
            continue
        f_rows.append(beta)
        f_dates.append(d)
        resid_rows.append(pd.Series(yv - Xv @ beta,
                                    index=[tickers[j] for j in np.flatnonzero(ok)], name=d))

    if len(f_rows) < min_obs:
        raise ValueError(
            f"有效截面回归只有 {len(f_rows)} 天 < 下限 {min_obs}，"
            f"估不出可用的因子协方差（样本太短的协方差看起来和正常结果一样，"
            f"所以这里拒绝返回）。")

    fret = pd.DataFrame(f_rows, index=pd.DatetimeIndex(f_dates), columns=names)
    resid = pd.DataFrame(resid_rows)
    factor_cov = fret.cov()
    specific_var = resid.var(ddof=1).reindex(prices.columns).fillna(0.0)

    last_exp = pd.concat([exps[k].iloc[-1].rename(k) for k in STYLE_FACTORS], axis=1)
    if sec is not None:
        last_exp = pd.concat([last_exp, sec.reindex(last_exp.index)], axis=1)
    last_exp = last_exp.reindex(columns=names).fillna(0.0)

    logger.info("[risk_engine] 风险模型：%d 天截面回归 | %d 因子（行业=%s）| %d 只股票",
                len(f_rows), len(names), sec is not None, last_exp.shape[0])
    return RiskModel(exposures=last_exp, factor_cov=factor_cov,
                     specific_var=specific_var, n_obs=len(f_rows),
                     sector_included=sec is not None)
