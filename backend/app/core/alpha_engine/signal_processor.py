"""
signal_processor.py — 策略级信号处理管道

在 DSL AST 执行之后、组合构建之前，对原始信号施加动态超参数处理。
全程使用 NumPy/Pandas 向量化实现，严禁 Python 级别的 T 轴循环。

处理顺序（业界标准）：
  A. Truncation   — 按行分位数截断极值
  B. Decay        — 线性加权衰减平滑（ts_decay_linear）
  C. Neutralize   — 行业中性化（ind_neutralize）
  D. Delay        — 执行延迟偏移（shift）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Literal

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# SimulationConfig — 策略超参数数据类
# ---------------------------------------------------------------------------

@dataclass
class SimulationConfig:
    """
    策略仿真超参数，传入 SignalProcessor 和 RealisticBacktester。

    Parameters
    ----------
    delay             : 执行延迟（天）。1 = T 日信号、T+1 日执行（默认）。
    decay_window      : ts_decay_linear 平滑窗口。0 = 不衰减。
    truncation_min_q  : 截断下分位数（0~1）。None = 不截断。
    truncation_max_q  : 截断上分位数（0~1）。None = 不截断。
    neutralize_groups : (N,) int 数组，资产→行业标签映射。None = 不中性化。
    portfolio_mode    : 组合构建模式，"long_short" 或 "decile"。
    top_pct           : decile 模式下多头/空头各取前 top_pct 比例（默认 0.10）。
    market_neutral    : 是否在组合构建后施加市场中性约束（默认 True）。
    """
    delay:             int                    = 1
    decay_window:      int                    = 0
    truncation_min_q:  Optional[float]        = 0.05
    truncation_max_q:  Optional[float]        = 0.95
    neutralize_groups: Optional[np.ndarray]   = field(default=None, repr=False)
    portfolio_mode:    Literal["long_short", "decile"] = "long_short"
    top_pct:           float                  = 0.10
    market_neutral:    bool                   = True

    def __post_init__(self) -> None:
        if self.delay < 0:
            raise ValueError(f"delay 必须 >= 0，当前={self.delay}")
        if self.decay_window < 0:
            raise ValueError(f"decay_window 必须 >= 0，当前={self.decay_window}")
        if self.truncation_min_q is not None and not (0.0 <= self.truncation_min_q < 0.5):
            raise ValueError(f"truncation_min_q 应在 [0, 0.5)，当前={self.truncation_min_q}")
        if self.truncation_max_q is not None and not (0.5 < self.truncation_max_q <= 1.0):
            raise ValueError(f"truncation_max_q 应在 (0.5, 1]，当前={self.truncation_max_q}")
        if self.top_pct <= 0 or self.top_pct >= 0.5:
            raise ValueError(f"top_pct 应在 (0, 0.5)，当前={self.top_pct}")


# ---------------------------------------------------------------------------
# SignalProcessor — 4 步向量化信号处理管道
# ---------------------------------------------------------------------------

class SignalProcessor:
    """
    对原始 DSL 信号（T×N DataFrame）依次施加 4 步后处理管道。

    所有操作全程向量化（NumPy/Pandas），无 Python for-over-T 循环。

    Parameters
    ----------
    config : SimulationConfig
    """

    def __init__(self, config: SimulationConfig) -> None:
        self.cfg = config

    # ------------------------------------------------------------------
    # 主接口
    # ------------------------------------------------------------------

    def process(self, signal: pd.DataFrame) -> pd.DataFrame:
        """
        依序执行 Truncation → Decay → Neutralize → Delay。

        Parameters
        ----------
        signal : (T, N) raw signal DataFrame，index=DatetimeIndex

        Returns
        -------
        pd.DataFrame  处理后信号，shape 与输入相同，index 保持不变。
        """
        out = signal.copy()

        # Step A: Truncation
        if self.cfg.truncation_min_q is not None or self.cfg.truncation_max_q is not None:
            out = self._truncate(out)

        # Step B: Decay
        if self.cfg.decay_window > 1:
            out = self._decay(out)

        # Step C: Neutralization
        if self.cfg.neutralize_groups is not None:
            out = self._neutralize(out)

        # Step D: Delay
        if self.cfg.delay > 0:
            out = self._delay(out)

        return out

    # ------------------------------------------------------------------
    # Step A — Truncation（按行分位数截断，全向量化）
    # ------------------------------------------------------------------

    def _truncate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Winsorise each row by cross-sectional quantiles.

        Only rows with at least one finite value receive meaningful bounds;
        all-NaN burn-in rows are left untouched (they stay NaN), which avoids
        the NumPy "All-NaN slice encountered" RuntimeWarning.
        """
        arr = df.to_numpy(dtype=float)          # (T, N)

        lo_arr = np.full(arr.shape[0], -np.inf)
        hi_arr = np.full(arr.shape[0],  np.inf)

        # Rows that contain at least one finite (non-NaN) value
        has_valid = np.isfinite(arr).any(axis=1)  # (T,) bool

        if has_valid.any():
            valid_arr = arr[has_valid]             # (T_valid, N)
            if self.cfg.truncation_min_q is not None:
                lo_arr[has_valid] = np.nanpercentile(
                    valid_arr, self.cfg.truncation_min_q * 100, axis=1,
                )
            if self.cfg.truncation_max_q is not None:
                hi_arr[has_valid] = np.nanpercentile(
                    valid_arr, self.cfg.truncation_max_q * 100, axis=1,
                )

        # Broadcast clip: (T,) → (T, 1) → (T, N)
        clipped = np.clip(arr, lo_arr[:, np.newaxis], hi_arr[:, np.newaxis])
        return pd.DataFrame(clipped, index=df.index, columns=df.columns)

    # ------------------------------------------------------------------
    # Step B — Decay（线性衰减，复用 fast_ops.ts_decay_linear）
    # ------------------------------------------------------------------

    def _decay(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        应用线性加权移动平均平滑信号。
        直接复用 fast_ops.ts_decay_linear（einsum + stride_tricks，全向量化）。
        """
        from app.core.alpha_engine.fast_ops import ts_decay_linear

        arr = df.to_numpy(dtype=float)          # (T, N)
        smoothed = ts_decay_linear(arr, self.cfg.decay_window)
        return pd.DataFrame(smoothed, index=df.index, columns=df.columns)

    # ------------------------------------------------------------------
    # Step C — Neutralization（行业中性化，复用 fast_ops.ind_neutralize）
    # ------------------------------------------------------------------

    def _neutralize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        按行业分组去均值（每组减去组内截面均值）。
        复用 fast_ops.ind_neutralize（G 外循环，T 内 NumPy，无 T 轴 Python 循环）。
        """
        from app.core.alpha_engine.fast_ops import ind_neutralize

        arr     = df.to_numpy(dtype=float)      # (T, N)
        groups  = np.asarray(self.cfg.neutralize_groups, dtype=int)
        neutral = ind_neutralize(arr, groups)
        return pd.DataFrame(neutral, index=df.index, columns=df.columns)

    # ------------------------------------------------------------------
    # Step D — Delay（执行延迟，Pandas shift，全向量化）
    # ------------------------------------------------------------------

    def _delay(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        向前偏移 delay 天，模拟 T 日收盘信号在 T+delay 日才能执行。
        pd.DataFrame.shift() 是 Pandas 内置向量化操作。
        """
        return df.shift(self.cfg.delay)
