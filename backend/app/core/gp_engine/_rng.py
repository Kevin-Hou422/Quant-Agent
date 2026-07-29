"""
_rng.py — GP 可绑定共享随机源（R-N1 修复，2026-07-30）

背景：`generate_random_alpha` / `mutations.*` / `_generate_diverse_seeds` 此前直接
用全局 `random`，使 GP 输出**跨进程/跨调用不可复现**（E4 当年只把 PopulationEvolver
改成实例级 RNG，这些上游/变异辅助漏改）。逐函数加 rng 参数需改 ~20 个签名、风险高。

本模块提供一个**可绑定的进程级随机源**：默认是一个新 `random.Random()`（行为等同
旧全局 random）；确定性入口在开始前调用 `bind(rng)` 绑定一个已播种的 `random.Random`，
使其后的 choice/random/uniform 完全确定。

并发安全：GP 全部入口由 `router._gp_lock` 串行化，测试顺序执行，故进程级共享随机源
不存在并发竞争（这与 E4 反对的"全局 random.seed 被并发覆盖"不同——那里是无锁的
进程级 seed；这里是串行化保护下的显式绑定）。
"""

from __future__ import annotations

import random as _random_mod
from typing import Any, Sequence

_active: _random_mod.Random = _random_mod.Random()


def bind(rng: _random_mod.Random) -> None:
    """将后续 choice/random/uniform 绑定到给定的已播种 Random 实例。"""
    global _active
    _active = rng


def bind_seed(seed: int) -> _random_mod.Random:
    """便捷：绑定一个 Random(seed) 并返回它。"""
    r = _random_mod.Random(seed)
    bind(r)
    return r


def current() -> _random_mod.Random:
    return _active


# --- random API 镜像（GP 仅用到这三个）---

def choice(seq: Sequence[Any]) -> Any:
    return _active.choice(seq)


def random() -> float:
    return _active.random()


def uniform(a: float, b: float) -> float:
    return _active.uniform(a, b)
