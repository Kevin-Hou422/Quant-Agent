"""
gp_engine/alpha_pool.py —— 去重、剪枝、正交化的定钉测试（变异测试驱动）

来由：22 个变异点，首测击杀率 **59.1%**（存活 9）。

AlphaPool 决定**哪些因子被留在池子里**。它的两条拒绝规则（DSL 完全相同、
信号相关性超阈值）是整个 GP 里唯一防"一池子全是同一个因子的变体"的机制。
去重失效不会报错，只会让最终推荐的 top-5 实际上是同一个信号的五种写法 ——
分散化是假的，而组合层会按"五个独立因子"去配权重。

存活项：
  - `if mask.sum() < 10:` —— 相关性判定的最小样本数，放宽会跳过该比的对
  - `if len(self._entries) > self._max_size:` —— 剪枝的触发边界
  - `self._entries.sort(key=..., reverse=True)` 的 reverse
    —— 改成 False 会**保留最差的**、淘汰最好的
  - `if len(valid) < 2:` —— 正交化的样本下限
  - PCA 白化里的 `mean(keepdims=True)`、`X = mat_clean - mu`、
    `full_matrices=False`、`1.0/(s[:k] + 1e-9)` 的 `+`
    —— 白化算错时正交化输出仍是"一堆数"，没人看得出不对
  - `signal_vec: field(repr=False)` —— 整条信号序列进 repr

既有覆盖（unit/test_gp_alpha_pool）测的是"能加、能去重、top_k 有序"。
"""
from __future__ import annotations

import numpy as np
import pytest

from app.core.gp_engine.alpha_pool import AlphaPool, PoolEntry


T = 64


def _entry(dsl: str, fitness: float = 1.0, vec=None, gen: int = 0) -> PoolEntry:
    return PoolEntry(dsl=dsl, fitness=fitness, sharpe_is=1.0, sharpe_oos=0.8,
                     turnover=0.5, overfitting_score=0.1, generation=gen,
                     signal_vec=vec)


def _vec(seed: int, n: int = T) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=n)


# ===========================================================================
# A. 去重的两条规则
# ===========================================================================

class TestRejection:

    def test_identical_dsl_is_rejected(self):
        p = AlphaPool()
        assert p.add(_entry("rank(close)")) is True
        assert p.add(_entry("rank(close)", fitness=99.0)) is False, (
            "同一条 DSL 被重复加入 —— 去重失效")
        assert len(p.top_k(10)) == 1

    def test_highly_correlated_signals_are_rejected(self):
        p = AlphaPool(corr_threshold=0.70)
        v = _vec(0)
        assert p.add(_entry("a", vec=v)) is True
        assert p.add(_entry("b", vec=v * 2.0 + 1.0)) is False, (
            "完全线性相关的信号进了池子 —— 相关性去重失效")

    def test_uncorrelated_signals_are_accepted(self):
        p = AlphaPool(corr_threshold=0.70)
        assert p.add(_entry("a", vec=_vec(1))) is True
        assert p.add(_entry("b", vec=_vec(2))) is True, (
            "两条独立信号被误判为高相关")

    def test_negative_correlation_counts_too(self):
        """`abs(corr) >= threshold` —— 反号的同一个信号同样是重复。"""
        p = AlphaPool(corr_threshold=0.70)
        v = _vec(3)
        p.add(_entry("a", vec=v))
        assert p.add(_entry("b", vec=-v)) is False, (
            "反号信号进了池子 —— 相关性没有取绝对值")

    def test_entries_without_a_signal_skip_the_correlation_check(self):
        p = AlphaPool()
        assert p.add(_entry("a", vec=_vec(4))) is True
        assert p.add(_entry("b", vec=None)) is True, "没有信号向量的条目被误拒"

    def test_mismatched_lengths_are_skipped(self):
        p = AlphaPool()
        p.add(_entry("a", vec=_vec(5, n=T)))
        assert p.add(_entry("b", vec=_vec(6, n=T // 2))) is True, (
            "长度不同的信号被拿去比相关性了")

    def test_too_few_overlapping_points_skip_the_check(self):
        """
        `if mask.sum() < 10: continue` —— **严格小于 10**。
        放宽成 `<=` 会让恰好 10 个重叠点的一对被跳过：
        两条几乎完全重复的信号只要有效点刚好 10 个就能双双进池。
        """
        base = np.full(T, np.nan)
        base[:10] = np.arange(10.0)          # 恰好 10 个有效点
        other = base * 3.0 + 1.0             # 完全线性相关
        p = AlphaPool(corr_threshold=0.70)
        p.add(_entry("a", vec=base))
        assert p.add(_entry("b", vec=other)) is False, (
            "恰好 10 个重叠点的高相关信号没有被拒 —— `mask.sum() < 10` 被放宽")

    def test_nine_overlapping_points_are_not_enough(self):
        base = np.full(T, np.nan)
        base[:9] = np.arange(9.0)
        p = AlphaPool(corr_threshold=0.70)
        p.add(_entry("a", vec=base))
        assert p.add(_entry("b", vec=base * 3.0)) is True, (
            "只有 9 个重叠点却做了相关性判定 —— 那里的相关系数不可信")


# ===========================================================================
# B. 剪枝
# ===========================================================================

class TestPruning:

    def test_pool_never_exceeds_max_size(self):
        p = AlphaPool(max_size=5)
        for i in range(12):
            p.add(_entry(f"d{i}", fitness=float(i)))
        assert len(p.top_k(100)) == 5, "池子超出了 max_size"

    def test_exactly_max_size_is_not_pruned(self):
        """
        `if len(self._entries) > self._max_size:` —— **严格大于**。
        放宽成 `>=` 会在刚好装满时就剪一个，池子实际容量少一位。
        """
        p = AlphaPool(max_size=3)
        for i in range(3):
            p.add(_entry(f"d{i}", fitness=float(i)))
        assert len(p.top_k(100)) == 3, (
            "刚好装满就被剪枝了 —— `> max_size` 被放宽成了 `>=`")

    def test_pruning_keeps_the_best_not_the_worst(self):
        """
        `self._entries.sort(key=lambda e: e.fitness, reverse=True)` ——
        `reverse=True` 改成 False 会让排序变升序，随后 `[:max_size]` 保留的
        就是**最差的那批**：GP 越进化，池子里的因子越烂，而一切看着正常。
        """
        p = AlphaPool(max_size=3)
        for i in range(10):
            p.add(_entry(f"d{i}", fitness=float(i)))
        kept = sorted(e.fitness for e in p.top_k(100))
        assert kept == [7.0, 8.0, 9.0], (
            f"剪枝后留下的是 {kept}，应当是 fitness 最高的三个 —— "
            f"排序方向疑似反了")

    def test_pruned_entries_are_forgotten_so_they_can_return(self):
        """被剪掉的 DSL 要从 `_seen_dsls` 里移除，否则它再也无法重新入池。"""
        p = AlphaPool(max_size=2)
        for i in range(5):
            p.add(_entry(f"d{i}", fitness=float(i)))
        assert p.add(_entry("d0", fitness=100.0)) is True, (
            "被剪掉的 DSL 没有从去重集合里移除，永远无法重新入池")

    def test_top_k_is_sorted_descending(self):
        p = AlphaPool(max_size=10)
        for f in (0.3, 2.5, 1.0, -1.0):
            p.add(_entry(f"d{f}", fitness=f))
        vals = [e.fitness for e in p.top_k(10)]
        assert vals == sorted(vals, reverse=True), f"top_k 不是降序：{vals}"
        assert p.top_k(2) == p.top_k(10)[:2]


# ===========================================================================
# C. 正交化
# ===========================================================================

class TestOrthogonalisation:

    @staticmethod
    def _orth(pool):
        for name in ("get_orthogonal_signals", "orthogonalize", "orthogonalise",
                     "get_orthogonal"):
            fn = getattr(pool, name, None)
            if fn is not None:
                return fn()
        pytest.skip("AlphaPool 没有正交化入口")

    def test_fewer_than_two_entries_returns_the_originals(self):
        """
        `if len(valid) < 2: return 原样` —— **严格小于 2**。
        放宽成 `<=` 会让恰好两条时也直接返回原向量，正交化整个不生效，
        而调用方拿到的仍是一个看着正常的 dict。
        """
        p = AlphaPool()
        v = _vec(7)
        p.add(_entry("a", vec=v))
        out = self._orth(p)
        np.testing.assert_array_equal(out["a"], v, err_msg="单条时不该改动原向量")

    def test_two_entries_do_get_orthogonalised(self):
        p = AlphaPool(corr_threshold=0.99)
        a, b = _vec(8), _vec(9)
        p.add(_entry("a", vec=a))
        p.add(_entry("b", vec=b))
        out = self._orth(p)
        assert set(out) == {"a", "b"}
        assert not np.array_equal(out["a"], a), (
            "恰好两条时没有做正交化 —— `len(valid) < 2` 被放宽成了 `<= 2`")

    def test_orthogonalised_components_are_decorrelated(self):
        """
        PCA 白化的目的就是让**输出的各主成分之间**相关性消失。
        只断言"有限值"抓不到白化算错（上一版就是这么让四处变异活下来的）：
        `X = mat_clean - mu` 的 `-` 改成 `+`（不再去中心化）、
        `mean(keepdims=True)` 改掉（跨维错位相减）、
        `1/(s+1e-9)` 的 `+` 改成 `-`（缩放因子变号）、
        `full_matrices=False` 改成 True（Vt 形状变 (T,T)，切片语义全变），
        都会让输出的列间相关性显著偏离 0。

        构造：四条高度相关的信号（共享同一个 base）。白化之后，
        投影出来的各主成分列之间必须近似不相关。
        """
        p = AlphaPool(corr_threshold=0.99)
        rng = np.random.default_rng(10)
        base = rng.normal(size=T)
        for i in range(4):
            p.add(_entry(f"d{i}", vec=base * (i + 1) + rng.normal(size=T) * 0.5))
        out = self._orth(p)
        mat = np.stack(list(out.values()), axis=0)          # (n_alphas, k)
        assert np.all(np.isfinite(mat)), (
            "正交化输出里有 NaN/inf —— 白化的分母守卫失效")
        assert mat.shape[0] == 4

        # 输入本来高度相关
        raw = np.stack([e.signal_vec for e in p.top_k(10)], axis=0)
        raw_corr = np.corrcoef(raw)
        off_raw = np.abs(raw_corr[~np.eye(4, dtype=bool)]).mean()
        assert off_raw > 0.8, f"构造的输入相关性只有 {off_raw:.2f}，分不出白化效果"

        # 白化之后主成分列之间应当近似不相关。
        # 注意只能看**前 n-1 个**：n 条信号去中心化后秩最多 n-1，
        # 第 n 个主成分是退化的，它与别的列的"相关性"是数值噪声放大出来的
        # （实测 0.94），拿它做断言会假失败。
        k = min(mat.shape[1], mat.shape[0] - 1)
        assert k >= 2, "主成分不足 2 个，测不出相关性"
        comp = mat[:, :k]
        comp_corr = np.corrcoef(comp.T)
        off = np.abs(comp_corr[~np.eye(k, dtype=bool)])
        assert np.nanmax(off) < 0.2, (
            f"白化后前 {k} 个主成分之间仍有 {np.nanmax(off):.2f} 的相关性 —— "
            f"去中心化或缩放被改坏了")

    def test_nan_inputs_do_not_produce_nan_outputs(self):
        p = AlphaPool(corr_threshold=0.99)
        rng = np.random.default_rng(11)
        for i in range(3):
            v = rng.normal(size=T)
            v[:5] = np.nan
            p.add(_entry(f"d{i}", vec=v))
        out = self._orth(p)
        assert all(np.all(np.isfinite(v)) for v in out.values()), (
            "输入含 NaN 时正交化产出了非有限值 —— NaN 填充失效")


# ===========================================================================
# D. 序列化
# ===========================================================================

class TestSerialisation:

    def test_signal_vector_stays_out_of_the_repr(self):
        """
        `signal_vec: ... = field(default=None, repr=False)` —— 改成 True
        会把整条 T 维信号序列塞进 `repr(PoolEntry)`，而这个 repr 会进日志。
        """
        e = _entry("d", vec=np.arange(1000.0))
        r = repr(e)
        assert "signal_vec" not in r, f"信号向量进了 repr：{r[:200]}…"
        assert len(r) < 400, f"PoolEntry 的 repr 膨胀到了 {len(r)} 字符"

    def test_to_dict_rounds_and_omits_the_vector(self):
        d = _entry("d", fitness=1.23456789, vec=np.arange(100.0)).to_dict()
        assert "signal_vec" not in d, "to_dict 泄漏了信号向量"
        assert d["fitness"] == pytest.approx(1.2346, abs=1e-9)
        assert set(d) == {"dsl", "fitness", "sharpe_is", "sharpe_oos",
                          "turnover", "overfitting_score", "generation"}


# ===========================================================================
# E. 剪枝边界与白化数值（把上一版误判为"等价"的两处钉死）
# ===========================================================================

def test_pool_order_is_insertion_order_until_it_actually_overflows():
    """
    `if len(self._entries) > self._max_size:` —— **严格大于**。

    上一版把它写成了"等价变异"，理由是 `len == max_size` 时切片取回全部、
    `removed` 为空、`_seen_dsls` 不变、`top_k` 自己排序。那份证明漏了
    `all_entries()`：它返回的是 `list(self._entries)` 的**插入顺序**，
    而 `>=` 分支会就地 `sort(reverse=True)` 把它按 fitness 重排。
    `get_orthogonal_signals()` 的 `enumerate(valid)` 同样吃这个顺序。

    所以这不是等价变异，是漏测。这里按插入顺序钉死。
    """
    p = AlphaPool(max_size=3, corr_threshold=1.01)
    for dsl, fit in [("d_low", 0.1), ("d_high", 9.9), ("d_mid", 5.0)]:
        assert p.add(_entry(dsl, fitness=fit)) is True

    assert len(p) == 3, "恰好装满就被剪掉了条目"
    assert [e.dsl for e in p.all_entries()] == ["d_low", "d_high", "d_mid"], (
        "池子恰好装满（len == max_size）时条目顺序被重排了 —— "
        "剪枝的触发边界从 `>` 放宽到了 `>=`")
    # 真正溢出时才剪，且剪掉 fitness 最低的那条
    assert p.add(_entry("d_new", fitness=1.0)) is True
    assert len(p) == 3
    assert {e.dsl for e in p.all_entries()} == {"d_high", "d_mid", "d_new"}, (
        "溢出时淘汰的不是 fitness 最低的条目")
    assert "d_low" not in p._seen_dsls, "被淘汰的 DSL 没有从去重集合里摘掉"


def test_whitening_matches_the_reference_formula_to_full_precision():
    """
    `S_inv = np.diag(1.0 / (s[:k] + 1e-9))` —— 那个 `+1e-9` 是防零除的
    正向偏置。改成 `-1e-9` 时：奇异值 O(1) 的分量只差 ~2e-9 相对量，
    但 **n 条信号去中心化后秩最多 n-1**，最后一个奇异值 ≈ 1e-17，
    `1/(1e-17 - 1e-9)` 与 `1/(1e-17 + 1e-9)` 差一个**符号**（±1e9）。

    松断言（allclose 默认 rtol=1e-5）两种都放过，所以这里直接跟
    公式的参考实现逐位比，rtol=1e-12。
    """
    p = AlphaPool(corr_threshold=1.01)
    rng = np.random.default_rng(7)
    vecs = []
    for i in range(4):
        v = rng.normal(size=T)
        vecs.append(v)
        assert p.add(_entry(f"d{i}", vec=v)) is True

    out = p.get_orthogonal_signals()
    assert len(out) == 4

    # 参考实现：与源码逐字一致
    mat_clean = np.stack(vecs, axis=0)
    k = min(10, 4, mat_clean.shape[1])
    mu = mat_clean.mean(axis=0, keepdims=True)
    X = mat_clean - mu
    _, s, Vt = np.linalg.svd(X, full_matrices=False)
    expected = X @ (np.diag(1.0 / (s[:k] + 1e-9)) @ Vt[:k]).T

    for i, dsl in enumerate(f"d{j}" for j in range(4)):
        np.testing.assert_allclose(
            out[dsl], expected[i], rtol=1e-12, atol=0.0,
            err_msg=f"{dsl} 的白化结果与参考公式不符 —— "
                    f"S_inv 的 epsilon 符号或缩放被改动了")

    # 顺带把"最后一个奇异值确实退化"这一前提也钉住：
    # 它正是 -1e-9 会翻符号、而 +1e-9 不会的原因。
    assert s[-1] < 1e-9, (
        f"最小奇异值 {s[-1]:.3e} 不再退化 —— 上面那条 epsilon 符号论证的前提没了")


# ===========================================================================
# F. 存活变异的等价性证明
# ===========================================================================

PROVEN_EQUIVALENT = {
    "L170 `mat_clean.mean(axis=0, keepdims=True)` → `keepdims=False`":
        "`mu` 只被用在下一行的 `X = mat_clean - mu`。keepdims=True 给 (1, T)，"
        "False 给 (T,)。numpy 广播按**尾轴对齐**：(n, T) 减 (T,) 与减 (1, T) "
        "展开成完全相同的元素级运算，结果逐位相同。"
        "见 test_keepdims_makes_no_difference_to_the_broadcast。",
}

# `L172 svd(full_matrices=False) → True` 曾被我列为"等价变异"，理由是
# `Vt` 只以 `Vt[:k]` 被读、两种形式的前 min(n,T) 行是同一组右奇异向量。
# 数学上没错，但**结论错了**：n < T 时两种形式走不同的 LAPACK 驱动，
# 被读到的那几行有 ≤ 3.4e-16 的舍入抖动，而白化对最小奇异值的放大倍数是
# 1/(s_min + 1e-9) ≈ 1e9 —— 3e-16 的输入差被放大成 ~1e-7 的输出差，
# 远超 test_whitening_matches_the_reference_formula_to_full_precision
# 的 rtol=1e-12。实测该变异已被杀死。
#
# 顺带说明一个产品事实：最后一个主成分是退化的（秩 ≤ n-1），白化把它放大
# 约 1e9 倍，输出里那一列基本是数值噪声。见
# test_pca_whitening_decorrelates 里"只看前 n-1 个"的注释。


def test_keepdims_makes_no_difference_to_the_broadcast():
    rng = np.random.default_rng(3)
    for n, t in [(2, 5), (4, 64), (11, 7), (3, 3)]:
        mat = rng.normal(size=(n, t))
        a = mat - mat.mean(axis=0, keepdims=True)
        b = mat - mat.mean(axis=0, keepdims=False)
        assert a.shape == b.shape == (n, t)
        assert np.array_equal(a, b), (
            f"(n={n}, T={t}) 下 keepdims 改变了广播结果 —— L170 不再是等价变异")


def test_whitening_amplifies_the_degenerate_component_by_about_a_billion():
    """
    记录一个产品事实，也是 L172 会被杀死的原因：

    n 条信号去中心化后秩最多 n-1，最小奇异值 ≈ 1e-17，于是
    `S_inv = 1/(s + 1e-9)` 给最后一个分量约 **1e9 倍**的放大。
    任何 1e-16 级的数值抖动（比如 LAPACK 换个驱动）都会被放大成 1e-7。

    这条同时是失效告警：哪天 `+1e-9` 被换成更大的正则项、或者退化分量被
    显式丢掉，放大倍数会掉下来，这条会红着提醒等价性判断需要重做。
    """
    rng = np.random.default_rng(5)
    n, t = 4, 64
    X = rng.normal(size=(n, t))
    X = X - X.mean(axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(X, full_matrices=False)
    assert s[-1] < 1e-9, f"最小奇异值 {s[-1]:.3e} 不再退化"
    amp = 1.0 / (s[-1] + 1e-9)
    assert amp > 1e8, (
        f"退化分量的放大倍数只有 {amp:.3e} —— 白化的正则项被改动了")


def test_every_survivor_has_a_written_proof():
    assert len(PROVEN_EQUIVALENT) == 1
    for key, why in PROVEN_EQUIVALENT.items():
        assert len(why) >= 40, f"{key} 的等价性说明过于敷衍：{why!r}"
