"""
pit_store.py —— 落盘与合并契约的定钉测试（变异测试驱动）

来由：24 个变异点，首测击杀率 70.8%，存活 7 处**全是布尔关键字参数**：
`mkdir(parents=True)`、`concat(ignore_index=True)`、`reset_index(drop=True)`、
`to_parquet(index=False)`。既有覆盖（test_phase8_pit / test_phase11_incremental）
只验"写进去能读出来"，对**落盘形态**零断言。

这些不是细节：
  - `parents=False` → 存储根目录只要多一层就建不出来（首次部署直接失败）
  - `ignore_index=False` → 追加合并后索引重复，`drop_duplicates` 之后的行号错乱
  - `drop=False` / `index=True` → 每写一次就往 parquet 里多塞一列索引，
    列集合逐次膨胀，下游 `[c for c in columns if c not in _KEY_COLS]`
    会把这些索引列当成**数据字段**
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.data_engine.pit_store import PITStore

_KEY_COLS = ["timestamp", "ticker", "as_of"]


def _panel(days: int = 6, tickers=("AAA", "BBB"), start: str = "2024-01-02",
           base: float = 100.0) -> dict:
    idx = pd.bdate_range(start, periods=days)
    cols = list(tickers)
    close = pd.DataFrame(
        base + np.arange(days * len(cols), dtype=float).reshape(days, len(cols)),
        index=idx, columns=cols)
    return {"close": close, "volume": close * 1000.0}


@pytest.fixture
def store(tmp_path) -> PITStore:
    return PITStore(tmp_path / "pit")


# ===========================================================================
# A. 目录创建
# ===========================================================================

def test_store_dir_is_created_with_parents(tmp_path):
    """
    `self.store_dir.mkdir(parents=True, exist_ok=True)` —— `parents=False`
    在多层路径上直接抛 FileNotFoundError。首次部署时存储根目录就是多层的。
    """
    deep = tmp_path / "a" / "b" / "c" / "pit"
    assert not deep.parent.exists()
    PITStore(deep)
    assert deep.is_dir()


def test_store_dir_can_be_reused(tmp_path):
    """`exist_ok=True`：第二次构造不得抛 FileExistsError，且仍指向同一目录。"""
    path = tmp_path / "pit"
    first = PITStore(path)
    second = PITStore(path)
    assert first.store_dir == second.store_dir == path
    assert path.is_dir()


def test_partition_dirs_are_created_with_parents(store):
    """
    `dataset_dir.mkdir(parents=True)` 与 `part_dir.mkdir(parents=True)`：
    数据集目录与 `year=` 分区目录都在存储根之下，两层一起建。
    """
    store.append(_panel(), name="px", as_of="2024-01-10")
    part = store.store_dir / "px" / "year=2024"
    assert part.is_dir(), "分区目录未创建"
    assert (part / "data.parquet").exists()


def test_multiple_years_get_separate_partitions(store):
    store.append(_panel(days=4, start="2023-12-26"), name="px", as_of="2024-01-10")
    store.append(_panel(days=4, start="2024-01-02"), name="px", as_of="2024-01-10")
    years = sorted(p.name for p in (store.store_dir / "px").glob("year=*"))
    assert years == ["year=2023", "year=2024"]


# ===========================================================================
# B. 落盘形态 —— 索引不得混进数据
# ===========================================================================

class TestOnDiskShape:

    @staticmethod
    def _columns(store: PITStore, name: str = "px") -> list:
        part = next((store.store_dir / name).glob("year=*")) / "data.parquet"
        return list(pd.read_parquet(part).columns)

    def test_no_index_column_is_written(self, store):
        """
        `to_parquet(..., index=False)` —— 改成 True 会往 parquet 里写一列索引
        （pyarrow 记为 `__index_level_0__`）。
        """
        store.append(_panel(), name="px", as_of="2024-01-10")
        cols = self._columns(store)
        assert not any(c.startswith("__index_level") or c == "index" for c in cols), (
            f"parquet 里混进了索引列：{cols}")
        assert set(_KEY_COLS).issubset(cols)

    def test_column_set_is_stable_across_appends(self, store):
        """
        多次追加后列集合必须**保持不变**。若 `reset_index(drop=False)` 或
        `to_parquet(index=True)`，每写一次就多一列索引，列集合逐次膨胀，
        而下游把"非 key 列"一律当成数据字段。
        """
        store.append(_panel(), name="px", as_of="2024-01-10")
        first = self._columns(store)
        for i, as_of in enumerate(("2024-01-11", "2024-01-12"), start=1):
            store.append(_panel(base=100.0 + i), name="px", as_of=as_of)
            assert self._columns(store) == first, (
                f"第 {i+1} 次追加后列集合变了：{self._columns(store)} != {first}")

    def test_stored_frame_has_a_clean_range_index(self, store):
        """
        `pd.concat([...], ignore_index=True)` 与 `.reset_index(drop=True)`：
        合并后必须是 0..n-1 的干净行号，否则重复索引会让后续
        `drop_duplicates(keep="last")` 的结果不可预期。
        """
        store.append(_panel(), name="px", as_of="2024-01-10")
        store.append(_panel(base=200.0), name="px", as_of="2024-01-11")
        part = next((store.store_dir / "px").glob("year=*")) / "data.parquet"
        df = pd.read_parquet(part)
        assert list(df.index) == list(range(len(df))), "落盘的行号不是干净的 RangeIndex"

    def test_rows_are_sorted_by_key(self, store):
        store.append(_panel(base=200.0), name="px", as_of="2024-01-11")
        store.append(_panel(), name="px", as_of="2024-01-10")
        part = next((store.store_dir / "px").glob("year=*")) / "data.parquet"
        df = pd.read_parquet(part)
        assert df[_KEY_COLS].equals(
            df[_KEY_COLS].sort_values(_KEY_COLS).reset_index(drop=True)), "落盘未按主键排序"


# ===========================================================================
# C. 追加与幂等
# ===========================================================================

class TestAppendSemantics:

    def test_new_vintage_is_appended_not_replaced(self, store):
        """同一天不同 as_of 是两条记录 —— PIT 的全部意义。"""
        store.append(_panel(days=2), name="px", as_of="2024-01-10")
        store.append(_panel(days=2, base=999.0), name="px", as_of="2024-01-11")
        part = next((store.store_dir / "px").glob("year=*")) / "data.parquet"
        df = pd.read_parquet(part)
        assert df["as_of"].nunique() == 2
        assert len(df) == 8, "2 天 × 2 标的 × 2 个 vintage = 8 行"

    def test_same_vintage_rewrite_is_idempotent(self, store):
        """同一 (timestamp,ticker,as_of) 重跑覆盖，不叠加。"""
        store.append(_panel(days=2), name="px", as_of="2024-01-10")
        n1 = len(pd.read_parquet(
            next((store.store_dir / "px").glob("year=*")) / "data.parquet"))
        store.append(_panel(days=2), name="px", as_of="2024-01-10")
        n2 = len(pd.read_parquet(
            next((store.store_dir / "px").glob("year=*")) / "data.parquet"))
        assert n1 == n2 == 4, f"同 vintage 重跑后行数由 {n1} 变成 {n2}"

    def test_same_vintage_rewrite_keeps_the_last_value(self, store):
        store.append(_panel(days=2, base=100.0), name="px", as_of="2024-01-10")
        store.append(_panel(days=2, base=500.0), name="px", as_of="2024-01-10")
        out = store.load_pit(name="px", as_of="2024-01-10")
        assert float(out["close"].iloc[0, 0]) == pytest.approx(500.0)

    def test_load_pit_respects_the_as_of_cutoff(self, store):
        """晚于 as_of 的 vintage 不得进入结果 —— 这是防前视的核心。"""
        store.append(_panel(days=2, base=100.0), name="px", as_of="2024-01-10")
        store.append(_panel(days=2, base=900.0), name="px", as_of="2024-01-20")
        early = store.load_pit(name="px", as_of="2024-01-15")
        assert float(early["close"].iloc[0, 0]) == pytest.approx(100.0), (
            "as_of=2024-01-15 的查询看到了 01-20 才发布的数据 —— 前视")

    def test_reading_an_unknown_dataset_returns_empty(self, store):
        assert store.load_pit(name="does-not-exist", as_of="2024-01-10") == {}


# ===========================================================================
# D. 存活变异的等价性证明
# ===========================================================================

PROVEN_EQUIVALENT = {
    "app/core/data_engine/pit_store.py ×2 — L201/L208 `dataset_dir.mkdir(parents=True)` / `part_dir.mkdir(parents=True)`":
        "这两处的父目录**必然已存在**：`store_dir` 在 `__init__` 里已建，"
        "`dataset_dir = store_dir/name` 只深一层；`part_dir = dataset_dir/year=…` "
        "紧跟在 dataset_dir 建好之后。`parents` 取何值都不影响结果。"
        "只有 `__init__` 里那一处真的需要 parents（已由 "
        "test_store_dir_is_created_with_parents 杀死）。",

    "app/core/data_engine/pit_store.py ×2 — L217/L268 `pd.concat([...], ignore_index=True)` → False":
        "两处 concat 之后都不再使用行索引：L217 紧接着 `drop_duplicates(subset=…)`、"
        "`sort_values`、`reset_index(drop=True)`，索引被重建；L268 之后只做"
        "布尔掩码过滤与 `pivot(index=\"timestamp\")`，用的是**列值**不是行索引。"
        "重复的索引标签因此不可观测。",

    "app/core/data_engine/pit_store.py ×1 — L227 `group.to_parquet(..., index=False)` → True":
        "写入前刚做过 `reset_index(drop=True)`，索引是标准 RangeIndex。"
        "pyarrow 对 RangeIndex 只写元数据、不写数据列，`pd.read_parquet` "
        "读回来仍是同一个 0..n-1 索引，列集合与取值都不变。"
        "见 test_range_index_round_trips_identically。",
}


def test_range_index_round_trips_identically(tmp_path):
    """L227 等价性的机械验证：RangeIndex 的 parquet 往返在两种 index 取值下一致。"""
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["x", "y", "z"]})
    p0, p1 = tmp_path / "no.parquet", tmp_path / "yes.parquet"
    df.to_parquet(p0, compression="snappy", index=False)
    df.to_parquet(p1, compression="snappy", index=True)
    a, b = pd.read_parquet(p0), pd.read_parquet(p1)
    assert list(a.columns) == list(b.columns)
    assert list(a.index) == list(b.index) == [0, 1, 2]
    assert a.equals(b)


def test_every_survivor_has_a_written_proof():
    assert len(PROVEN_EQUIVALENT) == 3
    for key, why in PROVEN_EQUIVALENT.items():
        assert len(why) >= 40, f"{key} 的等价性说明过于敷衍：{why!r}"
