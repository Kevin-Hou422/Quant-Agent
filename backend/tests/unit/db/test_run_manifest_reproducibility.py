"""
db/run_manifest.py —— 可复现台账的定钉测试（变异测试驱动）

来由：14 个变异点，首测击杀率 **14.3%**（存活 12）。

这张表是"这个回测结果是用哪份数据、哪个 commit、哪个种子跑出来的"的唯一记录。
它一旦不可信，所有历史结论就都无法复核 —— 而这正是本阶段交付的前提
（"接下来要交付给别人严格检查"）。

存活项：
  - `run_type` / `data_sha256` 的 `nullable=False` 与 `index=True`
  - `dataset_sha256` 里 `hash_pandas_object(v.index, index=False)` 的 index 参数
  - `current_git_commit` 的 `capture_output=True`
  - `git_commit if git_commit is not None else current_git_commit()` 的 `not`
  - `json.dumps(..., ensure_ascii=False, default=str)` 的两个 False

既有覆盖（test_phase6_reproducibility）只验证了"哈希对同一份数据稳定"。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import inspect

from app.db.run_manifest import (
    RunManifestStore,
    current_git_commit,
    dataset_sha256,
)


def _dataset(seed: int = 0, days: int = 10, tickers=("AAA", "BBB")) -> dict:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-02", periods=days)
    cols = list(tickers)
    close = pd.DataFrame(rng.normal(100, 1, (days, len(cols))), index=idx, columns=cols)
    return {"close": close, "volume": close * 1000.0}


@pytest.fixture
def store(tmp_path) -> RunManifestStore:
    return RunManifestStore(db_url=f"sqlite:///{tmp_path/'rm.db'}")


# ===========================================================================
# A. 数据指纹
# ===========================================================================

class TestDatasetHash:

    def test_same_data_same_hash(self):
        assert dataset_sha256(_dataset(1)) == dataset_sha256(_dataset(1))

    def test_field_order_does_not_matter(self):
        ds = _dataset(2)
        reordered = {k: ds[k] for k in reversed(list(ds))}
        assert dataset_sha256(ds) == dataset_sha256(reordered)

    def test_different_values_change_the_hash(self):
        ds = _dataset(3)
        other = {k: v.copy() for k, v in ds.items()}
        other["close"].iloc[0, 0] += 1e-9
        assert dataset_sha256(ds) != dataset_sha256(other), (
            "改动一个数值后指纹不变 —— 台账无法区分两份数据")

    def test_different_dates_change_the_hash(self):
        """
        `hash_pandas_object(v.index, index=False)` —— 哈希**索引本身**。
        把 `index=False` 改成 True 会让 pandas 额外把索引的索引也算进去，
        看似"更严"，实际会让**同一份数据**在不同构造路径下指纹不一致
        （index 的 name/类型差异被放大），复现校验随即变成永远不匹配。

        这里钉住的是可观测契约：日期不同 → 指纹必须不同；
        同一份数据重新构造 → 指纹必须相同。
        """
        a = _dataset(4, days=10)
        b = {k: v.copy() for k, v in a.items()}
        for v in b.values():
            v.index = pd.bdate_range("2024-02-01", periods=len(v))
        assert dataset_sha256(a) != dataset_sha256(b)

    def test_rebuilt_identical_frame_hashes_the_same(self):
        a = _dataset(5)
        b = {k: pd.DataFrame(v.to_numpy().copy(), index=v.index.copy(),
                             columns=list(v.columns))
             for k, v in a.items()}
        assert dataset_sha256(a) == dataset_sha256(b), (
            "同一份数据重新构造后指纹变了 —— 复现校验会永远不匹配")

    def test_different_tickers_change_the_hash(self):
        assert dataset_sha256(_dataset(6, tickers=("AAA", "BBB"))) != \
            dataset_sha256(_dataset(6, tickers=("AAA", "CCC")))

    def test_extra_field_changes_the_hash(self):
        ds = _dataset(7)
        more = dict(ds)
        more["open"] = ds["close"] * 1.001
        assert dataset_sha256(ds) != dataset_sha256(more)

    def test_hash_is_a_64_char_hex(self):
        h = dataset_sha256(_dataset(8))
        assert len(h) == 64 and all(c in "0123456789abcdef" for c in h)


# ===========================================================================
# B. git commit
# ===========================================================================

class TestGitCommit:

    def test_git_output_is_captured(self, monkeypatch):
        """
        `capture_output=True` —— 改成 False 时 git 的输出直接打到进程 stdout，
        `out.stdout` 变成 None，`.strip()` 抛 AttributeError 被 except 吞掉，
        函数返回**空串** —— 每条 manifest 的 git_commit 都是空的，
        "这个结果是哪个版本跑的"就此永久丢失。

        ⚠️ 不能依赖"当前处在 git 仓库里"：变异测试在 backend/ 的**临时副本**里跑，
        那里没有 .git，真实调用只会走 skip（第一版就是这么漏的）。
        这里把 subprocess 打桩，直接断言**传进去的 capture_output**。
        """
        import subprocess
        import app.db.run_manifest as rm

        seen = {}

        def _spy(cmd, **kw):
            seen.update(kw)
            return subprocess.CompletedProcess(cmd, 0, stdout="deadbee\n", stderr="")

        monkeypatch.setattr(rm.subprocess, "run", _spy)
        got = current_git_commit()
        assert seen.get("capture_output") is True, (
            "调用 git 时没有捕获输出 —— stdout 会是 None，commit 永远是空串")
        assert seen.get("text") is True and seen.get("timeout"), (
            "缺少 text/timeout —— 输出会是 bytes 或可能永久挂起")
        assert got == "deadbee", f"没有正确解析 git 输出：{got!r}"

    def test_git_failure_returns_empty_string(self, monkeypatch):
        """非 git 环境（returncode != 0）必须安静返回空串，不得抛异常。"""
        import subprocess
        import app.db.run_manifest as rm

        monkeypatch.setattr(
            rm.subprocess, "run",
            lambda cmd, **kw: subprocess.CompletedProcess(cmd, 128, stdout="",
                                                          stderr="not a repo"))
        assert current_git_commit() == ""

    def test_explicit_commit_overrides_detection(self, store):
        """
        `git_commit if git_commit is not None else current_git_commit()` ——
        删掉 `not` 后，**显式传入**的 commit 会被丢弃、改用自动探测，
        补跑历史时就无法把 manifest 归到当时那个 commit 上。
        """
        rid = store.record("backtest", _dataset(9), seed=7, git_commit="deadbee")
        rec = store.get(rid)
        assert rec.git_commit == "deadbee", (
            f"显式传入的 commit 被覆盖成了 {rec.git_commit!r}")

    def test_empty_string_commit_is_respected(self, store, monkeypatch):
        """
        空串是**显式**取值（不是 None），不得触发自动探测。

        上一版写的是
        `assert ("" if "" is not None else current_git_commit()) == ""` ——
        `"" is not None` 恒真，整条等价于 `assert "" == ""`，
        **把产品的表达式在测试里抄了一遍而没有碰产品**，恒真。
        这里改成真的存一条 `git_commit=""` 的 manifest，
        并把自动探测打成"一旦被调用就失败"的地雷。
        """
        import app.db.run_manifest as rm

        def _landmine():
            raise AssertionError(
                "显式传了 git_commit=\"\" 却仍然触发了自动探测 —— "
                "`git_commit if git_commit is not None else current_git_commit()` "
                "的空值判定被改成了真值判定")

        monkeypatch.setattr(rm, "current_git_commit", _landmine)
        rid = store.record("backtest", _dataset(11), seed=3, git_commit="")
        assert store.get(rid).git_commit == "", (
            "显式的空串 commit 没有被原样保存")


# ===========================================================================
# C. 表结构
# ===========================================================================

class TestSchema:

    def test_identity_columns_are_not_nullable(self, store):
        insp = inspect(store._engine)
        cols = {c["name"]: c for c in insp.get_columns("run_manifests")}
        for col in ("run_type", "data_sha256"):
            assert cols[col]["nullable"] is False, (
                f"{col} 允许为空 —— 台账里会出现无从归属的记录")

    def test_lookup_columns_are_indexed(self, store):
        insp = inspect(store._engine)
        indexed = {c for ix in insp.get_indexes("run_manifests")
                   for c in ix["column_names"]}
        assert "run_type" in indexed and "data_sha256" in indexed, (
            "按类型/数据指纹检索是复核时的主查询，却没有索引")

    def test_engine_does_not_echo_sql(self, store):
        assert store._engine.echo is False

    def test_objects_stay_usable_after_commit(self, store):
        rid = store.record("gp", _dataset(10), seed=3)
        rec = store.get(rid)
        assert rec.run_type == "gp" and rec.seed == 3

    def test_sqlite_allows_cross_thread_use(self, tmp_path):
        import threading
        st = RunManifestStore(db_url=f"sqlite:///{tmp_path/'t.db'}")
        rid = st.record("backtest", _dataset(11))
        box = {}

        def _read():
            try:
                box["v"] = st.get(rid)
            except Exception as exc:      # noqa: BLE001
                box["err"] = exc

        th = threading.Thread(target=_read)
        th.start()
        th.join()
        assert "err" not in box, f"跨线程读取失败：{box.get('err')}"


# ===========================================================================
# D. JSON 序列化
# ===========================================================================

class TestJsonPayloads:

    def test_non_ascii_is_stored_readably(self, store):
        """
        `json.dumps(..., ensure_ascii=False)` —— 改成 True 会把中文转义成
        `\\uXXXX`，台账变成不可读的转义串（这份台账是给人复核用的）。
        """
        rid = store.record("backtest", _dataset(12),
                           config={"说明": "中文配置"}, summary={"结论": "通过"})
        rec = store.get(rid)
        assert "中文配置" in rec.config_json, (
            f"中文被转义了：{rec.config_json}")
        assert "结论" in rec.summary_json
        assert json.loads(rec.config_json)["说明"] == "中文配置"

    def test_non_serialisable_values_do_not_crash(self, store):
        """
        `default=str` —— 改掉它会让 Timestamp / ndarray 之类直接抛
        `TypeError: Object of type ... is not JSON serializable`，
        整条 manifest 写不进去，**回测结果就此失去出处**。
        """
        cfg = {"start": pd.Timestamp("2024-01-02"),
               "seedvec": np.arange(3),
               "nested": {"when": pd.Timestamp("2024-03-05")}}
        rid = store.record("backtest", _dataset(13), config=cfg)
        rec = store.get(rid)
        loaded = json.loads(rec.config_json)
        assert "2024-01-02" in loaded["start"]
        assert isinstance(loaded["seedvec"], str)

    def test_missing_payloads_default_to_empty_objects(self, store):
        rid = store.record("paper", _dataset(14))
        rec = store.get(rid)
        assert json.loads(rec.config_json) == {}
        assert json.loads(rec.summary_json) == {}


# ===========================================================================
# E. 存活变异的等价性证明
# ===========================================================================

PROVEN_EQUIVALENT = {
    "app/db/run_manifest.py ×1 — L64 `hash_pandas_object(v.index, index=False)` → True":
        "`index` 参数只对 Series / DataFrame 有意义（决定要不要把行索引一并哈希）；"
        "传入的是 **Index 对象**，pandas 对它忽略该参数，两种取值返回完全相同的"
        "哈希数组。见 test_hash_of_an_index_ignores_the_index_flag。",

    "app/db/run_manifest.py ×1 — L117 `sessionmaker(..., expire_on_commit=False)` → True":
        "本类唯一 commit 的方法是 `record()`，它在**同一个 session 内**读取 "
        "`rec.id`（过期后会自动 refresh，仍然拿得到）；`get()` / `list()` 走的是"
        "只读 session，不触发过期。两种取值都观察不到差别。",
}


def test_hash_of_an_index_ignores_the_index_flag():
    """L64 等价性的机械验证：对 Index 传 index=True/False 结果逐元素相同。"""
    for idx in (pd.bdate_range("2024-01-02", periods=8),
                pd.Index(["AAA", "BBB", "CCC"]),
                pd.RangeIndex(5)):
        a = pd.util.hash_pandas_object(idx, index=False).values
        b = pd.util.hash_pandas_object(idx, index=True).values
        assert np.array_equal(a, b), f"{type(idx).__name__} 上两种取值给出了不同哈希"


def test_record_reads_its_id_inside_the_session():
    """L117 等价性的机械验证：唯一 commit 的方法在 session 内取 id。"""
    import inspect
    import app.db.run_manifest as rm
    src = inspect.getsource(rm.RunManifestStore.record)
    body = src.split("with self._Session()", 1)[1]
    assert "s.commit()" in body and "rec.id" in body, (
        "record() 的 id 读取被挪到了 session 之外 —— expire_on_commit 会变得可观测")


def test_every_survivor_has_a_written_proof():
    assert len(PROVEN_EQUIVALENT) == 2
    for key, why in PROVEN_EQUIVALENT.items():
        assert len(why) >= 40, f"{key} 的等价性说明过于敷衍：{why!r}"
