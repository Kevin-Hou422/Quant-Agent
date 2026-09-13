"""
顺序依赖体检插件（`-p shuffle_check`）

本仓**没有** pytest-randomly，顺序依赖此前没有任何机制在防。
2026-09 目录重组把 `unit/db/` 排到 `unit/trading_context/` 前面，
立刻暴露出一条"结论取决于前面跑过什么"的用例：
共享 session 级 DB 里残留的 active 策略配置，让边际准入分支被整个跳过。

用法：

    python -m pytest tests/ -q -p shuffle_check                    # 默认种子
    SHUFFLE_SEED=123 python -m pytest tests/ -q -p shuffle_check   # 指定种子
    SHUFFLE_DUMP=order.txt python -m pytest tests/ --co -q -p shuffle_check

种子会打印在头部；失败后用同一个种子 + `SHUFFLE_DUMP` 导出执行顺序，
即可二分定位到底是哪一条前置用例把状态弄脏了。
"""
import os
import random

SEED = int(os.environ.get("SHUFFLE_SEED", "20260913"))


def pytest_collection_modifyitems(session, config, items):
    random.Random(SEED).shuffle(items)
    print(f"\n[shuffle_check] seed={SEED} 打乱了 {len(items)} 个用例的执行顺序")

    dump = os.environ.get("SHUFFLE_DUMP")
    if dump:
        with open(dump, "w", encoding="utf-8") as fh:
            for it in items:
                fh.write(it.nodeid + "\n")
        print(f"[shuffle_check] 执行顺序已导出到 {dump}")
