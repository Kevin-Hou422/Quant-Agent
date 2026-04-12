# re-export: utils 语义位置入口
from app.core.alpha_engine.fast_ops import *  # noqa: F401, F403
from app.core.alpha_engine.fast_ops import (  # noqa: F401
    bn_ts_mean, bn_ts_std, ts_delta, bn_ts_rank, ts_decay_linear, ind_neutralize,
)
# 别名：统一为 ts_ 前缀方便外部使用
ts_mean = bn_ts_mean
ts_std  = bn_ts_std
ts_rank = bn_ts_rank
