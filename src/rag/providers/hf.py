"""HuggingFace 相关的环境变量处理 —— embedding 与 rerank 都要走这一步。

`huggingface_hub` 只认环境变量，读不到我们的 yaml（实测），所以模型加载前必须先落到
`os.environ`。这里放成公共函数是为了 embedding 与 rerank 两处**同一套口径**：
少写一次就会出现「embedding 走镜像、rerank 却去直连 huggingface.co」这种半死不活的状态
（直连不通，表现是首次加载卡住或超时，而不是一句能看懂的报错）。
"""

from __future__ import annotations

import os


def export_hf_env(*, endpoint: str | None, disable_xet: bool) -> None:
    """把 HF 镜像与 Xet 开关写进环境变量。

    配置值本身来自 ``${HF_ENDPOINT:-…}`` 插值，已经是「真实环境变量优先」，
    所以直接赋值不会盖掉用户显式设过的值。
    """
    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint
    if disable_xet:
        os.environ["HF_HUB_DISABLE_XET"] = "1"
