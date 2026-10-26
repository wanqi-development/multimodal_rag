"""可观测性：日志（一类一个文件）与 trace（LangSmith）。

日志分三类，**共用同一套关联 id**（`request_id` / `run_id`），各落一个文件
（落点表见 `obs/setup.py`；2026-10-06 起不再进 PG）：

    interface.log   接口：一次 HTTP 请求一行
    agent.log       agent 运行：模型#n / 工具#n 各一行 + 一轮的汇总行
    channel.log     查询通道：一次查询 × 每一路通道一行（candidates / kept / latency / error）
    system.log      第三方库与其余系统日志（不含上面三类）

日志是观测，不碰网络也不碰数据库；handler 自己的故障由 stdlib 报出来（见 `log.py`）。
token 那类要给统计用的账在**会话级**（`persistence/chat.py` 的 `token_usage`），不从日志出。
"""

from rag.obs.log import (
    current_ids,
    current_source,
    new_id,
    one_line,
    read_tail,
    record_agent_llm,
    record_agent_run,
    record_agent_tool,
    record_channels,
    record_interface,
    row_of,
    set_content,
    set_ids,
    set_source,
    split_ids,
)
from rag.obs.pg import PgLogHandler
from rag.obs.setup import configure_logging, get_logger, reset_logging

__all__ = [
    "PgLogHandler",
    "configure_logging",
    "current_ids",
    "current_source",
    "get_logger",
    "new_id",
    "one_line",
    "read_tail",
    "record_agent_llm",
    "record_agent_run",
    "record_agent_tool",
    "record_channels",
    "record_interface",
    "row_of",
    "reset_logging",
    "set_content",
    "set_ids",
    "set_source",
    "split_ids",
]
