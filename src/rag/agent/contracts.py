"""子 Agent 共用的结构化输出。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class EvidenceFinding(BaseModel):
    """一条可以被主 Agent 引用的发现。"""

    claim: str = Field(min_length=1, description="这条证据支持的结论")
    citation_indices: list[int] = Field(
        default_factory=list,
        description="工具返回的引用编号，例如 [1, 3]；没有来源就留空",
    )
    source: str = Field(default="", description="文档、业务工具或图关系的可读来源")


class EvidenceResult(BaseModel):
    """子 Agent 的统一返回。主 Agent 负责融合，不直接暴露子 Agent 的中间话术。"""

    findings: list[EvidenceFinding] = Field(default_factory=list)
    limitations: str = Field(default="", description="缺失、冲突或无法确认的内容")
    confidence: Literal["high", "medium", "low"] = "medium"
