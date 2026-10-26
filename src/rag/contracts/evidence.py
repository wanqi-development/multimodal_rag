"""统一证据账本：语料块与网络结果共用同一套引用编号。"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from rag.contracts.document import Citation


class EvidenceLedger(BaseModel):
    """一次回答的证据表。

    工具或装配层只负责登记证据，编号由这里统一发。语料块、图片描述块和网页结果
    共用同一个 `[n]` 空间，生成、前端与评测不再分别维护三套来源格式。
    """

    items: dict[int, Citation] = Field(default_factory=dict)

    @property
    def next_index(self) -> int:
        return max(self.items, default=0) + 1

    def issue(
        self,
        *,
        source_type: str = "corpus",
        chunk_id: str = "",
        source_uri: str = "",
        url: str = "",
        title: str = "",
        snippet: str = "",
        heading_path: str = "",
    ) -> int:
        """登记一条证据；同一来源复用同一个编号。"""
        identity = self._identity(
            source_type=source_type,
            chunk_id=chunk_id,
            source_uri=source_uri,
            url=url,
            heading_path=heading_path,
        )
        for number, item in self.items.items():
            if self._item_identity(item) == identity:
                return number

        number = self.next_index
        self.items[number] = Citation(
            index=number,
            source_type=source_type,
            chunk_id=chunk_id,
            source_uri=source_uri or url,
            url=url,
            title=title,
            snippet=snippet,
            heading_path=heading_path,
        )
        return number

    def issue_web(self, *, url: str, title: str = "", snippet: str = "") -> int:
        """登记一条网络证据。URL 是网络来源的稳定主键。"""
        return self.issue(
            source_type="web",
            source_uri=url or title,
            url=url,
            title=title,
            snippet=snippet,
        )

    def cited_in(self, answer: str) -> list[Citation]:
        """只返回答案里真的引用到的条目，按编号排序。"""
        used = {int(match) for match in re.findall(r"\[(\d+)\]", answer)}
        return [self.items[number] for number in sorted(used) if number in self.items]

    def records(self) -> list[Citation]:
        return [self.items[number] for number in sorted(self.items)]

    def to_payload(self) -> dict[str, Any]:
        """可落 store 的完整快照，保留来源类型、URL、标题与章节。"""
        return {"items": [item.model_dump(mode="json") for item in self.records()]}

    @classmethod
    def from_payload(cls, payload: dict[str, Any] | None) -> EvidenceLedger:
        """读取快照；旧版缺字段的记录按 Citation 默认值兼容。"""
        ledger = cls()
        for record in (payload or {}).get("items") or []:
            item = Citation.model_validate(record)
            ledger.items[item.index] = item
        return ledger

    @staticmethod
    def _identity(
        *,
        source_type: str,
        chunk_id: str,
        source_uri: str,
        url: str,
        heading_path: str,
    ) -> tuple[str, str, str]:
        primary = chunk_id or url or source_uri
        return source_type or "corpus", primary, heading_path

    @classmethod
    def _item_identity(cls, item: Citation) -> tuple[str, str, str]:
        return cls._identity(
            source_type=item.source_type,
            chunk_id=item.chunk_id,
            source_uri=item.source_uri,
            url=item.url,
            heading_path=item.heading_path,
        )
