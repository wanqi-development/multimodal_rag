"""上传件的人工目录：**纯元数据 —— 磁盘上不建目录、文件一个都不挪**。

**为什么不落成真的目录树**（使用者 2026-10-08 定）：`doc_id` 是**相对路径**的 sha1
（``loaders.base.doc_id_for``），把 `第一章…-1a2b.pdf` 挪进 `Flink/` 之后 doc_id 立刻变 ——
`data/stages/<doc_id>` 的解析产物、Milvus 里那批块、`_index.json` 的记录会一起变孤儿。
另外上传目录顶层已经被「抓来的图片」占着（`*_files/` 子目录），
``LocalUploadStore.list()`` 也只扫顶层文件。

所以目录是**页面上的一层组织**：不碰 doc_id、不碰 Milvus、不碰 stages，**也不影响检索**
（要「只在某个目录里搜」是检索侧加过滤，另算一件事）。三条约束（使用者定）：

1. **最多三层** —— 建第 4 层、把子树移到会超过三层的位置，都当场拒掉；
2. **一份文档只在一个目录里**（不做多归属、不做标签）；
3. 同级目录不能重名。

落点：`data/uploads/_folders.json`，跟上传件放在一起 —— 以后换文件服务器
（MinIO / 共享盘）时这份元数据跟着文件走，不用另建库。**丢了当没有**：目录全没了、
文档全回「未分类」，与 `_index.json` 同一个立场（附属信息坏了不该让整页打不开）。

没做**并发写保护**（读-改-写，与 `_index.json` 一样）：现在只有一个进程在写；
多 worker 那天要靠文件服务器的原子操作或换库，见 ``uploads.py`` 的模块说明。
"""

from __future__ import annotations

import json
import os
import re
import secrets
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# 与 `_index.json` 并列放在上传目录里：`_` 开头 + `.json`，所以不会被当成上传件列出来
FOLDERS_NAME = "_folders.json"

# **最多三层**（使用者 2026-10-08 定）：再深在页面上也看不清了。
MAX_DEPTH = 3

# 目录名上限：够长到写清「Flink 官方文档 2.5」，又不至于把左栏撑爆。
MAX_NAME = 60

# 控制字符会把 json 与页面都搞乱，直接挡在门外
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


class FolderError(ValueError):
    """目录操作被拒（重名 / 超过三层 / 移进自己的子孙…）。**说清原因，别静默吞。**"""


@dataclass(frozen=True)
class Folder:
    """一个目录。`parent` 为 None 就是顶层。"""

    id: str
    name: str
    parent: str | None = None


class FolderStore:
    """`_folders.json` 的读写。一个上传目录一份。"""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ---- 读 ----

    def tree(self) -> list[Folder]:
        """全部目录，按**建的时候的顺序**（页面上再按名字排，这里不做展示的事）。"""
        return [
            Folder(id=str(item["id"]), name=str(item["name"]), parent=item.get("parent"))
            for item in self._load()["folders"]
        ]

    def members(self) -> dict[str, str]:
        """`doc_id → 目录 id`。没归属的文档不在里面（等价于「未分类」）。"""
        return dict(self._load()["members"])

    # ---- 写 ----

    def create(self, name: str, parent: str | None = None) -> Folder:
        payload = self._load()
        clean = self._clean_name(name)
        # 深度检查放在**建之前**：它认 parent 存不存在、也认「第 4 层」这道线
        self._child_depth(payload, parent)
        self._reject_same_name(payload, clean, parent, ignore=None)
        folder = Folder(id=_new_id(payload), name=clean, parent=parent)
        payload["folders"].append({"id": folder.id, "name": folder.name, "parent": parent})
        self._save(payload)
        return folder

    def update(
        self,
        folder_id: str,
        *,
        name: str | None = None,
        parent: str | None = None,
        move: bool = False,
    ) -> Folder:
        """改名、搬家，或两件一起做。

        `move=True` 才认 `parent`（包括「搬到顶层」的 None）—— 否则「只改名」会被当成
        「搬到顶层」，那是两个完全不同的动作。
        """
        payload = self._load()
        current = self._find(payload, folder_id)
        new_name = current["name"] if name is None else self._clean_name(name)
        new_parent = parent if move else current["parent"]
        if move:
            self._reject_cycle(payload, folder_id, new_parent)
            height = self._height(payload, folder_id)
            if self._child_depth(payload, new_parent) + height - 1 > MAX_DEPTH:
                raise FolderError(
                    f"最多 {MAX_DEPTH} 层：这个目录还有 {height - 1} 层子目录，"
                    f"搬过去就超过 {MAX_DEPTH} 层了"
                )
        self._reject_same_name(payload, new_name, new_parent, ignore=folder_id)
        current["name"] = new_name
        current["parent"] = new_parent
        self._save(payload)
        return Folder(id=str(current["id"]), name=new_name, parent=new_parent)

    def delete(self, folder_id: str) -> int:
        """删一个目录，返回**被移回「未分类」的文档数**。

        **里面有子目录就拒绝**（使用者定的）—— 悄悄连着子树一起删，等于一次点击改掉
        看不全的一大批归属。文档不进垃圾桶、只回未分类：目录是组织方式，不是存放位置。
        """
        payload = self._load()
        self._find(payload, folder_id)
        if any(item["parent"] == folder_id for item in payload["folders"]):
            raise FolderError("这个目录里还有子目录，先把它们删掉或搬走")
        payload["folders"] = [item for item in payload["folders"] if item["id"] != folder_id]
        moved = [key for key, value in payload["members"].items() if value == folder_id]
        for key in moved:
            payload["members"].pop(key, None)
        self._save(payload)
        return len(moved)

    def assign(self, doc_ids: Sequence[str], folder_id: str | None) -> int:
        """把几份文档放进一个目录；`folder_id=None` 就是移回「未分类」。返回改动条数。"""
        payload = self._load()
        if folder_id is not None:
            self._find(payload, folder_id)
        changed = 0
        for doc_id in doc_ids:
            if folder_id is None:
                changed += 1 if payload["members"].pop(doc_id, None) is not None else 0
            elif payload["members"].get(doc_id) != folder_id:
                payload["members"][doc_id] = folder_id
                changed += 1
        self._save(payload)
        return changed

    def forget(self, doc_ids: Sequence[str]) -> None:
        """文档被删掉时跟着清归属（否则 `_folders.json` 里会堆死记录）。"""
        payload = self._load()
        if not any(doc_id in payload["members"] for doc_id in doc_ids):
            return
        for doc_id in doc_ids:
            payload["members"].pop(doc_id, None)
        self._save(payload)

    # ---- 内部 ----

    def _load(self) -> dict[str, Any]:
        """读。**缺失 / 坏了 / 形状不对都当空表**：目录丢了不该让整页打不开。"""
        empty: dict[str, Any] = {"folders": [], "members": {}}
        path = self.root / FOLDERS_NAME
        if not path.is_file():
            return empty
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return empty
        if not isinstance(payload, dict):
            return empty
        folders = [
            {
                "id": str(item["id"]),
                "name": str(item["name"]),
                "parent": item.get("parent"),
            }
            for item in payload.get("folders") or []
            if isinstance(item, dict) and item.get("id") and item.get("name")
        ]
        members = {
            str(key): str(value)
            for key, value in (payload.get("members") or {}).items()
            if isinstance(value, str)
        }
        return {"folders": folders, "members": members}

    def _save(self, payload: dict[str, Any]) -> None:
        path = self.root / FOLDERS_NAME
        tmp = path.with_suffix(path.suffix + ".part")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)

    def _find(self, payload: dict[str, Any], folder_id: str | None) -> dict[str, Any]:
        for item in payload["folders"]:
            if item["id"] == folder_id:
                return item
        raise FolderError(f"没有这个目录：{folder_id!r}")

    def _clean_name(self, name: str) -> str:
        clean = str(name).strip()
        if not clean:
            raise FolderError("目录名不能是空的")
        if len(clean) > MAX_NAME:
            raise FolderError(f"目录名最多 {MAX_NAME} 个字")
        if _CONTROL.search(clean):
            raise FolderError("目录名里有控制字符")
        return clean

    def _reject_same_name(
        self,
        payload: dict[str, Any],
        name: str,
        parent: str | None,
        *,
        ignore: str | None,
    ) -> None:
        for item in payload["folders"]:
            if item["id"] == ignore or item["parent"] != parent:
                continue
            if str(item["name"]).strip() == name:
                where = "顶层" if parent is None else f"「{self._name_of(payload, parent)}」里"
                raise FolderError(f"{where}已经有一个叫「{name}」的目录了")

    def _name_of(self, payload: dict[str, Any], folder_id: str) -> str:
        for item in payload["folders"]:
            if item["id"] == folder_id:
                return str(item["name"])
        return folder_id

    def _child_depth(self, payload: dict[str, Any], parent: str | None) -> int:
        """把新目录建在 `parent` 下面，它会是第几层（顶层 = 1）。超过三层当场拒。"""
        if parent is None:
            return 1
        depth = self._depth(payload, parent) + 1
        if depth > MAX_DEPTH:
            raise FolderError(
                f"最多 {MAX_DEPTH} 层：「{self._name_of(payload, parent)}」已经在第 "
                f"{depth - 1} 层了，不能再往下建"
            )
        return depth

    def _depth(self, payload: dict[str, Any], folder_id: str) -> int:
        """这个目录在第几层。**带上限的向上走**：文件被手改坏了成环也不会转不出来。"""
        depth = 1
        parent = self._find(payload, folder_id)["parent"]
        while parent is not None and depth <= MAX_DEPTH + 1:
            depth += 1
            parent = self._find(payload, parent)["parent"]
        return depth

    def _height(self, payload: dict[str, Any], folder_id: str) -> int:
        """这个目录往下还有几层（自己算 1 层）。带上限，防止坏文件里的环。"""
        height = 1
        frontier = [folder_id]
        while frontier and height <= MAX_DEPTH + 1:
            frontier = [
                str(item["id"]) for item in payload["folders"] if item["parent"] in frontier
            ]
            if frontier:
                height += 1
        return height

    def _reject_cycle(self, payload: dict[str, Any], folder_id: str, parent: str | None) -> None:
        """搬到自己的子孙下面 → 那棵子树会从树上掉下来。"""
        seen = parent
        hops = 0
        while seen is not None and hops <= MAX_DEPTH + 1:
            if seen == folder_id:
                raise FolderError("不能把目录搬到它自己（或它的子目录）里面")
            seen = self._find(payload, seen)["parent"]
            hops += 1


def _new_id(payload: dict[str, Any]) -> str:
    """目录 id：随机 8 位十六进制。撞了就再抽（树很小，几乎不会走到）。"""
    taken = {str(item["id"]) for item in payload["folders"]}
    for _ in range(100):
        candidate = f"f{secrets.token_hex(4)}"
        if candidate not in taken:
            return candidate
    raise FolderError("目录 id 抽不出来（树太大了？）")


def build_folder_store(settings: Any) -> FolderStore:
    """按配置建目录表。**与上传件同一个落点**（`ingest.uploads.dir`）。"""
    return FolderStore(settings.ingest.uploads.dir)
