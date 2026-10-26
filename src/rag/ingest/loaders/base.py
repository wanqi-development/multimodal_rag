"""loader 的公共部分：注册表、后缀→mime、编码嗅探、文档 id。

loader 是「读文件」这一步，不是 contracts 里的 Parser。OCR 兜底本来是按 Parser
设计的（见 contracts/protocols.py 里 Parser 的注释），但 Parser 只拿得到 Document、
拿不到原始文件，而把整份 PDF 交给产线需要文件本身。所以 OCR 落在了 loader 这一层的
分支上（见 pdf.py / image.py），Parser 协议依旧空着。
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel, Field

from rag.config.schema import IngestConfig
from rag.contracts import Document, ImageAsset, Position
from rag.ingest.assets import AssetStore, ExtractedText, extract_images, line_positions
from rag.ingest.cleaning import clean_ocr_markdown
from rag.ingest.headings import promote_numbered_headings

if TYPE_CHECKING:
    from rag.ingest.ocr_client import OcrEngine, OcrPage

_MIME_BY_SUFFIX: dict[str, str] = {
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".txt": "text/plain",
    ".log": "text/plain",
    ".rst": "text/x-rst",
    ".csv": "text/csv",
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".xlsm": "application/vnd.ms-excel.sheet.macroEnabled.12",
    ".html": "text/html",
    ".htm": "text/html",
    # 图片全靠 OCR 转文字，没有文字层可抽（见 image.py）。
    # 不加 .webp：官方只声明「图像文件（含 TIFF）」，webp 没验过，不猜。
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".bmp": "image/bmp",
}

# 上面那张表的反向视图（Content-Type → 后缀），由 `suffix_for_mime` 用。
# 一对多时 `setdefault` 保留**先出现**的那个后缀。
_SUFFIX_BY_MIME: dict[str, str] = {}
for _suffix, _mime in _MIME_BY_SUFFIX.items():
    _SUFFIX_BY_MIME.setdefault(_mime, _suffix)


class Loader(Protocol):
    """把一个文件读成 Document。"""

    name: str
    suffixes: tuple[str, ...]

    def load(self, path: Path, *, root: Path | None = None) -> Document: ...


# 后缀 → loader 类（不是实例）。实例化交给 build_loader —— 见那里的注释。
LOADERS: dict[str, type] = {}


def register_loader(*suffixes: str) -> Callable[[type], type]:
    """把实现登记到它声明的后缀上。

    存**类**而不是实例：PDF / 图片这两种 loader 要收配置和一个 OCR 客户端，
    没法在 import 时无参构造。这样也与 chunker / embedder / retriever 的写法一致了。

    后缀冲突直接报错，不静默覆盖 —— 否则会出现「加了新格式，老格式悄悄改了行为」
    这种难查的问题。
    """

    def deco(cls: type) -> type:
        for suffix in suffixes:
            existing = LOADERS.get(suffix)
            if existing is not None:
                raise ValueError(
                    f"后缀 {suffix!r} 已被 {existing.name!r} 占用，{cls.__name__} 注册失败"
                )
            LOADERS[suffix] = cls
        return cls

    return deco


def build_loader(
    suffix: str,
    cfg: IngestConfig,
    ocr: OcrEngine | None = None,
    assets: object | None = None,
) -> Loader:
    """按后缀建一个 loader。

    loader 的构造函数统一收「ingest 配置 + OCR 客户端 + 资产存储」，便于工厂统一构造；
    各自只取用需要的那部分（TextLoader 三样都不需要）。

    ocr 是运行时协作者而不是配置，所以单独一个入参 —— 与 build_retriever 多收
    index / embedder 是同一个先例。只有 PDF 与图片用得上它。

    `assets` 同理：它按 `storage.minio` 建成（html / docx 里抽出来的图要落对地方），
    而 `cfg` 只有 `ingest` 那一段 —— 所以由调用方（`run_parse`）建好传进来。
    不传时那两个 loader 会**自己按配置建**，且 `backend=minio` 时直接报错，
    绝不悄悄写到本地盘上。
    """
    cls = LOADERS.get(suffix.lower())
    if cls is None:
        raise KeyError(f"不支持的后缀 {suffix!r}；已注册: {sorted(LOADERS)}")
    return cls(cfg, ocr, assets)


def mime_for(path: Path) -> str:
    return _MIME_BY_SUFFIX.get(path.suffix.lower(), "application/octet-stream")


def suffix_for_mime(mime: str) -> str:
    """Content-Type → 后缀。**这张反向表由 `_MIME_BY_SUFFIX` 生成**，不另立一份名单。

    「链接导入」那条路要用它：服务器给的 `Content-Type: application/pdf` 是我们唯一能
    定后缀的线索（URL 可能是 `/download?id=7` 这种，没有后缀）。一张表生成两个方向，
    加了新格式只改一处。

    一对多时取表里**先出现**的那个（`.md` 在 `.markdown` 前、`.jpg` 在 `.jpeg` 前）——
    表里的顺序就是偏好顺序。认不出来返回空串：调用方要么另找线索，要么拒绝，别猜。
    """
    return _SUFFIX_BY_MIME.get(mime.split(";")[0].strip().lower(), "")


def image_suffixes() -> tuple[str, ...]:
    """属于 ``image/*`` 的后缀 —— 判断「这个后缀能不能当图片本体」时用它。

    同样是从 `_MIME_BY_SUFFIX` 生成，不另立名单。`.svg` / `.webp` 不在里面（前者是矢量图，
    后者官方没声明支持）：VLM 描述与多模态 embedding 都是按栅格图设计的，收了也喂不进去。
    """
    return tuple(suffix for suffix, mime in _MIME_BY_SUFFIX.items() if mime.startswith("image/"))


def data_hash(data: bytes) -> str:
    """一段字节的 sha1[:16]，与 `ingest.incremental.file_hash` **同一个算法**。

    给「手上有字节、还没落盘」的场景用（抓回来的页面与图片、上传的字节）。落盘那条路
    用 `file_hash(path)`：它按 1MB 分块读，不把大文件整个读进内存。
    """
    return hashlib.sha1(data).hexdigest()[:16]


def rel_uri(path: Path, root: Path | None) -> str:
    """source_uri：相对语料根目录的路径。

    相对路径而非绝对：引用里展示 `docs/a.md` 比 `D:/.../docs/a.md` 可读，
    而且同一份语料换机器 ingest 时 source_uri 不变，引用不会失效。
    """
    if root is not None:
        try:
            return path.relative_to(root).as_posix()
        except ValueError:
            # 调用方直接传了 root 之外的文件，退回原样
            pass
    return path.as_posix()


def doc_id_for(path: Path, root: Path | None) -> str:
    """由「相对路径」算 doc_id，**不含内容**。

    用路径而非内容哈希：同一份文档重新 ingest 时 doc_id 不变，chunk_id 也就不变，
    upsert 原地覆盖；若用内容哈希，改一个字就产生一批新 chunk，旧的变孤儿常驻库中。
    """
    return hashlib.sha1(rel_uri(path, root).encode("utf-8")).hexdigest()[:16]


_ENCODINGS = ("utf-8", "gb18030")

# **每种格式各自**的 loader 版本号。它按文件后缀进 ``ingest.stages.parse_key``：
# 改了哪种 loader 的解析结果（清洗、标题识别、表格转换……），那种格式的解析产物就该重做。
#
# 为什么按格式分开、不共用一个版本号：PDF 重解析要过 OCR / VLM，**是按量计费的那一步**
# （Kafka 那份 396 页花了十几分钟 GPU）。共用一个号的话，改 html 的标题清洗也会让
# 所有 PDF 产物失效，白白重跑一遍。实测踩过一次（html 去掉 `¶` 之后，产物还是旧的）。
#
#   pdf   1：初始
#   html  2：标题剥掉锚点/折叠箭头（¶ ▹ ▾）
LOADER_VERSIONS: dict[str, str] = {
    "pdf": "1",
    "html": "2",
    "docx": "1",
    "xlsx": "1",
    "text": "1",
    "image": "1",
}


def loader_version(suffix: str) -> str:
    """按后缀取 loader 版本；没登记的格式按 "1"（稳定值，不是每次都变）。"""
    return LOADER_VERSIONS.get(suffix.lstrip(".").lower(), "1")


def read_text_any_encoding(path: Path) -> str:
    """文件按 utf-8 → gb18030 依次尝试（见 `decode_text`）。"""
    return decode_text(path.read_bytes())


def decode_text(raw: bytes) -> str:
    """按 utf-8 → gb18030 依次尝试解码。

    中文语料里 GBK 系编码很常见，只按 utf-8 读会抛异常或整篇变替换字符。
    gb18030 覆盖 GBK/GB2312；两档都失败说明不是文本（如误改名成 .txt 的二进制），
    此时用替换字符解码兜底，让流程能跑完 —— 空正文会被 load_paths 拦下报出来。

    「手上有字节、还没落盘」的场景（抓回来的页面）走这一个，别各自抄一遍。
    """
    for encoding in _ENCODINGS:
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def to_markdown_table(rows: list[list[str]]) -> str:
    """二维表 → markdown 竖线表。docx / html / xlsx 三条路共用。

    它们与 OCR 那条路**恰好相反**：产线会给合并单元格（rowspan/colspan），
    竖线表表达不了，所以那边保留 HTML；而这边的结构是我们自己解析出来的，
    本来就没有 span 信息，竖线表更省 token、也更干净。

    单元格里的 ``|`` 会转义、换行会被压平 —— 一个裸 ``|`` 就能把整张表切错列。
    """
    cleaned = [
        [cell.replace("|", "\\|").replace("\n", " ").strip() for cell in row] for row in rows
    ]
    # 全空行不要：Excel 的已用区域经常包着一堆空行
    cleaned = [row for row in cleaned if any(row)]
    if not cleaned:
        return ""
    width = max(len(row) for row in cleaned)
    padded = [row + [""] * (width - len(row)) for row in cleaned]
    header, *body = padded
    lines = ["| " + " | ".join(header) + " |", "|" + " --- |" * width]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def normalize_newlines(text: str) -> str:
    """把 CRLF / CR 统一成 LF。

    Windows 上编辑的 .md/.txt 是 CRLF。虽然切块的分隔符判定容得下它（实测
    段落边界仍能正确断开），但 ``\\r`` 会原样存进库、出现在引用片段里，
    也可能被分词器当成一个独立的字符。在入口处归一化，不用下游各自防备。
    """
    return text.replace("\r\n", "\n").replace("\r", "\n")


def pages_to_text(pages: list[OcrPage], *, store: AssetStore, doc_id: str) -> ExtractedText:
    """产线返回的逐页 markdown → 一份可入库的正文（含图片落盘与清洗）。

    PDF 与图片两条 OCR 路径共用同一个入口，**清洗只在这里做一次** —— 两个调用点
    各写一遍的话，迟早会出现「一半文档洗过、一半没洗」这种极难排查的差异。

    空白页丢掉：产线对空白页会返回空串，留着只会在正文里堆出一串空行。

    图片在这里就从正文里挪走：base64 换成短标记、本体进资产目录。标记留在原位，
    所以切块之后还能知道每张图夹在哪两段之间。

    **标题兜底放在拼完之后**（`promote_numbered_headings`）：漏标的那串编号行会跨页
    （实测 `1./2./3.` 在第 9 页、编号接得上的 `4.` 在第 10 页），逐页做就看不见
    「下一个标题的编号接得上」这个判据。它一行都不增删，所以逐行打点仍然按页算得出来。
    """
    pieces: list[str] = []
    page_numbers: list[int] = []
    images: list[ImageAsset] = []
    for page in pages:
        if not page.markdown.strip():
            continue
        extracted = extract_images(
            page.markdown,
            page.images,
            store=store,
            doc_id=doc_id,
            # 页码从 1 起：给人看的，也是资产文件名的前缀
            page=page.page_index + 1,
        )
        text = clean_ocr_markdown(normalize_newlines(extracted.text))
        if not text:
            continue
        pieces.append(text)
        page_numbers.append(page.page_index + 1)
        images.extend(extracted.images)
    text = promote_numbered_headings("\n\n".join(pieces))
    return ExtractedText(text, images, _positions_by_page(text, pieces, page_numbers))


def _positions_by_page(text: str, pieces: list[str], page_numbers: list[int]) -> list[Position]:
    """逐行打点：``offset → (第几页, 第几行)``。

    ``pieces`` 是**变换前**的逐页文本。能这么用是因为标题兜底**一行都不增删** ——
    每页占几行照旧成立，变的只有偏移。
    """
    lines = text.split("\n")
    out: list[Position] = []
    cursor = 0
    offset = 0
    for piece, page in zip(pieces, page_numbers, strict=True):
        count = piece.count("\n") + 1
        chunk = "\n".join(lines[cursor : cursor + count])
        out.extend(line_positions(chunk, page=page, start=offset))
        cursor += count
        offset += len(chunk) + 2  # 页与页之间用 "\n\n" 连接
    return out


class SkippedFile(BaseModel):
    path: str
    reason: str


class LoadReport(BaseModel):
    """批量加载结果。

    skipped 不是装饰：一个文件没解析出正文，若静默跳过，用户会以为 ingest 成功了。
    """

    documents: list[Document] = Field(default_factory=list)
    skipped: list[SkippedFile] = Field(default_factory=list)
