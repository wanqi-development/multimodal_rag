"""API key 的**落库加密**（2026-10-09 定的 B 方案）。

* **AES-256-GCM**；每条密文自带随机 96-bit nonce；**AAD 绑 `provider_key`**
  （防止把 A 家的密文挪到 B 家去用）；存储格式 `v1:base64(nonce):base64(ct+tag)`，
  `v1` 是版本前缀（换算法/换主密钥留了升级位）。
* **页面永不回显明文**，连遮罩都不给：只回 `sha256(明文)` 前 8 位当**指纹**，
  作用是你换 key 时能确认"确实是另一把"。
* **主密钥是部署侧机密**：先看部署配置的 `RAG_SECRET_KEY`，没有就读库里的
  `app_secret`（部署时运维写入，应用只读）；两处都没有 → 返回 None，
  调用方据此**禁用页面填 key**（退回只读），而不是报错崩掉。
"""

from __future__ import annotations

import base64
import hashlib
import os
from typing import Any

# 密文版本前缀；换算法时写 v2，读的时候按前缀分派
PREFIX = "v1"
# 库里 `app_secret` 那一行的名字（部署时运维写入）
MASTER_SECRET_NAME = "rag_secret_key"
# 主密钥长度（AES-256）
KEY_BYTES = 32


class SecretError(RuntimeError):
    """加密/解密/主密钥相关的错误（都带人话原因，别把密文或明文带进消息）。"""


def fingerprint(key: str) -> str:
    """只回前 8 位十六进制 —— 能核对"是不是同一把"，反推不出明文。"""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:8]


def parse_master_key(raw: str) -> bytes:
    """把配置/库里的主密钥文本解析成 32 字节：接受 base64 或 hex，长度不够就报错。"""
    text = (raw or "").strip()
    if not text:
        raise SecretError("主密钥是空的")
    for decode in (base64.b64decode, bytes.fromhex):
        try:
            key = decode(text)
        except Exception:  # noqa: BLE001 —— 两种编码都试，都不行才算错
            continue
        if len(key) >= KEY_BYTES:
            return key[:KEY_BYTES]
    raise SecretError(f"主密钥要 {KEY_BYTES} 字节（base64 或 hex 编码后的文本）")


def encrypt_key(key: str, *, master: bytes, provider_key: str) -> str:
    """明文 → `v1:nonce:ct`。AAD 用 provider_key，换个 provider 解密必然失败。"""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    nonce = os.urandom(12)
    cipher = AESGCM(master).encrypt(nonce, key.encode("utf-8"), provider_key.encode("utf-8"))
    return f"{PREFIX}:{_b64(nonce)}:{_b64(cipher)}"


def decrypt_key(cipher: str, *, master: bytes, provider_key: str) -> str:
    """`v1:nonce:ct` → 明文。**失败只报原因，不回显密文与明文。**"""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    parts = (cipher or "").split(":")
    if len(parts) != 3 or parts[0] != PREFIX:
        raise SecretError("密文格式不对（要 v1:nonce:ct）")
    try:
        plain = AESGCM(master).decrypt(
            _unb64(parts[1]), _unb64(parts[2]), provider_key.encode("utf-8")
        )
    except Exception as exc:  # noqa: BLE001 —— 主密钥换了 / AAD 对不上 / 数据被改过
        raise SecretError(
            f"解密失败（主密钥换过、或这条密文不属于 {provider_key}）：{type(exc).__name__}"
        ) from exc
    return plain.decode("utf-8")


def load_master_key(store: Any = None) -> bytes | None:
    """部署配置优先、库里兜底；两处都没有返回 None（调用方据此禁用页面填 key）。"""
    raw = os.environ.get("RAG_SECRET_KEY", "").strip()
    if raw:
        return parse_master_key(raw)
    if store is not None:
        stored = store.secret(MASTER_SECRET_NAME)
        if stored:
            return parse_master_key(stored)
    return None


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _unb64(text: str) -> bytes:
    try:
        return base64.b64decode(text)
    except Exception as exc:  # noqa: BLE001
        raise SecretError("密文里的 base64 解不开") from exc
