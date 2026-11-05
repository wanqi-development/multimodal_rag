"""生成后的答案忠实度核验。"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable
from typing import Any

FaithfulnessFn = Callable[[str, str, list[str]], Awaitable[float]]


class AnswerVerifier:
    """用 ragas Faithfulness 判断答案是否被检索上下文支持。

    只做核验，不自动联网。模型或依赖不可用时返回 ``unavailable``，不影响已经生成的答案。
    """

    def __init__(
        self,
        *,
        threshold: float,
        timeout_s: float,
        faithfulness: FaithfulnessFn | None = None,
        setup_error: str = "",
    ) -> None:
        self.threshold = threshold
        self.timeout_s = timeout_s
        self._faithfulness = faithfulness
        self._setup_error = setup_error

    async def verify(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        *,
        retrieval_incomplete: bool = False,
    ) -> dict[str, Any]:
        """仅以真实语料核验；检索不完整且无证据时提示用户授权联网，不自动联网。"""
        if retrieval_incomplete and not contexts:
            return {
                "status": "low",
                "score": None,
                "threshold": self.threshold,
                "message": "本次检索不完整，未能取得可核验资料，是否允许联网搜索？",
                "can_search_web": True,
            }
        if not answer.strip():
            return {
                "status": "low",
                "score": None,
                "threshold": self.threshold,
                "message": "没有生成可核验的答案，是否允许联网搜索？",
                "can_search_web": True,
            }
        if not contexts:
            return {
                "status": "low",
                "score": None,
                "threshold": self.threshold,
                "message": "没有检索到能够回答该问题的资料，是否允许联网搜索？",
                "can_search_web": True,
            }
        if self._faithfulness is None:
            return {
                "status": "unavailable",
                "score": None,
                "threshold": self.threshold,
                "message": f"答案核验暂不可用：{self._setup_error or 'ragas 未安装或未配置'}",
                "can_search_web": False,
            }
        try:
            score = float(
                await asyncio.wait_for(
                    self._faithfulness(question, answer, contexts),
                    timeout=self.timeout_s,
                )
            )
        except Exception as exc:  # noqa: BLE001 —— 核验失败不能推翻已有答案
            return {
                "status": "unavailable",
                "score": None,
                "threshold": self.threshold,
                "message": f"答案核验暂不可用：{type(exc).__name__}: {exc}",
                "can_search_web": False,
            }
        if not math.isfinite(score):
            return {
                "status": "unavailable",
                "score": None,
                "threshold": self.threshold,
                "message": "答案核验没有返回有效分数",
                "can_search_web": False,
            }
        supported = score >= self.threshold
        return {
            "status": "supported" if supported else "low",
            "score": round(score, 6),
            "threshold": self.threshold,
            "message": (
                "当前回答已得到检索资料支持。"
                if supported
                else "当前回答未得到检索资料的充分支持，是否允许联网核实？"
            ),
            "can_search_web": not supported,
        }


def build_answer_verifier(settings: Any, secrets: Any) -> AnswerVerifier | None:
    cfg = settings.answer_verification
    if not cfg.enabled:
        return None
    try:
        from rag.eval.ragas_backend import build_faithfulness

        faithfulness = build_faithfulness(settings, secrets)
    except Exception as exc:  # noqa: BLE001 —— 可选项缺失时降级为“核验不可用”
        return AnswerVerifier(
            threshold=cfg.threshold,
            timeout_s=cfg.timeout_s,
            setup_error=f"{type(exc).__name__}: {exc}",
        )
    return AnswerVerifier(
        threshold=cfg.threshold,
        timeout_s=cfg.timeout_s,
        faithfulness=faithfulness,
    )
