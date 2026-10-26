"""依赖注入：鉴权、限流、组件取用。

三个 FastAPI 依赖，顺序有意：**先鉴权、再限流** —— 没通过鉴权的请求不该占配额，
否则有人拿错 key 乱打，正经客户端的额度会被挤掉。
"""

from __future__ import annotations

import hmac

from fastapi import Depends, Header, HTTPException, Request, status

from rag.api.ratelimit import identity_of

# 没配 RAG_API_KEY 时允许的来源。**默认只放行本机**：没配 key 就全放行等于裸奔。
LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost", "testclient"}


def get_components(request: Request):
    """取服务级单例（在 lifespan 里建好，挂 `app.state` 上）。"""
    return request.app.state.components


async def require_api_key(
    request: Request,
    x_api_key: str | None = Header(default=None),
    components=Depends(get_components),
) -> None:
    """`X-API-Key` 校验。配了 key 就一律要；没配就只允许本机。

    比较用 `hmac.compare_digest`：`==` 会在第一个不同的字节就返回，理论上能从响应时间
    里一位一位试出 key。这里不是高危场景，但一行就能避免的事没理由不做。

    **注意：2026-10-06 起这个依赖没有再挂在 router 上** —— 使用者定的是不用
    `RAG_API_KEY`，注册后即可使用。留着是因为用户系统那一批要在这里换成 token 校验，
    替换点明确比删掉再写一遍强。限流那边仍然按身份算（`identity_of` 会退回按 IP）。
    """
    expected = components.secrets.rag_api_key
    if expected:
        if not x_api_key or not hmac.compare_digest(x_api_key, expected):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "X-API-Key 不正确")
        return
    host = request.client.host if request.client else None
    if host not in LOCAL_HOSTS:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "服务没配 RAG_API_KEY，只允许本机访问（.env 里配上 key 才能对外）",
        )


async def enforce_rate_limit(
    request: Request,
    x_api_key: str | None = Header(default=None),
    components=Depends(get_components),
) -> None:
    """每分钟 N 次。超了 429，并带上 `Retry-After`（窗口固定 60 秒）。"""
    host = request.client.host if request.client else None
    allowed, count = await components.limiter.hit(identity_of(x_api_key, host))
    if not allowed:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            f"超过限额（{components.settings.api.rate_limit_per_min}/分钟，本窗口第 {count} 次）",
            headers={"Retry-After": "60"},
        )
