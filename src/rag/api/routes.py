"""HTTP 路由聚合入口。

端点按领域放在同目录的 ``*_routes.py``；这一层只负责组合，并把限流统一挂到所有
子路由上。鉴权约定见 ``deps.py``。
"""

from fastapi import APIRouter, Depends

from rag.api.deps import enforce_rate_limit
from rag.api.documents_routes import documents_router
from rag.api.search_chat_routes import search_chat_router
from rag.api.sessions_agent_routes import sessions_agent_router
from rag.api.system_routes import system_router

router = APIRouter(dependencies=[Depends(enforce_rate_limit)])
router.include_router(system_router)
router.include_router(search_chat_router)
router.include_router(documents_router)
router.include_router(sessions_agent_router)
