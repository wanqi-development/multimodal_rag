"""服务层：FastAPI（M4）。

    app = create_app(load_settings())
    uvicorn.run(app, host=..., port=...)      # 见 cli/main.py 的 `rag serve`

端点按领域见 `routes.py`；`/agent` 是 Web 唯一问答入口。服务层刻意做薄 —— 它只负责
**建一次重对象**、翻请求/响应、翻状态码；真正的活儿在 retrieval / ingest / graphs 那几层。
"""

from rag.api.app import Components, ServiceUnavailable, create_app, create_app_from_env

__all__ = ["Components", "ServiceUnavailable", "create_app", "create_app_from_env"]
