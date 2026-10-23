"""
loopback_guard.py — 关键动作只接受本机回环客户端（2026-10-22 用户决定 D11）

用户原话："首轮仅本机回环，不扩远程。保留真实监听参数和关键动作检查；需要远程访问时先做认证授权，
CORS 不能代替认证。"

服务没有认证。监听地址只允许回环（`main._assert_safe_bind`）是第一道；这里是第二道：所有改状态的请求
（POST / PUT / PATCH / DELETE —— 审批、激活、验证、熔断、接受差异、删除、跑 GP……）只要客户端不是回环地址，
一律 403。CORS 只决定浏览器能不能读响应，不证明请求者是谁 —— 被 CORS 放行的来源同样要过这一道。
读请求不在此列（远程能不能连上由监听地址决定）；CORS 预检（OPTIONS）放行。
"""

from __future__ import annotations

import ipaddress
import json
import logging

logger = logging.getLogger(__name__)

STATE_CHANGING = frozenset({"POST", "PUT", "PATCH", "DELETE"})


def is_loopback(host: str) -> bool:
    if not host:
        return False
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.split("%", 1)[0]).is_loopback
    except ValueError:
        logger.info("[security] 客户端地址 %r 不是 IP，按非回环处理", host)
        return False


class LoopbackGuardMiddleware:
    """纯 ASGI 中间件：改状态的请求来自非回环客户端 → 403。"""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") == "http" and scope.get("method", "").upper() in STATE_CHANGING:
            host = (scope.get("client") or ("", 0))[0] or ""
            if not is_loopback(host):
                logger.error("[security] 拒绝来自非回环客户端 %s 的 %s %s（服务无认证，关键动作只接受本机）",
                             host, scope.get("method"), scope.get("path"))
                body = json.dumps({"detail": "关键动作只接受本机回环客户端（服务没有认证；远程访问要先做认证授权，"
                                             "CORS 不能代替认证）"}, ensure_ascii=False).encode("utf-8")
                await send({"type": "http.response.start", "status": 403,
                            "headers": [(b"content-type", b"application/json; charset=utf-8"),
                                        (b"content-length", str(len(body)).encode())]})
                await send({"type": "http.response.body", "body": body})
                return
        await self.app(scope, receive, send)
