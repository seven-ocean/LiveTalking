#!/usr/bin/env python3
"""单端口 TCP 多路复用：让 51000 一个端口同时承载 HTTP/WSS 与 TURN over TCP。

现场通常只放通一个端口（SSH 隧道也只转一条 TCP）。WebRTC 的媒体面必须走
TURN/TCP 才能穿过这种链路，于是同一端口上会同时出现两种协议：

    首字节 0x00            → STUN/TURN 报文（Binding/Allocate）→ coturn
    首字节 0x40 ~ 0x7F     → TURN ChannelData                  → coturn
    其它（HTTP 方法字母）  → nginx（页面 / /prod-api / /zltalk-ws / /live）

为什么不用 nginx 的 stream 模块：原版 nginx 只能按 SNI（TLS）分流，
而 TURN 是明文 TCP，没有可用的分流键；所以这里用一个很小的应用无关转发层。

用法：
    port_mux.py --listen 0.0.0.0:51000 --http 127.0.0.1:51090 --turn 127.0.0.1:3478
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import socket
import sys

LOG = logging.getLogger("port_mux")
PEEK_TIMEOUT = float(os.environ.get("MUX_PEEK_TIMEOUT", "10"))
CONNECT_TIMEOUT = float(os.environ.get("MUX_CONNECT_TIMEOUT", "5"))


def parse_addr(text: str, default_port: int) -> tuple[str, int]:
    host, _, port = text.rpartition(":")
    if not host:
        return text, default_port
    return host, int(port)


def tune_tcp(writer) -> None:
    """关掉 Nagle。

    单端口方案里，媒体（TURN/TCP）要经过这一层转发。Nagle 会把连续的
    小包（RTP 约 1200 字节，远小于 loopback/LAN 的 MSS）攒着等 ACK，
    延迟被拉到 40~100ms 且成串到达 —— 客户端表现为 jitter 从 30ms 起
    一路涨到几百 ms、jitter buffer 越垫越高、最后丢帧/补音（电流音）。
    实时媒体必须关闭 Nagle。
    """
    try:
        sock = writer.get_extra_info("socket")
        if sock is not None:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except Exception:
        pass


def is_turn(first_byte: int) -> bool:
    """只看连接的首字节：STUN/TURN 报文类型 < 0x4000，高字节恒为 0x00。

    不能把 0x40~0x7F 也算成 TURN：TURN ChannelData 确实落在这个区间，
    但它只会出现在「已经分配过」的同一条连接上，而 HTTP 方法字母
    （G/P/O/H/D/C/T/U）也正好在 0x40~0x7F —— 按区间判断会把所有 HTTP 抢走。
    """
    return first_byte == 0x00


async def pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while True:
            data = await reader.read(65536)
            if not data:
                break
            writer.write(data)
            await writer.drain()
    except Exception:
        pass
    finally:
        try:
            writer.close()
        except Exception:
            pass


async def handle(client_reader: asyncio.StreamReader,
                 client_writer: asyncio.StreamWriter,
                 http_backend: tuple[str, int],
                 turn_backend: tuple[str, int]) -> None:
    peer = client_writer.get_extra_info("peername")
    try:
        first = await asyncio.wait_for(client_reader.readexactly(1), timeout=PEEK_TIMEOUT)
    except Exception:
        try:
            client_writer.close()
        except Exception:
            pass
        return

    kind = "turn" if is_turn(first[0]) else "http"
    backend = turn_backend if kind == "turn" else http_backend
    try:
        up_reader, up_writer = await asyncio.wait_for(
            asyncio.open_connection(*backend), timeout=CONNECT_TIMEOUT)
    except Exception as exc:
        LOG.warning("backend %s:%s 不可用（%s），断开 %s", backend[0], backend[1], exc, peer)
        try:
            client_writer.close()
        except Exception:
            pass
        return

    tune_tcp(client_writer)
    tune_tcp(up_writer)
    LOG.debug("%s -> %s (%s)", peer, backend, kind)
    up_writer.write(first)
    await asyncio.gather(
        pipe(client_reader, up_writer),
        pipe(up_reader, client_writer),
        return_exceptions=True,
    )


async def main_async(args) -> None:
    http_backend = parse_addr(args.http, 80)
    turn_backend = parse_addr(args.turn, 3478)
    listen_host, listen_port = parse_addr(args.listen, 51000)
    server = await asyncio.start_server(
        lambda r, w: handle(r, w, http_backend, turn_backend),
        host=listen_host, port=listen_port, backlog=512,
    )
    LOG.info("port_mux listening on %s:%s -> http %s:%s / turn %s:%s",
             listen_host, listen_port, *http_backend, *turn_backend)
    async with server:
        await server.serve_forever()


def main() -> int:
    parser = argparse.ArgumentParser(description="单端口 TCP 多路复用（HTTP + TURN/TCP）")
    parser.add_argument("--listen", default=os.environ.get("MUX_LISTEN", "0.0.0.0:51000"))
    parser.add_argument("--http", default=os.environ.get("MUX_HTTP", "127.0.0.1:51090"))
    parser.add_argument("--turn", default=os.environ.get("MUX_TURN", "127.0.0.1:3478"))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
        stream=sys.stdout,
    )
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
