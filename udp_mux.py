#!/usr/bin/env python3
"""UDP 侧的单端口转发：51000/udp → coturn 3478/udp。

为什么需要：单端口方案里 WebRTC 媒体只能走 TURN/TCP，而 TCP 会把
实时的小包攒起来（Nagle / 队头阻塞 / 拥塞窗口），客户端表现为 jitter
从 30ms 一路涨到几百 ms、jitter buffer 越垫越高，最后丢帧与补音。
TCP 与 UDP 端口号互不冲突，因此同一个 51000 可以同时监听 TCP（HTTP/TURN）
和 UDP（TURN over UDP），让浏览器优先走 UDP。

实现：按客户端地址做 NAT（每个客户端一条上游 socket），
只转发 TURN 报文，不做任何解析。
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time

LOG = logging.getLogger("udp_mux")
IDLE_TIMEOUT = float(os.environ.get("UDP_MUX_IDLE", "300"))


class _Upstream(asyncio.DatagramProtocol):
    def __init__(self, listener: "asyncio.DatagramTransport", client_addr):
        self.listener = listener
        self.client_addr = client_addr
        self.transport = None
        self.last = time.monotonic()

    def connection_made(self, transport):
        self.transport = transport

    def datagram_received(self, data, addr):
        self.last = time.monotonic()
        try:
            self.listener.sendto(data, self.client_addr)
        except Exception:
            pass


class _Listener(asyncio.DatagramProtocol):
    def __init__(self, loop, upstream_addr):
        self.loop = loop
        self.upstream_addr = upstream_addr
        self.transport = None
        self.peers = {}  # client_addr -> _Upstream

    def connection_made(self, transport):
        self.transport = transport

    def datagram_received(self, data, addr):
        peer = self.peers.get(addr)
        if peer is not None and peer.transport is not None:
            peer.last = time.monotonic()
            try:
                peer.transport.sendto(data)
            except Exception:
                pass
            return
        self.loop.create_task(self._new_peer(data, addr))

    async def _new_peer(self, data, addr):
        if addr in self.peers:
            return
        proto = _Upstream(self.transport, addr)
        peername_before = None
        try:
            transport, _ = await self.loop.create_datagram_endpoint(
                lambda: proto, remote_addr=self.upstream_addr)
        except Exception as exc:
            LOG.warning("upstream %s 不可用（%s）", self.upstream_addr, exc)
            return
        self.peers[addr] = proto
        transport.sendto(data)
        LOG.debug("client %s -> %s", addr, self.upstream_addr)

    async def reap(self):
        while True:
            await asyncio.sleep(60)
            now = time.monotonic()
            for addr, peer in list(self.peers.items()):
                if now - peer.last > IDLE_TIMEOUT:
                    try:
                        if peer.transport is not None:
                            peer.transport.close()
                    except Exception:
                        pass
                    self.peers.pop(addr, None)


def parse_addr(text: str, default_port: int):
    host, _, port = text.rpartition(":")
    if not host:
        return text, default_port
    return host, int(port)


async def main_async(listen: str, upstream: str) -> None:
    listen_host, listen_port = parse_addr(listen, 51000)
    up_host, up_port = parse_addr(upstream, 3478)
    loop = asyncio.get_running_loop()
    proto = _Listener(loop, (up_host, up_port))
    transport, _ = await loop.create_datagram_endpoint(
        lambda: proto, local_addr=(listen_host, listen_port))
    LOG.info("udp_mux listening on %s:%s -> %s:%s", listen_host, listen_port, up_host, up_port)
    reaper = asyncio.ensure_future(proto.reap())
    try:
        await asyncio.Event().wait()
    finally:
        reaper.cancel()
        transport.close()


def main() -> int:
    p = argparse.ArgumentParser(description="UDP 单端口转发（51000/udp → coturn）")
    p.add_argument("--listen", default=os.environ.get("UDP_MUX_LISTEN", "0.0.0.0:51000"))
    p.add_argument("--upstream", default=os.environ.get("UDP_MUX_UPSTREAM", "127.0.0.1:3478"))
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
        stream=sys.stdout,
    )
    try:
        asyncio.run(main_async(args.listen, args.upstream))
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
