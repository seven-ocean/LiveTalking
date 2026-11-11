#!/usr/bin/env python3
"""常驻 TCP 代理：对外 51004，转发到后端 zl-talking 51014。

进程级切换模型时，后端会退出并重启。代理保持 51004 监听，
对 /offer、/warmup 这类小请求做缓冲和重试，等新后端起来后重放，
避免 relay/前端看到 connection reset。
"""
import argparse
import asyncio
import logging
import re

LOG = logging.getLogger("tcp_proxy")
RETRY_SECONDS = 90.0
CONNECT_RETRY_SECONDS = 60.0


async def connect_backend(host, port, timeout=CONNECT_RETRY_SECONDS):
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while True:
        try:
            return await asyncio.open_connection(host, port)
        except Exception:
            if loop.time() >= deadline:
                raise
            await asyncio.sleep(0.2)


async def pipe(reader, writer):
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


def _content_length(header: bytes) -> int:
    match = re.search(br"Content-Length:\s*(\d+)", header, re.I)
    return int(match.group(1)) if match else 0


async def read_http_response(reader):
    header = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=90)
    length = _content_length(header)
    body = b""
    if length:
        while len(body) < length:
            chunk = await asyncio.wait_for(reader.read(65536), timeout=90)
            if not chunk:
                break
            body += chunk
    else:
        while True:
            chunk = await asyncio.wait_for(reader.read(65536), timeout=5)
            if not chunk:
                break
            body += chunk
    return header + body


async def handle(client_reader, client_writer, backend_host, backend_port):
    loop = asyncio.get_event_loop()
    try:
        first = b""
        while b"\r\n\r\n" not in first:
            chunk = await asyncio.wait_for(client_reader.read(65536), timeout=15)
            if not chunk:
                client_writer.close()
                return
            first += chunk

        header, _, body = first.partition(b"\r\n\r\n")
        request_line = header.split(b"\r\n", 1)[0]
        is_switchable = request_line.startswith(b"POST /offer ") or request_line.startswith(b"POST /warmup ")
        if not is_switchable:
            backend_reader, backend_writer = await connect_backend(backend_host, backend_port)
            backend_writer.write(first)
            await backend_writer.drain()
            await asyncio.gather(
                pipe(client_reader, backend_writer),
                pipe(backend_reader, client_writer),
            )
            return

        need = _content_length(header)
        while len(body) < need:
            chunk = await asyncio.wait_for(client_reader.read(65536), timeout=15)
            if not chunk:
                break
            body += chunk
        request = header + b"\r\n\r\n" + body

        deadline = loop.time() + RETRY_SECONDS
        while loop.time() < deadline:
            try:
                backend_reader, backend_writer = await connect_backend(backend_host, backend_port)
                backend_writer.write(request)
                await backend_writer.drain()
                response = await read_http_response(backend_reader)
                try:
                    backend_writer.close()
                except Exception:
                    pass
                client_writer.write(response)
                await client_writer.drain()
                LOG.info("forwarded switchable request: %s", request_line.decode("latin1", "ignore"))
                return
            except Exception as exc:
                LOG.warning("backend request failed, retrying: %s", exc)
                await asyncio.sleep(0.3)
        LOG.error("backend unavailable, giving up: %s", request_line.decode("latin1", "ignore"))
    except Exception:
        LOG.exception("proxy connection error")
    finally:
        try:
            client_writer.close()
        except Exception:
            pass


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen", type=int, default=51004)
    parser.add_argument("--backend", type=int, default=51014)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")

    async def on_client(reader, writer):
        await handle(reader, writer, args.host, args.backend)

    server = await asyncio.start_server(on_client, "0.0.0.0", args.listen)
    LOG.info("tcp proxy listening on 0.0.0.0:%d -> %s:%d", args.listen, args.host, args.backend)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
