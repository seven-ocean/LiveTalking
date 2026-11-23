# AGENTS.md — zl-talking（Python WebRTC 服务端）

分支：`feature/aihumanV2-MetaXC500`。**客户特供版，不轻易切换分支**；操作前 `git branch --show-current`，不是预期分支立即停止并询问用户。核心：aiortc / HumanPlayer / SDP / WS 降级。

## 启动与端口

- `tcp_proxy.py`：51004（对外代理，nginx `/zltalk-ws/` 指向这里）
- `app.py --listenport`：51014（内部 backend，`TALK_PORT+10`）
- 媒体：aiortc 随机 UDP 端口（默认 `32768–60999`）
- 本地 TURN：3478（port_mux/udp_mux 在 51000 上分流）

## 验证

```bash
cd zl-talking
python3 -m py_compile server/sdp_sanitize.py server/rtc_manager.py
python3 tests/test_sdp_sanitize.py
```

## SDP 不变量（非常重要）

- `server/sdp_sanitize.py` 必须保持纯 Python、无 aiortc 依赖，方便单测。
- `_negotiate()` 顺序：`_rewrite_sdp_public_ip()` → `_normalize_answer_sdp()`（内部调用 sanitizer）。
- sanitizer 必须保证：
  - fingerprint 只留一条会话级 sha-256；
  - `a=setup` 只留会话级；
  - `ice-ufrag`/`ice-pwd` 只留媒体级、每段一对，处理重复/拼接/脏字符；
  - 过滤 tcp、回环、链路本地、docker 网桥候选；
  - CRLF + 末尾 CRLF。
- 不要只修 fingerprint 或只修 setup；Electron 25 会继续报下一条 `Invalid SDP line`。
- 改 sanitizer 必须补/更新 `tests/test_sdp_sanitize.py`，并跑通。

## 会话/降级

- `REUSE_SESSION_ON_OFFER` 复用前检查 `HumanPlayer.closed`；已关闭 player 不能复用。
- WS 降级 `ws_video.py` 复用 sessionid；原会话被 reaper 回收时允许不带 sessionid 自建/选最近会话。
- 不要删除 `isStandbyActive` 相关的服务端兼容（视频源切换）。

## 离线约束

- 不使用公网 STUN/TURN；`TURN_ICE_SERVERS` 服务端可为空。
- 不下载模型/依赖；使用本地 models/wheels。
- 不改 `runtime/conf`；配置由安装包/ops.sh 管理。
