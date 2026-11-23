# AGENTS.md — zl-talking（Python WebRTC 服务端）

分支：`feature/aihumanV2-MetaXC500`。**客户特供版，不轻易切换分支**；操作前 `git branch --show-current`，不是预期分支立即停止并询问用户。核心：aiortc / HumanPlayer / SDP / WS 降级。

## 通用交付约束（客户离线版）

- 客户线上只通过 release 包更新，不直接改线上文件。
- 每次客户更新 = release id `YYYY.MM.DD-NN` + git tag `release/<id>` + `releases/<id>/MANIFEST.json`/`CHANGELOG.md`/包 sha256。
- 交付包只放在 `releases/<release-id>/`，不要放 `/data/shuziren/zl` 根目录；根目录只保留源码、安装树和工具。
- 全量包由 `aihuman_offline_real` 生成，固定三件：`part1-app`（源码+离线依赖+中间件+运行环境）、`part2-models`、`part3-avatars`；不要拆成更多分卷。
- 交付包必须含 `apply.sh`/`verify.sh`/`rollback.sh`/`README.txt`/`SHA256SUMS`/`source-patches/`；前端改动含 `web-dist/`。
- 禁止生产密码/密钥入库/入包；禁止提交 `.ipynb_checkpoints/`、`dist/`、`node_modules/`、`runtime/data/`、`logs/`。
- 离线：不用公网 STUN/CDN/在线依赖；依赖来自本地 wheels/仓库。
- 先对比真实生产 `runtime/conf`，不按默认模板猜拓扑；`web-dist` 删旧再整体替换，`index.html` no-store。
- 必须备份、可回滚、验证：构建/语法、单测、sha256、mock 安装根、`nginx -t`、端口/健康检查。
- 分支纪律：当前 `zl-*` 分支是客户特供版，不轻易切换/新建/合并；改动前 `git branch --show-current`；不一致立即停止并询问用户。
- 端口基线：51000/TCP（port_mux：web、`/prod-api`、`/zltalk-ws`、TURN/TCP）、51000/UDP（TURN/UDP）、51001/TCP（`/ws/desktop` 等）、32768–60999/UDP（WebRTC 直连媒体）；内部端口只本机。
- 已知坑：51000 可能是 port_mux→51090；51001 可能缺 `/prod-api`；111=zl-ai 51011 不在；turn/udpmux 要显式重启；Electron 25 需全量 SDP sanitize；待机只在 live 异常时出现且保留唤醒词。

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
