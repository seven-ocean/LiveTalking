# AGENTS.md — zl-talking（Python WebRTC 服务端）

分支：`feature/aihumanV2-MetaXC500`。**客户特供版，不轻易切换分支**；操作前 `git branch --show-current`，不是预期分支立即停止并询问用户。核心：aiortc / HumanPlayer / SDP / WS 降级。

## 通用交付约束（客户离线版）

- 客户线上只通过 release 包更新，不直接改线上文件。
- 增量包：release id `YYYY.MM.DD-NN` + git tag `patch/<id>` + `releases/patch-<id>/`（`patch-<id>.tar.gz`、MANIFEST、CHANGELOG、sha256）。
- 全量包：`releases/full-<version>-<id>/`，固定三件 `full-<version>-<id>-part1-app/part2-models/part3-avatars.tar.gz`，tag `full/<version>-<id>`。
- 交付包只放在 `releases/patch-<id>/` 或 `releases/full-<version>-<id>/`，不要放 `/data/shuziren/zl` 根目录；根目录只保留源码、安装树和工具。
- 全量包由 `aihuman_offline_real` 生成，固定三件：`part1-app`（源码+离线依赖+中间件+运行环境）、`part2-models`、`part3-avatars`；不要拆成更多分卷。
- 全量包 `part3-avatars` 只保留生产在用形象；测试/重复形象（如 `wav2lip256_avatar10_test`、`musetalk_avater_test_5`）不打包。
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
- **容量回收不抢在线会话**（2026-11-25）：
  - `/human`、`/humanaudio`、`/interrupt_talk`、`/is_speaking`、`/set_audiotype`、`/record` 必须
    `session_manager.touch(sessionid)`；否则 `/metrics` 的 `idle` 永远停在创建时刻，LRU 会把在线会话当最旧。
  - `_has_capacity()` / `evict_idle_session()` / gpu_lb 满员回收只能动「无活 PC + 无 WS 租约 +
    空闲 ≥ `TALK_QUEUE_EVICT_IDLE_SECONDS`（默认 5s）」的会话；没有就返回 503 `queue=true` 排队。
    绝不能有无条件 `evict_oldest()` 兜底——单卡 `max_session=1` 时会把正在出画面的用户踢掉，
    表现为画面冻结、zl-ai `/humanaudio` 报 `session not found`。
  - `/stop_session` 必须 `drop_session_pcs()`，否则 track 会重新拉起 render 循环。

## WebRTC 码率护栏（2026-11-24）

- 参数来源：AI 能力中心 → human（数字人推理）→ `default_params.launch`
  （`webrtc_bitrate`/`webrtc_min_bitrate` kbps，`webrtc_bitrate_cap_factor`、
  `webrtc_maxrate_factor`、`webrtc_bufsize_ms`），保存后写 `runtime/conf/zl-talking.launch.json`。
  `app.py` **优先 CLI，其次直接读该 launch.json**（客户现场 ops.sh 可能未更新，不要去覆盖客户的启动脚本），
  再回落环境变量/代码默认。改完需重启 zl-talking。
- `server/rtc_bitrate_guard.py` 是运行时 monkeypatch（aiortc 1.14.0）：
  - `target_bitrate` setter 夹在 `[最低码率, 目标 × 上探倍数]`，REMB 不能再顶到 4.5M；
  - 编码器重建时给 libx264 设 `maxrate = 上限 × 峰值倍数`、`bufsize = 上限 × 窗口`；
  - 目标码率变化小于 25% 不重建编码器（避免 REMB 抖动造成 IDR 风暴），累计目标超过阈值才重建；
  - `ZL_WEBRTC_RC_GUARD=0` 可整体关闭，`ZL_WEBRTC_RC_REBUILD_RATIO` 调整重建阈值。
- 验收看 `[perf]`：`vcap` = 上探上限，`vtb` = 当前编码目标，`vbps` = 只统计 `outbound-rtp`
  的真实视频码率，`kf` = 5s 内关键帧/重建次数。正常：`vtb ≤ vcap`、`vbps` 在目标附近、
  `kf ≤ 1~2/5s`、`vgap≈40ms`、`dropped=0`。
- 不要直接改 venv 里的 aiortc 文件；护栏必须在进程内安装，升级 aiortc 时先回归本文件。

## 离线约束

- 不使用公网 STUN/TURN；`TURN_ICE_SERVERS` 服务端可为空。
- 不下载模型/依赖；使用本地 models/wheels。
- 不改 `runtime/conf`；配置由安装包/ops.sh 管理。
- `web/` 页面只允许引用 `web/vendor/` 下的本地静态资源（Bootstrap 5.3、Bootstrap Icons 1.10、
  jQuery 3.6/2.1.1、SockJS、Inter 字体）；禁止新增 CDN、Google Fonts、code.jquery.com 等在线地址。
  新增页面资源时先落到 `web/vendor/` 再引用。
