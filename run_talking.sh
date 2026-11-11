#!/usr/bin/env bash
# zl-talking 进程级切换 wrapper。
# app.py 检测到目标 avatar 属于另一个模型时，写入目标文件并以 42 退出；
# 这里读取目标 avatar 后重新启动 app，保证旧模型进程完全退出、资源由 OS 回收。
set -u

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR" || exit 1

PY="${1:?usage: run_talking.sh <python> <app args...>}"
shift

TARGET="${ZL_SWITCH_TARGET_FILE:-$DIR/../runtime/run/zl-talking.target}"
AVATAR="${TALK_AVATAR_ID:-wav2lip256_avatar6}"
# keep：不释放、不重启，两个模型在同一进程内共存（客户高配机默认）
# restart：切换时退出旧进程、重启新模型，释放 GPU/CUDA 资源（低配机）
MODE="${TALK_MODEL_SWITCH_MODE:-keep}"
case "$MODE" in
  restart|release) PROCESS_SWITCH=1 ;;
  *) PROCESS_SWITCH=0 ;;
esac
if [ -f "$TARGET" ]; then
  AVATAR="$(cat "$TARGET")"
  rm -f "$TARGET"
fi

child=""
cleanup() {
  [ -n "$child" ] && kill "$child" 2>/dev/null
  [ -n "$child" ] && wait "$child" 2>/dev/null
  exit 0
}
trap cleanup TERM INT

while true; do
  ZL_DEFER_MODEL_LOAD=1 ZL_PROCESS_SWITCH="$PROCESS_SWITCH" ZL_SWITCH_TARGET_FILE="$TARGET" \
    TALK_AVATAR_ID="$AVATAR" "$PY" "$@" --avatar_id "$AVATAR" &
  child=$!
  wait "$child"
  rc=$?
  child=""
  if [ "$rc" -eq 42 ] && [ -f "$TARGET" ]; then
    AVATAR="$(cat "$TARGET")"
    rm -f "$TARGET"
    echo "[run_talking] switch model -> $AVATAR"
    continue
  fi
  if [ "$rc" -ne 0 ]; then
    echo "[run_talking] app exited rc=$rc, restarting in 3s"
    sleep 3
    continue
  fi
  exit "$rc"
done
