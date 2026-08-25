# -*- coding: utf-8 -*-
###############################################################################
#  zl-talking PyArmor 加密脚本（授权加固 · 批次 4）
#  ---------------------------------------------------------------------------
#  用法：
#     python build_talking.py           # 最小加密：server/ + utils/
#     python build_talking.py --full     # 全量加密：+ avatars/ + tts/ + streamout/
#  ---------------------------------------------------------------------------
#  说明：
#   - 入口文件（app.py / config.py / license_check.py / registry.py / llm.py）
#     保留明文，业务核心包（server/、utils/，以及 --full 的 avatars/、tts/、
#     streamout/）用 PyArmor 加密，避免用户接触核心源码
#   - 输出到 dist-talking/，含明文入口 + 加密包 + web/ 静态资源
#   - 部署：拷贝 dist-talking/ 到目标机器（模型文件 models/ 单独拷贝）
#     启动：python dist-talking/app.py [--license_check=true] [--ZL_AI_SERVER=...]
#   - 注意：首次发布请先跑「最小加密」，验证启动 + /human 全链路正常后再 --full，
#     规避 torch multiprocessing(spawn) 对加密模块的兼容问题
###############################################################################

import argparse
import os
import shutil
import subprocess
import sys

OUT_DIR = "dist-talking"

# 保留明文的入口 / 配置文件
PLAIN_FILES = [
    "app.py",
    "config.py",
    "license_check.py",
    "registry.py",
    "llm.py",
    "requirements.txt",
    ".env.example",
    "Dockerfile",
]

# 拷贝到发布目录的静态/数据目录（模型 models/ 体积大，单独拷贝）
COPY_DIRS = ["web", "data"]


def ensure_pyarmor():
    """确保 pyarmor 可用（缺失则自动安装）。"""
    try:
        subprocess.run(
            [sys.executable, "-m", "pyarmor", "--version"],
            check=True,
            capture_output=True,
        )
    except Exception:
        print("[build] 未检测到 pyarmor，开始安装: pip install pyarmor")
        subprocess.run([sys.executable, "-m", "pip", "install", "pyarmor"], check=True)


def main():
    parser = argparse.ArgumentParser(description="zl-talking PyArmor 加密构建")
    parser.add_argument(
        "--full",
        action="store_true",
        help="全量加密（含 avatars/ tts/ streamout/），默认仅 server/ + utils/",
    )
    args = parser.parse_args()

    ensure_pyarmor()
    shutil.rmtree(OUT_DIR, ignore_errors=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    targets = ["server", "utils"]
    if args.full:
        targets += ["avatars", "tts", "streamout"]

    cmd = [sys.executable, "-m", "pyarmor", "gen", "-O", OUT_DIR, "-r"] + targets
    print(f"[build] 执行加密: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # 拷贝明文入口 / 配置文件
    for name in PLAIN_FILES:
        if os.path.exists(name):
            shutil.copy(name, OUT_DIR)
            print(f"[build] 拷贝明文入口: {name}")

    # 拷贝静态 / 数据目录
    for name in COPY_DIRS:
        if os.path.isdir(name):
            shutil.copytree(name, os.path.join(OUT_DIR, name), dirs_exist_ok=True)
            print(f"[build] 拷贝资源目录: {name}")

    print(f"[build] 完成。发布目录: {OUT_DIR}/")
    print(f"[build] 启动验证: python {OUT_DIR}/app.py [--license_check=true] --ZL_AI_SERVER=http://127.0.0.1:9306")


if __name__ == "__main__":
    main()
