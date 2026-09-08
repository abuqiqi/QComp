"""保存实验的结构化事件与终端输出。

本模块接收文件路径、事件名称和结构化字段，补充北京时间（UTC+8）后逐条写入并关闭文件。
调用方决定实验文件和记录时机；终端捕获上下文将进程输出同步写入文本日志，
退出时恢复输出，并保存异常堆栈。本模块不执行实验或计算指标。

主要内容：
- ``log_event``：创建父目录并追加一条 JSON 事件记录。
- ``capture_console``：在 Linux 实验进程中同步保存终端输出与异常。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


def log_event(path: str | Path, event: str, **fields: Any) -> None:
    """向文件追加一条带北京时间（UTC+8）的 JSON 事件，返回前关闭文件。

    参数：
        path: JSON Lines 文件路径；父目录自动创建，已有内容保留。
        event: 事件名称，例如 ``experiment_started``。
        **fields: 可被 JSON 序列化的实验字段，保存在记录的 ``fields`` 对象中。

    异常：
        TypeError: 字段包含无法序列化为 JSON 的对象时抛出。
        ValueError: 字段包含 NaN 或无穷数值时抛出。
        OSError: 创建目录或写入文件失败时抛出。
    """

    record = {
        "timestamp": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "event": event,
        "fields": fields,
    }
    line = json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as stream:
        stream.write(line)


@contextmanager
def capture_console(path: str | Path) -> Iterator[None]:
    """将进程 stdout/stderr 同步追加到 path，保留终端显示并记录异常堆栈。

    参数：
        path: 终端日志路径；父目录自动创建。需要 Linux 的 ``tee`` 命令。

    文件描述符重定向覆盖已有 logging handler、进度条和子进程输出。
    此上下文用于独占进程的实验入口，不用于并发线程中的独立实验。
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout.flush()
    sys.stderr.flush()
    stdout_fd, stderr_fd = os.dup(1), os.dup(2)
    try:
        try:
            with subprocess.Popen(
                ["tee", "-a", "--", str(destination)],
                stdin=subprocess.PIPE,
                stdout=stdout_fd,
                stderr=stderr_fd,
            ) as process:
                assert process.stdin is not None
                try:
                    os.dup2(process.stdin.fileno(), 1)
                    os.dup2(process.stdin.fileno(), 2)
                    yield
                finally:
                    try:
                        sys.stdout.flush()
                        sys.stderr.flush()
                    finally:
                        os.dup2(stdout_fd, 1)
                        os.dup2(stderr_fd, 2)
                        process.stdin.close()
                if process.wait() != 0:
                    raise OSError(f"console logging failed: {destination}")
        except BaseException:
            # 等待 tee 写完后追加堆栈；异常继续交给调用方，终端只显示一次。
            with destination.open("a", encoding="utf-8") as stream:
                traceback.print_exc(file=stream)
            raise
    finally:
        os.close(stdout_fd)
        os.close(stderr_fd)
