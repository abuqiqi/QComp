"""将实验事件统一追加到 JSON Lines 日志。

本模块接收文件路径、事件名称和结构化字段，补充 UTC 时间后逐条写入并关闭文件。
调用方决定实验文件和记录时机；本模块不执行实验或计算指标。

主要内容：
- ``log_event``：创建父目录并追加一条 JSON 事件记录。
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


def log_event(path: str | Path, event: str, **fields: Any) -> None:
    """向文件追加一条带 UTC 时间的 JSON 事件，返回前关闭文件。

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

