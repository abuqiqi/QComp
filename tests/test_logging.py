"""验证公共 JSON Lines 日志的追加行为。

本模块用临时目录检查事件记录的可解析性、文件隔离和失败行为。
不加载模型或执行评测。

主要内容：
- ``LoggingTests``：验证公共事件日志。
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from qcomp import log_event


class LoggingTests(unittest.TestCase):
    """检查公共日志接口的文件输出。"""

    def test_append_and_independent_files(self) -> None:
        """追加保留原有记录，换行内容可解析且不同文件互不影响。"""

        with TemporaryDirectory() as directory:
            path = Path(directory) / "nested/first.jsonl"
            other = Path(directory) / "second.jsonl"
            log_event(path, "started", message="中文\n内容", timestamp="自定义字段")
            first = path.read_text(encoding="utf-8")
            log_event(str(path), "completed", latency_ms=1.23)
            log_event(other, "other")
            content = path.read_text(encoding="utf-8")
            self.assertTrue(content.startswith(first))
            records = [json.loads(line) for line in content.splitlines()]
            self.assertEqual([r["event"] for r in records], ["started", "completed"])
            self.assertEqual(records[0]["fields"]["message"], "中文\n内容")
            self.assertEqual(records[0]["fields"]["timestamp"], "自定义字段")
            self.assertEqual(datetime.fromisoformat(records[0]["timestamp"]).utcoffset(), timedelta(0))
            self.assertEqual(json.loads(other.read_text())["event"], "other")

    def test_serialization_failure_preserves_file(self) -> None:
        """不支持的对象或非有限数值不会向文件留下不完整记录。"""

        with TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            log_event(path, "started")
            original = path.read_bytes()
            for value, error in ((object(), TypeError), (float("nan"), ValueError)):
                with self.subTest(value=value), self.assertRaises(error):
                    log_event(path, "invalid", value=value)
                self.assertEqual(path.read_bytes(), original)
