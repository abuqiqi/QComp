"""验证公共 JSON Lines 日志的追加行为。

本模块用临时目录检查事件记录的可解析性、文件隔离和失败行为。
不加载模型或执行评测。

主要内容：
- ``LoggingTests``：验证公共事件日志。
"""

import json
import subprocess
import sys
import textwrap
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from qcomp import log_event


class LoggingTests(unittest.TestCase):
    """检查公共日志接口的文件输出。"""

    def test_append_and_independent_files(self) -> None:
        """追加保留原有记录，时间使用 UTC+8，换行可解析且不同文件互不影响。"""

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
            self.assertEqual(
                datetime.fromisoformat(records[0]["timestamp"]).utcoffset(),
                timedelta(hours=8),
            )
            self.assertEqual(json.loads(other.read_text())["event"], "other")

    def test_console_captures_process_output_and_failure(self) -> None:
        """子进程验证 stdout、已有 handler、原生输出、堆栈和退出后恢复。"""
        with TemporaryDirectory() as directory:
            path = Path(directory) / "console.log"
            source = textwrap.dedent("""\
                import logging
                import os
                import subprocess
                import sys
                from qcomp.logging import capture_console
                logging.basicConfig(level=logging.INFO)
                try:
                    with capture_console(sys.argv[1]):
                        print("stdout marker")
                        logging.info("existing handler marker")
                        os.write(2, b"native stderr marker\\n")
                        subprocess.run([sys.executable, "-c", "print('child marker')"], check=True)
                        raise RuntimeError("failure marker")
                except RuntimeError:
                    pass
                print("restored marker")
                with capture_console(sys.argv[1]):
                    print("append marker")
                """)
            result = subprocess.run(
                [sys.executable, "-c", source, str(path)],
                capture_output=True, text=True, timeout=30, check=True,
            )
            content = path.read_text()
            for marker in (
                "stdout marker", "existing handler marker", "native stderr marker",
                "child marker", "append marker",
            ):
                self.assertIn(marker, content)
                self.assertIn(marker, result.stdout)
            self.assertIn("Traceback", content)
            self.assertIn("RuntimeError: failure marker", content)
            self.assertNotIn("restored marker", content)
            self.assertIn("restored marker", result.stdout)

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
