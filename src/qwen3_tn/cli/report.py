"""Update the machine-controlled section of the weekly report."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from ..provenance import atomic_write_json, load_json
from ..reporting import update_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--state-file", type=Path, required=True)
    parser.add_argument(
        "--status", choices=("pending", "running", "success", "failure"), required=True
    )
    parser.add_argument("--stage", required=True)
    parser.add_argument("--start-time")
    parser.add_argument("--artifact-root")
    parser.add_argument("--log-path")
    parser.add_argument("--run-json")
    parser.add_argument("--decomposition-json")
    parser.add_argument("--verification-json")
    args = parser.parse_args()
    state = load_json(args.state_file) if args.state_file.is_file() else {}
    state.update({"status": args.status, "stage": args.stage})
    for key in (
        "artifact_root",
        "log_path",
        "run_json",
        "decomposition_json",
        "verification_json",
    ):
        value = getattr(args, key)
        if value is not None:
            state[key] = value
    if args.start_time is not None:
        if "start_time" not in state:
            state["start_time"] = args.start_time
            state["attempt_count"] = 1
        else:
            state["last_resume_time"] = args.start_time
            state["attempt_count"] = int(state.get("attempt_count", 1)) + 1
        state.pop("end_time", None)
    if args.status in {"success", "failure"}:
        state["end_time"] = (
            datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
        )
    atomic_write_json(args.state_file, state)
    update_report(
        args.report,
        state,
        run_json=args.run_json,
        decomposition_json=args.decomposition_json,
        verification_json=args.verification_json,
    )


if __name__ == "__main__":
    main()
