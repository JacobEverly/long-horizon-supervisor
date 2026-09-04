from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
from pathlib import Path

from horizon_supervisor.benchmark.model_catalog import ModelSpec
from horizon_supervisor.benchmark.runner import evaluate_one


async def _run(config_path: Path) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))

    def hard_timeout(_signum: int, _frame: object) -> None:
        os._exit(124)

    signal.signal(signal.SIGALRM, hard_timeout)
    signal.setitimer(signal.ITIMER_REAL, float(config["hard_timeout_seconds"]))
    model = ModelSpec(**config["model"])
    summary, _ = await evaluate_one(
        Path(config["run_root"]),
        model,
        config["task_id"],
        config["api_key_var"],
        config["per_run_cap_usd"],
        workspace_seed=(
            Path(config["workspace_seed"]) if config.get("workspace_seed") else None
        ),
        handoff_context=config.get("handoff_context", ""),
        handoff_evidence=config.get("handoff_evidence") or None,
        checkpoint_id=config.get("checkpoint_id"),
        max_turns=config.get("max_turns", 10),
    )
    Path(config["result_path"]).write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    signal.setitimer(signal.ITIMER_REAL, 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    asyncio.run(_run(args.config))


if __name__ == "__main__":
    main()
