from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from horizon_supervisor.benchmark.gate4 import run_recovery_gate
from horizon_supervisor.benchmark.gate5 import run_state_recovery_gate
from horizon_supervisor.benchmark.gate6 import run_gate6
from horizon_supervisor.benchmark.gate7 import (
    DEFAULT_SWITCHYARD_CONFIG,
    Gate7SmokeConfig,
    run_gate7_smoke,
)
from horizon_supervisor.benchmark.gate8 import (
    DEFAULT_BUDGET_CONTRACT,
    DEFAULT_ROUTES,
    DEFAULT_TASKS,
    Gate8PilotConfig,
    run_gate8_pilot,
)
from horizon_supervisor.benchmark.runner import run_mock, run_tournament


def main() -> None:
    parser = argparse.ArgumentParser(description="Run long-horizon supervisor benchmark gates")
    parser.add_argument(
        "gate",
        choices=[
            "gate1",
            "gate3",
            "gate4",
            "gate5",
            "gate6",
            "gate7-smoke",
            "gate8-pilot",
        ],
    )
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--budget-usd", type=float)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--source-gate4", type=Path)
    parser.add_argument("--replications", type=int, default=3)
    parser.add_argument("--task", default="log-summary-date-ranges")
    parser.add_argument("--route", default="gate7/stage-quality")
    parser.add_argument("--environment", default="daytona")
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--max-output-tokens", type=int, default=4_096)
    parser.add_argument("--request-timeout-seconds", type=int, default=1_200)
    parser.add_argument("--request-retry-attempts", type=int, default=1)
    parser.add_argument("--output-length-retry-attempts", type=int, default=1)
    parser.add_argument("--wall-timeout-seconds", type=int, default=4_200)
    parser.add_argument("--include-task", action="append", default=[])
    parser.add_argument("--include-route", action="append", default=[])
    parser.add_argument("--n-concurrent", type=int, default=2)
    parser.add_argument("--wave", type=int, default=1)
    parser.add_argument("--tasks-path", type=Path)
    parser.add_argument("--budget-contract", type=Path)
    parser.add_argument("--switchyard-config", type=Path)
    parser.add_argument("--model-roster", default="gate4")
    args = parser.parse_args()
    if args.gate == "gate1":
        result = asyncio.run(run_mock(args.artifacts / "gate1"))
    elif args.gate == "gate3":
        result = asyncio.run(
            run_tournament(
                args.artifacts,
                budget_usd=args.budget_usd or 50.0,
                resume_root=args.resume,
            )
        )
    elif args.gate == "gate4":
        result = asyncio.run(
            run_recovery_gate(
                args.artifacts,
                budget_usd=args.budget_usd or 5.0,
                resume_root=args.resume,
            )
        )
    elif args.gate == "gate5":
        if args.source_gate4 is None:
            parser.error("gate5 requires --source-gate4")
        result = asyncio.run(
            run_state_recovery_gate(
                args.artifacts,
                source_gate4_root=args.source_gate4,
                budget_usd=args.budget_usd or 2.0,
                resume_root=args.resume,
            )
        )
    elif args.gate == "gate6":
        if args.source_gate4 is None:
            parser.error("gate6 requires --source-gate4")
        result = asyncio.run(
            run_gate6(
                args.artifacts,
                source_gate4_root=args.source_gate4,
                replications_per_arm=args.replications,
                budget_usd=args.budget_usd or 3.0,
                resume_root=args.resume,
            )
        )
    elif args.gate == "gate7-smoke":
        result = run_gate7_smoke(
            Gate7SmokeConfig(
                artifacts_root=args.artifacts,
                task_name=args.task,
                route_id=args.route,
                environment=args.environment,
                max_turns=args.max_turns,
                max_output_tokens=args.max_output_tokens,
                authorized_model_budget_usd=args.budget_usd or 1.0,
                wall_timeout_seconds=args.wall_timeout_seconds,
            )
        )
    else:
        result = run_gate8_pilot(
            Gate8PilotConfig(
                artifacts_root=args.artifacts,
                wave=args.wave,
                tasks_path=(
                    args.tasks_path
                    or (
                        DEFAULT_TASKS
                        if args.wave == 1
                        else Path(
                            f"data/supervisor/terminal-bench-pro-wave-{args.wave}/tasks"
                        )
                    )
                ),
                budget_contract_path=(
                    args.budget_contract
                    or (
                        DEFAULT_BUDGET_CONTRACT
                        if args.wave == 1
                        else Path(f"benchmarks/gate8-wave{args.wave}-budget-v0.json")
                    )
                ),
                switchyard_config_path=(
                    args.switchyard_config or DEFAULT_SWITCHYARD_CONFIG
                ),
                route_ids=tuple(args.include_route) or DEFAULT_ROUTES,
                include_task_names=tuple(args.include_task),
                environment=args.environment,
                n_concurrent=args.n_concurrent,
                max_turns=args.max_turns,
                max_output_tokens=args.max_output_tokens,
                request_timeout_seconds=args.request_timeout_seconds,
                request_retry_attempts=args.request_retry_attempts,
                output_length_retry_attempts=args.output_length_retry_attempts,
                authorized_model_budget_usd=args.budget_usd or 50.0,
                wall_timeout_seconds=args.wall_timeout_seconds,
                model_roster=args.model_roster,
            )
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
