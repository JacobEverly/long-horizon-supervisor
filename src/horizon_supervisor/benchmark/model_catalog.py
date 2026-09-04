from __future__ import annotations

import json
import urllib.request
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    label: str
    role: str
    input_usd_per_token: float
    output_usd_per_token: float
    context_length: int
    tier: int = 0
    reasoning_effort: str | None = None
    cached_input_usd_per_token: float | None = None
    cache_write_input_usd_per_token: float | None = None

    @property
    def input_usd_per_million(self) -> float:
        return self.input_usd_per_token * 1_000_000

    @property
    def output_usd_per_million(self) -> float:
        return self.output_usd_per_token * 1_000_000


GATE3_MODEL_ROLES = {
    "moonshotai/kimi-k3": ("Kimi K3", "high-reasoning ceiling"),
    "z-ai/glm-5.2": ("GLM 5.2", "middle-tier generalist"),
    "deepseek/deepseek-v4-flash": ("DeepSeek V4 Flash", "fast reasoning value"),
    "qwen/qwen3-8b": ("Qwen3 8B", "small/cheap floor"),
}

# Exact, dated endpoints captured from OpenRouter's public catalog on 2026-08-20.
# Ordered from the compact floor to the expensive reasoning ceiling so recovery
# policies can step upward deterministically.
GATE4_MODEL_ROLES = {
    "qwen/qwen3.8-27b": ("Qwen3.8 27B", "compact reasoning floor", 0, "high"),
    "deepseek/deepseek-v4-flash-0731": (
        "DeepSeek V4 Flash 0731",
        "fast reasoning value",
        1,
        "high",
    ),
    "deepseek/deepseek-v4-pro-0813": (
        "DeepSeek V4 Pro 0813",
        "reasoning step-up",
        2,
        "high",
    ),
    "z-ai/glm-5.3": ("GLM 5.3", "long-horizon frontier", 3, "max"),
    "moonshotai/kimi-k3": ("Kimi K3", "high-reasoning ceiling", 4, "high"),
}

# Used only when the public catalog is temporarily unavailable. Values are per token.
GATE3_FALLBACK_PRICING = {
    "qwen/qwen3-8b": (0.000000117, 0.000000455, 131_072),
    "deepseek/deepseek-v4-flash": (0.0000000812, 0.0000001624, 1_048_576),
    "z-ai/glm-5.2": (0.000000966, 0.000003036, 1_048_576),
    "moonshotai/kimi-k3": (0.000003, 0.000015, 1_048_576),
}

GATE4_FALLBACK_PRICING = {
    "qwen/qwen3.8-27b": (0.00000045, 0.0000032, 1_000_000),
    "deepseek/deepseek-v4-flash-0731": (0.00000014, 0.00000028, 1_310_720),
    "deepseek/deepseek-v4-pro-0813": (0.000001188, 0.000003564, 1_048_576),
    "z-ai/glm-5.3": (0.0000014, 0.0000044, 1_048_576),
    "moonshotai/kimi-k3": (0.000003, 0.000015, 1_048_576),
}

SWISS_CHEESE_SMALL_MODEL_ID = "qwen/qwen3.5-9b"
SWISS_CHEESE_MODEL_ROLES = {
    **GATE4_MODEL_ROLES,
    SWISS_CHEESE_SMALL_MODEL_ID: (
        "Qwen3.5 9B",
        "small clean-start coverage probe",
        -1,
        "high",
    ),
}
SWISS_CHEESE_FALLBACK_PRICING = {
    **GATE4_FALLBACK_PRICING,
    SWISS_CHEESE_SMALL_MODEL_ID: (
        0.0000001,
        0.00000015,
        262_144,
    ),
}


def load_model_catalog(
    snapshot_path: Path | None = None, *, roster: str = "gate3"
) -> list[ModelSpec]:
    if roster == "gate3":
        roles = {
            model_id: (label, role, index, None)
            for index, (model_id, (label, role)) in enumerate(GATE3_MODEL_ROLES.items())
        }
        fallback = GATE3_FALLBACK_PRICING
    elif roster == "gate4":
        roles = GATE4_MODEL_ROLES
        fallback = GATE4_FALLBACK_PRICING
    elif roster == "swiss_cheese":
        roles = SWISS_CHEESE_MODEL_ROLES
        fallback = SWISS_CHEESE_FALLBACK_PRICING
    else:
        raise ValueError(f"unknown model roster: {roster}")
    source = "openrouter-public-catalog"
    try:
        request = urllib.request.Request(
            "https://openrouter.ai/api/v1/models",
            headers={"User-Agent": "long-horizon-supervisor/0.1"},
        )
        with urllib.request.urlopen(request, timeout=20) as response:
            payload = json.load(response)
        raw_by_id = {item["id"]: item for item in payload["data"]}
        rows = {}
        for model_id in roles:
            item = raw_by_id[model_id]
            if "tools" not in item.get("supported_parameters", []):
                raise RuntimeError(f"{model_id} no longer advertises tool support")
            rows[model_id] = (
                float(item["pricing"]["prompt"]),
                float(item["pricing"]["completion"]),
                int(item["context_length"]),
                float(
                    item["pricing"].get(
                        "input_cache_read", item["pricing"]["prompt"]
                    )
                ),
                float(
                    item["pricing"].get(
                        "input_cache_write", item["pricing"]["prompt"]
                    )
                ),
            )
    except Exception:
        source = "pinned-fallback"
        rows = fallback

    catalog = [
        ModelSpec(
            model_id=model_id,
            label=roles[model_id][0],
            role=roles[model_id][1],
            input_usd_per_token=rows[model_id][0],
            output_usd_per_token=rows[model_id][1],
            context_length=rows[model_id][2],
            tier=roles[model_id][2],
            reasoning_effort=roles[model_id][3],
            cached_input_usd_per_token=(
                rows[model_id][3] if len(rows[model_id]) > 3 else rows[model_id][0]
            ),
            cache_write_input_usd_per_token=(
                rows[model_id][4] if len(rows[model_id]) > 4 else rows[model_id][0]
            ),
        )
        for model_id in roles
    ]
    if snapshot_path is not None:
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text(
            json.dumps(
                {
                    "captured_at": datetime.now(UTC).isoformat(),
                    "source": source,
                    "roster": roster,
                    "models": [asdict(model) for model in catalog],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return catalog
