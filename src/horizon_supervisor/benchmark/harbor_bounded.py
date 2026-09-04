from __future__ import annotations

import os
import sys
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import Any

from tenacity import stop_after_attempt


def configure_harbor_retries(attempts: int) -> None:
    """Bound Harbor's two nested LLM retry decorators before the CLI starts."""
    if attempts < 1:
        raise ValueError("Harbor LLM attempts must be positive")

    from harbor.agents.terminus_2.terminus_2 import Terminus2
    from harbor.llms.lite_llm import LiteLLM

    Terminus2._query_llm.retry.stop = stop_after_attempt(attempts)
    LiteLLM.call.retry.stop = stop_after_attempt(attempts)


def _bound_output_length_recursion(
    original: Callable[..., Awaitable[Any]],
    corrective_attempts: int,
) -> Callable[..., Awaitable[Any]]:
    """Stop Terminus from recursively retrying the same truncated response forever."""
    if corrective_attempts < 0:
        raise ValueError("output-length corrective attempts cannot be negative")

    depth_attribute = "_horizon_output_length_retry_depth"

    @wraps(original)
    async def bounded(self: Any, *args: Any, **kwargs: Any) -> Any:
        depth = int(getattr(self, depth_attribute, 0))
        if depth > corrective_attempts:
            from harbor.llms.base import OutputLengthExceededError

            raise OutputLengthExceededError(
                "Horizon stopped repeated max-output retries after "
                f"{corrective_attempts} corrective attempt(s)."
            )
        setattr(self, depth_attribute, depth + 1)
        try:
            return await original(self, *args, **kwargs)
        finally:
            setattr(self, depth_attribute, depth)

    return bounded


def configure_output_length_retries(corrective_attempts: int) -> None:
    """Cap Terminus' recursive retry path for max-output truncation."""
    if corrective_attempts < 0:
        raise ValueError("output-length corrective attempts cannot be negative")

    from harbor.agents.terminus_2.terminus_2 import Terminus2

    current = Terminus2._query_llm
    original = getattr(current, "_horizon_original", current)
    bounded = _bound_output_length_recursion(original, corrective_attempts)
    bounded._horizon_original = original  # type: ignore[attr-defined]
    if hasattr(original, "retry"):
        bounded.retry = original.retry  # type: ignore[attr-defined]
    Terminus2._query_llm = bounded


def main() -> None:
    attempts = int(os.getenv("HORIZON_HARBOR_LLM_ATTEMPTS", "1"))
    output_length_retries = int(
        os.getenv("HORIZON_HARBOR_OUTPUT_LENGTH_RETRIES", "1")
    )
    configure_harbor_retries(attempts)
    configure_output_length_retries(output_length_retries)

    from harbor.cli.main import app

    sys.exit(app())


if __name__ == "__main__":
    main()
