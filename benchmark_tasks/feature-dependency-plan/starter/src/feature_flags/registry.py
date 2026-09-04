from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from feature_flags.errors import DependencyCycleError, UnknownFeatureError


@dataclass(frozen=True)
class Feature:
    name: str
    dependencies: tuple[str, ...] = ()


class FeatureRegistry:
    def __init__(self) -> None:
        self._features: dict[str, Feature] = {}

    def register(self, name: str, dependencies: Iterable[str] = ()) -> None:
        self._features[name] = Feature(name, tuple(dependencies))

    def evaluation_plan(
        self, requested: Iterable[str], enabled: Iterable[str] = ()
    ) -> list[str]:
        raise NotImplementedError
