class UnknownFeatureError(ValueError):
    def __init__(self, feature: str) -> None:
        self.feature = feature
        super().__init__(f"unknown feature: {feature}")


class DependencyCycleError(ValueError):
    def __init__(self, cycle: list[str]) -> None:
        self.cycle = cycle
        super().__init__("dependency cycle: " + " -> ".join(cycle))
