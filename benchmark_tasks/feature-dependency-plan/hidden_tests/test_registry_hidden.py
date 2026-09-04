import unittest

from feature_flags import DependencyCycleError, FeatureRegistry, UnknownFeatureError


class HiddenRegistryTests(unittest.TestCase):
    def test_registration_order_breaks_ties(self) -> None:
        registry = FeatureRegistry()
        registry.register("telemetry")
        registry.register("auth")
        registry.register("api", ["auth"])
        registry.register("dashboard", ["telemetry", "auth"])
        self.assertEqual(
            registry.evaluation_plan(["dashboard", "api"]),
            ["telemetry", "auth", "api", "dashboard"],
        )

    def test_shared_dependencies_and_duplicate_requests_emit_once(self) -> None:
        registry = FeatureRegistry()
        registry.register("base")
        registry.register("a", ["base"])
        registry.register("b", ["base"])
        self.assertEqual(registry.evaluation_plan(["a", "b", "a"]), ["base", "a", "b"])

    def test_unknown_requested_and_dependency_are_rejected(self) -> None:
        registry = FeatureRegistry()
        registry.register("known", ["missing-dependency"])
        with self.assertRaises(UnknownFeatureError) as requested:
            registry.evaluation_plan(["missing-request"])
        self.assertEqual(requested.exception.feature, "missing-request")
        with self.assertRaises(UnknownFeatureError) as dependency:
            registry.evaluation_plan(["known"])
        self.assertEqual(dependency.exception.feature, "missing-dependency")
        with self.assertRaises(UnknownFeatureError):
            registry.evaluation_plan(["known"], enabled=["missing-dependency"])

    def test_cycle_reports_closed_path(self) -> None:
        registry = FeatureRegistry()
        registry.register("a", ["b"])
        registry.register("b", ["c"])
        registry.register("c", ["a"])
        with self.assertRaises(DependencyCycleError) as raised:
            registry.evaluation_plan(["a"])
        cycle = raised.exception.cycle
        self.assertGreaterEqual(len(cycle), 4)
        self.assertEqual(cycle[0], cycle[-1])
        self.assertEqual(set(cycle[:-1]), {"a", "b", "c"})


if __name__ == "__main__":
    unittest.main()
