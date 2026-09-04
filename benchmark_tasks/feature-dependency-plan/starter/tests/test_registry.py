import unittest

from feature_flags import FeatureRegistry


class RegistryTests(unittest.TestCase):
    def test_dependencies_come_first(self) -> None:
        registry = FeatureRegistry()
        registry.register("database")
        registry.register("search", ["database"])
        registry.register("recommendations", ["search"])
        self.assertEqual(
            registry.evaluation_plan(["recommendations"]),
            ["database", "search", "recommendations"],
        )

    def test_enabled_dependencies_are_omitted(self) -> None:
        registry = FeatureRegistry()
        registry.register("base")
        registry.register("child", ["base"])
        self.assertEqual(registry.evaluation_plan(["child"], enabled=["base"]), ["child"])


if __name__ == "__main__":
    unittest.main()
