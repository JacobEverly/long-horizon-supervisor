import unittest

from quota import allocate_quota


class HiddenQuotaTests(unittest.TestCase):
    def test_partial_final_round_preserves_insertion_order(self) -> None:
        allocation = allocate_quota(
            {"alpha": 9, "beta": 9, "gamma": 9},
            capacity=5,
            weights={"alpha": 2, "beta": 2, "gamma": 2},
        )
        self.assertEqual(allocation, {"alpha": 2, "beta": 2, "gamma": 1})

    def test_zero_capacity_and_demand_are_retained(self) -> None:
        self.assertEqual(
            allocate_quota(
                {"zero": 0, "active": 4}, capacity=0, weights={"active": 1}
            ),
            {"zero": 0, "active": 0},
        )

    def test_invalid_inputs_are_rejected(self) -> None:
        cases = [
            ({"a": 1}, -1, {"a": 1}),
            ({"a": -1}, 1, {"a": 1}),
            ({"a": 1}, 1, {"a": 0}),
            ({"a": 1}, 1, {}),
            ({"a": 1}, 1, {"a": 1, "unknown": 1}),
            ({"a": True}, 1, {"a": 1}),
        ]
        for demands, capacity, weights in cases:
            with self.subTest(case=(demands, capacity, weights)):
                with self.assertRaises(ValueError):
                    allocate_quota(demands, capacity, weights)

    def test_capacity_larger_than_demand_does_not_overallocate(self) -> None:
        self.assertEqual(
            allocate_quota(
                {"alpha": 2, "beta": 1}, capacity=99, weights={"alpha": 3, "beta": 1}
            ),
            {"alpha": 2, "beta": 1},
        )


if __name__ == "__main__":
    unittest.main()
