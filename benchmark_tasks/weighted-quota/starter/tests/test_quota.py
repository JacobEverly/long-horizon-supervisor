import unittest

from quota import allocate_quota


class QuotaTests(unittest.TestCase):
    def test_repeated_weighted_rounds(self) -> None:
        allocation = allocate_quota(
            {"alpha": 10, "beta": 10}, capacity=6, weights={"alpha": 2, "beta": 1}
        )
        self.assertEqual(allocation, {"alpha": 4, "beta": 2})

    def test_satisfied_tenants_are_skipped(self) -> None:
        allocation = allocate_quota(
            {"alpha": 1, "beta": 5}, capacity=4, weights={"alpha": 2, "beta": 1}
        )
        self.assertEqual(allocation, {"alpha": 1, "beta": 3})


if __name__ == "__main__":
    unittest.main()
