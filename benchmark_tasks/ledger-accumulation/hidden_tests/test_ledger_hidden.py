import unittest
from decimal import Decimal

from ledger import posted_balance


class HiddenLedgerTests(unittest.TestCase):
    def test_empty_is_zero(self) -> None:
        self.assertEqual(posted_balance([]), Decimal("0.00"))

    def test_sums_negative_and_float_compatible_amounts(self) -> None:
        transactions = [
            {"amount": "10.005", "status": "posted"},
            {"amount": Decimal("-1.005"), "status": "posted"},
            {"amount": 0.5, "status": "posted"},
        ]
        self.assertEqual(posted_balance(transactions), Decimal("9.50"))

    def test_bankers_rounding_happens_after_accumulation(self) -> None:
        transactions = [
            {"amount": "0.005", "status": "posted"},
            {"amount": "0.010", "status": "posted"},
        ]
        self.assertEqual(posted_balance(transactions), Decimal("0.02"))


if __name__ == "__main__":
    unittest.main()
