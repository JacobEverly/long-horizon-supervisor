import unittest
from decimal import Decimal

from ledger import posted_balance


class LedgerTests(unittest.TestCase):
    def test_sums_posted_and_ignores_pending(self) -> None:
        transactions = [
            {"amount": "2.50", "status": "posted"},
            {"amount": "8.00", "status": "pending"},
            {"amount": "1.25", "status": "posted"},
        ]
        self.assertEqual(posted_balance(transactions), Decimal("3.75"))


if __name__ == "__main__":
    unittest.main()
