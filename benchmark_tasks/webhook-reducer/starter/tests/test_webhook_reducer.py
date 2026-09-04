import unittest
from decimal import Decimal

from webhook_reducer import replay_balance


class WebhookReducerTests(unittest.TestCase):
    def test_events_are_replayed_by_version(self) -> None:
        events = [
            {"id": "two", "version": 2, "kind": "debit", "amount": "4"},
            {"id": "one", "version": 1, "kind": "credit", "amount": "10"},
        ]
        self.assertEqual(replay_balance(events), Decimal("6.00"))

    def test_identical_delivery_is_idempotent(self) -> None:
        event = {"id": "one", "version": 1, "kind": "credit", "amount": "2.345"}
        self.assertEqual(replay_balance([event, dict(event)]), Decimal("2.34"))


if __name__ == "__main__":
    unittest.main()
