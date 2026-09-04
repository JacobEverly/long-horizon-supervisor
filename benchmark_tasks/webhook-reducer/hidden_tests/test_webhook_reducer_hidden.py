import unittest
from decimal import Decimal

from webhook_reducer import DuplicateEventError, VersionGapError, replay_balance


class HiddenWebhookReducerTests(unittest.TestCase):
    def test_conflicting_event_id_or_version_is_rejected(self) -> None:
        first = {"id": "one", "version": 1, "kind": "credit", "amount": "5"}
        with self.assertRaises(DuplicateEventError):
            replay_balance([first, first | {"amount": "6"}])
        with self.assertRaises(DuplicateEventError):
            replay_balance([first, {"id": "other", **(first | {"id": "other"})}])

    def test_version_gap_is_rejected(self) -> None:
        with self.assertRaises(VersionGapError):
            replay_balance(
                [{"id": "two", "version": 2, "kind": "credit", "amount": "1"}]
            )

    def test_invalid_kind_negative_amount_and_overdraft_are_rejected(self) -> None:
        base = {"id": "one", "version": 1, "kind": "credit", "amount": "1"}
        for event in (
            base | {"kind": "refund"},
            base | {"amount": "-1"},
            base | {"kind": "debit", "amount": "1"},
        ):
            with self.subTest(event=event):
                with self.assertRaises(ValueError):
                    replay_balance([event])

    def test_bankers_rounding_happens_after_replay(self) -> None:
        events = [
            {"id": "one", "version": 1, "kind": "credit", "amount": 0.105},
            {"id": "two", "version": 2, "kind": "credit", "amount": "0.10"},
        ]
        self.assertEqual(replay_balance(events), Decimal("0.20"))


if __name__ == "__main__":
    unittest.main()
