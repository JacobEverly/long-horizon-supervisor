import unittest

from idempotency import IdempotencyStore


class IdempotencyTests(unittest.TestCase):
    def test_pending_then_replay(self) -> None:
        store = IdempotencyStore(10)
        self.assertEqual(store.begin("key", "fp", 0).status, "started")
        self.assertEqual(store.begin("key", "fp", 1).status, "in_progress")
        store.complete("key", "fp", {"ok": True}, 2)
        result = store.begin("key", "fp", 3)
        self.assertEqual(result.status, "replay")
        self.assertEqual(result.response, {"ok": True})

    def test_different_fingerprint_conflicts(self) -> None:
        store = IdempotencyStore(10)
        store.begin("key", "first", 0)
        self.assertEqual(store.begin("key", "second", 1).status, "conflict")


if __name__ == "__main__":
    unittest.main()
