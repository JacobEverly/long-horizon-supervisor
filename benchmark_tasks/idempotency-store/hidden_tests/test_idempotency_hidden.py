import unittest

from idempotency import IdempotencyStore


class HiddenIdempotencyTests(unittest.TestCase):
    def test_falsy_completed_responses_are_replayed_exactly(self) -> None:
        for response in (None, False, 0, ""):
            with self.subTest(response=response):
                store = IdempotencyStore(10)
                store.begin("key", "fp", 0)
                store.complete("key", "fp", response, 1)
                result = store.begin("key", "fp", 2)
                self.assertEqual(result.status, "replay")
                self.assertEqual(result.response, response)

    def test_expiration_allows_a_new_fingerprint(self) -> None:
        store = IdempotencyStore(5)
        store.begin("key", "first", 0)
        self.assertEqual(store.begin("key", "second", 5).status, "started")

    def test_completion_requires_live_matching_pending_record(self) -> None:
        store = IdempotencyStore(5)
        store.begin("key", "first", 0)
        with self.assertRaises(ValueError):
            store.complete("key", "other", "x", 1)
        with self.assertRaises(KeyError):
            store.complete("key", "first", "x", 5)

    def test_ttl_is_validated_and_completion_refreshes_expiry(self) -> None:
        with self.assertRaises(ValueError):
            IdempotencyStore(0)
        store = IdempotencyStore(5)
        store.begin("key", "fp", 0)
        store.complete("key", "fp", "done", 4)
        self.assertEqual(store.begin("key", "fp", 8).status, "replay")


if __name__ == "__main__":
    unittest.main()
