import unittest

from cache import ManualClock, TTLCache


class HiddenCacheTests(unittest.TestCase):
    def test_falsy_and_none_values_are_cached(self) -> None:
        for value in (0, "", False, None):
            with self.subTest(value=value):
                clock = ManualClock()
                cache = TTLCache(clock, 5)
                calls = []
                self.assertEqual(cache.get_or_set("x", lambda: value), value)
                self.assertEqual(
                    cache.get_or_set("x", lambda: calls.append("called") or "new"), value
                )
                self.assertEqual(calls, [])

    def test_expired_entry_is_removed_and_touch_fails(self) -> None:
        clock = ManualClock()
        cache = TTLCache[str](clock, 2)
        cache.set("x", "old")
        clock.advance(2)
        self.assertFalse(cache.touch("x"))
        self.assertNotIn("x", cache._entries)

    def test_loader_runs_once_after_expiration(self) -> None:
        clock = ManualClock()
        cache = TTLCache[str](clock, 1)
        cache.set("x", "old")
        clock.advance(2)
        calls = []
        self.assertEqual(cache.get_or_set("x", lambda: calls.append(1) or "new"), "new")
        self.assertEqual(cache.get_or_set("x", lambda: calls.append(2) or "bad"), "new")
        self.assertEqual(calls, [1])


if __name__ == "__main__":
    unittest.main()
