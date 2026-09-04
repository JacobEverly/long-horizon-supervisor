import unittest

from cache import ManualClock, TTLCache


class CacheTests(unittest.TestCase):
    def test_get_does_not_slide_expiration(self) -> None:
        clock = ManualClock()
        cache = TTLCache[str](clock, 10)
        cache.set("key", "value")
        clock.advance(8)
        self.assertEqual(cache.get("key"), "value")
        clock.advance(3)
        self.assertIsNone(cache.get("key"))

    def test_touch_extends_expiration(self) -> None:
        clock = ManualClock()
        cache = TTLCache[str](clock, 10)
        cache.set("key", "value")
        clock.advance(8)
        self.assertTrue(cache.touch("key"))
        clock.advance(3)
        self.assertEqual(cache.get("key"), "value")


if __name__ == "__main__":
    unittest.main()
