import math
import unittest

from retry_policy import decide_retry


class HiddenRetryPolicyTests(unittest.TestCase):
    def test_zero_retry_after_is_a_real_override(self) -> None:
        decision = decide_retry(
            attempt=2,
            status_code=429,
            base_delay=5,
            max_delay=30,
            max_attempts=4,
            retry_after=0,
        )
        self.assertEqual(decision.delay_seconds, 0)

    def test_retry_after_and_exponential_delay_are_capped(self) -> None:
        override = decide_retry(
            attempt=1,
            status_code=503,
            base_delay=1,
            max_delay=3,
            max_attempts=5,
            retry_after=10,
        )
        exponential = decide_retry(
            attempt=4,
            status_code=408,
            base_delay=1,
            max_delay=3,
            max_attempts=5,
        )
        self.assertEqual(override.delay_seconds, 3)
        self.assertEqual(exponential.delay_seconds, 3)

    def test_invalid_inputs_are_rejected(self) -> None:
        base = dict(
            attempt=1,
            status_code=500,
            base_delay=1,
            max_delay=5,
            max_attempts=3,
        )
        for change in (
            {"attempt": 0},
            {"base_delay": -1},
            {"max_delay": -1},
            {"max_attempts": 0},
            {"retry_after": -1},
            {"retry_after": math.inf},
            {"retry_after": math.nan},
        ):
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    decide_retry(**(base | change))


if __name__ == "__main__":
    unittest.main()
