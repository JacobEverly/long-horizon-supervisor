import unittest

from retry_policy import decide_retry


class RetryPolicyTests(unittest.TestCase):
    def test_first_retry_uses_base_delay(self) -> None:
        decision = decide_retry(
            attempt=1,
            status_code=500,
            base_delay=2,
            max_delay=30,
            max_attempts=4,
        )
        self.assertTrue(decision.retry)
        self.assertEqual(decision.delay_seconds, 2)

    def test_attempt_at_limit_is_not_retried(self) -> None:
        decision = decide_retry(
            attempt=3,
            status_code=503,
            base_delay=1,
            max_delay=30,
            max_attempts=3,
        )
        self.assertFalse(decision.retry)

    def test_transport_and_conflict_are_retryable(self) -> None:
        for status in (None, 409):
            with self.subTest(status=status):
                self.assertTrue(
                    decide_retry(
                        attempt=1,
                        status_code=status,
                        base_delay=1,
                        max_delay=10,
                        max_attempts=3,
                    ).retry
                )


if __name__ == "__main__":
    unittest.main()
