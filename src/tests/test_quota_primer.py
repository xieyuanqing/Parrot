import unittest
from unittest.mock import patch

from src import quota_primer


class QuotaPrimerDueReasonTest(unittest.TestCase):
    def setUp(self):
        self.cfg = {
            "graceSeconds": 60,
            "minIntervalSeconds": 300,
            "bootstrapWhenUnknown": False,
            "claudeZeroUtilFallback": True,
            "includeQuotaDisabledAfterReset": True,
        }
        self.acc = {"email": "a@example.com", "disabled_reason": None}

    def _due(self, row, acc=None, cfg=None, *, account_key="claude:a@example.com", now: float = 1_800_000_000):
        with patch.object(quota_primer, "_load_state", return_value={}):
            return quota_primer._due_reason(
                account_key,
                row,
                self.acc if acc is None else acc,
                self.cfg if cfg is None else cfg,
                now=now,
            )

    def test_due_after_known_reset(self):
        row = {"five_hour_reset": "2026-01-01T00:00:00Z", "last_passive_update_at": 0}
        self.assertEqual(self._due(row), (True, "reset_due"))

    def test_skip_when_reset_is_future(self):
        row = {"five_hour_reset": "2099-01-01T00:00:00Z", "last_passive_update_at": 0}
        self.assertEqual(self._due(row), (False, "skip:reset_future"))

    def test_skip_when_model_request_already_happened_after_reset(self):
        row = {"five_hour_reset": "2026-01-01T00:00:00Z", "last_passive_update_at": 1_800_000_000_000}
        self.assertEqual(self._due(row), (False, "skip:already_used_after_reset"))

    def test_never_primes_user_or_auth_disabled_accounts(self):
        row = {"five_hour_reset": "2026-01-01T00:00:00Z", "last_passive_update_at": 0}
        self.assertEqual(self._due(row, {"disabled_reason": "user"}), (False, "skip:user"))
        self.assertEqual(self._due(row, {"disabled_reason": "auth_error"}), (False, "skip:auth_error"))

    def test_quota_disabled_can_prime_after_reset(self):
        row = {"five_hour_reset": "2026-01-01T00:00:00Z", "last_passive_update_at": 0}
        acc = {"email": "a@example.com", "disabled_reason": "quota", "disabled_until": "2026-01-01T00:00:00Z"}
        self.assertEqual(self._due(row, acc), (True, "reset_due"))

    def test_claude_zero_util_fallback_when_reset_is_missing(self):
        row = {"five_hour_reset": None, "five_hour_util": 0.0, "last_passive_update_at": 0}
        self.assertEqual(self._due(row), (True, "zero_util_fallback"))

    def test_claude_zero_util_fallback_skips_recent_model_request(self):
        row = {"five_hour_reset": None, "five_hour_util": 0.0, "last_passive_update_at": 1_799_999_900_000}
        self.assertEqual(self._due(row), (False, "skip:recent_model_request"))

    def test_zero_util_fallback_is_claude_only(self):
        row = {"five_hour_reset": None, "five_hour_util": 0.0, "last_passive_update_at": 0}
        self.assertEqual(
            self._due(row, account_key="openai:a@example.com"),
            (False, "skip:no_known_reset"),
        )

    def test_missing_reset_with_nonzero_util_does_not_prime_without_known_reset(self):
        row = {"five_hour_reset": None, "five_hour_util": 12.0, "last_passive_update_at": 0}
        self.assertEqual(self._due(row), (False, "skip:no_known_reset"))

    def test_unknown_bootstrap_still_requires_explicit_enable(self):
        row = {"five_hour_reset": None, "five_hour_util": 12.0, "last_passive_update_at": 0}
        cfg = dict(self.cfg)
        cfg["bootstrapWhenUnknown"] = True
        self.assertEqual(self._due(row, cfg=cfg), (True, "unknown_bootstrap"))


if __name__ == "__main__":
    unittest.main()
