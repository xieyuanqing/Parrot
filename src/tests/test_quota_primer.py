import asyncio
import unittest
from typing import Any, cast
from unittest.mock import patch

from src import quota_primer
from src.channel.base import UpstreamRequest


class QuotaPrimerDueReasonTest(unittest.TestCase):
    def setUp(self):
        self.cfg = {
            "postResetDelaySeconds": 360,
            "minIntervalSeconds": 300,
            "bootstrapWhenUnknown": False,
            "claudeZeroUtilFallback": True,
            "includeQuotaDisabledAfterReset": True,
        }
        self.acc = {"email": "a@example.com", "disabled_reason": None}

    def _due(self, row, acc=None, cfg=None, state=None, *, account_key="claude:a@example.com", now: float = 1_800_000_000):
        with patch.object(quota_primer, "_load_state", return_value={} if state is None else state):
            return quota_primer._due_reason(
                account_key,
                row,
                self.acc if acc is None else acc,
                self.cfg if cfg is None else cfg,
                now=now,
            )

    def test_due_only_after_known_reset_plus_delay(self):
        reset = quota_primer._parse_iso_utc("2026-01-01T00:00:00Z")
        self.assertIsNotNone(reset)
        reset = cast(float, reset)
        row = {"five_hour_reset": "2026-01-01T00:00:00Z", "last_passive_update_at": 0}
        self.assertEqual(self._due(row, now=reset + 359), (False, "skip:post_reset_delay"))
        self.assertEqual(self._due(row, now=reset + 360), (True, "reset_due"))

    def test_skip_when_reset_is_future(self):
        row = {"five_hour_reset": "2099-01-01T00:00:00Z", "last_passive_update_at": 0}
        self.assertEqual(self._due(row), (False, "skip:post_reset_delay"))

    def test_next_reset_returned_by_success_becomes_next_cycle(self):
        old_reset = "2026-01-01T00:00:00Z"
        next_reset = "2026-01-01T05:06:00Z"
        state = {
            "last_trigger_reset": old_reset,
            "last_success_at": cast(float, quota_primer._parse_iso_utc(old_reset)) + 360,
            "observed_next_reset": next_reset,
        }
        row = {"five_hour_reset": next_reset, "last_passive_update_at": 0}
        next_ts = quota_primer._parse_iso_utc(next_reset)
        self.assertIsNotNone(next_ts)
        next_ts = cast(float, next_ts)
        self.assertEqual(self._due(row, state=state, now=next_ts + 359), (False, "skip:post_reset_delay"))
        self.assertEqual(self._due(row, state=state, now=next_ts + 360), (True, "reset_due"))

    def test_same_reset_is_not_primed_twice_after_success(self):
        reset = "2026-01-01T00:00:00Z"
        row = {"five_hour_reset": reset, "last_passive_update_at": 0}
        state = {"last_trigger_reset": reset, "last_success_at": 1_799_999_000}
        self.assertEqual(self._due(row, state=state), (False, "skip:reset_already_primed"))

    def test_legacy_next_reset_marker_does_not_block_new_cycle(self):
        reset = "2026-01-01T00:00:00Z"
        row = {"five_hour_reset": reset, "last_passive_update_at": 0}
        state = {"last_reset_primed": reset, "last_prime_at": 1_700_000_000, "last_ok": True}
        self.assertEqual(self._due(row, state=state), (True, "reset_due"))

    def test_failure_uses_short_retry_window_not_success_interval(self):
        row = {"five_hour_reset": "2026-01-01T00:00:00Z", "last_passive_update_at": 0}
        waiting = {"last_ok": False, "last_attempt_at": 1_799_999_900, "next_retry_at": 1_800_000_200}
        self.assertEqual(self._due(row, state=waiting), (False, "skip:retry_backoff"))
        self.assertEqual(self._due(row, state=waiting, now=1_800_000_201), (True, "reset_due"))

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

    def test_loop_sleep_seconds_applies_jitter(self):
        with patch.object(quota_primer.random, "uniform", return_value=42.0):
            self.assertEqual(
                quota_primer._loop_sleep_seconds({"intervalSeconds": 600, "intervalJitterSeconds": 90}),
                642.0,
            )

    def test_loop_sleep_seconds_has_floor(self):
        with patch.object(quota_primer.random, "uniform", return_value=-90.0):
            self.assertEqual(
                quota_primer._loop_sleep_seconds({"intervalSeconds": 60, "intervalJitterSeconds": 90}),
                30.0,
            )

    def test_failure_retry_delay_is_exponential_and_capped(self):
        cfg = {"failureRetrySeconds": 300, "failureRetryMaxSeconds": 1800}
        self.assertEqual(quota_primer._failure_retry_delay(cfg, 1), 300)
        self.assertEqual(quota_primer._failure_retry_delay(cfg, 2), 600)
        self.assertEqual(quota_primer._failure_retry_delay(cfg, 3), 1200)
        self.assertEqual(quota_primer._failure_retry_delay(cfg, 4), 1800)
        self.assertEqual(quota_primer._failure_retry_delay(cfg, 8), 1800)

    def test_success_state_records_trigger_reset_not_returned_next_reset(self):
        patch_data = quota_primer._result_state_patch(
            {}, {"ok": True, "reason": "primed", "model": "claude-test"},
            reason="reset_due", trigger_reset="2026-01-01T00:00:00Z",
            trigger_reset_at=cast(float, quota_primer._parse_iso_utc("2026-01-01T00:00:00Z")),
            reset_after="2026-01-01T05:00:00Z", now_ts=1000,
            cfg=self.cfg,
        )
        self.assertEqual(patch_data["last_trigger_reset"], "2026-01-01T00:00:00Z")
        self.assertEqual(patch_data["observed_next_reset"], "2026-01-01T05:00:00Z")
        self.assertEqual(
            patch_data["next_cycle_reset_at"],
            int(cast(float, quota_primer._parse_iso_utc("2026-01-01T05:00:00Z"))),
        )
        self.assertEqual(patch_data["last_success_at"], 1000)
        self.assertEqual(patch_data["failure_count"], 0)
        self.assertEqual(patch_data["next_retry_at"], 0)

    def test_failure_state_retries_without_overwriting_last_success(self):
        previous = {"last_success_at": 500, "failure_count": 1}
        patch_data = quota_primer._result_state_patch(
            previous, {"ok": False, "reason": "HTTP 503", "model": "claude-test"},
            reason="reset_due", trigger_reset="2026-01-01T00:00:00Z",
            trigger_reset_at=cast(float, quota_primer._parse_iso_utc("2026-01-01T00:00:00Z")),
            reset_after="2026-01-01T00:00:00Z", now_ts=1000,
            cfg={**self.cfg, "failureRetrySeconds": 300, "failureRetryMaxSeconds": 1800},
        )
        self.assertNotIn("last_success_at", patch_data)
        self.assertNotIn("last_trigger_reset", patch_data)
        self.assertEqual(patch_data["failure_count"], 2)
        self.assertEqual(patch_data["next_retry_at"], 1600)

    def test_persisted_next_reset_survives_missing_quota_cache_reset(self):
        target = cast(float, quota_primer._parse_iso_utc("2026-01-01T05:00:00Z"))
        row = {"five_hour_reset": None, "five_hour_util": 0.0, "last_passive_update_at": 0}
        state = {"next_cycle_reset_at": target, "last_success_at": target - 18_000}
        self.assertEqual(self._due(row, state=state, now=target + 359), (False, "skip:post_reset_delay"))
        self.assertEqual(self._due(row, state=state, now=target + 360), (True, "reset_due"))

    def test_success_without_advanced_reset_schedules_next_five_hour_cycle(self):
        trigger = cast(float, quota_primer._parse_iso_utc("2026-01-01T00:00:00Z"))
        patch_data = quota_primer._result_state_patch(
            {}, {"ok": True, "reason": "primed", "model": "claude-test"},
            reason="reset_due", trigger_reset="2026-01-01T00:00:00Z",
            trigger_reset_at=trigger, reset_after="2026-01-01T00:00:00Z",
            now_ts=1000, cfg={**self.cfg, "windowSeconds": 18_000},
        )
        self.assertEqual(patch_data["next_cycle_reset_at"], 19_000)

    def test_newer_quota_reset_from_user_traffic_moves_cycle_forward(self):
        persisted = cast(float, quota_primer._parse_iso_utc("2026-01-01T05:00:00Z"))
        newer = "2026-01-01T05:20:00Z"
        newer_ts = cast(float, quota_primer._parse_iso_utc(newer))
        row = {"five_hour_reset": newer, "last_passive_update_at": 0}
        state = {"next_cycle_reset_at": persisted}
        self.assertEqual(self._due(row, state=state, now=persisted + 360), (False, "skip:post_reset_delay"))
        self.assertEqual(self._due(row, state=state, now=newer_ts + 360), (True, "reset_due"))

    def test_prime_claude_uses_plain_hello_without_artificial_token_cap(self):
        class FakeChannel:
            key = "oauth:claude:a@example.com"
            account_key = "claude:a@example.com"
            email = "a@example.com"
            models = ["claude-test"]

            async def build_upstream_request(self, body, model, *, ingress_protocol="anthropic"):
                self.seen_body = body
                self.seen_model = model
                self.seen_ingress = ingress_protocol
                return UpstreamRequest(url="https://example.test/v1/messages", headers={}, body=b"{}")

        class FakeResponse:
            status_code = 200
            headers = {}

        class FakeClient:
            def __init__(self):
                self.post_called = False

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def post(self, url, *, headers=None, content=None):
                self.post_called = True
                self.url = url
                self.headers = headers
                self.content = content
                return FakeResponse()

        ch = FakeChannel()
        client = FakeClient()
        with patch.object(quota_primer.network, "async_client", return_value=client):
            result = asyncio.run(quota_primer._prime_claude(cast(Any, ch), timeout_s=1))

        self.assertEqual(result["ok"], True)
        self.assertEqual(ch.seen_ingress, "anthropic")
        self.assertEqual(ch.seen_model, "claude-test")
        self.assertEqual(ch.seen_body["messages"], [{"role": "user", "content": "hello"}])
        self.assertEqual(ch.seen_body["stream"], False)
        self.assertNotIn("max_tokens", ch.seen_body)
        self.assertTrue(client.post_called)

    def test_quota_disabled_with_no_reset_does_not_use_zero_util_fallback(self):
        row = {"five_hour_reset": None, "five_hour_util": 0.0, "last_passive_update_at": 0}
        acc = {"email": "a@example.com", "disabled_reason": "quota", "disabled_until": None}
        self.assertEqual(self._due(row, acc), (False, "skip:quota_disabled_no_reset"))

    def test_quota_disabled_with_known_reset_still_primes_after_delay(self):
        reset = cast(float, quota_primer._parse_iso_utc("2026-01-01T00:00:00Z"))
        row = {"five_hour_reset": "2026-01-01T00:00:00Z", "five_hour_util": 0.0, "last_passive_update_at": 0}
        acc = {"email": "a@example.com", "disabled_reason": "quota", "disabled_until": "2026-01-01T00:00:00Z"}
        self.assertEqual(self._due(row, acc, now=reset + 359), (False, "skip:post_reset_delay"))
        self.assertEqual(self._due(row, acc, now=reset + 360), (True, "reset_due"))

    def test_openai_http_success_without_quota_headers_still_completes_cycle(self):
        class FakeChannel:
            models = ["gpt-test"]

            async def probe_usage(self, *, timeout_s):
                self.timeout_s = timeout_s
                return {
                    "ok": False,
                    "request_ok": True,
                    "reason": "upstream 200 but no x-codex-* headers",
                }

        ch = FakeChannel()
        result = asyncio.run(quota_primer._prime_openai(cast(Any, ch), timeout_s=7))
        self.assertEqual(result["ok"], True)
        self.assertEqual(result["reason"], "upstream 200 but no x-codex-* headers")
        self.assertEqual(ch.timeout_s, 7)


if __name__ == "__main__":
    unittest.main()
