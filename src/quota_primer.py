"""OAuth quota rolling-window primer.

Some subscription backends use a rolling 5h window that only starts on the first
model request after the previous window refreshes. This module can send one
ordinary short request a few minutes after a known 5h reset time so the next
rolling window starts even if no user traffic arrives immediately.  A successful
request normally returns the next reset timestamp, which becomes the anchor for
the following cycle.

Safety defaults:
- disabled by default;
- only OAuth channels are considered;
- user/auth disabled accounts are never touched;
- quota-disabled accounts are touched only after their reset time passed and only
  when quotaPrimer.includeQuotaDisabledAfterReset is true;
- one successful primer is recorded per observed reset timestamp.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import time
from datetime import datetime, timezone
from typing import Any

import httpx

from . import config, network, oauth_manager, state_db
from .anthropic import rate_limit_headers as anthropic_rl
from .channel.oauth_channel import OAuthChannel
from .channel.openai_oauth_channel import OpenAIOAuthChannel
from .channel import registry

_STATE_PREFIX = "quota_primer:"


def _cfg() -> dict:
    base = {
        "enabled": False,
        "intervalSeconds": 60,
        "intervalJitterSeconds": 10,
        "initialDelaySeconds": 90,
        "postResetDelaySeconds": 360,
        "windowSeconds": 18_000,
        "minIntervalSeconds": 17_400,  # 4h50m guard against bad/moving reset data
        "failureRetrySeconds": 300,
        "failureRetryMaxSeconds": 1_800,
        "timeoutSeconds": 20,
        "bootstrapWhenUnknown": False,
        "claudeZeroUtilFallback": True,
        "includeQuotaDisabledAfterReset": True,
        "providers": {"claude": True, "openai": True},
        "notify": False,
    }
    cur = config.get().get("quotaPrimer") or {}
    out = dict(base)
    out.update({k: v for k, v in cur.items() if k != "providers"})
    providers = dict(base["providers"])
    providers.update(cur.get("providers") or {})
    out["providers"] = providers
    return out


def _now() -> float:
    return time.time()


def _parse_iso_utc(value: Any) -> float | None:
    if not value:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        if s.endswith("Z"):
            dt = datetime.fromisoformat(s[:-1] + "+00:00")
        else:
            dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return None


def _state_key(account_key: str) -> str:
    return _STATE_PREFIX + account_key


def _load_state(account_key: str) -> dict:
    raw = state_db.schema_meta_get(_state_key(account_key))
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_state(account_key: str, patch: dict) -> dict:
    data = _load_state(account_key)
    data.update(patch)
    data["updated_at"] = int(_now())
    state_db.schema_meta_set(_state_key(account_key), json.dumps(data, ensure_ascii=False, sort_keys=True))
    return data


def _account_provider(ch: Any) -> str:
    if isinstance(ch, OpenAIOAuthChannel):
        return "openai"
    return "claude"


def _last_model_request_at(row: dict | None) -> float | None:
    if not row:
        return None
    raw = row.get("last_passive_update_at")
    try:
        ms = int(raw or 0)
    except Exception:
        return None
    return (ms / 1000.0) if ms > 0 else None


def _provider_from_account_key(account_key: str) -> str:
    return str(account_key or "").split(":", 1)[0] or "claude"


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _recent_model_request(row: dict | None, now: float, min_interval: int) -> bool:
    last_model_at = _last_model_request_at(row)
    return last_model_at is not None and now - last_model_at < min_interval


def _successful_prime_at(state: dict) -> float:
    """Read the successful-prime timestamp, including legacy persisted state."""
    value = state.get("last_success_at")
    if value is None and state.get("last_ok") is True:
        value = state.get("last_prime_at")
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _failure_retry_delay(cfg: dict, failure_count: int) -> int:
    base = max(30, int(cfg.get("failureRetrySeconds", 300) or 300))
    cap = max(base, int(cfg.get("failureRetryMaxSeconds", 1_800) or 1_800))
    exponent = max(0, min(int(failure_count) - 1, 10))
    return min(cap, base * (2 ** exponent))


def _effective_reset_at(row: dict | None, state: dict) -> float | None:
    """Return the newest reset anchor from quota cache or persisted scheduler state."""
    candidates = [
        _parse_iso_utc((row or {}).get("five_hour_reset")),
        _float_or_none(state.get("next_cycle_reset_at")),
    ]
    valid = [value for value in candidates if value is not None and value > 0]
    return max(valid) if valid else None


def _result_state_patch(previous_state: dict, result: dict, *, reason: str,
                        trigger_reset: Any, trigger_reset_at: float | None,
                        reset_after: Any, now_ts: int,
                        cfg: dict) -> dict:
    """Build the persistent scheduler state for one real primer attempt."""
    patch = {
        "last_attempt_at": now_ts,
        "last_reason": reason,
        "last_ok": bool(result.get("ok")),
        "last_result": result.get("reason"),
        "last_model": result.get("model"),
    }
    if result.get("ok"):
        returned_reset_at = _parse_iso_utc(reset_after)
        # If the response did not advance the reset timestamp, keep the cycle
        # alive with the known 5h window length.  A later real quota snapshot may
        # move this target forward, and _effective_reset_at() will prefer it.
        if returned_reset_at is None or (
            trigger_reset_at is not None and returned_reset_at <= trigger_reset_at
        ):
            returned_reset_at = float(now_ts + max(300, int(cfg.get("windowSeconds", 18_000) or 18_000)))
        patch.update({
            "last_prime_at": now_ts,  # retained for compatibility
            "last_success_at": now_ts,
            # This is the reset that caused this attempt, not the newly returned
            # reset.  The latter is the anchor of the following 5h cycle.
            "last_trigger_reset": trigger_reset,
            "last_trigger_at": trigger_reset_at,
            "observed_next_reset": reset_after,
            "next_cycle_reset_at": int(returned_reset_at),
            "failure_count": 0,
            "next_retry_at": 0,
        })
    else:
        failure_count = int(previous_state.get("failure_count") or 0) + 1
        patch.update({
            "failure_count": failure_count,
            "next_retry_at": now_ts + _failure_retry_delay(cfg, failure_count),
        })
    return patch


def _loop_sleep_seconds(cfg: dict) -> float:
    interval = int(cfg.get("intervalSeconds", 60) or 60)
    jitter = max(0, int(cfg.get("intervalJitterSeconds", 0) or 0))
    if jitter:
        interval += random.uniform(-jitter, jitter)
    return max(30.0, float(interval))


def _due_reason(account_key: str, row: dict | None, acc: dict | None, cfg: dict,
                *, now: float | None = None) -> tuple[bool, str]:
    """Pure-ish due decision, exposed for tests.

    Returns (due, reason). Reasons beginning with "skip:" are intentionally not
    due; "reset_due" and "unknown_bootstrap" are due.
    """
    now = _now() if now is None else now
    state = _load_state(account_key)

    if acc is None:
        return False, "skip:missing_account"
    if acc.get("disabled_reason") in ("user", "auth_error"):
        return False, f"skip:{acc.get('disabled_reason')}"

    next_retry_at = _float_or_none(state.get("next_retry_at")) or 0
    if next_retry_at > now:
        return False, "skip:retry_backoff"

    last_success = _successful_prime_at(state)
    min_interval = max(300, int(cfg.get("minIntervalSeconds", 17_400) or 17_400))

    reset_iso = (row or {}).get("five_hour_reset") if row else None
    reset_ts = _effective_reset_at(row, state)

    disabled_reason = acc.get("disabled_reason")
    disabled_until_ts = _parse_iso_utc(acc.get("disabled_until"))
    if disabled_reason == "quota":
        if not cfg.get("includeQuotaDisabledAfterReset", True):
            return False, "skip:quota_disabled"
        if disabled_until_ts is not None and disabled_until_ts > now:
            return False, "skip:quota_reset_future"

    if reset_ts is None:
        # quota-disabled accounts must only be primed against a known reset
        # timestamp.  The zero-util fallback is for healthy Claude accounts that
        # show util=0 but haven't started the next rolling window yet; a
        # quota-disabled account with no reset data should not get exploratory
        # requests.
        if disabled_reason == "quota":
            return False, "skip:quota_disabled_no_reset"
        if last_success and now - last_success < min_interval:
            return False, "skip:min_interval"
        provider = _provider_from_account_key(account_key)
        util = _float_or_none((row or {}).get("five_hour_util"))
        # Claude usage may show five_hour_util=0 after refresh but omit the next
        # five_hour_reset until the first model request starts the rolling window.
        if provider == "claude" and cfg.get("claudeZeroUtilFallback", True) and util is not None and util <= 0:
            if _recent_model_request(row, now, min_interval):
                return False, "skip:recent_model_request"
            return True, "zero_util_fallback"
        if cfg.get("bootstrapWhenUnknown"):
            return True, "unknown_bootstrap"
        return False, "skip:no_known_reset"

    post_reset_delay = max(0, int(cfg.get("postResetDelaySeconds", 360) or 360))
    if now < reset_ts + post_reset_delay:
        return False, "skip:post_reset_delay"

    # New state records the reset that triggered the successful request.  The old
    # implementation stored the reset returned *after* success, which is actually
    # the next cycle and must not suppress it after an upgrade.
    last_trigger_at = _float_or_none(state.get("last_trigger_at"))
    legacy_trigger_at = _parse_iso_utc(state.get("last_trigger_reset"))
    handled_at = last_trigger_at if last_trigger_at is not None else legacy_trigger_at
    if handled_at is not None and int(handled_at) == int(reset_ts):
        return False, "skip:reset_already_primed"

    last_model_at = _last_model_request_at(row)
    if last_model_at is not None and last_model_at >= reset_ts:
        return False, "skip:already_used_after_reset"

    return True, "reset_due"


async def _prime_claude(ch: OAuthChannel, *, timeout_s: float) -> dict:
    model = ch.models[0] if ch.models else "claude-sonnet-4-5"
    body = {
        "model": model,
        "stream": False,
        "messages": [{"role": "user", "content": "hello"}],
    }
    req = await ch.build_upstream_request(body, model, ingress_protocol="anthropic")
    try:
        async with network.async_client(
            timeout=timeout_s,
            proxy_purpose="oauth_anthropic",
            proxy_channel=ch.key,
            proxy_model=model,
        ) as client:
            # Do not set an artificial max_tokens cap. Some upstream quota
            # backends do not treat max_tokens=1 requests as normal usage. Use a
            # short hello prompt and let the response complete so the request is
            # observed as an ordinary model call.
            resp = await client.post(req.url, headers=req.headers, content=req.body)
            status = resp.status_code
            headers_snapshot = dict(resp.headers)
    except httpx.TimeoutException:
        return {"ok": False, "reason": f"timeout > {timeout_s}s", "model": model}
    except Exception as exc:
        return {"ok": False, "reason": str(exc)[:200], "model": model}

    patch = anthropic_rl.parse_rate_limit_headers(headers_snapshot) or {}
    if patch:
        state_db.quota_patch_passive(ch.account_key, patch, email=ch.email)
        try:
            threshold = float((config.get().get("quotaMonitor") or {}).get("disableThresholdPercent", 95))
            oauth_manager.evaluate_and_toggle_by_usage(ch.account_key, patch, threshold=threshold, fresh=True)
        except Exception as exc:
            print(f"[quota_primer] Claude quota evaluation failed for {ch.account_key}: {exc}")

    if status != 200:
        return {"ok": False, "reason": f"HTTP {status}", "model": model}
    return {"ok": True, "reason": "primed", "model": model, "headers": bool(patch)}


async def _prime_openai(ch: OpenAIOAuthChannel, *, timeout_s: float) -> dict:
    result = await ch.probe_usage(timeout_s=timeout_s)
    if result.get("ok"):
        try:
            row = state_db.quota_load(ch.account_key) or {}
            threshold = float((config.get().get("quotaMonitor") or {}).get("disableThresholdPercent", 95))
            oauth_manager.evaluate_and_toggle_by_usage(ch.account_key, row, threshold=threshold, fresh=True)
        except Exception as exc:
            print(f"[quota_primer] OpenAI quota evaluation failed for {ch.account_key}: {exc}")
    # A completed HTTP 200 model request starts the rolling window even when the
    # response omitted quota headers.  Treat it as a scheduler success to avoid
    # repeatedly spending tokens; quota refresh can catch up on later traffic.
    request_ok = bool(result.get("ok") or result.get("request_ok"))
    return {"ok": request_ok, "reason": str(result.get("reason") or ""), "model": (ch.models[0] if ch.models else "")}


async def primer_once() -> dict[str, str]:
    cfg = _cfg()
    if not cfg.get("enabled"):
        return {"_": "disabled"}
    if os.environ.get("PARROT_NO_PRIMER") == "1":
        return {"_": "env_disabled"}

    providers = cfg.get("providers") or {}
    timeout_s = float(cfg.get("timeoutSeconds", 20) or 20)
    out: dict[str, str] = {}

    for ch in registry.all_channels():
        if not isinstance(ch, (OAuthChannel, OpenAIOAuthChannel)):
            continue
        provider = _account_provider(ch)
        if not providers.get(provider, True):
            out[ch.key] = "skipped:provider_disabled"
            continue
        acc = oauth_manager.get_account(ch.account_key)
        row = state_db.quota_load(ch.account_key)
        due, reason = _due_reason(ch.account_key, row, acc, cfg)
        if not due:
            out[ch.key] = reason
            continue

        try:
            if isinstance(ch, OpenAIOAuthChannel):
                result = await _prime_openai(ch, timeout_s=timeout_s)
            else:
                result = await _prime_claude(ch, timeout_s=timeout_s)
        except Exception as exc:
            result = {"ok": False, "reason": str(exc)[:200], "model": ""}

        row_after = state_db.quota_load(ch.account_key) or row or {}
        previous_state = _load_state(ch.account_key)
        trigger_reset = (row or {}).get("five_hour_reset") if row else None
        trigger_reset_at = _effective_reset_at(row, previous_state)
        reset_after = row_after.get("five_hour_reset") or ((row or {}).get("five_hour_reset") if row else None)
        now_ts = int(_now())
        state_patch = _result_state_patch(
            previous_state,
            result,
            reason=reason,
            trigger_reset=trigger_reset,
            trigger_reset_at=trigger_reset_at,
            reset_after=reset_after,
            now_ts=now_ts,
            cfg=cfg,
        )
        _save_state(ch.account_key, state_patch)

        status = "ok" if result.get("ok") else "failed"
        out[ch.key] = f"{status}:{result.get('reason') or ''}"
        print(
            f"[quota_primer] {status} provider={provider} account={getattr(ch, 'email', '?')} "
            f"model={result.get('model') or '-'} trigger={reason} result={result.get('reason') or ''}"
        )
        if cfg.get("notify"):
            try:
                from . import notifier
                if result.get("ok"):
                    notifier.notify_event(
                        "quota_primer",
                        f"✅ <b>5h 窗口已启动</b>\n"
                        f"账户: <code>{notifier.escape_html(getattr(ch, 'email', ch.account_key))}</code>\n"
                        f"触发: {reason} · 模型: <code>{result.get('model') or '-'}</code>",
                    )
                else:
                    notifier.notify_event(
                        "quota_primer",
                        f"❌ <b>5h 窗口启动失败</b>\n"
                        f"账户: <code>{notifier.escape_html(getattr(ch, 'email', ch.account_key))}</code>\n"
                        f"触发: {reason} · 原因: {notifier.escape_html(str(result.get('reason') or ''))[:200]}",
                    )
            except Exception as exc:
                print(f"[quota_primer] notify failed: {exc}")
        # Avoid a burst across multiple accounts. This is intentionally small;
        # each actual upstream request is still gated by minInterval/reset state.
        await asyncio.sleep(2)

    return out


async def primer_loop() -> None:
    try:
        delay = int(_cfg().get("initialDelaySeconds", 90) or 90)
    except Exception:
        delay = 90
    await asyncio.sleep(max(0, delay))
    while True:
        try:
            cfg = _cfg()
            if cfg.get("enabled") and os.environ.get("PARROT_NO_PRIMER") != "1":
                await primer_once()
            sleep_for = _loop_sleep_seconds(cfg)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            print(f"[quota_primer] loop iteration failed: {exc}")
            sleep_for = 60
        await asyncio.sleep(sleep_for)
