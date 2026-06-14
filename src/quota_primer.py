"""OAuth quota rolling-window primer.

Some subscription backends use a rolling 5h window that only starts on the first
model request after the previous window refreshes. This module can send one
minimal request after a known 5h reset time has passed so the next rolling window
starts even if no user traffic arrives immediately.

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
        "initialDelaySeconds": 90,
        "graceSeconds": 60,
        "minIntervalSeconds": 17_400,  # 4h50m guard against bad/moving reset data
        "timeoutSeconds": 20,
        "maxTokens": 1,
        "bootstrapWhenUnknown": False,
        "claudeZeroUtilFallback": True,
        "halfHourSlotFallback": True,
        "halfHourSlotWindowSeconds": 120,
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


def _near_half_hour_slot(now: float, window_seconds: int) -> bool:
    if window_seconds <= 0:
        return False
    minute = int(now // 60)
    seconds_into_half_hour = (minute % 30) * 60 + int(now % 60)
    return seconds_into_half_hour <= window_seconds or seconds_into_half_hour >= 1800 - window_seconds


def _recent_model_request(row: dict | None, now: float, min_interval: int) -> bool:
    last_model_at = _last_model_request_at(row)
    return last_model_at is not None and now - last_model_at < min_interval


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

    last_prime = float(state.get("last_prime_at") or 0)
    min_interval = max(300, int(cfg.get("minIntervalSeconds", 17_400) or 17_400))
    if last_prime and now - last_prime < min_interval:
        return False, "skip:min_interval"

    reset_iso = (row or {}).get("five_hour_reset") if row else None
    reset_ts = _parse_iso_utc(reset_iso)

    disabled_reason = acc.get("disabled_reason")
    disabled_until_ts = _parse_iso_utc(acc.get("disabled_until"))
    if disabled_reason == "quota":
        if not cfg.get("includeQuotaDisabledAfterReset", True):
            return False, "skip:quota_disabled"
        if disabled_until_ts is not None and disabled_until_ts > now:
            return False, "skip:quota_reset_future"

    if reset_ts is None:
        provider = _provider_from_account_key(account_key)
        util = _float_or_none((row or {}).get("five_hour_util"))
        # Claude usage may show five_hour_util=0 after refresh but omit the next
        # five_hour_reset until the first model request starts the rolling window.
        if provider == "claude" and cfg.get("claudeZeroUtilFallback", True) and util is not None and util <= 0:
            if _recent_model_request(row, now, min_interval):
                return False, "skip:recent_model_request"
            return True, "zero_util_fallback"
        # Last-resort observation-based fallback: Claude/ChatGPT subscription
        # refreshes often happen around :00 / :30. Only use it near those slots
        # and only when there has been no model request for nearly a full 5h.
        if cfg.get("halfHourSlotFallback", True) and row is not None:
            slot_window = max(0, int(cfg.get("halfHourSlotWindowSeconds", 120) or 120))
            if _near_half_hour_slot(now, slot_window):
                if _recent_model_request(row, now, min_interval):
                    return False, "skip:recent_model_request"
                return True, "half_hour_slot_fallback"
        if cfg.get("bootstrapWhenUnknown"):
            return True, "unknown_bootstrap"
        return False, "skip:no_known_reset"

    grace = max(0, int(cfg.get("graceSeconds", 60) or 60))
    if reset_ts > now + grace:
        return False, "skip:reset_future"

    if str(state.get("last_reset_primed") or "") == str(reset_iso):
        return False, "skip:reset_already_primed"

    last_model_at = _last_model_request_at(row)
    if last_model_at is not None and last_model_at >= reset_ts:
        return False, "skip:already_used_after_reset"

    return True, "reset_due"


async def _prime_claude(ch: OAuthChannel, *, timeout_s: float, max_tokens: int) -> dict:
    model = ch.models[0] if ch.models else "claude-sonnet-4-5"
    body = {
        "model": model,
        "max_tokens": max(1, int(max_tokens or 1)),
        "stream": True,
        "messages": [{"role": "user", "content": "."}],
    }
    req = await ch.build_upstream_request(body, model, ingress_protocol="anthropic")
    try:
        async with network.async_client(
            timeout=timeout_s,
            proxy_purpose="oauth_anthropic",
            proxy_channel=ch.key,
            proxy_model=model,
        ) as client:
            async with client.stream("POST", req.url, headers=req.headers, content=req.body) as resp:
                status = resp.status_code
                headers_snapshot = dict(resp.headers)
                # Consume at most one chunk so the upstream sees a normal tiny request,
                # then close. max_tokens=1 keeps spend minimal.
                try:
                    async for _chunk in resp.aiter_bytes():
                        break
                except Exception:
                    pass
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
    return {"ok": bool(result.get("ok")), "reason": str(result.get("reason") or ""), "model": (ch.models[0] if ch.models else "")}


async def primer_once() -> dict[str, str]:
    cfg = _cfg()
    if not cfg.get("enabled"):
        return {"_": "disabled"}
    if os.environ.get("PARROT_NO_PRIMER") == "1":
        return {"_": "env_disabled"}

    providers = cfg.get("providers") or {}
    timeout_s = float(cfg.get("timeoutSeconds", 20) or 20)
    max_tokens = int(cfg.get("maxTokens", 1) or 1)
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
                result = await _prime_claude(ch, timeout_s=timeout_s, max_tokens=max_tokens)
        except Exception as exc:
            result = {"ok": False, "reason": str(exc)[:200], "model": ""}

        row_after = state_db.quota_load(ch.account_key) or row or {}
        reset_after = row_after.get("five_hour_reset") or ((row or {}).get("five_hour_reset") if row else None)
        state_patch = {
            "last_prime_at": int(_now()),
            "last_reason": reason,
            "last_ok": bool(result.get("ok")),
            "last_result": result.get("reason"),
            "last_model": result.get("model"),
        }
        if result.get("ok") and reset_after:
            state_patch["last_reset_primed"] = reset_after
        _save_state(ch.account_key, state_patch)

        status = "ok" if result.get("ok") else "failed"
        out[ch.key] = f"{status}:{result.get('reason') or ''}"
        print(
            f"[quota_primer] {status} provider={provider} account={getattr(ch, 'email', '?')} "
            f"model={result.get('model') or '-'} trigger={reason} result={result.get('reason') or ''}"
        )
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
            interval = int(cfg.get("intervalSeconds", 60) or 60)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            print(f"[quota_primer] loop iteration failed: {exc}")
            interval = 60
        await asyncio.sleep(max(30, interval))
