"""Pacing and fatal-error tripwires for pipeline.stage_send.

FATAL-ERROR DETECTION -- two layers, because string-matching alone is unreliable:

  1. By exception TYPE (is_fatal_exception): smtplib raises typed exceptions for
     authentication failures and server-level rejections regardless of what words the
     server happens to put in its message. Gmail's real throttle message is
     "454 4.7.0 Too many login attempts" -- note it says "too many login attempts",
     NOT "too many requests", so a string check for the latter phrase misses it
     entirely. Checking the exception's TYPE (smtplib.SMTPAuthenticationError,
     SMTPServerDisconnected, SMTPResponseException carrying a 421/450/452/454 code)
     catches this regardless of exact wording.

  2. By message TEXT (is_fatal_provider_error): the fallback for transports (SendGrid's
     urllib-based HTTPError) that do not raise a typed exception per failure kind --
     SendGrid embeds the real HTTP status as a literal string ("SendGrid 429: ...",
     "SendGrid 403: ...") which this layer matches directly.

stage_send tries the typed check first (needs the original exception object, not just
its string form) and falls back to the text check when no typed exception is available
or matches nothing.

Ported from the sibling `newsletter-cc-cl` project's `campaign.py`, which sends real
email in batches and has already been tuned against real bounce/throttle incidents.
That file is the reference; read its module docstring and MAX_CONSECUTIVE_FAILURES
comment for the full reasoning. This module carries over the two ideas that matter
for a bulk send:

  PACING       batches of short intra-batch delays, separated by a long inter-batch
               pause -- so hundreds of sends do not fire back-to-back in one burst,
               which is the pattern most likely to get a sending domain flagged or
               rate-limited in the first place.

  TRIPWIRES    a provider fault (429 rate limit, expired/invalid key, insufficient
               balance) fails EVERY remaining send identically. Retrying it, or even
               just continuing to the next recipient at full speed, does not help --
               and for a 429 specifically, hammering a host that just told you to
               slow down is how a temporary throttle turns into a longer ban. So a
               fatal pattern stops the WHOLE run immediately, not after N failures.
               A consecutive-failure breaker also exists for the ordinary case where
               failures are real but not obviously a provider fault (many bad
               addresses in a row, a dead SMTP relay) -- three in a row is a pattern,
               not bad luck.

TUNING (env vars, all optional -- same names/shape as campaign.py's CLI flags so the
reasoning transfers directly):
    CMI_SEND_BATCH_SIZE        sends per batch                    (default 10)
    CMI_SEND_SHORT_DELAY_MIN   min seconds between sends IN a batch (default 5)
    CMI_SEND_SHORT_DELAY_MAX   max seconds between sends IN a batch (default 9)
    CMI_SEND_LONG_PAUSE_MIN    min seconds BETWEEN batches          (default 60)
    CMI_SEND_LONG_PAUSE_MAX    max seconds BETWEEN batches          (default 70)
    CMI_SEND_MAX_CONSEC_FAIL   abort after this many failures in a row (default 3)
    CMI_SEND_PACING_DISABLE=1  turn all delays off (tests only)

Intra/inter-batch figures match this project's newsletter-cc-cl sibling as run in
production, not campaign.py's own file defaults (which are lower and were tuned for
a different, already-warm sending domain -- see that file's "NOTE ON SEND PACING
HISTORY"). Matching the numbers actually in use elsewhere rather than that file's
committed defaults.
"""
from __future__ import annotations

import os


class SendAborted(Exception):
    """Raised to stop the whole send run, not just the current recipient."""


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def batch_size() -> int:
    return max(1, _int_env("CMI_SEND_BATCH_SIZE", 10))


def short_delay_range() -> tuple[float, float]:
    return (_float_env("CMI_SEND_SHORT_DELAY_MIN", 5.0),
            _float_env("CMI_SEND_SHORT_DELAY_MAX", 9.0))


def long_pause_range() -> tuple[float, float]:
    return (_float_env("CMI_SEND_LONG_PAUSE_MIN", 60.0),
            _float_env("CMI_SEND_LONG_PAUSE_MAX", 70.0))


def max_consecutive_failures() -> int:
    return max(1, _int_env("CMI_SEND_MAX_CONSEC_FAIL", 3))


def pacing_disabled() -> bool:
    return os.environ.get("CMI_SEND_PACING_DISABLE") == "1"


def chunk(items: list, size: int) -> list[list]:
    return [items[i:i + size] for i in range(0, len(items), size)]


# Provider/transport faults that will repeat for every remaining recipient in this
# run: the rate limit will not lift between sends, the key will not un-expire, the
# balance will not refill. Matched against the exception text so it works whichever
# transport (SendGrid HTTP error, SMTP error) raised it.
#
# 429 / "rate limit" / "too many requests" are the ones that matter most: retrying,
# or even just continuing to send at the same pace, risks turning a temporary
# throttle into a longer block -- see netpolite.py's PolitelyBlocked, which applies
# the identical reasoning to outbound news-fetch traffic in the sibling project.
#
# This text list is a FALLBACK, not the primary defence -- see is_fatal_exception()
# below for the type-based check that catches SMTP failures whose wording doesn't
# happen to match any of these phrases (e.g. Gmail's real "454 4.7.0 Too many login
# attempts", which contains neither "429" nor "too many requests" verbatim).
_FATAL_PATTERNS = (
    "429",
    "too many requests",
    "too many login attempts",   # real Gmail SMTP throttle wording
    "rate limit",
    "quota",
    "insufficient balance",
    "402",
    "421",   # SMTP: service not available / temporary system problem
    "450",   # SMTP: mailbox unavailable / throttled
    "452",   # SMTP: insufficient system storage / too many recipients
    "454",   # SMTP: temporary auth failure / too many login attempts
    "invalid_request_error",
    "authentication fails",
    "authentication failed",
    "invalid api key",
    "incorrect api key",
    "username and password not accepted",   # Gmail's literal SMTPAuthenticationError text
    "unauthorized",
    "403",  # SendGrid: unverified sender / forbidden -- will not clear mid-run either
)

# Exception TYPES that are always fatal regardless of their message wording. Checked
# against the real exception object (not its string form) in is_fatal_exception(),
# which is the primary defence for the SMTP transport -- smtplib's own exception
# hierarchy already distinguishes "this will never succeed without operator action"
# from "transient", so relying on it beats guessing at message text.
_FATAL_EXCEPTION_TYPE_NAMES = (
    "SMTPAuthenticationError",   # wrong/expired/revoked credentials -- will not clear
    "SMTPServerDisconnected",    # the server dropped/refused the connection outright
    "SMTPNotSupportedError",     # server rejects a required extension (e.g. AUTH)
)

# SMTP reply codes that are provider-level faults, not one-off recipient problems.
# 4xx codes here are the server's own "back off" signal (throttling, temporary auth
# lockout); repeating the same request into the same wall does not help. Recipient-
# specific rejections (e.g. 550 "no such user") are deliberately NOT listed -- those
# are real per-address failures, not a provider fault, and must keep counting toward
# the ordinary consecutive-failure breaker instead of aborting the whole run.
_FATAL_SMTP_CODES = (421, 450, 452, 454)


def is_fatal_exception(exc: BaseException | None) -> bool:
    """True when the exception's TYPE (not its message) marks a provider-level fault.

    Works even when the server's wording doesn't match any string in
    _FATAL_PATTERNS -- e.g. smtplib.SMTPAuthenticationError always means "credentials
    rejected" regardless of what text the server sent back.
    """
    if exc is None:
        return False
    if type(exc).__name__ in _FATAL_EXCEPTION_TYPE_NAMES:
        return True
    # smtplib.SMTPResponseException and its subclasses (SMTPDataError,
    # SMTPRecipientsRefused, etc.) carry the raw SMTP code as their first argument.
    code = getattr(exc, "smtp_code", None)
    if code is None and exc.args and isinstance(exc.args[0], int):
        code = exc.args[0]
    return code in _FATAL_SMTP_CODES


def is_fatal_provider_error(detail: str) -> bool:
    """True when this failure will repeat for every remaining recipient, so the run
    should stop now rather than burn through the rest at full speed.

    Text-based fallback -- prefer is_fatal_exception() when the original exception
    object is available (stage_send checks both)."""
    t = (detail or "").lower()
    return any(p in t for p in _FATAL_PATTERNS)
