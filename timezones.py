"""Country name -> IANA timezone + business-hours window, for the scheduler.

Business hours are the window in the lead's LOCAL time during which the newsletter
should land — chosen as a representative working-hours slot per country (typically
9am-6pm local, adjusted for known regional norms like Gulf Sun-Thu weeks). This is a
best-effort mapping for send-time targeting, not a legal/labor reference.

    from timezones import business_window, in_business_hours
    tz, start_hour, end_hour, workdays = business_window("India")
    in_business_hours("India", datetime.now(timezone.utc))
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

# Mon=0 ... Sun=6. Most of the world: Mon-Fri. Gulf countries traditionally run Sun-Thu.
MON_FRI = (0, 1, 2, 3, 4)
SUN_THU = (6, 0, 1, 2, 3)

# name (matches config.COUNTRIES / campaign.py's parsed `market` field) -> (IANA tz,
# business start hour, business end hour, workdays tuple). Falls back to UTC 9-18
# Mon-Fri via DEFAULT_WINDOW for any country not listed here.
COUNTRY_TZ: dict[str, tuple[str, int, int, tuple[int, ...]]] = {
    "Global": ("UTC", 9, 18, MON_FRI),
    "India": ("Asia/Kolkata", 9, 18, MON_FRI),
    "United States": ("America/New_York", 9, 18, MON_FRI),
    "United Kingdom": ("Europe/London", 9, 18, MON_FRI),
    "Canada": ("America/Toronto", 9, 18, MON_FRI),
    "Australia": ("Australia/Sydney", 9, 18, MON_FRI),
    "Singapore": ("Asia/Singapore", 9, 18, MON_FRI),
    "United Arab Emirates": ("Asia/Dubai", 9, 18, SUN_THU),
    "Saudi Arabia": ("Asia/Riyadh", 9, 17, SUN_THU),
    "Germany": ("Europe/Berlin", 9, 18, MON_FRI),
    "Japan": ("Asia/Tokyo", 9, 18, MON_FRI),
    "Brazil": ("America/Sao_Paulo", 9, 18, MON_FRI),
    "Nigeria": ("Africa/Lagos", 9, 17, MON_FRI),
    "Netherlands": ("Europe/Amsterdam", 9, 18, MON_FRI),
    "Denmark": ("Europe/Copenhagen", 9, 17, MON_FRI),
    "Switzerland": ("Europe/Zurich", 9, 18, MON_FRI),
    "Hungary": ("Europe/Budapest", 9, 17, MON_FRI),
    "Turkey": ("Europe/Istanbul", 9, 18, MON_FRI),
    "Finland": ("Europe/Helsinki", 9, 17, MON_FRI),
    "Portugal": ("Europe/Lisbon", 9, 18, MON_FRI),
    "Russian Federation": ("Europe/Moscow", 9, 18, MON_FRI),
    "Russia": ("Europe/Moscow", 9, 18, MON_FRI),
    "Bulgaria": ("Europe/Sofia", 9, 18, MON_FRI),
    "Spain": ("Europe/Madrid", 9, 18, MON_FRI),
    "South Korea": ("Asia/Seoul", 9, 18, MON_FRI),
    "Ireland": ("Europe/Dublin", 9, 18, MON_FRI),
    "Norway": ("Europe/Oslo", 9, 17, MON_FRI),
    "South Africa": ("Africa/Johannesburg", 8, 17, MON_FRI),
    "China": ("Asia/Shanghai", 9, 18, MON_FRI),
    "Malaysia": ("Asia/Kuala_Lumpur", 9, 18, MON_FRI),
    "American Samoa": ("Pacific/Pago_Pago", 9, 17, MON_FRI),
    "Latvia": ("Europe/Riga", 9, 18, MON_FRI),
    "Italy": ("Europe/Rome", 9, 18, MON_FRI),
    "Czech Republic": ("Europe/Prague", 9, 18, MON_FRI),
    "Egypt": ("Africa/Cairo", 9, 17, SUN_THU),
    "Hong Kong": ("Asia/Hong_Kong", 9, 18, MON_FRI),
    "Algeria": ("Africa/Algiers", 9, 17, SUN_THU),
    "Vietnam": ("Asia/Ho_Chi_Minh", 8, 17, MON_FRI),
    "Poland": ("Europe/Warsaw", 9, 18, MON_FRI),
    "Guernsey": ("Europe/London", 9, 18, MON_FRI),
    "France": ("Europe/Paris", 9, 18, MON_FRI),
    "Mexico": ("America/Mexico_City", 9, 18, MON_FRI),
    "Indonesia": ("Asia/Jakarta", 9, 17, MON_FRI),
    "Israel": ("Asia/Jerusalem", 9, 18, SUN_THU),
    "Sweden": ("Europe/Stockholm", 9, 17, MON_FRI),
    "Belgium": ("Europe/Brussels", 9, 18, MON_FRI),
    "Austria": ("Europe/Vienna", 9, 18, MON_FRI),
    "New Zealand": ("Pacific/Auckland", 9, 17, MON_FRI),
    "Thailand": ("Asia/Bangkok", 9, 18, MON_FRI),
    "Philippines": ("Asia/Manila", 9, 18, MON_FRI),
    "Qatar": ("Asia/Qatar", 8, 17, SUN_THU),
    "Kuwait": ("Asia/Kuwait", 8, 17, SUN_THU),
}

DEFAULT_WINDOW = ("UTC", 9, 18, MON_FRI)


def _normalize(country: str) -> str:
    return (country or "Global").strip()


def business_window(country: str) -> tuple[str, int, int, tuple[int, ...]]:
    """(IANA tz name, start hour, end hour, workday tuple) for a country name. Falls
    back to UTC 9am-6pm Mon-Fri for any country not in the table."""
    return COUNTRY_TZ.get(_normalize(country), DEFAULT_WINDOW)


def local_now(country: str) -> datetime:
    tz_name, *_ = business_window(country)
    return datetime.now(ZoneInfo(tz_name))


def in_business_hours(country: str, at: datetime | None = None) -> bool:
    """True if `at` (default: now) falls inside the lead's country's local business
    hours and on a business day. `at` may be naive (assumed UTC) or aware."""
    tz_name, start_h, end_h, workdays = business_window(country)
    if at is None:
        at = datetime.now(timezone.utc)
    elif at.tzinfo is None:
        at = at.replace(tzinfo=timezone.utc)
    local = at.astimezone(ZoneInfo(tz_name))
    return local.weekday() in workdays and start_h <= local.hour < end_h


def next_business_hour(country: str, after: datetime | None = None) -> datetime:
    """The next UTC datetime at or after `after` (default: now) that falls inside
    the lead's country's business hours. Used by the scheduler to decide when a
    lead not currently in-window should next be tried."""
    tz_name, start_h, end_h, workdays = business_window(country)
    if after is None:
        after = datetime.now(timezone.utc)
    elif after.tzinfo is None:
        after = after.replace(tzinfo=timezone.utc)
    local = after.astimezone(ZoneInfo(tz_name))

    for day_offset in range(8):  # covers any single weekend/holiday gap
        candidate_date = (local + timedelta(days=day_offset)).date()
        if day_offset == 0:
            if local.weekday() in workdays and local.hour < end_h:
                start_at = max(local.replace(hour=start_h, minute=0, second=0, microsecond=0), local)
                if start_at.weekday() in workdays:
                    return start_at.astimezone(timezone.utc)
            continue
        candidate = datetime(candidate_date.year, candidate_date.month, candidate_date.day,
                              start_h, 0, 0, tzinfo=ZoneInfo(tz_name))
        if candidate.weekday() in workdays:
            return candidate.astimezone(timezone.utc)
    return after  # unreachable in practice — workdays is never empty
