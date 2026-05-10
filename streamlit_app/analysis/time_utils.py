"""Calendar hours for capacity-factor denominators (matches ``build_derived_facts``)."""

from __future__ import annotations

import calendar

HOURS_NON_LEAP = 8760
HOURS_LEAP = 8784


def hours_per_calendar_year(year: int) -> int:
    y = int(year)
    return HOURS_LEAP if calendar.isleap(y) else HOURS_NON_LEAP
