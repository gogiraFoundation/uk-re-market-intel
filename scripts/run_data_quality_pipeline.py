#!/usr/bin/env python3
"""Cross-workbook data quality assessment and deterministic standardization.

Reads .xlsx / .xls / .ods under configured publisher folders, emits:
  - Cleaned sheet exports under ``cleaned_data/<publisher>/`` (Parquet + CSV)
  - Flat issue register CSV, per-workbook JSON reports, manifest, summary,
    canonicalization map and output checksums under ``dq_run_<RUN_ID>/``

Ambiguous cases (e.g. 01/02/2024 date order) are flagged, not coerced.

Implements the six fixes from the data-quality audit:
  1. HTML-disguised .xls files are parsed via ``pandas.read_html`` with
     sheet-name recovery from Microsoft's ``<x:Name>`` markers.
  2. Trailing empty rows/cols trimmed before processing; the real header row
     is auto-detected so metadata above the table is preserved separately.
  3. Identical ``raw__*`` shadow columns and all-empty ``*_iso_date`` columns
     are dropped.  CSVs are written with ``na_rep=""`` and ``float_format``.
  4. Parquet write failures raise an explicit ``parquet_write_failed`` issue
     and are reflected in the manifest (no silent CSV fallback claims).
  5. DESNZ-style footnote tokens (``[c]``, ``[low]``, ``..`` …) are stripped
     into a sidecar ``<col>_flag`` column; outliers use a robust log-MAD test.
  6. Canonicalization mappings persist to ``canonicalization_map.csv``; the
     ambiguous-date regex now requires a strict full-match.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import yaml as _yaml  # PyYAML; optional: schema contract check skips when absent.
except ImportError:  # pragma: no cover
    _yaml = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Constants (deterministic; extend in one place)
# ---------------------------------------------------------------------------

PLACEHOLDER_STRINGS: frozenset[str] = frozenset(
    {
        "",
        "n/a",
        "na",
        "#n/a",
        "#na",
        "null",
        "none",
        "nan",
        "-",
        "--",
        "unknown",
        "unk",
        "not available",
        "not applicable",
        "missing",
        "no data",
        "tbd",
        "tbc",
        ".",
    }
)

NUMERIC_SENTINELS: frozenset[float] = frozenset({-999.0, -9999.0, 999999.0})

# DESNZ / ONS footnote markers.  Compiled to a single regex so we can
# tolerate whitespace inside the brackets — e.g. ``[c ]`` and ``[ c]`` —
# which the original substring matcher missed (see EDA risks section 16).
_FOOTNOTE_SPECS: dict[str, str] = {
    r"\[\s*c\s*\]": "suppressed_confidential",
    r"\[\s*low\s*\]": "suppressed_low",
    r"\[\s*x\s*\]": "not_applicable",
    r"\[\s*r\s*\]": "revised",
    r"\[\s*p\s*\]": "provisional",
    r"\[\s*e\s*\]": "estimate",
    r"\[\s*w\s*\]": "withheld",
    r"\[\s*note\s*([1-9])\s*\]": "footnote_n",  # captures number into flag
    r"\.\.": "missing",
    r"\s*:\s*(?=\s|$)": "not_available",  # leading/standalone colon marker
}
DESNZ_FOOTNOTE_RE = re.compile(
    "|".join(f"(?P<g{i}>{pat})" for i, pat in enumerate(_FOOTNOTE_SPECS)),
    flags=re.IGNORECASE,
)
# Capture-group-free twin for the cheap pandas ``str.contains`` boolean
# mask — pandas warns on regex with capture groups even when only used for
# matching.  Strip ALL capture groups (named + numbered) by replacing each
# ``(`` not preceded by ``\`` and not already non-capturing with ``(?:``.
def _strip_capture_groups(pattern: str) -> str:
    return re.sub(r"(?<!\\)\((?!\?)", "(?:", pattern)


DESNZ_FOOTNOTE_DETECT_RE = re.compile(
    "|".join(f"(?:{_strip_capture_groups(pat)})" for pat in _FOOTNOTE_SPECS),
    flags=re.IGNORECASE,
)
_FOOTNOTE_GROUP_TO_CODE: list[str] = list(_FOOTNOTE_SPECS.values())

# Backwards-compat alias for any external callers still expecting the old
# substring map.
DESNZ_FOOTNOTES: dict[str, str] = {
    "[c]": "suppressed_confidential",
    "[low]": "suppressed_low",
    "[x]": "not_applicable",
    "[r]": "revised",
    "[p]": "provisional",
    "[e]": "estimate",
    "[w]": "withheld",
    "..": "missing",
}

# Mass → kilograms
MASS_TO_KG: dict[str, float] = {
    "kg": 1.0,
    "kilogram": 1.0,
    "kilograms": 1.0,
    "g": 0.001,
    "gram": 1e-3,
    "lb": 0.45359237,
    "lbs": 0.45359237,
    "pound": 0.45359237,
    "pounds": 0.45359237,
    "st": 6.35029318,
    "stone": 6.35029318,
    "tonne": 1000.0,
    "t": 1000.0,
}

# Length → centimeters
LENGTH_TO_CM: dict[str, float] = {
    "cm": 1.0,
    "mm": 0.1,
    "m": 100.0,
    "meter": 100.0,
    "metre": 100.0,
    "in": 2.54,
    "inch": 2.54,
    "inches": 2.54,
    "ft": 30.48,
    "foot": 30.48,
    "feet": 30.48,
}

TEMP_PATTERNS: dict[str, tuple[str, str]] = {
    "f": ("fahrenheit", "F"),
    "fahrenheit": ("fahrenheit", "F"),
    "°f": ("fahrenheit", "F"),
    "c": ("celsius", "C"),
    "celsius": ("celsius", "C"),
    "°c": ("celsius", "C"),
}

COUNTRY_ALIASES: dict[str, str] = {
    "usa": "United States",
    "u.s.a": "United States",
    "u.s.a.": "United States",
    "us": "United States",
    "united states": "United States",
    "united states of america": "United States",
    "uk": "United Kingdom",
    "u.k.": "United Kingdom",
    "united kingdom": "United Kingdom",
    "great britain": "United Kingdom",
}

# Statistical thresholds (named so EDA + pipeline use the same constants).
ROBUST_Z_THRESHOLD: float = 3.5  # Iglewicz–Hoaglin robust modified z-score
TUKEY_IQR_MULTIPLIER: float = 1.5
NUMERIC_DTYPE_PROMOTION_THRESHOLD: float = 0.95  # share of cells parseable as float

CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
HTML_TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")

UNIT_IN_TEXT_RE = re.compile(
    r"^\s*([+-]?\d+(?:[.,]\d+)?(?:[eE][+-]?\d+)?)\s*([a-zA-Z°µ/²³%]+)\s*$"
)

# Strict three-component date: same separator on both sides, all numeric parts.
STRICT_DATE_RE = re.compile(r"^(\d{1,4})([\-/.])(\d{1,2})\2(\d{1,4})$")


@dataclass
class Issue:
    workbook: str
    sheet: str
    column: str
    issue_code: str
    detail: str
    confidence: str  # high | medium | low
    rows_affected: int
    example_before: str = ""
    example_after: str = ""
    recommendation: str = ""

    def row_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _printable_path(p: Path) -> str:
    """Normalise Unicode hyphens etc. so manifest paths are shell-friendly."""
    return unicodedata.normalize("NFKC", str(p))


def snake_case_columns(cols: list[str]) -> list[str]:
    """Return positionally-aligned snake_case names; collisions are
    deduplicated with a numeric suffix.  Returning a *list* (one entry per
    input column) is essential because workbooks frequently arrive with
    duplicate header labels — a name-keyed dict would lose information."""
    out: list[str] = []
    used: set[str] = set()
    for c in cols:
        base = re.sub(r"[^a-zA-Z0-9]+", "_", str(c).strip().lower())
        base = re.sub(r"_+", "_", base).strip("_") or "column"
        candidate = base
        suffix = 2
        while candidate in used:
            candidate = f"{base}_{suffix}"
            suffix += 1
        used.add(candidate)
        out.append(candidate)
    return out


def strip_and_unicode_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = CONTROL_CHARS_RE.sub("", s)
    s = HTML_TAG_RE.sub("", s)
    return s.strip()


def is_placeholder_token(text: str) -> bool:
    t = text.strip().lower()
    return t in PLACEHOLDER_STRINGS


def strip_footnote(val: Any) -> tuple[Any, str | None]:
    """If a cell carries a known DESNZ/ONS footnote marker, strip it and
    return ``(remaining_value, flag_code)``.  Whitespace-tolerant: ``[c ]``
    and ``[ c]`` are matched the same as ``[c]``.  An empty remainder
    becomes ``None`` so it can be coerced to NA downstream."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return val, None
    if not isinstance(val, str):
        return val, None
    m = DESNZ_FOOTNOTE_RE.search(val)
    if m is None:
        return val, None
    # Find which named group fired -> map back to flag code.
    code = None
    for name, raw in m.groupdict().items():
        if raw is not None:
            idx = int(name[1:])
            code = _FOOTNOTE_GROUP_TO_CODE[idx]
            if code == "footnote_n" and m.lastindex is not None:
                # `[note N]` → footnote_<N>
                num = m.group(m.lastindex + 1) if m.lastindex + 1 <= (m.re.groups) else None
                if num and num.isdigit():
                    code = f"footnote_{num}"
                else:
                    code = "footnote"
            break
    cleaned = DESNZ_FOOTNOTE_RE.sub(" ", val)
    cleaned = WS_RE.sub(" ", cleaned).strip()
    return (cleaned if cleaned else None), code


def parse_number_loose(val: Any) -> float | None:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    if isinstance(val, (int, np.integer, float, np.floating)) and not isinstance(val, bool):
        f = float(val)
        if f in NUMERIC_SENTINELS:
            return None
        return f
    s = strip_and_unicode_text(str(val))
    if not s or is_placeholder_token(s):
        return None
    s = s.replace(",", "")
    s = re.sub(r"[£$€]", "", s)
    s = s.replace("%", "")
    s = s.strip()
    try:
        f = float(s)
        if f in NUMERIC_SENTINELS:
            return None
        return f
    except ValueError:
        return None


def try_parse_unambiguous_date(val: Any) -> tuple[pd.Timestamp | None, str]:
    """Return (ts, status). status: ok | empty | ambiguous | failed.

    A bare three-part numeric token is only treated as a date when the
    *whole* string matches ``STRICT_DATE_RE`` with a single separator on both
    sides — this prevents values like ``11-29.99`` (a kWh range) being
    misread as 1999-11-29.
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None, "empty"
    if isinstance(val, (pd.Timestamp, datetime)):
        ts = pd.Timestamp(val)
        if pd.isna(ts):
            return None, "empty"
        return ts.tz_localize(None) if ts.tzinfo else ts, "ok"
    if isinstance(val, (int, np.integer)):
        n = int(val)
        if abs(n) > 10_000_000_000:  # ms
            ts = pd.to_datetime(n, unit="ms", utc=True)
        elif abs(n) > 1_000_000_000:  # s heuristic
            ts = pd.to_datetime(n, unit="s", utc=True)
        else:
            return None, "failed"
        return ts.tz_convert(None) if ts.tzinfo else ts, "ok"

    s = strip_and_unicode_text(str(val))
    if not s or is_placeholder_token(s):
        return None, "empty"

    m = STRICT_DATE_RE.match(s)
    if m is not None:
        a, _sep, b, y = int(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4))
        if y < 100:
            y += 2000 if y < 70 else 1900
        if a > 12 and b <= 12:
            day, month = a, b
        elif b > 12 and a <= 12:
            month, day = a, b
        elif a <= 12 and b <= 12:
            return None, "ambiguous"
        else:
            return None, "failed"
        try:
            return pd.Timestamp(year=y, month=month, day=day), "ok"
        except ValueError:
            return None, "failed"

    # Reject any value that doesn't even contain a digit — pandas otherwise
    # happily parses 'May' / 'June' as the current year's date.
    if not re.search(r"\d", s):
        return None, "failed"
    ts = pd.to_datetime(s, errors="coerce", utc=True, format="mixed")
    if pd.isna(ts):
        return None, "failed"
    return ts.tz_convert(None) if ts.tzinfo else ts, "ok"


def fahrenheit_to_celsius(f: float) -> float:
    return (f - 32.0) * 5.0 / 9.0


def detect_mass_length_unit_token(token: str) -> tuple[str | None, str | None]:
    t = token.lower().strip()
    if t in MASS_TO_KG:
        return "mass_kg", t
    if t in LENGTH_TO_CM:
        return "length_cm", t
    return None, None


def normalize_embedded_quantity(val: Any) -> tuple[Any, str | None, str | None]:
    """If cell is '10 kg', return (10.0 * factor_to_canonical, 'mass_kg', detail)."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return val, None, None
    s = strip_and_unicode_text(str(val))
    m = UNIT_IN_TEXT_RE.match(s)
    if not m:
        return val, None, None
    num_s, unit = m.group(1), m.group(2).lower()
    try:
        num = float(num_s.replace(",", "."))
    except ValueError:
        return val, None, None
    kind, utok = detect_mass_length_unit_token(unit)
    if kind == "mass_kg":
        assert utok is not None
        kg = num * MASS_TO_KG.get(utok, 1.0)
        return kg, "mass_kg", f"converted {num} {utok} → kg ({kg})"
    if kind == "length_cm":
        assert utok is not None
        cm = num * LENGTH_TO_CM.get(utok, 1.0)
        return cm, "length_cm", f"converted {num} {utok} → cm ({cm})"
    return val, None, None


def column_name_suggests_temperature(name: str) -> bool:
    n = name.lower()
    return "temp" in n or "temperature" in n


def column_name_suggests_currency(name: str) -> bool:
    n = name.lower()
    return any(x in n for x in ("price", "cost", "revenue", "gbp", "usd", "eur", "£", "fee"))


def robust_outlier_mask(series: pd.Series, threshold: float = ROBUST_Z_THRESHOLD) -> pd.Series:
    """Iglewicz–Hoaglin robust z-score on log-transformed positives.

    Returns a boolean mask aligned to ``series``.  Tukey IQR×1.5 over-flags
    monotonically growing time-series (renewable capacity, prices, etc.);
    log-MAD with the named ``ROBUST_Z_THRESHOLD`` keeps only genuinely
    extreme observations.
    """
    s = pd.to_numeric(series, errors="coerce")
    s_pos = s[s > 0].dropna()
    if s_pos.size < 12:
        return pd.Series(False, index=series.index)
    log_s = np.log(s_pos.astype(float))
    med = float(log_s.median())
    mad = float((log_s - med).abs().median())
    if mad == 0:
        return pd.Series(False, index=series.index)
    log_full = np.log(s.where(s > 0).astype(float))
    z = 0.6745 * (log_full - med) / mad
    return (z.abs() > threshold).fillna(False)


def unify_categorical_value(raw: str) -> tuple[str, bool]:
    """Title-case + known country aliases; returns (canonical, changed)."""
    s = strip_and_unicode_text(raw)
    if not s:
        return s, False
    key = s.lower().replace(".", "")
    if key in COUNTRY_ALIASES:
        return COUNTRY_ALIASES[key], True
    # Preserve likely acronyms (SIC, CO2, UK, …) — do not title-case.
    if s.isupper() and 1 < len(s) <= 12 and s.replace("-", "").isalnum():
        return s, False
    if s.isupper() and len(s) > 1:
        t = s.title()
        return t, t != s
    return s, False


def trim_empty_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Trim trailing fully-empty rows and columns.

    openpyxl reports ``max_column`` based on cell formatting, not data, so
    DESNZ workbooks routinely come back with 16,384 columns.  We treat
    whitespace-only strings as empty for this purpose.
    """
    if df.empty:
        return df
    # Treat None/NaN/whitespace-only strings as empty without using
    # ``df.replace(..., regex=True)`` (deprecated in pandas 2.3).
    def _is_empty(v: Any) -> bool:
        if v is None:
            return True
        if isinstance(v, float) and pd.isna(v):
            return True
        if isinstance(v, str) and not v.strip():
            return True
        return False

    empty_mask = df.map(_is_empty).to_numpy()
    nonempty_col = ~empty_mask.all(axis=0)
    nonempty_row = ~empty_mask.all(axis=1)
    if not nonempty_col.any() or not nonempty_row.any():
        return df.iloc[0:0, 0:0]
    last_col = int(np.where(nonempty_col)[0].max())
    last_row = int(np.where(nonempty_row)[0].max())
    return df.iloc[: last_row + 1, : last_col + 1]


_ONS_METADATA_LABELS: tuple[str, ...] = (
    "title",
    "cdid",
    "source dataset id",
    "preunit",
    "unit",
    "release date",
    "next release",
    "important notes",
    "important note",
    "key word",
)


def _strip_ons_timeseries_header(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Detect the canonical ONS time-series metadata block (Title / CDID /
    Source dataset ID / PreUnit / Unit / Release date / Next release /
    Important notes) at the top of an ONS series workbook and split it off
    into a separate metadata frame.  Returns ``(data_df, metadata_df)``.

    Detection rule: the first column of the first ~12 rows must contain
    multiple known ONS label tokens.  No-op on non-ONS sheets."""
    if df.empty or df.shape[1] < 1:
        return df, None
    head = df.head(12)
    first_col = head.iloc[:, 0].astype("string").str.strip().str.lower()
    label_hits = first_col.isin(_ONS_METADATA_LABELS).sum()
    if label_hits < 3:
        return df, None
    # Find last metadata row (heuristic: last row whose first cell is one of
    # the known labels).
    matches = first_col[first_col.isin(_ONS_METADATA_LABELS)]
    if matches.empty:
        return df, None
    last_meta_idx = int(matches.index.max())
    meta = df.iloc[: last_meta_idx + 1].reset_index(drop=True)
    data = df.iloc[last_meta_idx + 1 :].reset_index(drop=True)
    return data, meta


def _is_text_cell(v: Any) -> bool:
    if v is None:
        return False
    if isinstance(v, float) and pd.isna(v):
        return False
    if isinstance(v, (int, np.integer, float, np.floating, bool)):
        return False
    s = str(v).strip()
    if not s:
        return False
    cleaned = s.replace(",", "").replace(".", "").replace("-", "").replace("+", "")
    return not cleaned.isdigit()


def _is_numeric_like(v: Any) -> bool:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return False
    if isinstance(v, (int, np.integer, float, np.floating)) and not isinstance(v, bool):
        return True
    try:
        float(str(v).replace(",", "").replace("£", "").replace("$", "").replace("€", "").strip())
        return True
    except (ValueError, TypeError):
        return False


def _looks_like_real_header(cols: list[Any]) -> bool:
    """True iff pandas' parsed header row appears to be genuine column names.

    Used to gate header re-detection: when ``pd.read_excel`` already returned
    sensible column labels (IRENA, MCS Monthly, …) we must not run the row-
    based detector, as that would misclassify the first row of *data* as the
    header.
    """
    if not cols:
        return False
    n = len(cols)
    suspicious = 0
    for c in cols:
        if c is None:
            suspicious += 1
            continue
        if isinstance(c, float) and pd.isna(c):
            suspicious += 1
            continue
        s = str(c).strip()
        if not s:
            suspicious += 1
            continue
        if s.startswith("Unnamed:"):
            suspicious += 1
            continue
        try:
            float(s.replace(",", ""))
            suspicious += 1
            continue
        except ValueError:
            pass
        if len(s) > 100:  # absurdly long ⇒ likely a sheet title bled into row 0
            suspicious += 1
    return suspicious / n < 0.3


def detect_header_row(df: pd.DataFrame, max_scan: int = 16) -> int:
    """Pick the row most plausibly carrying column headers.

    A header row has:
      - a high fraction of *text* cells (not numbers),
      - mostly *unique* values (DESNZ-style "notes pasted across N columns"
        rows score very low here — they have one distinct value),
      - rows below that are *predominantly numeric*.
    """
    n = min(len(df), max_scan)
    if n < 2:
        return 0
    best_score = -1.0
    best_idx = 0
    for i in range(n):
        row = df.iloc[i]
        n_filled = int(row.notna().sum())
        if n_filled == 0:
            continue
        text_vals = [str(v).strip().lower() for v in row if _is_text_cell(v)]
        n_text = len(text_vals)
        n_distinct = len(set(text_vals))
        text_density = n_text / n_filled
        unique_ratio = n_distinct / max(n_text, 1)
        below = df.iloc[i + 1 : i + 6]
        if below.empty:
            num_below = 0.0
        else:
            num_cells = sum(_is_numeric_like(v) for _, r in below.iterrows() for v in r)
            num_below = num_cells / max(below.size, 1)
        # Cap effective filled-count at a sensible header width (≤200) to
        # prevent absurdly wide note rows from dominating.
        filled_factor = min(n_filled, 200) / 200
        score = text_density * unique_ratio * (0.4 + num_below) * filled_factor
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx


# ---------------------------------------------------------------------------
# Workbook IO
# ---------------------------------------------------------------------------


def _read_html_xls(path: Path) -> dict[str, pd.DataFrame]:
    """Parse Microsoft's HTML-as-Excel files; recover sheet names from the
    ``<x:Name>`` markers Excel embeds in its HTML export."""
    raw = path.read_bytes()
    try:
        tables = pd.read_html(path, displayed_only=False)  # bs4 + lxml or html5lib
    except ImportError as exc:
        raise RuntimeError(
            "HTML-disguised .xls detected but no HTML parser is installed. "
            "Run: pip install -r requirements-dq.txt "
            "(needs beautifulsoup4 + lxml, or html5lib)."
        ) from exc
    if not tables:
        raise ValueError("HTML-as-Excel file contained no <table> elements")
    x_names = [m.decode("utf-8", "replace") for m in re.findall(rb"<x:Name>([^<]+)</x:Name>", raw)]
    out: dict[str, pd.DataFrame] = {}
    used: set[str] = set()
    for i, t in enumerate(tables):
        base = x_names[i] if i < len(x_names) else f"html_table_{i}"
        base = re.sub(r"\s+", " ", base).strip() or f"html_table_{i}"
        name = base
        suffix = 2
        while name in used:
            name = f"{base}_{suffix}"
            suffix += 1
        used.add(name)
        out[name] = t
    return out


def read_workbook(path: Path) -> dict[str, pd.DataFrame]:
    suf = path.suffix.lower()
    if suf == ".xlsx":
        return pd.read_excel(path, sheet_name=None, engine="openpyxl", dtype=object)
    if suf == ".ods":
        return pd.read_excel(path, sheet_name=None, engine="odf", dtype=object)
    if suf == ".xls":
        head = path.read_bytes()[:2048].lower()
        looks_html = b"<html" in head or b"<!doctype html" in head or b"<table" in head
        if looks_html:
            return _read_html_xls(path)
        try:
            return pd.read_excel(path, sheet_name=None, engine="xlrd", dtype=object)
        except Exception as exc:  # last-ditch: maybe HTML without recognisable preamble
            head_full = path.read_bytes()[:8192].lower()
            if b"<html" in head_full or b"<table" in head_full:
                return _read_html_xls(path)
            raise exc
    raise ValueError(f"Unsupported suffix: {suf}")


# ---------------------------------------------------------------------------
# Sheet-level processing
# ---------------------------------------------------------------------------


def _apply_footnote_split(out: pd.DataFrame, col: str) -> tuple[pd.Series, pd.Series, int]:
    """Strip footnote markers from a column.  Vectorised on the string
    fast-path: only cells where ``DESNZ_FOOTNOTE_RE`` matches are routed
    through the slower per-cell ``strip_footnote`` helper.  Returns
    ``(value_series, flag_series, n_changed)``.  Flag series may be entirely
    NA — caller is responsible for dropping it in that case.

    On the IRENA Country sheet (~92k rows, 17 cols) this changes the
    pipeline's footnote pass from a per-cell python loop into a
    pandas-level boolean mask, eliminating the O(rows × cols) hot-spot
    flagged in the codebase review."""
    raw = out[col]
    s = raw.astype("object")
    is_str_mask = s.map(lambda v: isinstance(v, str))
    if not bool(is_str_mask.any()):
        return raw, pd.Series([None] * len(raw), index=raw.index, dtype="object"), 0

    str_view = s.where(is_str_mask)
    candidates = str_view.dropna()
    hit_mask = candidates.str.contains(DESNZ_FOOTNOTE_DETECT_RE, na=False, regex=True)
    hit_index = candidates.index[hit_mask]
    if hit_index.empty:
        return raw, pd.Series([None] * len(raw), index=raw.index, dtype="object"), 0

    values = raw.copy()
    flags = pd.Series([None] * len(raw), index=raw.index, dtype="object")
    for idx in hit_index:
        new_v, code = strip_footnote(raw.loc[idx])
        values.loc[idx] = new_v
        flags.loc[idx] = code
    return values, flags, int(hit_index.size)


def process_sheet(
    workbook_rel: str,
    sheet_name: str,
    df: pd.DataFrame,
    issues: list[Issue],
    canon_log: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Return ``(cleaned, metadata_above_header_or_None)``.

    ``cleaned`` contains the canonicalised data with raw__ shadows kept only
    where they actually differ from the cleaned column, and ``*_iso_date`` /
    ``*_flag`` sidecars only when populated.
    """
    if df.empty:
        return df, None

    metadata_block: pd.DataFrame | None = None
    parsed_cols = list(df.columns)
    if _looks_like_real_header([str(c) for c in parsed_cols]):
        # ``pd.read_excel`` already promoted a sensible header row.  Don't
        # second-guess it — running the detector here would mistake row 0
        # of the body for a header.
        df.columns = [str(c) for c in parsed_cols]
    else:
        # Header is absent or junk (sheet title, "Unnamed:" everywhere) —
        # locate the real header row in the body.  Critically we extract
        # this BEFORE trim_empty_frame, because DESNZ sheets often paste a
        # notes row across thousands of columns which would otherwise hide
        # the truly empty trailing columns from the trim.
        hdr_idx = detect_header_row(df)
        if hdr_idx > 0:
            metadata_block = df.iloc[:hdr_idx].reset_index(drop=True)
        new_header = [
            str(v).strip() if v is not None and not (isinstance(v, float) and pd.isna(v)) else ""
            for v in df.iloc[hdr_idx].tolist()
        ]
        df = df.iloc[hdr_idx + 1 :].reset_index(drop=True)
        df.columns = new_header

    df = trim_empty_frame(df)
    if df.empty:
        return df, metadata_block

    # ONS-time-series detector: ONS releases ship "Important notes" + 6 metadata
    # rows (CDID / Source dataset ID / PreUnit / Unit / Release date / Next
    # release) at the top of every series block.  Strip them into the metadata
    # sidecar so the data block starts at the first observation row.
    df, ons_meta = _strip_ons_timeseries_header(df)
    if ons_meta is not None and not ons_meta.empty:
        if metadata_block is None or metadata_block.empty:
            metadata_block = ons_meta
        else:
            metadata_block = pd.concat([metadata_block, ons_meta], ignore_index=True)
        issues.append(
            Issue(
                workbook_rel,
                sheet_name,
                "*",
                "ons_metadata_stripped",
                f"ONS time-series metadata rows ({len(ons_meta)}) extracted into __metadata.csv.",
                "low",
                int(len(ons_meta)),
                recommendation="See __metadata.csv sidecar for CDID / Source / Unit / Release date.",
            )
        )

    if df.empty:
        return df, metadata_block

    new_names = snake_case_columns([str(c) for c in df.columns])
    # Snapshot raw values BEFORE any renaming so duplicate input labels
    # cannot collapse into one Series.  Index by position, not by label.
    raw_block = pd.DataFrame(
        {
            f"raw__{new_names[i]}": pd.Series(df.iloc[:, i].values, index=df.index)
            for i in range(df.shape[1])
        }
    )
    out = df.copy()
    out.columns = new_names
    out = pd.concat([out, raw_block], axis=1)

    data_cols = [c for c in out.columns if not str(c).startswith("raw__")]
    n_rows = len(out)

    # Full-row duplicates (data columns only; excludes raw__ backups).
    dup_mask = out[data_cols].duplicated(keep=False)
    ndup = int(dup_mask.sum())
    if ndup:
        issues.append(
            Issue(
                workbook_rel,
                sheet_name,
                "*",
                "duplicate_row",
                "Exact duplicate rows (all columns)",
                "high",
                ndup,
                recommendation="Review keys; keep first or aggregate.",
            )
        )

    flag_cols: list[str] = []
    iso_cols: list[str] = []

    for col in list(data_cols):
        # ----- Strip footnote markers FIRST -------------------------------
        new_vals, new_flags, n_footnote = _apply_footnote_split(out, col)
        if n_footnote:
            flag_col = f"{col}_flag"
            out[col] = new_vals
            out[flag_col] = new_flags
            flag_cols.append(flag_col)
            issues.append(
                Issue(
                    workbook_rel,
                    sheet_name,
                    col,
                    "footnote_marker_stripped",
                    "DESNZ/ONS footnote markers extracted into "
                    f"{flag_col}; remaining cell value retained.",
                    "high",
                    n_footnote,
                    recommendation="Use *_flag to filter or join provenance.",
                )
            )

        series = out[col]
        str_series = series.map(lambda x: "" if pd.isna(x) else strip_and_unicode_text(str(x)))

        # Placeholders → NA
        ph_mask = str_series.map(lambda t: bool(t) and is_placeholder_token(t))
        n_ph = int(ph_mask.sum())
        if n_ph:
            issues.append(
                Issue(
                    workbook_rel,
                    sheet_name,
                    col,
                    "placeholder_to_na",
                    "Placeholder / sentinel string treated as missing",
                    "high",
                    n_ph,
                    recommendation="Impute, drop, or flag downstream.",
                )
            )
            out.loc[ph_mask, col] = pd.NA

        # Numeric sentinel
        num_try = pd.to_numeric(out[col], errors="coerce")
        sent_mask = num_try.isin(list(NUMERIC_SENTINELS))
        if sent_mask.any():
            issues.append(
                Issue(
                    workbook_rel,
                    sheet_name,
                    col,
                    "numeric_sentinel",
                    f"Values in {NUMERIC_SENTINELS} set to NA",
                    "high",
                    int(sent_mask.sum()),
                )
            )
            out.loc[sent_mask, col] = pd.NA

        # Mixed string/numeric object column (after footnote strip + sentinel + placeholder)
        if series.dtype == object:
            cleaned_str = out[col].map(lambda x: "" if pd.isna(x) else strip_and_unicode_text(str(x)))
            parsed = cleaned_str.map(parse_number_loose)
            non_null_str = cleaned_str[cleaned_str != ""]
            if not non_null_str.empty:
                numeric_ratio = parsed.notna().sum() / max(len(non_null_str), 1)
                if 0.2 < numeric_ratio < 1.0:
                    issues.append(
                        Issue(
                            workbook_rel,
                            sheet_name,
                            col,
                            "mixed_type_numeric_string",
                            "Column mixes numeric-like and text tokens",
                            "medium",
                            int(len(non_null_str)),
                            recommendation="Parse per-row or split columns.",
                        )
                    )

        # Embedded units in text cells
        emb_changes = 0
        str_for_unit = out[col].astype("string")
        unit_match = str_for_unit.str.match(UNIT_IN_TEXT_RE, na=False)
        if unit_match.any():
            idxs = str_for_unit.index[unit_match]
            for idx in idxs:
                v = out.at[idx, col]
                new_v, kind, _detail = normalize_embedded_quantity(v)
                if kind and new_v != v:
                    out.at[idx, col] = new_v
                    emb_changes += 1
        if emb_changes:
            issues.append(
                Issue(
                    workbook_rel,
                    sheet_name,
                    col,
                    "unit_embedded_normalized",
                    "Mass/length embedded in text normalized to kg or cm",
                    "high",
                    emb_changes,
                    recommendation="Verify unit assumptions in header/context.",
                )
            )

        # Dates — only emit helper column if something actually parsed
        date_iso_col = f"{col}_iso_date"
        amb = ok = 0
        iso_vals: list[Any] = []
        for v in out[col].tolist():
            ts, st = try_parse_unambiguous_date(v)
            if st == "ok" and ts is not None:
                iso_vals.append(ts.strftime("%Y-%m-%d"))
                ok += 1
            elif st == "ambiguous":
                iso_vals.append(pd.NA)
                amb += 1
            else:
                iso_vals.append(pd.NA)
        if ok:
            out[date_iso_col] = iso_vals
            iso_cols.append(date_iso_col)
            issues.append(
                Issue(
                    workbook_rel,
                    sheet_name,
                    col,
                    "date_normalized_iso8601",
                    f"Parsed {ok} unambiguous dates to {date_iso_col}",
                    "high",
                    ok,
                )
            )
        if amb:
            issues.append(
                Issue(
                    workbook_rel,
                    sheet_name,
                    col,
                    "ambiguous_date",
                    "Day/month order unclear — not coerced; review locale context.",
                    "low",
                    amb,
                    recommendation="Human review with locale context.",
                )
            )

        # Temperature
        if column_name_suggests_temperature(col):
            f_conv = 0
            ex_before = ""
            ex_after = ""
            for idx, v in zip(out.index, out[col].tolist(), strict=True):
                fn = parse_number_loose(v)
                if fn is None:
                    continue
                low = str(v).lower()
                if "°f" in low or "fahrenheit" in low:
                    new_v = fahrenheit_to_celsius(fn)
                    out.at[idx, col] = new_v
                    f_conv += 1
                    if not ex_before:
                        ex_before, ex_after = str(v), str(new_v)
            if f_conv:
                issues.append(
                    Issue(
                        workbook_rel,
                        sheet_name,
                        col,
                        "temperature_f_to_c",
                        "Converted Fahrenheit to Celsius (column name + explicit F hint in cell)",
                        "medium",
                        f_conv,
                        example_before=ex_before,
                        example_after=ex_after,
                    )
                )

        # Currency
        if column_name_suggests_currency(col):
            symbols: set[str] = set()
            for idx in out.index:
                v = out.at[idx, col]
                if v is None or pd.isna(v):
                    continue
                s = str(v)
                for sym in ("£", "$", "€"):
                    if sym in s:
                        symbols.add(sym)
            if len(symbols) > 1:
                issues.append(
                    Issue(
                        workbook_rel,
                        sheet_name,
                        col,
                        "multi_currency_detected",
                        f"Multiple currency symbols in column: {symbols}",
                        "medium",
                        n_rows,
                        recommendation="Provide FX table to unify; not auto-converted.",
                    )
                )

        # Categorical unification
        if out[col].dtype == object:
            uniq = out[col].dropna().astype(str).unique()
            if 2 <= len(uniq) <= 200:
                changed = 0
                mapping_ex: dict[str, str] = {}
                for u in uniq:
                    new_u, did = unify_categorical_value(u)
                    if did:
                        mapping_ex[str(u)] = new_u
                        n_rows_changed = int(out[col].astype(str).eq(u).sum())
                        changed += n_rows_changed
                        out[col] = out[col].astype(str).replace({u: new_u})
                        canon_log.append(
                            {
                                "workbook": workbook_rel,
                                "sheet": sheet_name,
                                "column": col,
                                "from_value": u,
                                "to_value": new_u,
                                "rows_affected": n_rows_changed,
                            }
                        )
                if mapping_ex:
                    ex = json.dumps(mapping_ex, ensure_ascii=False)[:500]
                    issues.append(
                        Issue(
                            workbook_rel,
                            sheet_name,
                            col,
                            "categorical_canonicalized",
                            f"Examples: {ex}",
                            "medium",
                            int(changed),
                            recommendation="Full mapping persisted in canonicalization_map.csv.",
                        )
                    )

        # Robust outliers
        numcol = pd.to_numeric(out[col], errors="coerce")
        if numcol.notna().sum() >= 12:
            om = robust_outlier_mask(out[col])
            n_out = int(om.sum())
            if n_out:
                issues.append(
                    Issue(
                        workbook_rel,
                        sheet_name,
                        col,
                        "outlier_robust_log_mad",
                        "Iglewicz–Hoaglin |z|>3.5 on log positives (flag only)",
                        "low",
                        n_out,
                        recommendation="Inspect for data entry errors.",
                    )
                )

        # Negative-where-positive-expected
        if any(k in col.lower() for k in ("count", "number", "install", "price", "mw", "gwh")):
            neg = numcol < 0
            if neg.any():
                issues.append(
                    Issue(
                        workbook_rel,
                        sheet_name,
                        col,
                        "negative_value_suspicious",
                        "Negative values in non-negative context (flag only)",
                        "medium",
                        int(neg.sum()),
                    )
                )

    # ----- Prune redundant raw__ and empty *_iso_date columns ----------------
    for c in list(out.columns):
        if not c.startswith("raw__"):
            continue
        sister = c[len("raw__") :]
        if sister not in out.columns:
            continue
        a = out[c].astype("string").fillna("")
        b = out[sister].astype("string").fillna("")
        if a.equals(b):
            out.drop(columns=c, inplace=True)
    for c in list(iso_cols):
        if c in out.columns and out[c].notna().sum() == 0:
            out.drop(columns=c, inplace=True)
    for c in list(flag_cols):
        if c in out.columns and out[c].notna().sum() == 0:
            out.drop(columns=c, inplace=True)

    out.attrs["dq_duplicate_rows"] = int(ndup) if ndup else 0
    return out, metadata_block


def _canonicalize_dtypes_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """Cast object columns that are >=95% numeric to nullable Float64; force
    raw__ / _flag columns to ``string``.  Pyarrow refuses to write a column
    holding both Python str and float together — this prevents it."""
    out = df.copy()
    for c in out.columns:
        if str(c).startswith("raw__") or str(c).endswith("_flag"):
            out[c] = out[c].astype("string")
            continue
        col = out[c]
        if col.dtype == object:
            num = pd.to_numeric(col, errors="coerce")
            non_null = col.notna().sum()
            if non_null and num.notna().sum() / non_null >= 0.95:
                out[c] = num.astype("Float64")
            else:
                out[c] = col.astype("string")
    return out


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def _safe_stem(rel: str, sheet_name: str, max_len: int = 180) -> str:
    clean_sheet = re.sub(r"\s+", " ", str(sheet_name)).strip()
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", f"{Path(rel).stem}__{clean_sheet}")
    return safe.strip("_")[:max_len] or "sheet"


def _validate_against_registry(registry_path: Path, cleaned_root: Path) -> list[Issue]:
    """Validate every cleaned parquet against ``config/dataset_registry.yml``.

    Returns an ``Issue`` list (each contract failure is a high-severity
    ``schema_contract_violation``).  Missing PyYAML / missing registry are
    treated as soft skips: the pipeline still completes."""
    if _yaml is None or not registry_path.exists():
        return []
    try:
        registry = _yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        return [Issue(str(registry_path), "*", "*", "schema_registry_unreadable",
                      f"{type(exc).__name__}: {exc}", "high", 0)]

    issues: list[Issue] = []
    datasets = registry.get("datasets") or []
    for spec in datasets:
        wb = spec.get("workbook", "")
        sh = spec.get("sheet", "")
        spec.get("publisher", "")
        if not wb or not sh:
            continue
        wb_stem = Path(wb).stem
        wb_safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", wb_stem).strip("_")
        sh_safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", sh).strip("_")
        candidates = list(cleaned_root.rglob(f"{wb_safe}__{sh_safe}.parquet"))
        if not candidates:
            issues.append(Issue(wb, sh, "*", "schema_contract_violation",
                                f"Expected output not found: {wb_safe}__{sh_safe}.parquet",
                                "high", 0,
                                recommendation="Verify upstream sheet name; update config/dataset_registry.yml if rename is intentional."))
            continue
        parquet = candidates[0]
        try:
            df = pd.read_parquet(parquet)
        except Exception as exc:
            issues.append(Issue(wb, sh, "*", "schema_contract_violation",
                                f"Cannot read cleaned parquet: {type(exc).__name__}: {exc}",
                                "high", 0))
            continue

        required = spec.get("required_columns") or []
        missing = [c for c in required if c not in df.columns]
        if missing:
            issues.append(Issue(wb, sh, ",".join(missing),
                                "schema_contract_violation",
                                f"Missing required columns: {missing}",
                                "high", len(missing),
                                recommendation="Upstream column rename or pipeline regression — investigate."))

        min_rows = spec.get("min_rows")
        if isinstance(min_rows, int) and len(df) < min_rows:
            issues.append(Issue(wb, sh, "*", "schema_contract_violation",
                                f"Row count below floor: {len(df)} < {min_rows}",
                                "high", int(min_rows - len(df))))

        max_null_pct = spec.get("max_null_pct")
        if isinstance(max_null_pct, (int, float)) and not df.empty:
            pct = float(df.isna().mean().mean()) * 100
            if pct > float(max_null_pct):
                issues.append(Issue(wb, sh, "*", "schema_contract_violation",
                                    f"Null share {pct:.1f}% exceeds ceiling {float(max_null_pct):.1f}%",
                                    "medium", int(round(pct - float(max_null_pct)))))

        for col, bounds in (spec.get("numeric_columns") or {}).items():
            if col not in df.columns:
                continue
            arr = pd.to_numeric(df[col], errors="coerce").dropna()
            if arr.empty:
                continue
            lo = bounds.get("min")
            hi = bounds.get("max")
            n_below = int((arr < lo).sum()) if lo is not None else 0
            n_above = int((arr > hi).sum()) if hi is not None else 0
            if n_below + n_above > 0:
                issues.append(Issue(wb, sh, col, "schema_contract_violation",
                                    f"{n_below} below {lo} and {n_above} above {hi} (unit={bounds.get('unit', '?')})",
                                    "medium", n_below + n_above,
                                    recommendation="Verify unit; check raw__ shadow column for original value."))
    return issues


def _publisher_dir(rel: str) -> str:
    parts = Path(rel).parts
    return parts[0] if len(parts) > 1 else "_root"


def run_pipeline(
    repo_root: Path,
    audit_root: Path,
    cleaned_root: Path,
    roots: list[str],
    run_id: str,
) -> None:
    audit_root.mkdir(parents=True, exist_ok=True)
    cleaned_root.mkdir(parents=True, exist_ok=True)
    reports_root = audit_root / "reports"
    reports_root.mkdir(parents=True, exist_ok=True)

    all_issues: list[Issue] = []
    canon_log: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "tool": "scripts/run_data_quality_pipeline.py",
        "tool_version": "2.0.0",
        "repo_root": _printable_path(repo_root.resolve()),
        "audit_outputs": _printable_path(audit_root.resolve()),
        "cleaned_outputs": _printable_path(cleaned_root.resolve()),
        "workbooks": [],
    }

    for root_name in roots:
        base = repo_root / root_name
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if path.suffix.lower() not in {".xlsx", ".xls", ".ods"}:
                continue
            rel = path.relative_to(repo_root).as_posix()
            whash = _sha256(path)
            wb_entry: dict[str, Any] = {
                "path": rel,
                "sha256": whash,
                "sheets": [],
            }
            try:
                sheets = read_workbook(path)
            except Exception as exc:
                all_issues.append(
                    Issue(rel, "*", "*", "read_error", f"{type(exc).__name__}: {exc}", "high", 0)
                )
                manifest["workbooks"].append(wb_entry)
                continue

            publisher_dir = cleaned_root / _publisher_dir(rel)
            publisher_dir.mkdir(parents=True, exist_ok=True)

            for sheet_name, raw_df in sheets.items():
                issues_before = len(all_issues)
                cleaned, metadata_block = process_sheet(
                    rel, str(sheet_name), raw_df.copy(), all_issues, canon_log
                )

                stem = _safe_stem(rel, str(sheet_name))
                parquet_path = publisher_dir / f"{stem}.parquet"
                csv_path = publisher_dir / f"{stem}.csv"
                metadata_path = publisher_dir / f"{stem}__metadata.csv"

                parquet_written = False
                parquet_err: str | None = None
                # Canonicalize once: pyarrow refuses mixed-dtype object columns,
                # and ``to_csv(float_format=...)`` only respects floats living in
                # genuinely numeric dtypes — object columns get ``str()`` which
                # surfaces float-repr artefacts like ``52571.034999999996``.
                cleaned_typed = (
                    _canonicalize_dtypes_for_parquet(cleaned) if not cleaned.empty else cleaned
                )
                if not cleaned_typed.empty:
                    try:
                        cleaned_typed.to_parquet(
                            parquet_path, index=False, compression="zstd"
                        )
                        parquet_written = True
                    except Exception as exc:
                        parquet_err = f"{type(exc).__name__}: {exc}"
                        all_issues.append(
                            Issue(
                                rel,
                                str(sheet_name),
                                "*",
                                "parquet_write_failed",
                                parquet_err,
                                "high",
                                0,
                                recommendation="Inspect dtype mix; CSV is still authoritative.",
                            )
                        )
                cleaned_typed.to_csv(
                    csv_path,
                    index=False,
                    na_rep="",
                    float_format="%.10g",
                )
                if metadata_block is not None and not metadata_block.empty:
                    # Metadata rows can span the full Excel-max width; trim.
                    trim_empty_frame(metadata_block).to_csv(
                        metadata_path, index=False, header=False, na_rep=""
                    )

                sheet_report: dict[str, Any] = {
                    "sheet": str(sheet_name),
                    "rows_in": int(len(raw_df)),
                    "cols_in": int(raw_df.shape[1]),
                    "rows_out": int(len(cleaned)),
                    "cols_out": int(cleaned.shape[1]),
                    "issues_raised": len(all_issues) - issues_before,
                    "outputs": {
                        "csv": _printable_path(csv_path.relative_to(repo_root)),
                    },
                }
                if parquet_written:
                    sheet_report["outputs"]["parquet"] = _printable_path(
                        parquet_path.relative_to(repo_root)
                    )
                if parquet_err:
                    sheet_report["parquet_error"] = parquet_err
                if metadata_block is not None and not metadata_block.empty:
                    sheet_report["outputs"]["metadata"] = _printable_path(
                        metadata_path.relative_to(repo_root)
                    )
                wb_entry["sheets"].append(sheet_report)

            per_path = reports_root / f"{re.sub(r'[^a-zA-Z0-9_-]+', '_', rel)}.json"
            per_path.write_text(json.dumps(wb_entry, indent=2), encoding="utf-8")
            manifest["workbooks"].append(wb_entry)

    # Schema contract validation pass — runs after every sheet is written so it
    # can read the on-disk parquet artefacts directly.
    contract_issues = _validate_against_registry(repo_root / "config" / "dataset_registry.yml", cleaned_root)
    all_issues.extend(contract_issues)

    issues_df = pd.DataFrame([i.row_dict() for i in all_issues])
    issues_df.to_csv(audit_root / "issues_register.csv", index=False, na_rep="")
    pd.DataFrame(canon_log).to_csv(audit_root / "canonicalization_map.csv", index=False, na_rep="")

    manifest_path = audit_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    summary_lines = [
        f"# Data quality run `{run_id}`",
        "",
        f"- Workbooks processed: {len(manifest['workbooks'])}",
        f"- Sheets cleaned: {sum(len(w['sheets']) for w in manifest['workbooks'])}",
        f"- Total issues logged: {len(all_issues)}",
        f"- Canonicalization mappings: {len(canon_log)}",
        f"- Cleaned outputs root: `{_printable_path(cleaned_root)}`",
        "",
        "## Issue codes (issues / rows affected)",
        "",
    ]
    if not issues_df.empty:
        agg = (
            issues_df.groupby("issue_code")
            .agg(issues=("issue_code", "size"), rows_affected=("rows_affected", "sum"))
            .sort_values("issues", ascending=False)
        )
        for code, row in agg.iterrows():
            summary_lines.append(
                f"- `{code}`: {int(row['issues'])} issues / {int(row['rows_affected'])} rows"
            )
    else:
        summary_lines.append("- (none)")
    summary_lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- `{audit_root.name}/issues_register.csv` — flat audit trail",
            f"- `{audit_root.name}/canonicalization_map.csv` — every from/to value mapping",
            f"- `{audit_root.name}/reports/*.json` — per-workbook metadata",
            f"- `{audit_root.name}/manifest.json` — full run manifest with output map",
            f"- `{audit_root.name}/SHA256SUMS` — checksums of every cleaned file",
            f"- `{cleaned_root.name}/<publisher>/*.parquet` (+ matching `.csv`) — cleaned data",
            f"- `{cleaned_root.name}/<publisher>/*__metadata.csv` — header-row metadata recovered",
            "",
            "## Risks",
            "",
            "- Ambiguous dates are **not** coerced; review `ambiguous_date` rows.",
            "- Embedded unit conversion assumes stated token (kg, lb, etc.).",
            "- Temperature conversion uses column-name heuristic.",
            "- No FX rates: multi-currency columns are flagged only.",
            "- HTML-as-Excel files use sheet-name recovery from `<x:Name>` markers; ",
            "  files without those markers fall back to `html_table_<n>`.",
        ]
    )
    (audit_root / "SUMMARY.md").write_text("\n".join(summary_lines), encoding="utf-8")

    sums_lines: list[str] = []
    for p in sorted(cleaned_root.rglob("*")):
        if p.is_file():
            sums_lines.append(f"{_sha256(p)}  {p.relative_to(repo_root).as_posix()}")
    (audit_root / "SHA256SUMS").write_text("\n".join(sums_lines) + ("\n" if sums_lines else ""), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Repository root (contains DESNZ, ONS, …)",
    )
    parser.add_argument(
        "--audit-out",
        type=Path,
        default=None,
        help="Audit output directory (default: <repo>/dq_run_<RUN_ID>)",
    )
    parser.add_argument(
        "--cleaned-out",
        type=Path,
        default=None,
        help="Cleaned data output directory (default: <repo>/cleaned_data)",
    )
    parser.add_argument(
        "--roots",
        nargs="*",
        default=["DESNZ", "MCS", "ONS", "Ofgem", "IRENA"],
        help="Publisher folders under repo root",
    )
    args = parser.parse_args()
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    repo = args.repo.resolve()
    audit = (args.audit_out or repo / f"dq_run_{run_id}").resolve()
    cleaned = (args.cleaned_out or repo / "cleaned_data").resolve()
    run_pipeline(repo, audit, cleaned, args.roots, run_id)
    print(f"Done.\n  Audit:   {audit}\n  Cleaned: {cleaned}")


if __name__ == "__main__":
    main()
