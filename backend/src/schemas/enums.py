"""Enumerations matching the production Law Lens schema.

Integer codes are the wire format so downstream consumers get stable,
sortable values without re-fetching display labels.
"""

from __future__ import annotations

from enum import IntEnum


class BenchType(IntEnum):
    UNKNOWN = 0
    SINGLE = 1
    DIVISION = 2
    THREE_JUDGE = 3
    FIVE_JUDGE = 5
    SEVEN_JUDGE = 7
    NINE_JUDGE = 9
    CONSTITUTION = 11


class Outcome(IntEnum):
    UNKNOWN = 0
    ALLOWED = 1
    DISMISSED = 2
    PARTIALLY_ALLOWED = 3
    REMANDED = 4
    DISPOSED = 5
    REFERRED = 6
    WITHDRAWN = 7
    INFRUCTUOUS = 8


class DecisionType(IntEnum):
    UNKNOWN = 0
    ORDER = 1
    JUDGMENT = 2


SUBJECT_CATEGORIES: dict[str, int] = {
    "Criminal": 1,
    "Constitutional": 2,
    "Civil": 3,
    "Tax": 4,
    "Service": 5,
    "Family": 6,
    "Company": 7,
    "Intellectual Property": 8,
    "Arbitration": 9,
    "Environmental": 10,
    "Labour": 11,
    "Property": 12,
    "Contract": 13,
    "Banking": 14,
    "Consumer": 15,
    "Administrative": 16,
    "Election": 17,
    "Insolvency": 18,
    "Education": 19,
    "Other": 99,
}

SUBJECT_LABELS: dict[int, str] = {v: k for k, v in SUBJECT_CATEGORIES.items()}

OUTCOME_LABELS: dict[int, str] = {
    0: "Unknown",
    1: "Allowed",
    2: "Dismissed",
    3: "Partially Allowed",
    4: "Remanded",
    5: "Disposed",
    6: "Referred to Larger Bench",
    7: "Withdrawn",
    8: "Infructuous",
}

BENCH_TYPE_LABELS: dict[int, str] = {
    0: "Unknown",
    1: "Single Judge",
    2: "Division Bench",
    3: "Three-Judge Bench",
    5: "Five-Judge Bench",
    7: "Seven-Judge Bench",
    9: "Nine-Judge Bench",
    11: "Constitution Bench",
}


def category_code(label: str | None) -> int:
    if not label:
        return SUBJECT_CATEGORIES["Other"]
    return SUBJECT_CATEGORIES.get(label.strip(), SUBJECT_CATEGORIES["Other"])


def category_label(code: int | None) -> str:
    if code is None:
        return "Other"
    return SUBJECT_LABELS.get(code, "Other")


def infer_bench_type(bench_string: str | None) -> BenchType:
    if not bench_string:
        return BenchType.UNKNOWN

    s = bench_string.replace(" and ", ", ").replace(" & ", ", ")
    judges = [j.strip() for j in s.split(",") if j.strip()]
    real_judges = [
        j for j in judges
        if len(j) > 2 and not j.upper().rstrip(".") in {"J", "JJ", "CJI", "CJ"}
    ]
    n = len(real_judges) or len(judges)

    if n <= 1:
        return BenchType.SINGLE
    if n == 2:
        return BenchType.DIVISION
    if n == 3:
        return BenchType.THREE_JUDGE
    if n == 5:
        return BenchType.FIVE_JUDGE
    if n == 7:
        return BenchType.SEVEN_JUDGE
    if n == 9:
        return BenchType.NINE_JUDGE
    if n >= 5:
        return BenchType.CONSTITUTION
    return BenchType.UNKNOWN
