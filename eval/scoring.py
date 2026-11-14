"""How an eval answer is scored. Every check is a plain rule, so a score
can always be traced back to why it passed or failed.

For each case:
- tools_ok:   the agent called every tool in expected_tools (extra tools are allowed),
              and no more tool calls in total than max_tool_calls, when that is set
- sources_ok: when expected_sources is set, at least one of those policy files
              came back from search_docs
- answer_ok:  every answer check passes:
    must_mention   each group needs at least one of its phrases in the answer
                   (a phrase must start at a word, and a number phrase must be
                   the whole number: "25" doesn't match "$175" or "25.50")
    truth/number   the truth-SQL number appears in the answer (converted to
                   another currency first when convert_to is set). With a
                   "unit" (e.g. "order"), the number has to be the count of
                   that thing ("3 orders", "orders: 3"), so a "3" from
                   "3-5 business days" doesn't count; a different count of
                   it ("7 orders") fails the check
    truth/names    every name from the truth SQL appears in the answer; an
                   empty truth fails (the database isn't seeded). With a
                   universe (all names of that kind), naming any other one fails
    fx             the converted amount appears in the answer
    expect_refusal the answer says the information isn't available
    must_not_mention  none of these phrases appear (e.g. "has been deleted")
    must_not_match    none of these regular expressions match (e.g. a made-up
                      percentage in a refusal)
- passed: tools_ok and sources_ok and answer_ok

Dates written with month names ("January 1", "1 March 2026") are taken out
before numbers are read, like ISO dates, so a question's own dates aren't
mistaken for answer numbers.
"""

import re
from dataclasses import dataclass, field

REFUSAL_PHRASES = [
    "don't have",
    "do not have",
    "doesn't have",
    "does not have",
    "not available",
    "no information",
    "no data",
    "isn't stored",
    "not stored",
    "doesn't store",
    "does not store",
    "don't store",
    "do not store",
    "can't answer",
    "cannot answer",
    "unable to",
    "not able to",
    "don't track",
    "do not track",
    "doesn't track",
    "does not track",
    "not tracked",
    "isn't tracked",
    "no record",
    "don't record",
    "do not record",
    "doesn't record",
    "does not record",
    "not recorded",
    "don't keep track",
    "do not keep track",
    "doesn't keep track",
    "does not keep track",
    "can't find",
    "cannot find",
    "couldn't find",
    "could not find",
    "isn't something",
    "not something i can",
    "no way to",
    "can't provide",
    "cannot provide",
    "can't calculate",
    "cannot calculate",
    "doesn't contain",
    "does not contain",
    "can't tell",
    "cannot tell",
]

# Thousands commas only in real groups: "26,568", Indian-style "1,23,456", but "12,15,18" is
# three numbers, not 121518.
NUMBER = re.compile(r"(?<![\w.])-?(?:\d{1,2}(?:,\d{2})+,\d{3}(?!\d)|\d{1,3}(?:,\d{3})+(?!\d)|\d+)(?:\.\d+)?")
# "USD173.50", "Rs.23,456": a currency written right before the number would hide it.
CURRENCY_PREFIX = re.compile(r"\b(USD|EUR|GBP|INR|AUD|CAD|CHF|JPY|Rs\.?)(?=\d)", re.IGNORECASE)
# "2026-02-18" holds no answer numbers; without this, "02" and "18" count as numbers.
ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
# Neither do "January 1", "Feb 18th, 2026" or "1 March 2026": without this, the "1"
# in "between January 1 and January 31" passed a wrong count when the answer was 1.
_MONTH = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?"
    r"|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?"
)
MONTH_DATE = re.compile(
    rf"\b{_MONTH}\s+\d{{1,2}}(?:st|nd|rd|th)?\b(?:,?\s+\d{{4}})?"
    rf"|\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?{_MONTH}(?:,?\s+\d{{4}})?\b",
    re.IGNORECASE,
)
NUMBER_WORDS = {
    word: value
    for value, word in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
        "fifteen sixteen seventeen eighteen nineteen twenty".split()
    )
}

# Models often write special Unicode characters that look like normal ones:
# a non-breaking hyphen in "USB‑C", a narrow no-break space in "Grace Okafor"
# or as a thousands separator in "26 568". They are turned into their plain
# versions before any check, so a correct answer isn't failed over a lookalike.
HYPHENS = dict.fromkeys(map(ord, "‐‑‒–−"), "-")
SPACES = "    "
QUOTES = {ord("‘"): "'", ord("’"): "'", ord("“"): '"', ord("”"): '"'}
THOUSANDS_SPACE = re.compile(rf"(?<=\d)[{SPACES}](?=\d{{3}}(?!\d))")


def normalize(text: str) -> str:
    text = THOUSANDS_SPACE.sub("", text)  # "26 568" -> "26568"
    text = re.sub(f"[{SPACES}]", " ", text)
    return text.translate(HYPHENS).translate(QUOTES)


def without_dates(text: str) -> str:
    text = CURRENCY_PREFIX.sub(r"\1 ", normalize(text))
    return MONTH_DATE.sub(" ", ISO_DATE.sub(" ", text))


def extract_numbers(text: str) -> list[float]:
    numbers = []
    for raw in NUMBER.findall(without_dates(text)):
        cleaned = raw.replace(",", "")
        try:
            numbers.append(float(cleaned))
        except ValueError:
            continue
    return numbers


def mentions_number(text: str, expected: float, rel_tol: float = 0.005, abs_tol: float = 0.011) -> bool:
    return any(close_enough(n, expected, rel_tol, abs_tol) for n in extract_numbers(text))


def close_enough(n: float, expected: float, rel_tol: float = 0.005, abs_tol: float = 0.011) -> bool:
    return abs(n - expected) <= max(abs_tol, abs(expected) * rel_tol)


# A count: digits or a number word, standing on its own (not part of "175" or "3.5").
_COUNT = r"(?<![\w.,$])(\d[\d,]*(?:\.\d+)?|" + "|".join(NUMBER_WORDS) + r")(?![\w%]|[.,]\d)"


def _count_value(raw: str) -> float:
    return float(NUMBER_WORDS[raw]) if raw in NUMBER_WORDS else float(raw.replace(",", ""))


def mentions_count(text: str, expected: float, unit: str) -> bool:
    """The answer gives `expected` as the number of `unit`s.

    "3 orders", "3 open delivery tickets", "1 cancelled order" (up to two words
    in between) count; so does "orders still processing: 3" when no number
    stands right before the unit anywhere. If one does and none of those is
    the expected count, the answer claims a different number and fails.
    """
    t = without_dates(text).lower()
    unit_re = rf"\b{re.escape(unit.lower())}(?:e?s)?\b"
    before = [
        _count_value(m.group(1))
        for m in re.finditer(_COUNT + r"(?:[\W_]+[a-z][a-z-]*){0,2}?[\W_]+" + unit_re, t)
    ]
    if before:
        return any(close_enough(n, expected) for n in before)
    for m in re.finditer(unit_re + r"[^.!?\n]{0,60}?" + _COUNT, t):
        if close_enough(_count_value(m.group(1)), expected):
            return True
    return False


def mentions(text: str, phrase: str) -> bool:
    """The phrase appears, starting at a word ("yes" isn't found in "eyes"). A phrase
    that starts or ends with a digit must be the whole number: "75" is found in "$75"
    and "$75.00" but not in "$175", "75,000" or "75.50"."""
    p = normalize(phrase).lower()
    if not p:
        return False
    pattern = re.escape(p)
    if p[0].isalnum():
        pattern = r"(?<!\w)" + (r"(?<!\d[.,])" if p[0].isdigit() else "") + pattern
    if p[-1].isdigit():
        pattern += r"(?!\d)(?!,\d)(?!\.(?!0+(?!\d))\d)"
    return re.search(pattern, normalize(text).lower()) is not None


def mentions_name(text: str, name: str) -> bool:
    """Names match ignoring case, and a note in brackets is optional:
    "Glass Food Containers" counts for "Glass Food Containers (Set of 4)"."""
    short = re.sub(r"\s*\(.*?\)", "", name).strip()
    return mentions(text, name) or (bool(short) and mentions(text, short))


def extra_names(text: str, expected: list, universe: list) -> list[str]:
    """Names from the universe that the answer mentions but the truth doesn't hold.

    The expected names are blanked out first, so "Yoga Mat" isn't found inside
    an expected "Cork Yoga Mat".
    """
    rest = normalize(text)
    for name in sorted((str(v) for v in expected), key=len, reverse=True):
        short = re.sub(r"\s*\(.*?\)", "", name).strip()
        for form in (name, short):
            if form:
                rest = re.sub(re.escape(normalize(form)), " ", rest, flags=re.IGNORECASE)
    wanted = {str(v) for v in expected}
    return [str(n) for n in universe if str(n) not in wanted and mentions_name(rest, str(n))]


def is_refusal(text: str) -> bool:
    lowered = normalize(text).lower()
    return any(p in lowered for p in REFUSAL_PHRASES)


@dataclass
class CaseScore:
    id: str
    category: str
    passed: bool
    tools_ok: bool
    sources_ok: bool
    answer_ok: bool
    failures: list[str] = field(default_factory=list)


def score_case(
    case: dict,
    answer: str,
    tools_used: list[str],
    sources: list[dict],
    truth_values: list[dict],
    fx_expected: float | None,
) -> CaseScore:
    """truth_values: one dict per truth entry: {"check", "values", "expected_number"}."""
    failures: list[str] = []

    missing_tools = [t for t in case.get("expected_tools", []) if t not in tools_used]
    tools_ok = not missing_tools
    if missing_tools:
        failures.append(f"didn't call {missing_tools}")
    limit = case.get("max_tool_calls")
    if limit is not None and len(tools_used) > limit:
        tools_ok = False
        failures.append(f"used {len(tools_used)} tool calls (limit {limit})")

    sources_ok = True
    if case.get("expected_sources"):
        found = {s.get("name") for s in sources if s.get("type") == "document"}
        sources_ok = bool(found & set(case["expected_sources"]))
        if not sources_ok:
            failures.append(f"search didn't return {case['expected_sources']}")

    answer_checks: list[bool] = []

    for group in case.get("must_mention", []):
        ok = any(mentions(answer, phrase) for phrase in group)
        answer_checks.append(ok)
        if not ok:
            failures.append(f"answer doesn't mention any of {group}")

    for truth in truth_values:
        if truth["check"] == "number":
            expected = truth["expected_number"]
            unit = truth.get("unit")
            if expected is None:
                ok = False
            elif unit:
                ok = mentions_count(answer, expected, unit)
            else:
                # A converted amount is checked loosely: the rate can move during a run.
                tol = 0.02 if truth.get("converted") else 0.005
                ok = mentions_number(answer, expected, rel_tol=tol)
            answer_checks.append(ok)
            if not ok:
                what = f"give {expected} as the number of {unit}s" if unit else f"contain {expected}"
                failures.append(f"answer doesn't {what}")
        elif truth["check"] == "names":
            if not truth["values"]:
                answer_checks.append(False)
                failures.append("the truth query found no names (is the database seeded?)")
                continue
            missing = [v for v in truth["values"] if not mentions_name(answer, str(v))]
            answer_checks.append(not missing)
            if missing:
                failures.append(f"answer is missing {missing}")
            extra = extra_names(answer, truth["values"], truth.get("universe") or [])
            answer_checks.append(not extra)
            if extra:
                failures.append(f"answer also names {extra}")

    if fx_expected is not None:
        ok = mentions_number(answer, fx_expected, rel_tol=0.02)
        answer_checks.append(ok)
        if not ok:
            failures.append(f"answer doesn't contain the converted amount ~{fx_expected}")

    if case.get("expect_refusal"):
        ok = is_refusal(answer)
        answer_checks.append(ok)
        if not ok:
            failures.append("answer doesn't say the information is unavailable")

    said = [phrase for phrase in case.get("must_not_mention", []) if mentions(answer, phrase)]
    answer_checks.append(not said)
    if said:
        failures.append(f"answer says {said}")

    plain = normalize(answer)
    matched = [rx for rx in case.get("must_not_match", []) if re.search(rx, plain, re.IGNORECASE)]
    answer_checks.append(not matched)
    if matched:
        failures.append(f"answer matches {matched}")

    answer_ok = all(answer_checks)
    return CaseScore(
        id=case["id"],
        category=case["category"],
        passed=tools_ok and sources_ok and answer_ok,
        tools_ok=tools_ok,
        sources_ok=sources_ok,
        answer_ok=answer_ok,
        failures=failures,
    )
