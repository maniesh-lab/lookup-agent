"""How an eval answer is scored. Every check is a plain rule, so a score
can always be traced back to why it passed or failed.

For each case:
- tools_ok:   the agent called every tool in expected_tools (extra tools are allowed)
- sources_ok: when expected_sources is set, at least one of those policy files
              came back from search_docs
- answer_ok:  every answer check passes:
    must_mention   each group needs at least one of its phrases in the answer
    truth/number   the truth-SQL number appears in the answer (converted to
                   another currency first when convert_to is set)
    truth/names    every name from the truth SQL appears in the answer
    fx             the converted amount appears in the answer
    expect_refusal the answer says the information isn't available
- passed: tools_ok and sources_ok and answer_ok
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
    "can't tell",
    "cannot tell",
]

NUMBER = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?")

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


def extract_numbers(text: str) -> list[float]:
    numbers = []
    for raw in NUMBER.findall(normalize(text)):
        cleaned = raw.replace(",", "")
        try:
            numbers.append(float(cleaned))
        except ValueError:
            continue
    return numbers


def mentions_number(text: str, expected: float, rel_tol: float = 0.005, abs_tol: float = 0.011) -> bool:
    for n in extract_numbers(text):
        if abs(n - expected) <= max(abs_tol, abs(expected) * rel_tol):
            return True
    return False


def mentions(text: str, phrase: str) -> bool:
    return normalize(phrase).lower() in normalize(text).lower()


def mentions_name(text: str, name: str) -> bool:
    """Names match ignoring case, and a note in brackets is optional:
    "Glass Food Containers" counts for "Glass Food Containers (Set of 4)"."""
    short = re.sub(r"\s*\(.*?\)", "", name).strip()
    return mentions(text, name) or (bool(short) and mentions(text, short))


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
            # A converted amount is checked loosely: the rate can move during a run.
            tol = 0.02 if truth.get("converted") else 0.005
            ok = expected is not None and mentions_number(answer, expected, rel_tol=tol)
            answer_checks.append(ok)
            if not ok:
                failures.append(f"answer doesn't contain {expected}")
        elif truth["check"] == "names":
            missing = [v for v in truth["values"] if not mentions_name(answer, str(v))]
            answer_checks.append(not missing)
            if missing:
                failures.append(f"answer is missing {missing}")

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
