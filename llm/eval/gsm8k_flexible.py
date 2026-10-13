"""Flexible GSM8K scoring of saved generations.

lm-eval's built-in filters mark a response wrong whenever its FORMAT differs
from the expected regex, even when the answer is correct. This scorer
re-reads each logged response and compares the answer numerically.

Rules:
  * answer extraction, in priority order:
      1. last \\boxed{...}
      2. last "The answer is[:]" / "answer:" match
      3. last number in the response
  * normalisation: strip $ , % whitespace and a trailing '.'; "30.0" == "30"
  * fractions are evaluated: "3/4" -> 0.75 (x/0 -> unparseable)
  * numeric comparison with a relative tolerance, not rounded to integer
  * an unparseable response counts as WRONG
"""
import math
import re
from fractions import Fraction

BOXED = re.compile(r"\\boxed\s*\{([^{}]*)\}")
ANSWER_IS = re.compile(r"(?:the\s+answer\s+is|answer)\s*[:=]?\s*\**\s*"
                       r"(-?\$?\s*[0-9][0-9,]*(?:\.[0-9]+)?(?:\s*/\s*-?[0-9]+)?)", re.I)
NUMBER = re.compile(r"-?\$?\s*[0-9][0-9,]*(?:\.[0-9]+)?(?:\s*/\s*-?[0-9]+)?")
_CLEAN = re.compile(r"[\s,$%]")


def to_number(raw):
    """'$1,234.50' -> 1234.5 ; '3/4' -> 0.75 ; junk -> None. No rounding."""
    if raw is None:
        return None
    s = _CLEAN.sub("", str(raw)).strip().rstrip(".")
    if s in ("", "-", ".", "-."):
        return None
    if "/" in s:                                    # fractions are evaluated
        try:
            f = Fraction(s)
        except (ValueError, ZeroDivisionError):
            return None
        return float(f)
    try:
        v = float(s)
    except ValueError:
        return None
    return None if math.isinf(v) or math.isnan(v) else v


def extract_answer(text):
    for pat in (BOXED, ANSWER_IS, NUMBER):
        hits = pat.findall(text or "")
        for cand in reversed(hits):                 # last match wins
            v = to_number(cand if isinstance(cand, str) else cand[0])
            if v is not None:
                return v
    return None


def gold_answer(target):
    s = str(target)
    if "####" in s:
        s = s.split("####")[-1]
    return to_number(s)


def equal(pred, gold):
    if pred is None or gold is None:
        return False                                # unparseable == wrong
    return math.isclose(pred, gold, rel_tol=1e-6, abs_tol=1e-6)
