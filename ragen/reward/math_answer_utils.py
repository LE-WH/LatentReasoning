"""Math answer extraction and normalization utilities.

Migrated from scalable-latent-reasoning/dual_cot/math_answer_utils.py
"""
from __future__ import annotations

import re
from math import gcd


MATH_PROMPT_POSTFIX = "\nProvide ONLY the final answer after thinking. Do not explain."

_EXPLICIT_RE = re.compile(
    r"(?:final answer(?:\s+is)?|answer(?:\s+is)?)\s*[:\-]?\s*(.+)",
    re.IGNORECASE | re.DOTALL,
)
_FRAC_RE = re.compile(r"-?\d+\s*/\s*-?\d+")
_DECIMAL_RE = re.compile(r"-?\d+(?:\.\d+)?")
_INTEGER_RE = re.compile(r"-?\d+")


def _strip_tex_noise(s: str) -> str:
    s = s.strip()
    s = s.replace("$", "")
    s = s.replace("\\!", "")
    s = s.replace("\\,", "")
    s = s.replace("\\;", "")
    s = s.replace("\\:", "")
    s = s.replace("\\left", "")
    s = s.replace("\\right", "")
    s = s.replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _remove_outer_braces(s: str) -> str:
    s = s.strip()
    while len(s) >= 2 and s[0] == "{" and s[-1] == "}":
        depth = 0
        ok = True
        for i, ch in enumerate(s):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and i != len(s) - 1:
                    ok = False
                    break
        if ok:
            s = s[1:-1].strip()
        else:
            break
    return s


def _read_braced(text: str, start_brace: int) -> tuple[str, int]:
    if start_brace >= len(text) or text[start_brace] != "{":
        raise ValueError("start_brace must point to '{'")

    depth = 0
    buf = []
    i = start_brace
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
            if depth > 1:
                buf.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(buf), i
            buf.append(ch)
        else:
            buf.append(ch)
        i += 1

    raise ValueError("unmatched brace while reading boxed answer")


def extract_last_boxed(text: str) -> str:
    """Robustly extract the LAST \\boxed{...}, supporting nested braces."""
    key = "\\boxed"
    last = ""

    i = 0
    while True:
        pos = text.find(key, i)
        if pos < 0:
            break

        j = pos + len(key)
        while j < len(text) and text[j].isspace():
            j += 1

        if j < len(text) and text[j] == "{":
            try:
                content, end_pos = _read_braced(text, j)
                last = content.strip()
                i = end_pos + 1
                continue
            except Exception:
                pass

        i = pos + len(key)

    return last


def _extract_from_explicit_marker(text: str) -> str:
    m = _EXPLICIT_RE.search(text)
    if not m:
        return ""

    candidate = m.group(1).strip()

    boxed = extract_last_boxed(candidate)
    if boxed:
        return boxed

    frac = _FRAC_RE.search(candidate)
    if frac:
        return frac.group(0)

    nums = _DECIMAL_RE.findall(candidate)
    if nums:
        return nums[-1]

    return candidate.strip()


def extract_final_answer(text: str) -> str:
    """Prefer: 1) last \\boxed{...} 2) explicit marker 3) last number."""
    text = text.strip()

    boxed = extract_last_boxed(text)
    if boxed:
        return boxed.strip()

    explicit = _extract_from_explicit_marker(text)
    if explicit:
        return explicit.strip()

    fracs = _FRAC_RE.findall(text)
    if fracs:
        return fracs[-1].strip()

    nums = _DECIMAL_RE.findall(text)
    if nums:
        return nums[-1].strip()

    ints = _INTEGER_RE.findall(text)
    if ints:
        return ints[-1].strip()

    return ""


def normalize_final_answer(s: str) -> str:
    s = _strip_tex_noise(s)
    s = _remove_outer_braces(s)
    s = s.strip()
    s = s.rstrip(".")
    s = s.replace(" ", "")

    # normalize fraction
    if _FRAC_RE.fullmatch(s):
        a_str, b_str = s.split("/")
        try:
            a = int(a_str)
            b = int(b_str)
            if b == 0:
                return s
            if b < 0:
                a = -a
                b = -b
            g = gcd(abs(a), abs(b))
            a //= g
            b //= g
            return f"{a}/{b}"
        except Exception:
            return s

    # normalize decimal/int
    try:
        x = float(s)
        if x.is_integer():
            return str(int(x))
        return str(x)
    except Exception:
        return s


def answers_equal(a: str, b: str) -> bool:
    return normalize_final_answer(a) == normalize_final_answer(b)


def extract_and_normalize(text: str) -> tuple[str, str]:
    raw = extract_final_answer(text)
    norm = normalize_final_answer(raw) if raw else ""
    return raw, norm
