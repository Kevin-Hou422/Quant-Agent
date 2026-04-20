"""
_helpers.py — module-level utility helpers for DSL parsing and JSON extraction.
"""
from __future__ import annotations

import json
import re
from typing import Optional


def _extract_balanced(text: str, start: int) -> Optional[str]:
    """
    Extract the first balanced-parenthesis DSL expression from text[start:].

    Handles optional leading '-' and arbitrarily nested parens
    (e.g. rank(ts_delta(log(close), 5))).
    """
    paren_idx = text.find("(", start)
    if paren_idx == -1:
        return None
    depth = 0
    for i in range(paren_idx, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return text[start : i + 1].strip()
    return None


def _safe_json_loads(s: str) -> dict:
    """Extract and parse the first JSON object from a string, ignoring Markdown fences."""
    m = re.search(r"\{.*?\}", s, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return {}
