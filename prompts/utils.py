"""Shared text helpers for diagnosis prompts and pipeline extraction."""

from __future__ import annotations

import json
import re
from typing import Iterable, List


def normalize_text(text: str) -> str:
    text = text or ""
    text = text.lower()
    text = re.sub(r"\*\*", "", text)
    text = re.sub(r"\([^)]*\)", "", text)
    text = re.sub(r"[`“”\"']", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text).strip()
    return text


def extract_final_diagnosis(output_text: str) -> str:
    if not isinstance(output_text, str):
        return ""
    match = re.search(r"final\s+diagnosis\s*[:\-–]?\s*\**\s*([^\n]+)", output_text, flags=re.I)
    if match:
        return match.group(1).strip()
    return output_text.splitlines()[0].strip() if output_text else ""


def extract_rationale(output_text: str) -> str:
    if not isinstance(output_text, str):
        return ""
    try:
        match = re.search(r"rationale\s*[:\-–]?\s*([\s\S]+)$", output_text, flags=re.I)
        return (match.group(1).strip() if match else output_text.strip())
    except Exception:
        return output_text.strip()


def extract_ranked_diagnoses(output_text: str, limit: int = 5) -> List[str]:
    if not isinstance(output_text, str):
        return []

    ranked: List[str] = []
    line_pattern = re.compile(r"^\s*(?:\d+\.|-)\s*Final diagnosis\s*[:\-–]\s*(.+)$", flags=re.I)
    for line in output_text.splitlines():
        match = line_pattern.search(line)
        if not match:
            continue
        diag = match.group(1).strip().rstrip(";.")
        if diag:
            ranked.append(diag)
        if len(ranked) >= limit:
            break

    if ranked:
        return ranked

    fallback = re.findall(r"final\s+diagnosis\s*[:\-–]\s*([^\n]+)", output_text, flags=re.I)
    cleaned: List[str] = []
    for item in fallback:
        diag = item.strip().rstrip(";.")
        if diag:
            cleaned.append(diag)
        if len(cleaned) >= limit:
            break
    return cleaned


def serialize_ranked_diagnoses(candidates: Iterable[str]) -> str:
    data = [str(c) for c in candidates if str(c).strip()]
    return json.dumps(data) if data else ""

