"""Assist composing a SeeClickFix issue report using Gemini (or local fallback).

Workflow:
- Load existing `city_issues.json` (JSON array or NDJSON) and look for similar issues.
- If a close match exists, show it and ask whether to reuse it.
- If not, prompt the user for missing fields and either call a Gemini-like LLM API
  (if `GEMINI_API_KEY` and `GEMINI_MODEL`/`GEMINI_ENDPOINT` are set) or build a draft locally.

This is a lightweight, self-contained helper. It does not auto-submit reports to SeeClickFix.
"""
from __future__ import annotations

import json
import os
import sys
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Optional

try:
    import requests
except Exception:
    requests = None  # optional, only needed for a configured Gemini endpoint


def load_issues(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            data = json.load(f)
            return data if isinstance(data, list) else [data]
        else:
            # treat as NDJSON
            out = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
            return out


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a or "", b or "").ratio()


def find_similar_issue(issues: Iterable[Dict[str, Any]], summary: str, address: Optional[str] = None,
                       threshold: float = 0.6) -> Optional[Dict[str, Any]]:
    best = None
    best_score = 0.0
    target = (summary or "") + " \u2014 " + (address or "")
    for it in issues:
        cand = (it.get("summary", "") or "") + " \u2014 " + (it.get("address", "") or "")
        score = similarity(target.lower(), cand.lower())
        if score > best_score:
            best_score = score
            best = it
    if best_score >= threshold:
        best_copy = dict(best)
        best_copy["_similarity_score"] = best_score
        return best_copy
    return None


# simple keyword synonym mapping for common issue types
KEYWORD_SYNONYMS = {
    "pothole": ["pothole", "potholes", "sinkhole", "road hole", "road damage"],
    "streetlight": ["streetlight", "light", "lamp", "lighting", "lights out", "alley light"],
    "graffiti": ["graffiti", "tagging", "vandalism", "spray paint"],
    "litter": ["litter", "trash", "garbage", "dumping"],
    "sign": ["sign", "stop sign", "street sign", "signage"],
}


def find_similar_by_keyword(issues: Iterable[Dict[str, Any]], summary: str, top_n: int = 5):
    """Return issues that contain keyword synonyms from the provided summary.

    Matches are based on substring presence in summary/description/address/request_type/tags.
    Returns a list of matching issue dicts sorted by match count.
    """
    if not summary:
        return []
    s = summary.lower()
    # detect which canonical keywords appear in the user's summary
    found_keys = set()
    for canon, syns in KEYWORD_SYNONYMS.items():
        for syn in syns:
            if syn in s:
                found_keys.add(canon)
                break

    # if none detected, try to look for any synonym substring tokens from summary words
    if not found_keys:
        words = [w.strip() for w in s.split() if len(w) > 3]
        for canon, syns in KEYWORD_SYNONYMS.items():
            for syn in syns:
                for w in words:
                    if w in syn:
                        found_keys.add(canon)
                        break
                if canon in found_keys:
                    break

    if not found_keys:
        return []

    matches = []
    for it in issues:
        score = 0
        hay = " ".join(
            [str(it.get("summary", "") or ""), str(it.get("description", "") or ""), str(it.get("address", "") or "")]
        ).lower()
        # also check tags and request_type
        if isinstance(it.get("tags"), list):
            hay += " " + " ".join([str(t).lower() for t in it.get("tags")])
        if it.get("request_type"):
            if isinstance(it.get("request_type"), dict):
                hay += " " + " ".join([str(v).lower() for v in it.get("request_type", {}).values()])
            else:
                hay += " " + str(it.get("request_type")).lower()

        for canon in found_keys:
            for syn in KEYWORD_SYNONYMS.get(canon, []):
                if syn in hay:
                    score += 1

        if score > 0:
            m = dict(it)
            m["_keyword_score"] = score
            matches.append(m)

    matches.sort(key=lambda x: x.get("_keyword_score", 0), reverse=True)
    return matches[:top_n]


def build_prompt(input_fields: Dict[str, Any], sample_issues: List[Dict[str, Any]]) -> str:
    # Create a concise instruction for the LLM to draft a SeeClickFix-style issue JSON
    prompt = (
        "You are a helpful assistant that drafts SeeClickFix issue reports as JSON. "
        "Given the following partial information, produce a single JSON object with these fields: "
        "summary (short title), description (detailed), address, lat (optional float), lng (optional float), "
        "tags (list of short strings), severity (one of low/medium/high), reporter_name (optional).\n\n"
    )
    prompt += "Partial input:\n" + json.dumps(input_fields, ensure_ascii=False, indent=2) + "\n\n"
    if sample_issues:
        prompt += "Example existing issues (short):\n"
        for s in sample_issues[:3]:
            prompt += "- " + (s.get("summary") or "<no summary>") + " | " + (s.get("address") or "") + "\n"
    prompt += (
        "\nReturn only a JSON object (no surrounding text). Keep descriptions realistic but concise. "
        "If lat/lng are unknown, omit them or set to null."
    )
    return prompt


def call_gemini(prompt: str) -> Optional[str]:
    """Call a Gemini-like endpoint if configured.

    Expects environment variables:
    - GEMINI_ENDPOINT: full URL to POST the prompt
    - GEMINI_API_KEY: API key for Authorization (Bearer)

    The function returns the raw text result from the model or None on failure.
    This is a best-effort integration; adjust to your provider's details.
    """
    endpoint = os.environ.get("GEMINI_ENDPOINT")
    api_key = os.environ.get("GEMINI_API_KEY")
    model = os.environ.get("GEMINI_MODEL") or "models/text-bison-001"
    if requests is None:
        return None

    # If a custom endpoint is provided, prefer that (user-managed URL)
    if endpoint and api_key:
        try:
            resp = requests.post(
                endpoint,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"prompt": prompt, "max_tokens": 600},
                timeout=20,
            )
            resp.raise_for_status()
            data = resp.json()
            # provider-specific: try common fields
            if isinstance(data, dict):
                return data.get("text") or data.get("output") or data.get("result") or json.dumps(data)
            return str(data)
        except Exception:
            return None

    # Otherwise, attempt Google Generative Language HTTP API (Gemini) using API key
    if api_key:
        # model should be like 'models/text-bison-001' or 'models/gemini-1.0'
        base = "https://generativelanguage.googleapis.com/v1beta2"
        url = f"{base}/{model}:generate?key={api_key}"
        payload = {
            "prompt": {"text": prompt},
            "max_output_tokens": 512,
            "temperature": 0.25,
        }
        try:
            resp = requests.post(url, json=payload, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            # Google-style: candidates -> content, or output -> text
            if isinstance(data, dict):
                # v1beta2 response often has 'candidates' with 'content'
                if "candidates" in data and isinstance(data["candidates"], list) and data["candidates"]:
                    cand = data["candidates"][0]
                    if isinstance(cand, dict) and "content" in cand:
                        return cand["content"]
                # older/newer shapes
                if "output" in data:
                    out = data["output"]
                    if isinstance(out, list) and out:
                        # some responses include 'content' inside items
                        first = out[0]
                        if isinstance(first, dict):
                            text = first.get("content") or first.get("text")
                            if text:
                                return text
                # fallback to common keys
                return data.get("text") or data.get("result") or json.dumps(data)
            return str(data)
        except Exception:
            return None

    return None


def draft_report(input_fields: Dict[str, Any], issues_sample: List[Dict[str, Any]]) -> Dict[str, Any]:
    prompt = build_prompt(input_fields, issues_sample)
    model_text = call_gemini(prompt)
    if model_text:
        # try to extract JSON from model_text
        text = model_text.strip()
        try:
            # if the model returned JSON directly
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except Exception:
            # attempt to find first { ... } block
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    obj = json.loads(text[start : end + 1])
                    if isinstance(obj, dict):
                        return obj
                except Exception:
                    pass

    # Fallback local drafting: expand description and normalize fields
    summary = input_fields.get("summary") or "(no summary)"
    addr = input_fields.get("address")
    desc = input_fields.get("description") or (
        f"Reported issue: {summary}. Location: {addr or 'unspecified'}. Please provide additional details."
    )
    tags = input_fields.get("tags") or []
    severity = input_fields.get("severity") or "medium"
    reporter = input_fields.get("reporter_name") or "anonymous"
    drafted = {
        "summary": summary,
        "description": desc,
        "address": addr,
        "lat": input_fields.get("lat"),
        "lng": input_fields.get("lng"),
        "tags": tags,
        "severity": severity,
        "reporter_name": reporter,
        "status": "Draft",
    }
    return drafted


def main(argv) -> int:
    issues_path = argv[1] if len(argv) > 1 else "city_issues.json"
    issues = load_issues(issues_path)

    print("Enter a short summary/title for the issue (e.g. 'broken streetlight on 3rd'):")
    summary = input("Summary: ").strip()
    if not summary:
        print("Summary is required.")
        return 2
    address = input("Address (optional): ").strip() or None
    description = input("Short description (optional): ").strip() or None

    # First check keyword-based matches (e.g., pothole)
    keyword_matches = find_similar_by_keyword(issues, summary)
    if keyword_matches:
        print("Found issues matching keywords related to your summary:")
        for m in keyword_matches:
            print(f"- score={m.get('_keyword_score')} id={m.get('id')} summary={m.get('summary')} address={m.get('address')}")
        usek = input("Use one of these existing issues instead of drafting? [y/N]: ").strip().lower()
        if usek == "y":
            pick = input("Enter the id of the issue to use (or press Enter to cancel): ").strip()
            if pick:
                for m in keyword_matches:
                    if str(m.get("id")) == pick:
                        print("Using existing issue. Exiting.")
                        print(json.dumps(m, ensure_ascii=False, indent=2))
                        return 0

    # Next try fuzzy similarity
    match = find_similar_issue(issues, summary, address, threshold=0.6)
    if match:
        print("Found a similar existing issue (score={:.2f}):".format(match.get("_similarity_score", 0)))
        print(json.dumps(match, ensure_ascii=False, indent=2))
        use = input("Use this existing issue instead of drafting? [y/N]: ").strip().lower()
        if use == "y":
            print("Use the existing issue as your report. Exiting.")
            return 0

    # collect optional extras
    tags_raw = input("Comma-separated tags (optional): ").strip()
    tags = [t.strip() for t in tags_raw.split(",") if t.strip()] if tags_raw else []
    severity = input("Severity [low/medium/high] (default medium): ").strip().lower() or "medium"
    reporter = input("Reporter name (optional): ").strip() or None

    input_fields = {
        "summary": summary,
        "address": address,
        "description": description,
        "tags": tags,
        "severity": severity,
        "reporter_name": reporter,
    }

    drafted = draft_report(input_fields, issues_sample=issues[:5])

    print("\nDrafted report:\n")
    print(json.dumps(drafted, ensure_ascii=False, indent=2))

    save = input("Save this draft to file? [Y/n]: ").strip().lower()
    if save in ("", "y", "yes"):
        out_path = input("Output path (default drafted_issue.json): ").strip() or "drafted_issue.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(drafted, f, ensure_ascii=False, indent=2)
        print(f"Draft saved to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
