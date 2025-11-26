"""Generate a schema report for city issues and check for similar issues.

Usage:
  python generate_report.py [--file PATH] [--summary "short title"]

If `--summary` is provided, the script will try keyword-based matching (pothole, streetlight, etc.)
and a fuzzy similarity check against existing `city_issues.json` (or the file you pass).
"""
from __future__ import annotations
import json
import sys
import os
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Optional
from dotenv import load_dotenv
try:
    from google import genai
    from google.genai import types
except Exception:
    genai = None
    types = None
try:
    import requests
except Exception:
    requests = None


def detect_file_mode(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        while True:
            ch = f.read(1)
            if not ch:
                return "empty"
            if ch.isspace():
                continue
            return "array" if ch == "[" else "ndjson"


def iter_records(path: str):
    mode = detect_file_mode(path)
    if mode == "empty":
        return
    if mode == "array":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                for obj in data:
                    yield obj
            else:
                yield data
    else:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue


def typename(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int) and not isinstance(v, bool):
        return "int"
    if isinstance(v, float):
        return "float"
    if isinstance(v, str):
        return "str"
    if isinstance(v, list):
        return "list"
    if isinstance(v, dict):
        return "object"
    return type(v).__name__


def infer_schema(path: str, max_samples_per_field: int = 3) -> dict:
    field_types = defaultdict(set)
    field_samples = defaultdict(list)
    total = 0
    for rec in iter_records(path):
        if not isinstance(rec, dict):
            continue
        total += 1
        for k, v in rec.items():
            t = typename(v)
            field_types[k].add(t)
            if len(field_samples[k]) < max_samples_per_field:
                field_samples[k].append(v)
    schema = {}
    for k in sorted(field_types.keys()):
        schema[k] = {"types": sorted(field_types[k]), "samples": field_samples[k]}
    return {"total_records": total, "fields": schema}


def pretty_print_schema(schema: dict) -> None:
    print(f"Total records: {schema.get('total_records',0)}\n")
    fields = schema.get("fields", {})
    for k, info in fields.items():
        types = ", ".join(info.get("types", []))
        print(f"- {k}: {types}")
        samples = info.get("samples", [])
        for s in samples:
            sval = json.dumps(s, ensure_ascii=False)
            if len(sval) > 120:
                sval = sval[:117] + "..."
            print(f"    sample: {sval}")


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


# keyword synonyms
KEYWORD_SYNONYMS = {
    "pothole": ["pothole", "potholes", "sinkhole", "road hole", "road damage"],
    "streetlight": ["streetlight", "light", "lamp", "lighting", "lights out", "alley light"],
    "graffiti": ["graffiti", "tagging", "vandalism", "spray paint"],
    "litter": ["litter", "trash", "garbage", "dumping"],
}


def find_similar_by_keyword(issues: Iterable[Dict[str, Any]], summary: str, top_n: int = 5):
    if not summary:
        return []
    s = summary.lower()
    found_keys = set()
    for canon, syns in KEYWORD_SYNONYMS.items():
        for syn in syns:
            if syn in s:
                found_keys.add(canon)
                break
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


def load_issues(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    return list(iter_records(path))


def build_prompt_for_gemini(summary: str, lat: float, lng: float, samples: List[Dict[str, Any]] | None = None) -> str:
    
    prompt = (
        "You are a helpful assistant that drafts SeeClickFix-style issue reports in JSON.\n"
        "Produce a single JSON object with fields: summary, description, address (optional), lat, lng, tags (array), severity (low/medium/high), reporter_name.\n"
    )
    prompt += f"Target summary: {summary}\nLocation: {lat},{lng}\n"
    if samples:
        prompt += "Existing similar issues (short):\n"
        for s in samples[:3]:
            prompt += f"- {s.get('summary','<no summary>')} | {s.get('address','')}\n"
    prompt += "\nReturn ONLY a JSON object (no explanation).\n"
    return prompt


def call_gemini(prompt: str) -> Optional[str]:
    """Call the Google `genai` SDK to generate content via Gemini.

    Requires `google-genai` installed and `GOOGLE_API_KEY` set in the environment (or .env).
    Returns a string with the model output, or None on error.
    """
    load_dotenv()
    if genai is None or types is None:
        return None
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    model = os.getenv("GEMINI_MODEL") or "gemini-2.5-flash"
    try:
        client = genai.Client(api_key=api_key)
        cfg = types.GenerateContentConfig(
            top_p=0.95,
            top_k=40,
            temperature=0.2,
            max_output_tokens=512,
            response_mime_type="application/json",
        )
        resp = client.models.generate_content(model=model, config=cfg, contents=[prompt])

        # SDK response may contain `candidates` or `output` depending on version
        if hasattr(resp, "candidates") and resp.candidates:
            first = resp.candidates[0]
            # candidate may expose `content`
            return getattr(first, "content", None) or getattr(first, "text", None) or str(first)

        if hasattr(resp, "output") and resp.output:
            first = resp.output[0]
            if isinstance(first, dict):
                return first.get("content") or first.get("text") or json.dumps(first)
            # if object-like, attempt attribute access
            return getattr(first, "content", None) or getattr(first, "text", None) or str(first)

        # fallback to string representation
        return str(resp)
    except Exception:
        return None


def main(argv) -> int:
    """Main entry: check for an issue (default 'pothole').

    Behavior:
    - Load `city_issues.json` (or a provided file with `-f/--file`).
    - Check for keyword matches and fuzzy matches for the target summary.
    - If a match exists, print `True` and exit 0.
    - If no match, build a draft report with lat/lng set to Long Beach coordinates,
      print the draft JSON, save it to `drafted_issue.json`, print `False`, and exit 0.
    """
    path = "city_issues.json"
    # default target issue summary
    target_summary = "streetlight"

    # simple arg parsing: allow --file PATH and optionally a summary string
    i = 1
    while i < len(argv):
        a = argv[i]
        if a in ("-f", "--file") and i + 1 < len(argv):
            path = argv[i + 1]
            i += 2
            continue
        if a in ("-s", "--summary") and i + 1 < len(argv):
            target_summary = argv[i + 1]
            i += 2
            continue
        # positional override
        target_summary = a
        i += 1

    issues = load_issues(path) if os.path.exists(path) else []

    # show inferred schema for visibility
    if os.path.exists(path):
        schema = infer_schema(path)
        pretty_print_schema(schema)

    print(f"\nChecking for issue: '{target_summary}'\n")

    # Keyword-based check first (good for 'pothole')
    k_matches = find_similar_by_keyword(issues, target_summary)
    if k_matches:
        print("Found keyword match(s):")
        for m in k_matches:
            print(f"- id={m.get('id')} score={m.get('_keyword_score')} summary={m.get('summary')} address={m.get('address')}")
        print(True)
        return 0

    # Fuzzy similarity fallback
    fmatch = find_similar_issue(issues, target_summary)
    if fmatch:
        print("Found fuzzy match:")
        print(json.dumps(fmatch, ensure_ascii=False, indent=2))
        print(True)
        return 0

    # No match found -> try Gemini to generate a draft; otherwise local draft
    lat, lng = 33.7701, -118.1937
    prompt = build_prompt_for_gemini(target_summary, lat, lng, samples=issues[:5] if issues else None)
    gemini_out = call_gemini(prompt)
    print(gemini_out)

    draft = None
    if gemini_out:
        # gemini_out may be an SDK object (not a plain string). Convert safely to text.
        if isinstance(gemini_out, str):
            text = gemini_out.strip()
        else:
            try:
                # try common attributes
                text = getattr(gemini_out, "content", None) or getattr(gemini_out, "text", None) or str(gemini_out)
            except Exception:
                text = str(gemini_out)
            text = text.strip()
        # Try direct JSON parse
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                draft = parsed
        except Exception:
            # find first {...} block
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    parsed = json.loads(text[start : end + 1])
                    if isinstance(parsed, dict):
                        draft = parsed
                except Exception:
                    draft = None

    if not draft:
        draft = {
            "summary": target_summary,
            "description": f"Auto-generated draft report for '{target_summary}' near ({lat}, {lng}).",
            "address": None,
            "lat": lat,
            "lng": lng,
            "tags": [target_summary],
            "severity": "medium",
            "reporter_name": "auto-draft",
            "status": "Draft",
        }

    # Ensure lat/lng
    if "lat" not in draft or draft.get("lat") is None:
        draft["lat"] = lat
    if "lng" not in draft or draft.get("lng") is None:
        draft["lng"] = lng

    out_path = "drafted_issue.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(draft, f, ensure_ascii=False, indent=2)

    if gemini_out:
        print("No existing match found. Gemini produced a draft (saved to", out_path, "):")
    else:
        print("No existing match found. Local draft created (saved to", out_path, "):")

    print(json.dumps(draft, ensure_ascii=False, indent=2))
    print(False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
