import time, requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import json

UA = {"User-Agent": "CivicAI-Student/1.0 (edu project)"}
BASE = "https://seeclickfix.com/api/v2"

# Session with retry for transient network errors
session = requests.Session()
retries = Retry(total=3, backoff_factor=1, status_forcelist=(429, 500, 502, 503, 504))
session.mount("https://", HTTPAdapter(max_retries=retries))

def find_response_site(lat, lng):
    r = requests.get(f"{BASE}/places", params={"lat": lat, "lng": lng}, 
                     headers=UA, timeout=15)
    r.raise_for_status()
    places = r.json().get("places", [])
    return places if places else None

def fetch_issues(place_url, status="open", per_page=50, max_pages=5):
    all_issues = []
    for page in range(1, max_pages + 1):
        try:
            r = session.get(f"{BASE}/issues",
                            params={"place_url": place_url, "status": status, "page": page, "per_page": per_page},
                            headers=UA, timeout=30)
            r.raise_for_status()
        except requests.exceptions.ReadTimeout:
            print(f"Read timeout fetching page {page}; skipping this page.")
            continue
        except requests.exceptions.RequestException as e:
            print(f"Request error fetching page {page}: {e}; skipping.")
            continue
        issues = r.json().get("issues") or []
        if not issues: break
        all_issues.extend(issues)
        time.sleep(3)  # be polite (~20 req/min public limit)

        with open("city_issues.json", "w", encoding="utf-8") as f:
            json.dump(issues, f, ensure_ascii=False, indent=2)
    return all_issues

def choose_best_place(places):
    # Preferred keywords for real cities
    priority_keywords = ["city", "town", "municipality"]
    # Avoid overly broad names
    blacklist = ["continental", "united states", "usa", "state", "county"]

    for p in places:
        name = p["name"].lower()
        place_type = p.get("type", "").lower()
        # Skip broad areas
        if any(bad in name for bad in blacklist):
            continue
        # Prefer city-like names or explicit type
        if any(k in place_type for k in priority_keywords) or "city" in name:
            return p

    # Fallback: pick first non-global entry
    for p in places:
        if not any(bad in p["name"].lower() for bad in blacklist):
            return p

    # Last resort: return the first (e.g., Continental US)
    return places[0] if places else None

if __name__ == "__main__":
    lat, lng = 33.7701, -118.1937  # Long Beach downtown
    slug = find_response_site(lat, lng)
    for s in slug:
        print(s['url_name'])
    if not slug:
        raise SystemExit("No SeeClickFix coverage detected around that point.")
    spot = choose_best_place(slug)
    issues = fetch_issues(spot['url_name'], status="open", per_page=50, max_pages=3)
    print(f"Place: {slug} | Open issues pulled: {len(issues)}")
    for i in issues[:10]:
        print(i.get("summary"), "—", i.get("address"), "|", i.get("status"))
