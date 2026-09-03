"""Phase 3 backend tests: deep_research /generate, /behavior, /more (with behavior), /regenerate (deep)."""
import os
import re
import pytest
import requests

BASE_URL = os.environ.get("REACT_APP_BACKEND_URL", "https://research-swipe-1.preview.emergentagent.com").rstrip("/")
DEEP_TIMEOUT = 240  # 60-90s expected, be generous
TOPIC = "how India UPI became the world's largest payment network"


# --- Module scope: single deep-research feed reused across tests to save API cost ---
@pytest.fixture(scope="module")
def deep_feed():
    """Generate a real deep-research feed on the UPI topic. Slow (~60-90s). Retries once on 502."""
    last_err = None
    for attempt in range(3):
        try:
            r = requests.post(
                f"{BASE_URL}/api/insights/generate",
                json={"topic": TOPIC},
                timeout=DEEP_TIMEOUT,
            )
            if r.status_code == 200:
                data = r.json()
                assert "session_id" in data
                assert len(data["insights"]) == 12, f"Expected 12 got {len(data['insights'])}"
                return data
            last_err = f"HTTP {r.status_code}: {r.text[:200]}"
        except Exception as e:
            last_err = str(e)
    pytest.fail(f"Deep generate failed after 3 attempts: {last_err}")


# ---------- /api/insights/generate quality assertions ----------
def test_deep_generate_returns_12_insights(deep_feed):
    assert len(deep_feed["insights"]) == 12
    assert deep_feed["topic"].lower().startswith("how india")


def test_deep_generate_insights_are_quantified(deep_feed):
    """At least half the insights should contain a number, year, or %-figure."""
    quantified = 0
    for ins in deep_feed["insights"]:
        text = (ins.get("text") or "") + " " + (ins.get("headline") or "")
        if re.search(r"\d{2,}", text) or re.search(r"\d+\s?%|\bbillion\b|\btrillion\b|\bmillion\b", text, re.I):
            quantified += 1
    assert quantified >= 8, f"Only {quantified}/12 insights are quantified; expected >=8"


def test_deep_generate_mentions_named_entities(deep_feed):
    """UPI-specific: at least one insight should reference NPCI, RBI, or a named bank/company."""
    joined = " ".join((i.get("text") or "") + " " + (i.get("headline") or "") for i in deep_feed["insights"])
    keywords = ["NPCI", "RBI", "UPI", "PhonePe", "Google Pay", "Paytm", "India"]
    hits = [k for k in keywords if k.lower() in joined.lower()]
    assert len(hits) >= 3, f"Expected >=3 named entities, got {hits}"


def test_deep_generate_cites_real_source_url(deep_feed):
    """At least one insight cites a real http(s) source URL."""
    urls = [i.get("source_url") for i in deep_feed["insights"] if i.get("source_url")]
    assert len(urls) >= 1, "Expected at least 1 insight with a source_url"
    assert all(u.startswith("http") for u in urls)


# ---------- /api/insights/behavior ----------
def test_behavior_save_event(deep_feed):
    r = requests.post(
        f"{BASE_URL}/api/insights/behavior",
        json={
            "session_id": deep_feed["session_id"],
            "insight_id": deep_feed["insights"][0]["id"],
            "topic": deep_feed["topic"],
            "headline": deep_feed["insights"][0].get("headline", ""),
            "event": "save",
        },
        timeout=30,
    )
    assert r.status_code == 200
    assert r.json() == {"ok": True}


def test_behavior_skip_with_dwell(deep_feed):
    r = requests.post(
        f"{BASE_URL}/api/insights/behavior",
        json={
            "session_id": deep_feed["session_id"],
            "insight_id": deep_feed["insights"][1]["id"],
            "headline": deep_feed["insights"][1].get("headline", ""),
            "event": "skip",
            "dwell_ms": 1200,
        },
        timeout=30,
    )
    assert r.status_code == 200
    assert r.json()["ok"] is True


@pytest.mark.parametrize("event", ["ask", "regenerate", "share", "dwell"])
def test_behavior_other_events(deep_feed, event):
    r = requests.post(
        f"{BASE_URL}/api/insights/behavior",
        json={
            "session_id": deep_feed["session_id"],
            "insight_id": deep_feed["insights"][2]["id"],
            "headline": deep_feed["insights"][2].get("headline", ""),
            "event": event,
            "dwell_ms": 8000 if event == "dwell" else None,
        },
        timeout=30,
    )
    assert r.status_code == 200
    assert r.json()["ok"] is True


# ---------- /api/insights/more with behavior + no repeats ----------
def test_more_returns_12_no_repeat(deep_feed):
    prev = [i["headline"] for i in deep_feed["insights"] if i.get("headline")]
    r = requests.post(
        f"{BASE_URL}/api/insights/more",
        json={
            "session_id": deep_feed["session_id"],
            "topic": deep_feed["topic"],
            "previous_headlines": prev,
        },
        timeout=DEEP_TIMEOUT,
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert len(data["insights"]) == 12
    new_headlines = {i["headline"].strip().lower() for i in data["insights"] if i.get("headline")}
    old_headlines = {h.strip().lower() for h in prev}
    overlap = new_headlines & old_headlines
    assert len(overlap) <= 1, f"Too many repeated headlines: {overlap}"


# ---------- /api/insights/regenerate deep ----------
def test_regenerate_preserves_id_new_text(deep_feed):
    target = deep_feed["insights"][3]
    r = requests.post(
        f"{BASE_URL}/api/insights/regenerate",
        json={
            "topic": deep_feed["topic"],
            "insight_id": target["id"],
            "previous_text": target["text"],
            "exclude": [i["headline"] for i in deep_feed["insights"] if i.get("headline")],
        },
        timeout=DEEP_TIMEOUT,
    )
    assert r.status_code == 200, r.text
    fresh = r.json()
    assert fresh["id"] == target["id"]
    assert fresh["text"].strip() != target["text"].strip()
    assert len(fresh["text"]) > 30


# ---------- /manifest.json PWA sanity ----------
def test_manifest_json_served():
    r = requests.get(f"{BASE_URL}/manifest.json", timeout=30)
    assert r.status_code == 200
    m = r.json()
    assert m.get("name") == "Insight Swipe"
    assert m.get("start_url") == "/"
    assert m.get("display") == "standalone"


def test_sw_js_served():
    r = requests.get(f"{BASE_URL}/sw.js", timeout=30)
    assert r.status_code == 200
    assert "serviceWorker" in r.text or "self.addEventListener" in r.text
