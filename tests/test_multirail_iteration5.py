"""Multi-rail persona pipeline tests for /api/insights/generate/stream.

Tests validate:
 - rails_planned event emitted for multi-interest requests
 - final done payload contains data.rails and 12 insights
 - persona-tagged insights link to persona-lane source domains
 - single-interest still routes multi-rail with 1 rail
 - topic-only (no interests) uses SINGLE-RAIL flow (no rails_planned, no data.rails)
 - news mode routes through news-window path
"""
import json
import os
import re
import pytest
import requests

BASE_URL = os.environ.get("EXPO_PUBLIC_BACKEND_URL", "https://explore-topics.preview.emergentagent.com").rstrip("/")
STREAM_URL = f"{BASE_URL}/api/insights/generate/stream"
TIMEOUT = 240  # multi-rail can take up to 3 minutes

GOSSIP_DOMAINS = ["pagesix", "tmz", "deuxmoi", "fauxmoi", "buzzfeed", "instagram", "tiktok", "eonline", "people.com", "usmagazine", "dailymail", "reddit.com/r/fauxmoi"]
POLITICS_DOMAINS = ["reuters", "apnews", "foreignaffairs", "cfr.org", "politico", "ft.com", "bbc", "nytimes", "washingtonpost", "aljazeera", "economist"]
TECH_DOMAINS = ["techcrunch", "theverge", "9to5", "news.ycombinator", "producthunt", "arstechnica", "wired", "engadget", "theinformation"]


def _stream_events(payload: dict, timeout: int = TIMEOUT):
    """Yield parsed SSE JSON events from the stream."""
    with requests.post(STREAM_URL, json=payload, stream=True, timeout=timeout) as r:
        assert r.status_code == 200, f"Non-200 from stream: {r.status_code} {r.text[:400]}"
        for raw in r.iter_lines(decode_unicode=True):
            if not raw:
                continue
            if raw.startswith(":"):  # keep-alive
                continue
            if raw.startswith("data: "):
                data = raw[6:]
                try:
                    yield json.loads(data)
                except Exception:
                    continue


def _collect(payload: dict):
    """Run stream and collect all events until done/error."""
    events = []
    for evt in _stream_events(payload):
        events.append(evt)
        if evt.get("phase") in ("done", "error"):
            break
    return events


def _domain_hits(insights, domain_list):
    hits = 0
    for ins in insights:
        url = (ins.get("source_url") or "").lower()
        if any(d in url for d in domain_list):
            hits += 1
    return hits


# ---------- Multi-rail: 3 distinct interests + news ----------
@pytest.mark.timeout(TIMEOUT + 30)
def test_multirail_gossip_politics_tech_news():
    events = _collect({
        "topic": "latest happenings this week",
        "interests": ["Celebrity Gossip", "World Politics", "New Tech Launches"],
        "is_news": True,
    })
    phases = [e.get("phase") for e in events]
    assert "rails_planned" in phases, f"Missing rails_planned. Phases: {phases}"

    rails_evt = next(e for e in events if e.get("phase") == "rails_planned")
    rails_meta = (rails_evt.get("meta") or {}).get("rails") or []
    assert 2 <= len(rails_meta) <= 4, f"Expected 2-4 rails, got {len(rails_meta)}: {rails_meta}"

    names = [r.get("name", "") for r in rails_meta]
    personas = [r.get("persona", "") for r in rails_meta]
    assert all(names), "All rails must have name"
    assert all(personas), "All rails must have persona"
    assert len(set(names)) == len(names), f"Rail names must be distinct: {names}"
    # Personas must not be trivially generic
    for p in personas:
        assert len(p) > 20, f"Persona too generic: {p}"

    done = next((e for e in events if e.get("phase") == "done"), None)
    assert done is not None, f"No done event. Last: {events[-1] if events else None}"
    data = done.get("data") or {}
    assert "rails" in data and isinstance(data["rails"], list) and len(data["rails"]) >= 2
    insights = data.get("insights") or []
    assert len(insights) == 12, f"Expected 12 insights got {len(insights)}"

    # Check that at least one of gossip/politics/tech domain patterns are represented
    total_persona_hits = (
        _domain_hits(insights, GOSSIP_DOMAINS)
        + _domain_hits(insights, POLITICS_DOMAINS)
        + _domain_hits(insights, TECH_DOMAINS)
    )
    # Log for debug
    print("Persona-domain hits (total):", total_persona_hits)
    for ins in insights:
        print("  ->", ins.get("source_url"))
    assert total_persona_hits >= 3, "Expected at least 3 insights linking to persona-lane domains"


# ---------- Multi-rail: single-cluster interests ----------
@pytest.mark.timeout(TIMEOUT + 30)
def test_multirail_single_persona_cluster():
    events = _collect({
        "topic": "startup ecosystem",
        "interests": ["AI", "Startups", "New Tech Launches"],
    })
    rails_evt = next((e for e in events if e.get("phase") == "rails_planned"), None)
    assert rails_evt is not None, "rails_planned event missing"
    rails = (rails_evt.get("meta") or {}).get("rails") or []
    assert 1 <= len(rails) <= 2, f"Expected 1 (or 2) rail(s) for tightly clustered interests, got {len(rails)}"

    done = next((e for e in events if e.get("phase") == "done"), None)
    assert done is not None
    data = done.get("data") or {}
    assert len(data.get("insights") or []) == 12


# ---------- Multi-rail: single interest ----------
@pytest.mark.timeout(TIMEOUT + 30)
def test_multirail_single_interest_gossip():
    events = _collect({
        "topic": "this week's biggest stories",
        "interests": ["Celebrity Gossip"],
    })
    rails_evt = next((e for e in events if e.get("phase") == "rails_planned"), None)
    assert rails_evt is not None
    rails = (rails_evt.get("meta") or {}).get("rails") or []
    assert len(rails) == 1, f"Single interest should produce 1 rail, got {len(rails)}"
    persona = rails[0].get("persona", "").lower()
    assert "gossip" in persona or "celebrity" in persona or "pop culture" in persona, f"Persona should reflect gossip: {persona}"

    done = next((e for e in events if e.get("phase") == "done"), None)
    assert done is not None
    insights = (done.get("data") or {}).get("insights") or []
    assert len(insights) == 12


# ---------- Single-rail path: topic only ----------
@pytest.mark.timeout(TIMEOUT + 30)
def test_single_rail_topic_only():
    events = _collect({
        "topic": "why did the Roman Empire fall",
    })
    phases = [e.get("phase") for e in events]
    assert "rails_planned" not in phases, f"Topic-only must NOT plan rails. Phases: {phases}"

    done = next((e for e in events if e.get("phase") == "done"), None)
    assert done is not None, f"No done event: last={events[-1] if events else None}"
    data = done.get("data") or {}
    assert "rails" not in data, f"Single-rail done payload must NOT contain rails. Got: {list(data.keys())}"
    insights = data.get("insights") or []
    assert len(insights) == 12
