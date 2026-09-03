"""Iteration 4 backend smoke: /api/insights/hot/regenerate-single + hot/stream."""
import os
import json
import time
import requests
import pytest

BASE_URL = os.environ.get("REACT_APP_BACKEND_URL", "https://interactive-news-hub-1.preview.emergentagent.com").rstrip("/")


def test_regenerate_single_returns_new_card():
    payload = {
        "topic": "India IPO rush",
        "headline": "12 firms lining up",
        "text": "Zepto and others plan Rs 25,000 cr IPO.",
        "source_title": "Mint",
        "steer": "give me the counter-intuitive angle",
    }
    r = requests.post(f"{BASE_URL}/api/insights/hot/regenerate-single", json=payload, timeout=30)
    assert r.status_code == 200, r.text
    data = r.json()
    assert "card" in data, data
    c = data["card"]
    for k in ("id", "topic", "headline", "text", "source_title"):
        assert k in c, f"missing {k}"
    assert isinstance(c["headline"], str) and len(c["headline"]) > 0
    assert c["headline"] != payload["headline"], "Headline unchanged"
    assert c["topic"] == payload["topic"]


def test_regenerate_single_no_steer():
    payload = {
        "topic": "RBI liquidity",
        "headline": "RBI just injected 23 billion",
        "text": "The central bank pumped in cash to ease overnight rates.",
        "source_title": "Reuters",
    }
    r = requests.post(f"{BASE_URL}/api/insights/hot/regenerate-single", json=payload, timeout=30)
    assert r.status_code == 200, r.text
    c = r.json()["card"]
    assert c["headline"] and c["text"]


def test_hot_stream_still_returns_card():
    """Ensure /api/insights/hot/stream emits at least one card event within 20s."""
    got_card = False
    with requests.get(f"{BASE_URL}/api/insights/hot/stream", stream=True, timeout=25) as r:
        assert r.status_code == 200
        start = time.time()
        for raw in r.iter_lines(decode_unicode=True):
            if raw and raw.startswith("data:"):
                try:
                    j = json.loads(raw[5:].strip())
                    if j.get("phase") == "card" and j.get("card"):
                        got_card = True
                        break
                except Exception:
                    pass
            if time.time() - start > 20:
                break
    assert got_card, "No card event received from hot/stream within 20s"
