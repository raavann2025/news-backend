"""
Iteration-5 focused backend tests for POST /api/insights/directions.
Time-boxed: skips the slow /generate/stream calls.

Scenarios:
  1. is_news:true payload → 200 with 4 directions.
  2. is_news defaults false, single 2-char interest via helper → 200 with 4 directions
     (topic='Deep dive on AI', interests=['AI']).
  3. is_news:false explicit → 200 with 4 directions (regression).
"""

import os
import requests

BASE_URL = os.environ["REACT_APP_BACKEND_URL"].rstrip("/")
API = f"{BASE_URL}/api"


class TestDirectionsIteration5:
    def test_directions_is_news_true(self):
        payload = {
            "topic": "OpenAI o1 reasoning model release",
            "interests": [
                "Recent AI Developments",
                "OpenAI o1 reasoning model release",
            ],
            "is_news": True,
        }
        r = requests.post(f"{API}/insights/directions", json=payload, timeout=90)
        assert r.status_code == 200, r.text
        d = r.json()
        assert d["topic"] == payload["topic"]
        dirs = d.get("directions") or []
        assert isinstance(dirs, list) and len(dirs) == 4, f"expected 4 directions, got {len(dirs)}"
        for opt in dirs:
            title = (opt.get("title") or "").strip()
            assert title and len(title) > 3
        print("[is_news=true] DIRECTION TITLES:", [d.get("title") for d in dirs])

    def test_directions_default_is_news_false_regression(self):
        payload = {"topic": "coffee history", "interests": []}
        r = requests.post(f"{API}/insights/directions", json=payload, timeout=90)
        assert r.status_code == 200, r.text
        d = r.json()
        dirs = d.get("directions") or []
        assert len(dirs) == 4, f"expected 4 directions, got {len(dirs)}"
        print("[is_news default] DIRECTION TITLES:", [d.get("title") for d in dirs])

    def test_directions_helper_padded_single_short_interest(self):
        # Mirrors what synthesizeTopicFromInterests(['AI']) sends now.
        payload = {"topic": "Deep dive on AI", "interests": ["AI"]}
        r = requests.post(f"{API}/insights/directions", json=payload, timeout=90)
        assert r.status_code == 200, r.text
        d = r.json()
        dirs = d.get("directions") or []
        assert len(dirs) == 4, f"expected 4 directions, got {len(dirs)}"
        for opt in dirs:
            assert (opt.get("title") or "").strip()
        print("[helper-padded AI] DIRECTION TITLES:", [d.get("title") for d in dirs])
