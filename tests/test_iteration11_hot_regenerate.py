"""Iteration 11 — Hot Now regenerate (fresh=1) + cache behaviour.

Covers:
  * GET /api/insights/hot/stream?fresh=1  → bypasses the 20-min cache
  * GET /api/insights/hot/stream          → serves cache on repeat calls
"""
import json
import os
import time

import pytest
import requests
from dotenv import dotenv_values

frontend_env = dotenv_values("/app/frontend/.env")
base_url = os.environ.get("REACT_APP_BACKEND_URL") or frontend_env.get("REACT_APP_BACKEND_URL")
if not base_url:
    raise RuntimeError("REACT_APP_BACKEND_URL missing")
BASE_URL = base_url.rstrip("/")
HOT_STREAM = f"{BASE_URL}/api/insights/hot/stream"


def consume_hot_stream(url, timeout=180):
    """Consume the SSE stream, return (events, first_card_latency, total_time)."""
    t0 = time.time()
    events = []
    first_card_at = None
    with requests.get(url, stream=True, timeout=timeout) as r:
        assert r.status_code == 200, f"{url} -> {r.status_code}"
        assert "text/event-stream" in r.headers.get("content-type", "")
        buf = ""
        for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
            if chunk is None:
                continue
            buf += chunk
            while "\n\n" in buf:
                raw, buf = buf.split("\n\n", 1)
                line = raw.strip()
                if not line or line.startswith(":"):
                    continue
                if not line.startswith("data:"):
                    continue
                try:
                    ev = json.loads(line[5:].strip())
                except Exception:
                    continue
                events.append(ev)
                if ev.get("phase") == "card" and first_card_at is None:
                    first_card_at = time.time() - t0
                if ev.get("phase") == "done":
                    return events, first_card_at, time.time() - t0
    return events, first_card_at, time.time() - t0


def cards_of(events):
    return [e["card"] for e in events if e.get("phase") == "card" and e.get("card")]


@pytest.fixture(scope="module")
def warm_cache():
    """Ensure the server cache is populated before the cache test."""
    events, _, _ = consume_hot_stream(HOT_STREAM)
    return cards_of(events)


class TestHotStreamFresh:
    """fresh=1 must always regenerate (cache-independent)."""

    def test_fresh_twice_in_quick_succession(self):
        ev1, first1, tot1 = consume_hot_stream(f"{HOT_STREAM}?fresh=1")
        c1 = cards_of(ev1)
        print(f"[fresh #1] cards={len(c1)} first_card={first1:.1f}s total={tot1:.1f}s")
        assert ev1 and ev1[0].get("phase") == "start"
        assert ev1[0].get("total") == 12
        assert len(c1) >= 8, f"only {len(c1)} cards on fresh call 1"
        assert any(e.get("phase") == "done" for e in ev1)

        # Immediately again — must NOT be served from cache
        ev2, first2, tot2 = consume_hot_stream(f"{HOT_STREAM}?fresh=1")
        c2 = cards_of(ev2)
        print(f"[fresh #2] cards={len(c2)} first_card={first2:.1f}s total={tot2:.1f}s")
        assert len(c2) >= 8, f"only {len(c2)} cards on fresh call 2"
        # A cached replay finishes in <1s; a real regeneration takes seconds.
        assert tot2 > 2.0, "fresh=1 appears to have been served from cache"

        # Card shape sanity
        for c in c1[:3]:
            assert c.get("headline") and isinstance(c["headline"], str)
            assert c.get("text") and isinstance(c["text"], str)
            assert "id" in c and "blueprint_index" in c
            assert "_id" not in c

        h1 = {c["headline"] for c in c1}
        h2 = {c["headline"] for c in c2}
        overlap = len(h1 & h2) / max(1, len(h1))
        print(f"[fresh] headline overlap between batches: {overlap:.0%}")

    def test_first_card_arrives_progressively(self):
        ev, first, tot = consume_hot_stream(f"{HOT_STREAM}?fresh=1")
        print(f"[progressive] first_card={first}s total={tot:.1f}s cards={len(cards_of(ev))}")
        assert first is not None, "no card event emitted"
        assert first < tot * 0.9 + 1, "cards not streamed progressively"
        assert first <= 25, f"first card took {first:.1f}s (>25s)"


class TestHotStreamCache:
    """No fresh param → cached replay is near-instant."""

    def test_cached_replay_is_fast(self, warm_cache):
        assert len(warm_cache) >= 8
        ev, first, tot = consume_hot_stream(HOT_STREAM)
        c = cards_of(ev)
        print(f"[cached] cards={len(c)} total={tot:.2f}s")
        assert len(c) == len(warm_cache)
        assert tot < 3.0, f"cached call took {tot:.2f}s"
        assert {x["headline"] for x in c} == {x["headline"] for x in warm_cache}
