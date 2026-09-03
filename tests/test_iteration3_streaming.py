"""
Iteration-3 backend tests — layered interests + streaming chips + News mode.

Scope (per review request):
- POST /api/interests/expand/stream (SSE) — start / chip / done events for both modes.
  Second call must return from_cache=True and still emit all chip events.
- GET  /api/interests/news/seed — >= 8 items, sanity content.
- POST /api/insights/directions — accepts {topic, interests: [...]}, returns 4 directions.
- POST /api/insights/generate/stream — accepts {topic, direction, interests}, streams to done.
"""

import os
import json
import time
import uuid
import requests
import pytest

BASE_URL = os.environ["REACT_APP_BACKEND_URL"].rstrip("/")
API = f"{BASE_URL}/api"


# ---------- helpers ----------

def _iter_sse(resp, deadline_s=170):
    """Yield decoded 'data:' payload strings from an SSE response."""
    buf = ""
    start = time.time()
    for chunk in resp.iter_content(chunk_size=1024, decode_unicode=True):
        if time.time() - start > deadline_s:
            return
        if not chunk:
            continue
        buf += chunk
        while "\n\n" in buf:
            block, buf = buf.split("\n\n", 1)
            for line in block.splitlines():
                line = line.strip()
                if line.startswith(":"):
                    continue  # keep-alive
                if line.startswith("data:"):
                    yield line[5:].strip()


def _consume_expand_stream(interest, parent_chain, mode, timeout=90):
    """POST /api/interests/expand/stream and return list of parsed events."""
    payload = {"interest": interest, "parent_chain": parent_chain, "mode": mode}
    events = []
    with requests.post(
        f"{API}/interests/expand/stream",
        json=payload,
        stream=True,
        timeout=timeout,
    ) as resp:
        assert resp.status_code == 200, resp.text
        assert "text/event-stream" in resp.headers.get("content-type", "")
        for data in _iter_sse(resp, deadline_s=timeout):
            try:
                ev = json.loads(data)
            except Exception:
                continue
            events.append(ev)
            if ev.get("phase") in ("done", "error"):
                break
    return events


# ---------- News seed ----------

class TestNewsSeed:
    def test_returns_at_least_8_items(self):
        r = requests.get(f"{API}/interests/news/seed", timeout=15)
        assert r.status_code == 200
        d = r.json()
        assert isinstance(d.get("interests"), list)
        assert len(d["interests"]) >= 8, f"expected >= 8, got {len(d['interests'])}"
        # Sanity — Recent AI Developments must be present (used in the test-id spec)
        assert "Recent AI Developments" in d["interests"]


# ---------- Streaming chip expansion ----------

class TestExpandStream:
    def test_stream_interests_mode_emits_chips_then_done(self):
        interest = "AI"
        # Force a fresh cache key with a UUID parent
        parent_chain = [f"test-{uuid.uuid4().hex[:8]}"]

        evs = _consume_expand_stream(interest, parent_chain, mode="interests")
        phases = [e.get("phase") for e in evs]

        assert "start" in phases, f"no start event: {phases}"
        assert "done" in phases, f"no done event: {phases}"
        chip_events = [e for e in evs if e.get("phase") == "chip"]
        assert len(chip_events) >= 6, f"expected >=6 chip events, got {len(chip_events)}: {phases}"
        # Each chip must have a non-empty label string
        for c in chip_events:
            assert isinstance(c.get("label"), str) and c["label"].strip()

        done = next(e for e in evs if e["phase"] == "done")
        subs = done.get("sub_interests") or []
        assert len(subs) >= 6 and len(subs) <= 8
        # first call should not come from cache
        assert done.get("from_cache") is False

        # Second call — must still stream all chips AND be from_cache=True
        evs2 = _consume_expand_stream(interest, parent_chain, mode="interests")
        chip_events2 = [e for e in evs2 if e.get("phase") == "chip"]
        assert len(chip_events2) == len(subs), (
            f"cached call should replay all {len(subs)} chips, got {len(chip_events2)}"
        )
        done2 = next(e for e in evs2 if e["phase"] == "done")
        assert done2.get("from_cache") is True
        assert done2.get("sub_interests") == subs

    def test_stream_news_mode_returns_specific_storylines(self):
        interest = "Recent AI Developments"
        parent_chain = [f"test-{uuid.uuid4().hex[:8]}"]

        evs = _consume_expand_stream(interest, parent_chain, mode="news", timeout=90)
        chip_events = [e for e in evs if e.get("phase") == "chip"]
        # news mode chip labels can be 2-6 words phrases
        assert len(chip_events) >= 6, f"news mode: expected >=6 chips got {len(chip_events)}"
        for c in chip_events:
            lbl = c.get("label")
            assert isinstance(lbl, str) and 2 <= len(lbl) <= 80

        done = next(e for e in evs if e["phase"] == "done")
        assert done.get("from_cache") is False
        assert len(done.get("sub_interests") or []) >= 6


# ---------- Directions with interests ----------

class TestDirectionsWithInterests:
    def test_directions_accept_interests_and_return_options(self):
        payload = {
            "topic": "how Amazon became successful",
            "interests": ["AI", "Business", "Economics"],
        }
        r = requests.post(f"{API}/insights/directions", json=payload, timeout=90)
        assert r.status_code == 200, r.text
        d = r.json()
        assert d["topic"] == payload["topic"]
        dirs = d["directions"]
        assert isinstance(dirs, list) and len(dirs) == 4
        for opt in dirs:
            assert opt.get("title")
            assert opt.get("subtitle") is not None  # may be empty but present


# ---------- Generate stream with direction + interests ----------

class TestGenerateStreamWithInterests:
    def test_stream_with_direction_and_interests_completes(self):
        payload = {
            "topic": "how Amazon became successful",
            "direction": "The unsung ops decisions behind Prime",
            "interests": ["AI", "Operations", "Logistics"],
        }
        with requests.post(
            f"{API}/insights/generate/stream",
            json=payload,
            stream=True,
            timeout=200,
        ) as resp:
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers.get("content-type", "")
            done_data = None
            err = None
            deadline = time.time() + 190
            for data in _iter_sse(resp, deadline_s=190):
                if time.time() > deadline:
                    break
                try:
                    ev = json.loads(data)
                except Exception:
                    continue
                p = ev.get("phase")
                if p == "done":
                    done_data = ev.get("data")
                    break
                if p == "error":
                    err = ev.get("message")
                    break

            assert err is None, f"stream errored: {err}"
            assert done_data is not None, "no done event"
            assert done_data.get("session_id")
            insights = done_data.get("insights") or []
            assert len(insights) >= 6, f"expected >=6 insights, got {len(insights)}"
            # Real content assertions
            first = insights[0]
            assert first.get("id")
            assert isinstance(first.get("text"), str) and len(first["text"]) > 40
