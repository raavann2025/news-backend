"""Tests for Back Story SSE endpoints introduced in iteration 2."""
import os
import json
import re
import pytest
import requests

BASE_URL = os.environ.get("REACT_APP_BACKEND_URL", "https://interactive-news-hub-1.preview.emergentagent.com").rstrip("/")


def _parse_sse(raw_text):
    events = []
    for line in raw_text.splitlines():
        if line.startswith("data: "):
            try:
                events.append(json.loads(line[6:]))
            except Exception:
                pass
    return events


class TestBackstoryStream:
    def test_backstory_stream_basic(self):
        payload = {
            "topic": "India IPO rush",
            "headline": "12 firms lining up",
            "text": "Zepto and others are eyeing Rs 25,000 cr IPO haul.",
            "source_title": "Moneycontrol",
            "source_url": "https://example.com/a",
        }
        r = requests.post(f"{BASE_URL}/api/insights/hot/backstory/stream", json=payload, stream=True, timeout=90)
        assert r.status_code == 200, r.text
        body = r.text
        events = _parse_sse(body)
        phases = [e.get("phase") for e in events]
        assert "start" in phases, f"missing start; got {phases[:5]}"
        assert "done" in phases or "error" in phases, f"missing done/error; got {phases[-5:]}"
        chunks = [e.get("text", "") for e in events if e.get("phase") == "chunk"]
        assert len(chunks) > 0, "no chunk events received"
        full = "".join(chunks)
        assert len(full) > 300, f"too short: {len(full)} chars"
        # look for at least two of the section markers
        markers = ["What actually happened", "Who is involved", "Why it matters"]
        found = sum(1 for m in markers if m in full)
        assert found >= 2, f"section markers missing; found {found} in text: {full[:400]}"

    def test_backstory_chat_stream_basic(self):
        payload = {
            "topic": "India IPO rush",
            "headline": "12 firms lining up",
            "text": "Zepto and others are eyeing Rs 25,000 cr IPO haul.",
            "question": "what happens next",
        }
        r = requests.post(f"{BASE_URL}/api/insights/hot/backstory/chat/stream", json=payload, stream=True, timeout=90)
        assert r.status_code == 200
        events = _parse_sse(r.text)
        phases = [e.get("phase") for e in events]
        assert "start" in phases
        assert "done" in phases or "error" in phases
        chunks = [e.get("text", "") for e in events if e.get("phase") == "chunk"]
        assert len(chunks) > 0
        full = "".join(chunks)
        assert len(full) > 80, f"reply too short: {full}"

    def test_backstory_stream_validation(self):
        # Missing required fields
        r = requests.post(f"{BASE_URL}/api/insights/hot/backstory/stream", json={"topic": "x"}, timeout=15)
        assert r.status_code in (400, 422)
