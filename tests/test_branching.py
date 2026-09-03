"""
Backend tests for Phase 7 branching endpoints:
- /api/insights/questions
- /api/insights/generate/stream (quality)
- /api/insights/deeper/stream
- /api/insights/explain/stream
- /api/insights/answer/stream
"""
import os
import re
import json
import time
import pytest
import requests

BASE_URL = os.environ.get("REACT_APP_BACKEND_URL", "https://research-swipe-1.preview.emergentagent.com").rstrip("/")

TIMEOUT_QUICK = 60
TIMEOUT_STREAM = 240

SEED_INSIGHT = {
    "id": "seed-1",
    "topic": "why do people bribe in India",
    "headline": "Petty bribes cost Indians ₹39,239 crore",
    "text": "A 2019 Transparency International India survey found 51% of Indians paid a bribe in the prior 12 months, and the total petty-bribe economy was estimated at ₹39,239 crore for that year. Karnataka and Andhra Pradesh scored worst on daily-service bribery.",
    "source_url": None,
    "source_title": None,
}


def _iter_sse(resp):
    """Yield parsed JSON events from an SSE HTTP response."""
    for raw in resp.iter_lines(decode_unicode=True):
        if not raw:
            continue
        if raw.startswith(":"):
            continue
        if raw.startswith("data:"):
            payload = raw[5:].strip()
            if payload == "[DONE]":
                return
            try:
                yield json.loads(payload)
            except Exception:
                continue


def _collect(resp, expected_final="done", timeout=TIMEOUT_STREAM):
    events = []
    start = time.time()
    for ev in _iter_sse(resp):
        events.append(ev)
        if ev.get("phase") in ("done", "error"):
            break
        if time.time() - start > timeout:
            break
    return events


# ---------- Quality checks helpers ----------
SPECIFIC_RE = re.compile(
    r"(\b\d{1,4}(?:[.,]\d+)?%?\b|\$\d|₹\d|€\d|£\d|"
    r"\b(19|20)\d{2}\b|"
    r"\b(million|billion|trillion|crore|lakh|percent|%)\b|"
    r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3}\b)"  # named entity
)

def has_specific(text: str) -> bool:
    return bool(SPECIFIC_RE.search(text or ""))


# ==================== Tests ====================

class TestQuestions:
    def test_questions_endpoint_returns_5(self):
        r = requests.post(
            f"{BASE_URL}/api/insights/questions",
            json={"topic": SEED_INSIGHT["topic"], "source_insight": SEED_INSIGHT},
            timeout=TIMEOUT_QUICK,
        )
        assert r.status_code == 200, r.text
        data = r.json()
        assert "questions" in data
        qs = data["questions"]
        assert isinstance(qs, list)
        assert len(qs) == 5, f"expected 5 questions, got {len(qs)}"
        for q in qs:
            words = q.split()
            assert 4 <= len(words) <= 20, f"question length off: '{q}' ({len(words)} words)"
            assert not q.lower().startswith(("is ", "are ", "does ", "do ", "can ", "will "))


class TestExplainStream:
    def test_explain_returns_one_insight(self):
        r = requests.post(
            f"{BASE_URL}/api/insights/explain/stream",
            json={
                "session_id": "test-explain-sess",
                "topic": SEED_INSIGHT["topic"],
                "source_insight": SEED_INSIGHT,
                "phrase": "Transparency International India",
            },
            stream=True,
            timeout=TIMEOUT_STREAM,
        )
        assert r.status_code == 200
        events = _collect(r)
        phases = [e.get("phase") for e in events]
        assert "start" in phases
        assert events[-1]["phase"] == "done", f"expected done, got {events[-1]}"
        insights = events[-1]["data"]["insights"]
        assert len(insights) == 1
        assert insights[0]["text"]
        assert has_specific(insights[0]["text"]), f"no specific in: {insights[0]['text']}"


class TestAnswerStream:
    def test_answer_returns_one_insight(self):
        r = requests.post(
            f"{BASE_URL}/api/insights/answer/stream",
            json={
                "session_id": "test-answer-sess",
                "topic": SEED_INSIGHT["topic"],
                "source_insight": SEED_INSIGHT,
                "question": "How does India's petty-bribe economy compare with other large democracies?",
                "previous_headlines": [],
            },
            stream=True,
            timeout=TIMEOUT_STREAM,
        )
        assert r.status_code == 200
        events = _collect(r)
        assert events[-1]["phase"] == "done", f"got: {events[-1]}"
        insights = events[-1]["data"]["insights"]
        assert len(insights) == 1
        assert insights[0]["text"]


class TestDeeperStream:
    def test_deeper_returns_12(self):
        r = requests.post(
            f"{BASE_URL}/api/insights/deeper/stream",
            json={
                "session_id": "test-deeper-sess",
                "topic": SEED_INSIGHT["topic"],
                "source_insight": SEED_INSIGHT,
                "previous_headlines": [SEED_INSIGHT["headline"]],
            },
            stream=True,
            timeout=TIMEOUT_STREAM,
        )
        assert r.status_code == 200
        events = _collect(r)
        assert events[-1]["phase"] == "done", f"got: {events[-1]}"
        insights = events[-1]["data"]["insights"]
        assert len(insights) >= 8, f"expected ~12 insights, got {len(insights)}"

        # None should be a near-duplicate of the parent headline
        parent_h = SEED_INSIGHT["headline"].lower()
        dups = [i for i in insights if i["headline"].lower() == parent_h]
        assert not dups, "deeper insights should not restate parent headline"

        # At least 70% should have a specific
        specific_count = sum(1 for i in insights if has_specific(i["text"]))
        assert specific_count >= len(insights) * 0.7, f"only {specific_count}/{len(insights)} have specifics"


class TestGenerateQuality:
    def test_generate_stream_quality(self):
        r = requests.post(
            f"{BASE_URL}/api/insights/generate/stream",
            json={"topic": "why do people bribe in India"},
            stream=True,
            timeout=TIMEOUT_STREAM,
        )
        assert r.status_code == 200
        events = _collect(r, timeout=TIMEOUT_STREAM)
        final = events[-1]
        assert final["phase"] == "done", f"got: {final}"
        insights = final["data"]["insights"]
        assert len(insights) == 12, f"expected 12, got {len(insights)}"

        # Every insight has a specific
        missing_specific = [i for i in insights if not has_specific(i["text"])]
        assert not missing_specific, (
            "insights missing specifics:\n" + "\n".join(f"- {i['headline']}: {i['text']}" for i in missing_specific)
        )

        # Angle diversity: headlines should be distinct
        headlines = [i["headline"].lower() for i in insights]
        assert len(set(headlines)) == len(headlines), "duplicate headlines"
