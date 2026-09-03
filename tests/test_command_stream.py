"""Phase 4 backend tests — unified /api/insights/command/stream endpoint.

Tests classification and streaming for chat/transform/steer/rewrite modes,
plus regression on the other /stream endpoints.
"""
import json
import os
import time
import pytest
import requests

BASE_URL = os.environ["REACT_APP_BACKEND_URL"].rstrip("/") if os.environ.get("REACT_APP_BACKEND_URL") else "https://research-swipe-1.preview.emergentagent.com"


def parse_sse(resp, timeout_s=180):
    """Yield decoded SSE JSON events from a streaming response. Also yield
    ':keep-alive' comments as {'phase': 'keep-alive'} so tests can assert them."""
    start = time.time()
    for raw in resp.iter_lines(decode_unicode=True):
        if time.time() - start > timeout_s:
            raise TimeoutError("SSE exceeded timeout")
        if raw is None:
            continue
        line = raw.strip("\r")
        if not line:
            continue
        if line.startswith(":"):
            yield {"phase": "keep-alive", "raw": line}
            continue
        if line.startswith("data:"):
            payload = line[5:].strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                yield json.loads(payload)
            except Exception:
                continue


def _src(headline="Amazon started as an online bookstore in 1994", text="Jeff Bezos launched Amazon in 1994."):
    return {
        "id": "test-insight-1",
        "topic": "amazon history",
        "headline": headline,
        "text": text,
        "source_url": None,
        "source_title": None,
    }


def _payload(message, insights_count=3):
    src = _src()
    return {
        "session_id": "test-session-cmd",
        "topic": "amazon history",
        "current_insight": src,
        "insights": [_src(f"headline {i}", f"text {i} with year 2020") for i in range(insights_count)],
        "message": message,
        "channel": "text",
    }


def _post_stream(path, body):
    return requests.post(f"{BASE_URL}{path}", json=body, stream=True, timeout=200)


# ============ /command/stream classification tests ============

class TestCommandStream:
    def test_chat_mode(self):
        r = _post_stream("/api/insights/command/stream", _payload("What year was Amazon founded?"))
        assert r.status_code == 200
        events = list(parse_sse(r))
        modes = [e for e in events if e.get("phase") == "mode"]
        assert modes and modes[0]["mode"] == "chat", f"Expected chat mode, got {modes}"
        deltas = [e for e in events if e.get("phase") == "delta"]
        assert len(deltas) > 0, "Expected delta events for chat"
        done = [e for e in events if e.get("phase") == "done"]
        assert done, "Expected done event"

    def test_transform_mode(self):
        body = _payload("use bullet points on all cards", insights_count=3)
        r = _post_stream("/api/insights/command/stream", body)
        assert r.status_code == 200
        events = list(parse_sse(r))
        modes = [e for e in events if e.get("phase") == "mode"]
        assert modes and modes[0]["mode"] == "transform", f"Expected transform, got {modes}"
        done = [e for e in events if e.get("phase") == "done"]
        assert done, "Expected done event"
        data = done[-1].get("data") or {}
        insights = data.get("insights", [])
        assert len(insights) == 3, f"Transform must return same count. Got {len(insights)}"

    def test_steer_mode(self):
        r = _post_stream("/api/insights/command/stream", _payload("focus on Japanese examples"))
        assert r.status_code == 200
        events = list(parse_sse(r), )
        modes = [e for e in events if e.get("phase") == "mode"]
        assert modes and modes[0]["mode"] == "steer", f"Expected steer, got {modes}"
        done = [e for e in events if e.get("phase") == "done"]
        assert done, "Expected done event"
        assert (done[-1].get("data") or {}).get("insights"), "Steer must return insights"

    def test_rewrite_mode(self):
        r = _post_stream("/api/insights/command/stream", _payload("explain this insight in simpler words"))
        assert r.status_code == 200
        events = list(parse_sse(r))
        modes = [e for e in events if e.get("phase") == "mode"]
        assert modes and modes[0]["mode"] == "rewrite", f"Expected rewrite, got {modes}"
        deltas = [e for e in events if e.get("phase") == "delta"]
        assert len(deltas) > 0, "Rewrite must stream delta tokens"
        done = [e for e in events if e.get("phase") == "done"]
        assert done, "Expected done event"
        data = done[-1].get("data") or {}
        assert "headline" in data and "text" in data


# ============ Regression: other stream endpoints ============

class TestStreamRegression:
    def test_generate_stream_still_works(self):
        r = _post_stream("/api/insights/generate/stream", {"topic": "how did LEGO become successful"})
        assert r.status_code == 200
        events = list(parse_sse(r, ), )
        # This one takes 60-90s, so we just want at least phase events + done
        phases = {e.get("phase") for e in events}
        assert "start" in phases or "planning" in phases or "synthesizing" in phases, f"phases={phases}"
        done = [e for e in events if e.get("phase") == "done"]
        assert done, f"Expected done. Events phases: {phases}"
        assert (done[-1].get("data") or {}).get("insights")

    def test_deeper_stream(self):
        body = {
            "session_id": "test-sess",
            "topic": "amazon history",
            "source_insight": _src(),
            "previous_headlines": [],
        }
        r = _post_stream("/api/insights/deeper/stream", body)
        assert r.status_code == 200
        events = list(parse_sse(r))
        done = [e for e in events if e.get("phase") == "done"]
        assert done, "deeper: no done event"
        assert (done[-1].get("data") or {}).get("insights")

    def test_explain_stream(self):
        body = {
            "session_id": "test-sess",
            "topic": "amazon history",
            "source_insight": _src(),
            "phrase": "bookstore",
        }
        r = _post_stream("/api/insights/explain/stream", body)
        assert r.status_code == 200
        events = list(parse_sse(r))
        done = [e for e in events if e.get("phase") == "done"]
        assert done, "explain: no done event"
        assert (done[-1].get("data") or {}).get("insights")

    def test_answer_stream(self):
        body = {
            "session_id": "test-sess",
            "topic": "amazon history",
            "source_insight": _src(),
            "question": "Why did Bezos pick books first?",
            "previous_headlines": [],
        }
        r = _post_stream("/api/insights/answer/stream", body)
        assert r.status_code == 200
        events = list(parse_sse(r))
        done = [e for e in events if e.get("phase") == "done"]
        assert done, "answer: no done event"

    def test_more_stream(self):
        body = {
            "session_id": "test-sess",
            "topic": "how did LEGO become successful",
            "previous_headlines": [],
        }
        r = _post_stream("/api/insights/more/stream", body)
        assert r.status_code == 200
        events = list(parse_sse(r))
        done = [e for e in events if e.get("phase") == "done"]
        assert done, "more: no done event"
