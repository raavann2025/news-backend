"""Phase 3 SSE streaming tests: /generate/stream, /more/stream, error phase, legacy fallback, behavior."""
import os
import json
import time
import pytest
import requests

BASE_URL = os.environ.get("REACT_APP_BACKEND_URL", "https://research-swipe-1.preview.emergentagent.com").rstrip("/")
DEEP_TIMEOUT = 240


def parse_sse(resp, max_seconds=180):
    """Yield parsed JSON events from an SSE response. Skips keep-alive comments."""
    start = time.time()
    buffer = ""
    for chunk in resp.iter_content(chunk_size=None, decode_unicode=True):
        if chunk is None:
            continue
        buffer += chunk
        while "\n\n" in buffer:
            raw, buffer = buffer.split("\n\n", 1)
            for line in raw.split("\n"):
                s = line.strip()
                if not s or s.startswith(":"):
                    continue
                if s.startswith("data:"):
                    data = s[5:].strip()
                    if not data:
                        continue
                    try:
                        yield json.loads(data)
                    except json.JSONDecodeError:
                        pass
        if time.time() - start > max_seconds:
            raise TimeoutError(f"SSE exceeded {max_seconds}s")


# ---------- /generate/stream ----------

@pytest.fixture(scope="module")
def stream_result_heavy():
    """Run a heavy topic through /generate/stream and collect events."""
    r = requests.post(
        f"{BASE_URL}/api/insights/generate/stream",
        json={"topic": "the semiconductor industry evolution"},
        stream=True,
        timeout=200,
    )
    assert r.status_code == 200, f"HTTP {r.status_code}: {r.text[:200]}"
    assert "text/event-stream" in r.headers.get("content-type", ""), r.headers
    events = []
    t0 = time.time()
    for ev in parse_sse(r, max_seconds=180):
        events.append(ev)
        if ev.get("phase") in ("done", "error"):
            break
    elapsed = time.time() - t0
    return {"events": events, "elapsed": elapsed}


def test_stream_generate_no_502_and_ordered_phases(stream_result_heavy):
    events = stream_result_heavy["events"]
    phases = [e.get("phase") for e in events]
    # Must contain start and done, no error
    assert "done" in phases, f"No done phase. Phases: {phases}"
    assert "error" not in phases, f"Got error phase: {events}"
    # 'start' should be first (or very early)
    assert phases[0] == "start", f"First phase should be start, got {phases[0]}"
    # At least 3 distinct phase labels
    distinct = set(phases)
    assert len(distinct) >= 3, f"Expected >=3 distinct phases, got {distinct}"
    # Should complete within reasonable time
    assert stream_result_heavy["elapsed"] < 180, f"Took {stream_result_heavy['elapsed']}s"


def test_stream_generate_done_payload(stream_result_heavy):
    done = [e for e in stream_result_heavy["events"] if e.get("phase") == "done"]
    assert done, "No done event"
    data = done[-1].get("data")
    assert data, "done event missing data payload"
    assert "session_id" in data
    assert "insights" in data
    assert len(data["insights"]) == 12
    # Quantified sanity
    joined = " ".join((i.get("text") or "") + " " + (i.get("headline") or "") for i in data["insights"])
    assert any(k in joined.lower() for k in ["semiconductor", "chip", "tsmc", "asml", "nm", "nanometer", "wafer", "fab"])


def test_stream_generate_expected_phase_labels(stream_result_heavy):
    """Phases should include planning, searching, synthesizing (or similar)."""
    phases = {e.get("phase") for e in stream_result_heavy["events"]}
    expected_any = {"planning", "searching", "synthesizing", "angles_ready", "sources_ready", "extracting", "extracted"}
    overlap = phases & expected_any
    assert len(overlap) >= 3, f"Expected >=3 of {expected_any}, got {phases}"


# ---------- vague topic error via SSE ----------

def test_stream_generate_vague_topic_emits_error_event():
    r = requests.post(
        f"{BASE_URL}/api/insights/generate/stream",
        json={"topic": "x"},
        stream=True,
        timeout=30,
    )
    assert r.status_code == 200
    assert "text/event-stream" in r.headers.get("content-type", "")
    events = list(parse_sse(r, max_seconds=15))
    assert events, "No events received for vague topic"
    err_events = [e for e in events if e.get("phase") == "error"]
    assert err_events, f"Expected error phase, got {events}"
    msg = err_events[0].get("message", "").lower()
    assert "vague" in msg, f"Expected 'vague' in message, got {msg!r}"


# ---------- /more/stream ----------

def test_stream_more_returns_12_insights(stream_result_heavy):
    done = [e for e in stream_result_heavy["events"] if e.get("phase") == "done"][-1]
    prev = done["data"]
    sid = prev["session_id"]
    topic = prev["topic"]
    prev_headlines = [i["headline"] for i in prev["insights"] if i.get("headline")]

    r = requests.post(
        f"{BASE_URL}/api/insights/more/stream",
        json={
            "session_id": sid,
            "topic": topic,
            "previous_headlines": prev_headlines,
        },
        stream=True,
        timeout=200,
    )
    assert r.status_code == 200
    events = []
    for ev in parse_sse(r, max_seconds=180):
        events.append(ev)
        if ev.get("phase") in ("done", "error"):
            break
    done_ev = [e for e in events if e.get("phase") == "done"]
    assert done_ev, f"No done event. Phases: {[e.get('phase') for e in events]}"
    data = done_ev[-1]["data"]
    assert len(data["insights"]) == 12
    new_headlines = {i["headline"].strip().lower() for i in data["insights"] if i.get("headline")}
    old = {h.strip().lower() for h in prev_headlines}
    overlap = new_headlines & old
    assert len(overlap) <= 2, f"Too many repeats: {overlap}"


# ---------- Legacy JSON /generate still works ----------

def test_legacy_generate_still_returns_json_12():
    r = requests.post(
        f"{BASE_URL}/api/insights/generate",
        json={"topic": "how Amazon became successful"},
        timeout=DEEP_TIMEOUT,
    )
    # Legacy may still hit 60s cloudflare timeout; accept 200 OR 502 as environmental.
    if r.status_code == 502:
        pytest.skip("Cloudflare 502 on legacy endpoint (expected env limitation)")
    assert r.status_code == 200, r.text[:300]
    data = r.json()
    assert len(data["insights"]) == 12
    assert "session_id" in data


# ---------- Behavior tracking regression ----------

def test_behavior_events_all_accepted():
    sid = "test-sse-session"
    for ev, dwell in [("save", None), ("skip", 1500), ("dwell", 8000), ("ask", None), ("regenerate", None), ("share", None)]:
        r = requests.post(
            f"{BASE_URL}/api/insights/behavior",
            json={
                "session_id": sid,
                "insight_id": f"test-ins-{ev}",
                "topic": "test",
                "headline": f"headline for {ev}",
                "event": ev,
                "dwell_ms": dwell,
            },
            timeout=15,
        )
        assert r.status_code == 200, f"{ev} failed: {r.text}"
        assert r.json() == {"ok": True}
