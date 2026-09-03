"""Phase 2 backend tests: TTS, STT, chat SSE stream, regenerate, more insights."""
import os
import json
import time
import pytest
import requests

BASE_URL = os.environ.get("REACT_APP_BACKEND_URL", "https://research-swipe-1.preview.emergentagent.com").rstrip("/")

TTS_PATH = "/tmp/tts.mp3"


# --- Module scope: generate one seed feed to reuse for /more test ---
@pytest.fixture(scope="module")
def seed_feed():
    r = requests.post(f"{BASE_URL}/api/insights/generate", json={"topic": "Nintendo business strategy"}, timeout=120)
    assert r.status_code == 200, r.text
    data = r.json()
    assert len(data["insights"]) == 12
    return data


# --- TTS ---
def test_tts_returns_audio_bytes():
    r = requests.post(
        f"{BASE_URL}/api/insights/tts",
        json={"text": "Nintendo was founded in 1889 as a playing card company.", "voice": "sage"},
        timeout=60,
    )
    assert r.status_code == 200, r.text
    assert r.headers.get("content-type", "").startswith("audio/mpeg")
    assert len(r.content) > 2000  # a few kb minimum
    with open(TTS_PATH, "wb") as f:
        f.write(r.content)


# --- STT ---
def test_stt_transcribes_uploaded_audio():
    assert os.path.exists(TTS_PATH), "TTS output missing; run TTS test first"
    with open(TTS_PATH, "rb") as f:
        r = requests.post(
            f"{BASE_URL}/api/insights/stt",
            files={"file": ("tts.mp3", f, "audio/mpeg")},
            timeout=90,
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "text" in body
    assert isinstance(body["text"], str)
    # Should contain some transcript content (allow lowercase match on a keyword)
    assert len(body["text"]) > 0


# --- Chat SSE ---
def test_chat_sse_streams_deltas(seed_feed):
    insight = seed_feed["insights"][0]
    payload = {
        "session_id": seed_feed["session_id"],
        "insight_id": insight["id"],
        "insight_text": insight["text"],
        "insight_headline": insight.get("headline", ""),
        "insight_topic": seed_feed["topic"],
        "source_url": insight.get("source_url"),
        "source_title": insight.get("source_title"),
        "history": [],
        "message": "Give me one concrete example that supports this insight.",
    }
    with requests.post(
        f"{BASE_URL}/api/insights/chat", json=payload, stream=True, timeout=90
    ) as r:
        assert r.status_code == 200, r.text
        assert "text/event-stream" in r.headers.get("content-type", "")
        deltas = 0
        done = False
        started = time.time()
        for line in r.iter_lines(decode_unicode=True):
            if line is None:
                continue
            if not line:
                continue
            if line.startswith("data: "):
                payload_str = line[6:]
                if payload_str.strip() == "[DONE]":
                    done = True
                    break
                try:
                    obj = json.loads(payload_str)
                except Exception:
                    continue
                if "delta" in obj and obj["delta"]:
                    deltas += 1
                if deltas >= 3 and time.time() - started > 2:
                    # got enough evidence of streaming
                    pass
            if time.time() - started > 60:
                break
        assert deltas >= 2, f"Expected multiple deltas, got {deltas}"


# --- Regenerate ---
def test_regenerate_returns_fresh_insight(seed_feed):
    insight = seed_feed["insights"][0]
    payload = {
        "topic": seed_feed["topic"],
        "insight_id": insight["id"],
        "previous_text": insight["text"],
        "exclude": [i["text"] for i in seed_feed["insights"][:3]],
    }
    r = requests.post(f"{BASE_URL}/api/insights/regenerate", json=payload, timeout=90)
    assert r.status_code == 200, r.text
    new_ins = r.json()
    assert new_ins["id"] == insight["id"], "ID must be preserved"
    assert new_ins["text"] and new_ins["text"] != insight["text"], "Text should be different"
    assert new_ins.get("headline")


# --- More (infinite) ---
def test_more_insights_returns_12_fresh(seed_feed):
    prev_headlines = [i["headline"] for i in seed_feed["insights"]]
    payload = {
        "session_id": seed_feed["session_id"],
        "topic": seed_feed["topic"],
        "previous_headlines": prev_headlines,
    }
    r = requests.post(f"{BASE_URL}/api/insights/more", json=payload, timeout=120)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["session_id"] == seed_feed["session_id"]
    assert len(data["insights"]) == 12
    new_headlines = {i["headline"].lower() for i in data["insights"]}
    old_headlines = {h.lower() for h in prev_headlines}
    overlap = new_headlines & old_headlines
    # Allow at most 1 accidental repeat, but should mostly be distinct
    assert len(overlap) <= 1, f"Too many repeat headlines: {overlap}"


def test_more_insights_with_steer(seed_feed):
    payload = {
        "session_id": seed_feed["session_id"],
        "topic": seed_feed["topic"],
        "previous_headlines": [i["headline"] for i in seed_feed["insights"]],
        "steer": "focus on Japan and cultural context",
    }
    r = requests.post(f"{BASE_URL}/api/insights/more", json=payload, timeout=120)
    assert r.status_code == 200, r.text
    data = r.json()
    assert len(data["insights"]) == 12
    # At least some insights should reference Japan/culture given steering
    combined = " ".join(i["text"].lower() + " " + i["headline"].lower() for i in data["insights"])
    assert "japan" in combined or "japanese" in combined, "Steer 'Japan' should influence output"
