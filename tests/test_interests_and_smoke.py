"""
Backend tests for iteration 2 — merged codebase.

Covers:
- NEW interest onboarding endpoints (/api/interests/seed, /expand, /me GET/POST/DELETE)
- EXISTING /api/insights/directions
- EXISTING /api/insights/generate/stream (SSE — asserts session_id + insights)
- EXISTING /api/insights/tts returns audio/mpeg
- Root /api/ health
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

def _sse_lines(resp):
    """Yield decoded, trimmed 'data:' payload strings from an SSE response."""
    buf = ""
    for chunk in resp.iter_content(chunk_size=1024, decode_unicode=True):
        if not chunk:
            continue
        buf += chunk
        while "\n\n" in buf:
            block, buf = buf.split("\n\n", 1)
            for line in block.splitlines():
                line = line.strip()
                if line.startswith("data:"):
                    yield line[5:].strip()


# ---------- Health ----------

class TestHealth:
    def test_root(self):
        r = requests.get(f"{API}/", timeout=15)
        assert r.status_code == 200
        d = r.json()
        assert d.get("status") == "ok"


# ---------- Interests: seed ----------

class TestInterestsSeed:
    def test_seed_returns_28_interests(self):
        r = requests.get(f"{API}/interests/seed", timeout=15)
        assert r.status_code == 200
        d = r.json()
        assert "interests" in d
        assert isinstance(d["interests"], list)
        assert len(d["interests"]) == 28, f"expected 28 seed interests got {len(d['interests'])}"
        # Sanity: a well-known chip
        assert "AI" in d["interests"]
        assert "Celebrity Gossip" in d["interests"]


# ---------- Interests: expand (Gemini 3 Flash) ----------

class TestInterestsExpand:
    def test_expand_returns_8_subs_and_second_call_cached(self):
        # Use a unique-ish topic to try to force a fresh LLM call the first time
        interest = f"AI"  # keep it a known seed, but the cache is keyed off the whole chain
        parent_chain = [f"test-{uuid.uuid4().hex[:8]}"]

        r1 = requests.post(
            f"{API}/interests/expand",
            json={"interest": interest, "parent_chain": parent_chain},
            timeout=60,
        )
        assert r1.status_code == 200, r1.text
        d1 = r1.json()
        assert d1["interest"] == interest
        subs = d1["sub_interests"]
        assert isinstance(subs, list) and len(subs) == 8, f"expected 8 subs got {len(subs)}"
        for s in subs:
            assert isinstance(s, str) and 1 <= len(s) <= 60
        assert d1["from_cache"] is False

        # Second call with same key should hit cache
        r2 = requests.post(
            f"{API}/interests/expand",
            json={"interest": interest, "parent_chain": parent_chain},
            timeout=30,
        )
        assert r2.status_code == 200
        d2 = r2.json()
        assert d2["from_cache"] is True
        assert d2["sub_interests"] == subs


# ---------- Interests: me CRUD ----------

class TestMyInterestsCRUD:
    def setup_method(self):
        self.device_id = f"TEST_{uuid.uuid4().hex}"

    def teardown_method(self):
        try:
            requests.delete(f"{API}/interests/me", params={"device_id": self.device_id}, timeout=15)
        except Exception:
            pass

    def test_get_empty_initially(self):
        r = requests.get(f"{API}/interests/me", params={"device_id": self.device_id}, timeout=15)
        assert r.status_code == 200
        d = r.json()
        assert d["device_id"] == self.device_id
        assert d["interests"] == []

    def test_save_and_read_back(self):
        payload = {"device_id": self.device_id, "interests": ["AI", "History", "AI", " philosophy "]}
        r = requests.post(f"{API}/interests/me", json=payload, timeout=15)
        assert r.status_code == 200
        d = r.json()
        # dedupe + strip on server side
        assert d["device_id"] == self.device_id
        assert len(d["interests"]) == 3  # AI, History, philosophy (deduped, trimmed)
        assert "updated_at" in d

        r2 = requests.get(f"{API}/interests/me", params={"device_id": self.device_id}, timeout=15)
        assert r2.status_code == 200
        d2 = r2.json()
        assert d2["interests"] == d["interests"]

    def test_delete_clears(self):
        requests.post(
            f"{API}/interests/me",
            json={"device_id": self.device_id, "interests": ["AI"]},
            timeout=15,
        )
        r = requests.delete(f"{API}/interests/me", params={"device_id": self.device_id}, timeout=15)
        assert r.status_code == 200
        assert r.json().get("ok") is True

        r2 = requests.get(f"{API}/interests/me", params={"device_id": self.device_id}, timeout=15)
        assert r2.status_code == 200
        assert r2.json()["interests"] == []


# ---------- Directions ----------

class TestDirections:
    def test_directions_for_valid_topic(self):
        r = requests.post(
            f"{API}/insights/directions",
            json={"topic": "how Amazon became successful"},
            timeout=90,
        )
        assert r.status_code == 200, r.text
        d = r.json()
        assert d["topic"] == "how Amazon became successful"
        dirs = d["directions"]
        assert isinstance(dirs, list) and len(dirs) == 4
        for opt in dirs:
            assert opt.get("title") and isinstance(opt["title"], str)

    def test_directions_vague_topic_422(self):
        r = requests.post(f"{API}/insights/directions", json={"topic": "idk"}, timeout=30)
        assert r.status_code == 422


# ---------- Generate stream (SSE) ----------

class TestGenerateStream:
    def test_generate_stream_returns_session_and_insights(self):
        # Use a short-ish topic that shouldn't be too slow
        with requests.post(
            f"{API}/insights/generate/stream",
            json={"topic": "history of the espresso machine"},
            stream=True,
            timeout=180,
        ) as resp:
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers.get("content-type", "")
            done_payload = None
            error_msg = None
            deadline = time.time() + 170
            for data in _sse_lines(resp):
                if time.time() > deadline:
                    break
                try:
                    ev = json.loads(data)
                except Exception:
                    continue
                phase = ev.get("phase")
                if phase == "done":
                    done_payload = ev.get("data")
                    break
                if phase == "error":
                    error_msg = ev.get("message")
                    break

            assert error_msg is None, f"stream errored: {error_msg}"
            assert done_payload is not None, "did not receive done event"
            assert "session_id" in done_payload
            insights = done_payload.get("insights", [])
            assert isinstance(insights, list)
            assert len(insights) >= 6, f"expected at least 6 insights got {len(insights)}"
            for ins in insights[:3]:
                assert ins.get("text")
                assert ins.get("id")


# ---------- TTS ----------

class TestTTS:
    def test_tts_returns_audio_mpeg(self):
        r = requests.post(
            f"{API}/insights/tts",
            json={"text": "Hello world."},
            timeout=45,
        )
        assert r.status_code == 200, r.text
        assert r.headers.get("content-type", "").startswith("audio/mpeg")
        assert len(r.content) > 1000, f"audio bytes too small: {len(r.content)}"

    def test_tts_empty_text_400(self):
        r = requests.post(f"{API}/insights/tts", json={"text": ""}, timeout=15)
        assert r.status_code == 400


# ---------- STT ----------

class TestSTT:
    def test_stt_rejects_bad_audio(self):
        # empty file — provider should error. Backend returns 500 on failure per current code.
        files = {"file": ("empty.webm", b"", "audio/webm")}
        r = requests.post(f"{API}/insights/stt", files=files, timeout=30)
        # Accept 4xx or 5xx — feature just needs to not hang, real STT tested via UI.
        assert r.status_code in (400, 422, 500, 502)
