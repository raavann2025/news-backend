"""Iteration 7 backend regression suite for Insight Swipe (post LLM key top-up).

Covers: health, interests (seed/expand/me), insights hot (+SSE), directions,
generate (+SSE), chat SSE, tts, save/saved/delete, more.
All requests go through the public preview URL (REACT_APP_BACKEND_URL).
"""
import json
import os
import uuid

import pytest
import requests
from dotenv import dotenv_values

frontend_env = dotenv_values("/app/frontend/.env")
base_url = os.environ.get("REACT_APP_BACKEND_URL") or frontend_env.get("REACT_APP_BACKEND_URL")
if not base_url:
    raise RuntimeError("REACT_APP_BACKEND_URL missing")
BASE_URL = base_url.rstrip("/")
API = f"{BASE_URL}/api"

DEVICE_ID = "qa-user-1"
LONG = 180
XLONG = 300


@pytest.fixture(scope="session")
def client():
    s = requests.Session()
    s.headers.update({"Content-Type": "application/json"})
    return s


# ---------------- health / interests ----------------
class TestHealthAndInterests:
    def test_root(self, client):
        r = client.get(f"{API}/", timeout=30)
        assert r.status_code == 200
        assert r.json().get("status") == "ok"

    def test_interest_seed(self, client):
        r = client.get(f"{API}/interests/seed", timeout=30)
        assert r.status_code == 200
        data = r.json()
        items = data.get("interests", data)
        assert isinstance(items, list) and len(items) > 0

    def test_news_seed(self, client):
        r = client.get(f"{API}/interests/news/seed", timeout=30)
        assert r.status_code == 200

    def test_expand_ai(self, client):
        r = client.post(f"{API}/interests/expand", json={"interest": "AI"}, timeout=LONG)
        assert r.status_code == 200, r.text[:400]
        data = r.json()
        assert data["interest"] == "AI"
        subs = data["sub_interests"]
        assert isinstance(subs, list) and len(subs) >= 4
        assert all(isinstance(s, str) and s.strip() for s in subs)

    def test_interests_me_roundtrip(self, client):
        r = client.post(
            f"{API}/interests/me",
            json={"device_id": DEVICE_ID, "interests": ["AI", "Space"]},
            timeout=60,
        )
        assert r.status_code == 200, r.text[:300]
        g = client.get(f"{API}/interests/me", params={"device_id": DEVICE_ID}, timeout=30)
        assert g.status_code == 200
        assert "AI" in g.json()["interests"]


# ---------------- hot feed ----------------
class TestHotFeed:
    def test_hot_cards(self, client):
        r = client.get(f"{API}/insights/hot", timeout=XLONG)
        assert r.status_code == 200, r.text[:400]
        cards = r.json()["cards"]
        assert len(cards) >= 8, f"expected >=8 hot cards, got {len(cards)}"
        c = cards[0]
        for f in ("id", "topic", "headline", "text", "category"):
            assert c.get(f), f"missing {f} in hot card"
        assert len(c["text"]) > 60

    def test_hot_stream(self, client):
        events = []
        with client.get(f"{API}/insights/hot/stream", stream=True, timeout=XLONG) as r:
            assert r.status_code == 200
            for line in r.iter_lines(decode_unicode=True):
                if line and line.startswith("data:"):
                    try:
                        payload = json.loads(line[5:].strip())
                    except Exception:
                        continue
                    if payload.get("type") == "card" or "card" in payload:
                        events.append(payload)
                    if payload.get("type") == "done" or len(events) >= 8:
                        break
        assert len(events) >= 8, f"only {len(events)} card events from hot/stream"


# ---------------- directions ----------------
class TestDirections:
    def test_directions(self, client):
        r = client.post(
            f"{API}/insights/directions",
            json={"topic": "NEET exam leak", "device_id": DEVICE_ID},
            timeout=XLONG,
        )
        assert r.status_code == 200, r.text[:400]
        data = r.json()
        assert data["topic"]
        dirs = data["directions"]
        assert 3 <= len(dirs) <= 10, f"got {len(dirs)} directions"
        assert all(d["title"].strip() and d["subtitle"].strip() for d in dirs)

    def test_directions_vague_topic_422(self, client):
        r = client.post(f"{API}/insights/directions", json={"topic": "stuff"}, timeout=60)
        assert r.status_code == 422


# ---------------- generate / more / save ----------------
@pytest.fixture(scope="class")
def generated(client):
    r = client.post(
        f"{API}/insights/generate",
        json={"topic": "how Nvidia became the AI chip leader", "interests": ["AI"], "device_id": DEVICE_ID},
        timeout=XLONG,
    )
    return r


class TestGenerateFlow:
    def test_generate(self, generated):
        assert generated.status_code == 200, generated.text[:400]
        data = generated.json()
        assert data["session_id"]
        ins = data["insights"]
        assert len(ins) >= 8, f"expected ~12 insights, got {len(ins)}"
        assert all(i["text"] and i["headline"] for i in ins)

    def test_generate_stream(self, client):
        """Backend emits phase events (searching/reading/synthesizing) then one
        terminal `done` event carrying the full 12-card payload."""
        phases = []
        done_payload = None
        err = None
        with client.post(
            f"{API}/insights/generate/stream",
            json={"topic": "the semiconductor industry evolution"},
            stream=True,
            timeout=XLONG,
        ) as r:
            assert r.status_code == 200
            for line in r.iter_lines(decode_unicode=True):
                if line and line.startswith("data:"):
                    try:
                        p = json.loads(line[5:].strip())
                    except Exception:
                        continue
                    phases.append(p.get("phase"))
                    if p.get("phase") == "error":
                        err = p.get("message")
                        break
                    if p.get("phase") == "done":
                        done_payload = p.get("data")
                        break
        assert err is None, f"generate/stream error event: {err}"
        assert done_payload, f"no done event; phases seen: {phases}"
        assert len(done_payload["insights"]) >= 8, len(done_payload["insights"])
        assert len(phases) >= 2, f"no progress phases emitted: {phases}"

    def test_save_list_delete(self, client, generated):
        if generated.status_code != 200:
            pytest.fail("generate failed; cannot test save flow")
        card = dict(generated.json()["insights"][0])
        card["id"] = f"TEST_{uuid.uuid4()}"
        card["topic"] = "TEST_" + card["topic"]
        s = client.post(f"{API}/insights/save", json={"insight": card}, timeout=60)
        assert s.status_code == 200, s.text[:300]
        assert s.json()["id"] == card["id"]

        g = client.get(f"{API}/insights/saved", params={"device_id": DEVICE_ID}, timeout=60)
        assert g.status_code == 200
        ids = [x["id"] for x in g.json()]
        assert card["id"] in ids, "saved card not returned by /insights/saved"

        d = client.delete(f"{API}/insights/saved/{card['id']}", timeout=60)
        assert d.status_code == 200
        g2 = client.get(f"{API}/insights/saved", timeout=60)
        assert card["id"] not in [x["id"] for x in g2.json()]

    def test_more(self, client, generated):
        if generated.status_code != 200:
            pytest.fail("generate failed; cannot test /more")
        data = generated.json()
        r = client.post(
            f"{API}/insights/more",
            json={
                "session_id": data["session_id"],
                "topic": data["topic"],
                "previous_headlines": [i["headline"] for i in data["insights"][:5]],
            },
            timeout=XLONG,
        )
        assert r.status_code == 200, r.text[:400]
        assert len(r.json()["insights"]) >= 5

    def test_chat_stream(self, client, generated):
        if generated.status_code != 200:
            pytest.fail("generate failed; cannot test chat")
        card = generated.json()["insights"][0]
        chunks = []
        with client.post(
            f"{API}/insights/chat",
            json={
                "session_id": str(uuid.uuid4()),
                "insight_id": card["id"],
                "insight_text": card["text"],
                "insight_headline": card["headline"],
                "insight_topic": card["topic"],
                "message": "Why does this matter?",
            },
            stream=True,
            timeout=XLONG,
        ) as r:
            assert r.status_code == 200, r.text[:300]
            for line in r.iter_lines(decode_unicode=True):
                if line and line.startswith("data:"):
                    chunks.append(line[5:])
                    if len("".join(chunks)) > 80:
                        break
        assert len("".join(chunks)) > 20, "chat stream returned no content"


# ---------------- tts ----------------
class TestTTS:
    def test_tts(self, client):
        r = client.post(
            f"{API}/insights/tts", json={"text": "Insight swipe test audio."}, timeout=XLONG
        )
        assert r.status_code == 200, r.text[:300]
        assert r.headers.get("content-type", "").startswith("audio/")
        assert len(r.content) > 1000

    def test_tts_empty_400(self, client):
        r = client.post(f"{API}/insights/tts", json={"text": "   "}, timeout=60)
        assert r.status_code == 400
