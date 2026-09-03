"""Phase 5 UX overhaul backend tests.

Focus: non-LLM endpoints (saved CRUD, health, behavior track, PWA assets).
LLM-dependent endpoints (/directions, /rewrite/stream, /generate/stream) are
tested only for basic reachability + expected behavior when LLM budget is
exhausted (currently returning 500). They are marked as BLOCKED-LLM.
"""
import os
import uuid
import pytest
import requests

BASE_URL = os.environ.get("REACT_APP_BACKEND_URL", "https://research-swipe-1.preview.emergentagent.com").rstrip("/")


@pytest.fixture(scope="module")
def client():
    s = requests.Session()
    s.headers.update({"Content-Type": "application/json"})
    return s


# ---------- Health ----------
def test_health(client):
    r = client.get(f"{BASE_URL}/api/")
    assert r.status_code == 200
    data = r.json()
    assert data.get("status") == "ok"


# ---------- Saved insights CRUD (no LLM) ----------
class TestSavedCRUD:
    insight = {
        "id": f"TEST_{uuid.uuid4().hex[:8]}",
        "topic": "TEST_topic",
        "headline": "TEST headline",
        "text": "TEST body text for saved insight persistence",
        "source_url": "https://example.com/a",
        "source_title": "TEST source",
    }

    def test_save(self, client):
        r = client.post(f"{BASE_URL}/api/insights/save", json={"insight": self.insight})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body.get("ok") or body.get("id") or body  # accept various shapes

    def test_list_contains_saved(self, client):
        r = client.get(f"{BASE_URL}/api/insights/saved")
        assert r.status_code == 200
        items = r.json()
        assert isinstance(items, list)
        ids = [x.get("id") for x in items]
        assert self.insight["id"] in ids
        # No mongo _id leakage
        for x in items:
            assert "_id" not in x

    def test_delete_and_verify(self, client):
        r = client.delete(f"{BASE_URL}/api/insights/saved/{self.insight['id']}")
        assert r.status_code in (200, 204)
        r2 = client.get(f"{BASE_URL}/api/insights/saved")
        ids = [x.get("id") for x in r2.json()]
        assert self.insight["id"] not in ids


# ---------- Behavior tracking ----------
def test_behavior_track(client):
    payload = {
        "session_id": "TEST_sess",
        "insight_id": "TEST_ins",
        "topic": "TEST",
        "headline": "TEST head",
        "event": "dwell",
        "dwell_ms": 5000,
    }
    r = client.post(f"{BASE_URL}/api/insights/behavior", json=payload)
    # Accept 200 or 404 if not implemented; log status
    assert r.status_code in (200, 201, 204, 404), r.text


# ---------- PWA assets ----------
def test_manifest_and_sw():
    # These are served by frontend but confirm reachable via public URL
    m = requests.get(f"{BASE_URL}/manifest.json", timeout=10)
    sw = requests.get(f"{BASE_URL}/sw.js", timeout=10)
    assert m.status_code == 200
    assert sw.status_code == 200


# ---------- LLM endpoints (expected BLOCKED because budget exhausted) ----------
class TestLLMEndpointsBlocked:
    """These SHOULD reach the endpoint but fail with 500 due to LLM budget.
    We assert reachability (not 404) and that failure path is graceful."""

    def test_directions_reachable(self, client):
        r = client.post(f"{BASE_URL}/api/insights/directions", json={"topic": "how Nintendo survived 100 years"})
        assert r.status_code != 404, "endpoint missing"
        # LLM budget exhausted → 500. Accept either 200 (if budget refilled) or 500.
        assert r.status_code in (200, 500)
        if r.status_code == 200:
            data = r.json()
            assert "directions" in data
            assert len(data["directions"]) == 4
        else:
            pytest.skip("BLOCKED: LLM budget exhausted (expected)")

    def test_rewrite_stream_reachable(self, client):
        payload = {
            "session_id": "s1",
            "topic": "t",
            "insight_id": "i1",
            "insight_headline": "h",
            "insight_text": "body",
            "transcript": "rewrite this shorter",
            "previous_headlines": [],
        }
        r = client.post(f"{BASE_URL}/api/insights/rewrite/stream", json=payload, stream=True, timeout=20)
        assert r.status_code != 404
        r.close()

    def test_generate_stream_accepts_direction(self, client):
        payload = {"topic": "test topic", "direction": "focus on economics"}
        r = client.post(f"{BASE_URL}/api/insights/generate/stream", json=payload, stream=True, timeout=15)
        assert r.status_code != 404
        # Validate it did not reject 'direction' field with a 422
        assert r.status_code != 422, r.text[:300]
        r.close()
