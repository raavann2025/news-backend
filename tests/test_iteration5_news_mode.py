"""
Iteration-5 backend tests — is_news flag end-to-end.

Scope:
- POST /api/insights/directions with is_news:true accepts the flag and returns 4 direction options.
- POST /api/insights/generate/stream with is_news:true completes end-to-end and produces >=6 insights.
- Baseline (is_news:false) still works for both.
"""

import os
import json
import time
import requests

BASE_URL = os.environ["REACT_APP_BACKEND_URL"].rstrip("/")
API = f"{BASE_URL}/api"


def _iter_sse(resp, deadline_s=190):
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
                    continue
                if line.startswith("data:"):
                    yield line[5:].strip()


class TestDirectionsIsNews:
    def test_directions_accept_is_news_true(self):
        payload = {
            "topic": "Recent AI Developments",
            "interests": ["Recent AI Developments"],
            "is_news": True,
        }
        r = requests.post(f"{API}/insights/directions", json=payload, timeout=90)
        assert r.status_code == 200, r.text
        d = r.json()
        assert d["topic"] == payload["topic"]
        dirs = d["directions"]
        assert isinstance(dirs, list) and len(dirs) == 4
        titles = [(opt.get("title") or "").strip() for opt in dirs]
        for t in titles:
            assert t and len(t) > 3
        # Print for manual audit of recency-flavored titles
        print("NEWS-MODE DIRECTIONS:", titles)

    def test_directions_is_news_false_still_works(self):
        payload = {
            "topic": "how coffee scaled globally",
            "interests": ["Coffee", "Business"],
            "is_news": False,
        }
        r = requests.post(f"{API}/insights/directions", json=payload, timeout=90)
        assert r.status_code == 200, r.text
        d = r.json()
        assert len(d["directions"]) == 4


class TestGenerateStreamIsNews:
    def test_stream_is_news_true_completes(self):
        payload = {
            "topic": "Recent AI Developments",
            "direction": "Safety debate after o1",
            "interests": ["Recent AI Developments"],
            "is_news": True,
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
            for data in _iter_sse(resp, deadline_s=190):
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
            first = insights[0]
            assert first.get("id")
            assert isinstance(first.get("text"), str) and len(first["text"]) > 40
            print("NEWS-MODE INSIGHT HEADLINES:", [i.get("headline") for i in insights[:6]])

    def test_stream_is_news_false_baseline(self):
        payload = {
            "topic": "Coffee history",
            "direction": None,
            "interests": [],
            "is_news": False,
        }
        with requests.post(
            f"{API}/insights/generate/stream",
            json=payload,
            stream=True,
            timeout=200,
        ) as resp:
            assert resp.status_code == 200
            done_data = None
            for data in _iter_sse(resp, deadline_s=190):
                try:
                    ev = json.loads(data)
                except Exception:
                    continue
                if ev.get("phase") == "done":
                    done_data = ev.get("data")
                    break
                if ev.get("phase") == "error":
                    raise AssertionError(ev.get("message"))
            assert done_data is not None
            assert len(done_data.get("insights") or []) >= 6
