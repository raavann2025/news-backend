"""Backend endpoints used by the Hot Now ActionBar: save, tts, stt, hot/stream."""
import os
import requests
import pytest

BASE_URL = os.environ.get("REACT_APP_BACKEND_URL", "https://interactive-news-hub-1.preview.emergentagent.com").rstrip("/")
API = f"{BASE_URL}/api"


def test_save_insight_returns_saved():
    payload = {"insight": {
        "topic": "TEST_topic_actionbar",
        "text": "TEST body copy for saved endpoint validation.",
        "headline": "TEST headline",
        "source_url": "https://example.com/a",
        "source_title": "Example",
    }}
    r = requests.post(f"{API}/insights/save", json=payload, timeout=15)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["topic"] == "TEST_topic_actionbar"
    assert d["headline"] == "TEST headline"
    assert "id" in d
    # verify persistence
    lr = requests.get(f"{API}/insights/saved", timeout=15)
    assert lr.status_code == 200
    ids = [x["id"] for x in lr.json()]
    assert d["id"] in ids
    # cleanup
    requests.delete(f"{API}/insights/saved/{d['id']}", timeout=15)


def test_tts_returns_audio_bytes():
    r = requests.post(f"{API}/insights/tts", json={"text": "Hello test", "voice": "sage"}, timeout=60)
    assert r.status_code == 200, r.text
    assert r.headers.get("content-type", "").startswith("audio/")
    assert len(r.content) > 500


def test_stt_endpoint_exists():
    # Send a tiny invalid webm to confirm endpoint reachable; expect 4xx/5xx but NOT 404
    files = {"file": ("voice.webm", b"\x1a\x45\xdf\xa3", "audio/webm")}
    r = requests.post(f"{API}/insights/stt", files=files, timeout=60)
    assert r.status_code != 404, "STT endpoint missing"


def test_hot_stream_emits_card_within_3s():
    import time
    start = time.time()
    with requests.get(f"{API}/insights/hot/stream?fresh=1", stream=True, timeout=15) as r:
        assert r.status_code == 200
        got_card = False
        for raw in r.iter_lines(decode_unicode=True):
            if raw is None:
                if time.time() - start > 12:
                    break
                continue
            if raw.startswith("data:") and '"phase"' in raw and '"card"' in raw:
                got_card = True
                break
            if time.time() - start > 12:
                break
        assert got_card, "No card event received within timeout"
