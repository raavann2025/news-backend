"""Backend smoke tests for the Deep Insight Swipe app.

Covers:
- GET /api/          -> service & status
- POST /api/interests/seed -> returns interests
- POST /api/insights/directions with topic 'AI startups' -> returns 4 directions
"""
import os
import requests

BASE_URL = os.environ.get("EXPO_PUBLIC_BACKEND_URL", "https://explore-topics.preview.emergentagent.com").rstrip("/")


def test_root_service_ok():
    r = requests.get(f"{BASE_URL}/api/", timeout=30)
    assert r.status_code == 200, r.text
    data = r.json()
    assert "service" in data or "status" in data, data


def test_interests_seed_returns_list():
    r = requests.post(f"{BASE_URL}/api/interests/seed", json={}, timeout=60)
    assert r.status_code == 200, r.text
    data = r.json()
    # accept either raw list or {interests: [...]}
    items = data if isinstance(data, list) else data.get("interests") or data.get("items")
    assert items and isinstance(items, list), data
    assert len(items) >= 1


def test_directions_ai_startups_returns_four():
    r = requests.post(
        f"{BASE_URL}/api/insights/directions",
        json={"topic": "AI startups"},
        timeout=120,
    )
    assert r.status_code == 200, r.text
    data = r.json()
    directions = data.get("directions") or data.get("items") or data
    assert isinstance(directions, list), f"not list: {data}"
    assert len(directions) == 4, f"expected 4 got {len(directions)}: {directions}"
