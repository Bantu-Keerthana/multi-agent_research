"""
Tests for the FastAPI endpoints.

Run: python -m pytest tests/test_api.py -v
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from api.server import app

client = TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_ok(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["version"] == "0.2.0"

    def test_health_includes_warnings(self):
        response = client.get("/health")
        data = response.json()
        assert "warnings" in data


class TestConfigEndpoint:
    def test_config_returns_tier_map(self):
        response = client.get("/config")
        assert response.status_code == 200
        data = response.json()
        assert "fast_model" in data
        assert "power_model" in data
        assert "tier_map" in data
        assert "planner" in data["tier_map"]


class TestResearchSync:
    def test_sync_returns_report(self):
        response = client.post(
            "/research/sync",
            json={"query": "What is Python?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "report" in data
        assert "Research Report" in data["report"]
        assert "tiering" in data

    def test_sync_includes_tiering(self):
        response = client.post(
            "/research/sync",
            json={"query": "Quick test"},
        )
        data = response.json()
        tiering = data["tiering"]
        assert "total_calls" in tiering
        assert "providers" in tiering


class TestResearchStream:
    def test_stream_returns_sse(self):
        response = client.post(
            "/research",
            json={"query": "What is AI?"},
        )
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        # Parse SSE events
        events = []
        for line in response.text.split("\n"):
            if line.startswith("event: "):
                events.append(line[7:])

        assert "pipeline_start" in events
        assert "done" in events


class TestHumanReviewAPI:
    def test_review_returns_session(self):
        response = client.post(
            "/research/review",
            json={"query": "Validate startup idea"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert data["status"] == "awaiting_review"
        assert "review_summary" in data

    def test_approve_with_valid_session(self):
        # Start a review
        response = client.post(
            "/research/review",
            json={"query": "Test review flow"},
        )
        session_id = response.json()["session_id"]

        # Approve it
        response = client.post(
            "/research/approve",
            json={"session_id": session_id, "approved": True},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "complete"
        assert "report" in data

    def test_reject_review(self):
        response = client.post(
            "/research/review",
            json={"query": "Test rejection"},
        )
        session_id = response.json()["session_id"]

        response = client.post(
            "/research/approve",
            json={"session_id": session_id, "approved": False},
        )
        assert response.status_code == 200
        assert response.json()["status"] == "rejected"

    def test_approve_invalid_session(self):
        response = client.post(
            "/research/approve",
            json={"session_id": "nonexistent", "approved": True},
        )
        assert response.status_code == 404


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
