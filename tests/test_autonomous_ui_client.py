import io
import json

from src.ui.api_client import ask_autonomous_question


class Response:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def read(self):
        return json.dumps(
            {"status": "completed", "answer": "P&L complete", "evidence": []}
        ).encode()


def test_autonomous_client_uses_new_endpoint(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data.decode())
        return Response()

    monkeypatch.setattr("src.ui.api_client.urlopen", fake_urlopen)
    result = ask_autonomous_question("Show P&L for August 2026")
    assert captured["url"].endswith("/api/v1/autonomous/ask")
    assert captured["payload"]["question"] == "Show P&L for August 2026"
    assert result["status"] == "completed"
