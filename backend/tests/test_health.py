def test_healthz(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["vectorstore_status"] in {"unknown", "starting", "ready", "failed"}
    assert "vectorstore_uri" in payload
    assert "vectorstore_error" in payload
    assert payload["embedding_status"] in {"unknown", "starting", "ready", "failed", "disabled"}
    assert "embedding_model" in payload
    assert "embedding_error" in payload
