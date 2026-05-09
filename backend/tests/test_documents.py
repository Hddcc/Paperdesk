from io import BytesIO


def test_document_upload_list_delete_flow(client):
    upload_response = client.post(
        "/api/documents/upload",
        files={"file": ("sample.pdf", BytesIO(b"%PDF-1.4 sample"), "application/pdf")},
    )
    assert upload_response.status_code == 200
    payload = upload_response.json()
    assert payload["display_name"] == "sample.pdf"

    list_response = client.get("/api/documents")
    assert list_response.status_code == 200
    documents = list_response.json()
    assert len(documents) == 1
    assert documents[0]["id"] == payload["id"]

    delete_response = client.delete(f"/api/documents/{payload['id']}")
    assert delete_response.status_code == 200

    list_again_response = client.get("/api/documents")
    assert list_again_response.status_code == 200
    assert list_again_response.json() == []

