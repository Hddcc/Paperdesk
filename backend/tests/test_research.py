from io import BytesIO


def test_research_stream_and_report_persistence(client):
    client.post(
        "/api/documents/upload",
        files={"file": ("library.pdf", BytesIO(b"%PDF-1.4 library"), "application/pdf")},
    )

    response = client.post(
        "/api/research/stream",
        json={"topic": "RAG 系统中的评估方法"},
        headers={"Accept": "text/event-stream"},
    )
    assert response.status_code == 200
    body = response.text
    assert '"type": "run_created"' in body
    assert '"type": "todo_list"' in body
    assert '"type": "final_report"' in body
    assert '"type": "done"' in body

    reports_response = client.get("/api/reports")
    assert reports_response.status_code == 200
    reports = reports_response.json()
    assert len(reports) == 1

    report_response = client.get(f"/api/reports/{reports[0]['id']}")
    assert report_response.status_code == 200
    report = report_response.json()
    assert report["topic"] == "RAG 系统中的评估方法"
    assert "PaperDesk 00/01 可运行骨架" in report["markdown"]
    assert "导出路径：" in report["markdown"]
