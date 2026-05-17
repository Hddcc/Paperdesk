from pathlib import Path

import pytest

from app.services.milvus_bootstrap_service import MilvusBootstrapService


def test_bootstrap_ignores_non_local_uri(tmp_path):
    service = MilvusBootstrapService(
        uri="http://milvus.example:19530",
        auto_start=True,
        timeout_seconds=1,
        runtime_dir=tmp_path,
        container_name="paperdesk-milvus",
        image="milvusdb/milvus:v2.6.14",
        docker_desktop_executable=tmp_path / "Docker Desktop.exe",
    )
    service.ensure_running()


def test_bootstrap_starts_container_when_local_port_is_closed(monkeypatch, tmp_path):
    service = MilvusBootstrapService(
        uri="http://127.0.0.1:19530",
        auto_start=True,
        timeout_seconds=1,
        runtime_dir=tmp_path,
        container_name="paperdesk-milvus",
        image="milvusdb/milvus:v2.6.14",
        docker_desktop_executable=tmp_path / "Docker Desktop.exe",
    )

    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(service, "_is_port_open", lambda host, port: bool(calls and calls[-1][0] == "wait"))
    monkeypatch.setattr(service, "_ensure_docker_ready", lambda: calls.append(("docker_ready", None)))
    monkeypatch.setattr(service, "_container_exists", lambda: False)
    monkeypatch.setattr(service, "_write_runtime_files", lambda: calls.append(("write_runtime", None)))
    monkeypatch.setattr(service, "_run", lambda command, action: calls.append((action, command)))
    monkeypatch.setattr(service, "_wait_for_port", lambda host, port: calls.append(("wait", (host, port))))

    service.ensure_running()

    assert calls[0][0] == "docker_ready"
    assert calls[1][0] == "write_runtime"
    assert calls[2][0] == "create Milvus container"
    assert calls[3] == ("wait", ("127.0.0.1", 19530))


def test_bootstrap_recreates_failed_container(monkeypatch, tmp_path):
    service = MilvusBootstrapService(
        uri="http://127.0.0.1:19530",
        auto_start=True,
        timeout_seconds=1,
        runtime_dir=tmp_path,
        container_name="paperdesk-milvus",
        image="milvusdb/milvus:v2.6.14",
        docker_desktop_executable=tmp_path / "Docker Desktop.exe",
    )

    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(service, "_is_port_open", lambda host, port: False)
    monkeypatch.setattr(service, "_ensure_docker_ready", lambda: calls.append(("docker_ready", None)))
    monkeypatch.setattr(service, "_container_exists", lambda: True)
    monkeypatch.setattr(service, "_container_running", lambda: False)
    monkeypatch.setattr(service, "_write_runtime_files", lambda: calls.append(("write_runtime", None)))
    monkeypatch.setattr(service, "_run", lambda command, action: calls.append((action, command)))
    monkeypatch.setattr(service, "_wait_for_port", lambda host, port: calls.append(("wait", (host, port))))

    service.ensure_running()

    assert calls[0][0] == "docker_ready"
    assert calls[1][0] == "remove failed Milvus container"
    assert calls[2][0] == "write_runtime"
    assert calls[3][0] == "create Milvus container"
    assert calls[4] == ("wait", ("127.0.0.1", 19530))


def test_bootstrap_runtime_files_use_container_etcd_path(tmp_path):
    service = MilvusBootstrapService(
        uri="http://127.0.0.1:19530",
        auto_start=True,
        timeout_seconds=1,
        runtime_dir=tmp_path,
        container_name="paperdesk-milvus",
        image="milvusdb/milvus:v2.6.14",
        docker_desktop_executable=tmp_path / "Docker Desktop.exe",
    )

    service._write_runtime_files()

    assert "dir: /var/lib/milvus/etcd" in (tmp_path / "embedEtcd.yaml").read_text(encoding="utf-8")


def test_wait_for_port_reports_container_exit(monkeypatch, tmp_path):
    service = MilvusBootstrapService(
        uri="http://127.0.0.1:19530",
        auto_start=True,
        timeout_seconds=30,
        runtime_dir=tmp_path,
        container_name="paperdesk-milvus",
        image="milvusdb/milvus:v2.6.14",
        docker_desktop_executable=tmp_path / "Docker Desktop.exe",
    )

    monkeypatch.setattr(service, "_is_port_open", lambda host, port: False)
    monkeypatch.setattr(service, "_container_state", lambda: {"Running": False, "ExitCode": 134})
    monkeypatch.setattr(service, "_tail_logs", lambda: "panic: etcdserver: leader changed")

    with pytest.raises(RuntimeError) as exc_info:
        service._wait_for_port("127.0.0.1", 19530)

    message = str(exc_info.value)
    assert "exited with code 134" in message
    assert "panic: etcdserver: leader changed" in message
