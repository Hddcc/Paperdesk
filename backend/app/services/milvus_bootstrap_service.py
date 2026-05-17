"""Helpers for starting a local Milvus container during backend startup."""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse


class MilvusBootstrapService:
    """Ensure a local Docker-backed Milvus instance is available when needed."""

    def __init__(
        self,
        *,
        uri: str,
        auto_start: bool,
        timeout_seconds: int,
        runtime_dir: Path,
        container_name: str,
        image: str,
        docker_desktop_executable: Path,
    ) -> None:
        self.uri = uri
        self.auto_start = auto_start
        self.timeout_seconds = timeout_seconds
        self.runtime_dir = runtime_dir
        self.container_name = container_name
        self.image = image
        self.docker_desktop_executable = docker_desktop_executable

    def ensure_running(self) -> None:
        endpoint = self._parse_local_http_endpoint()
        if not self.auto_start or endpoint is None:
            return

        host, port = endpoint
        if self._is_port_open(host, port):
            return

        self._ensure_docker_ready()
        if self._container_exists():
            if not self._container_running():
                self._run(
                    ["docker", "rm", "-f", self.container_name],
                    "remove failed Milvus container",
                )
                self._write_runtime_files()
                self._run(self._build_run_command(), "create Milvus container")
        else:
            self._write_runtime_files()
            self._run(self._build_run_command(), "create Milvus container")

        self._wait_for_port(host, port)

    def _parse_local_http_endpoint(self) -> tuple[str, int] | None:
        parsed = urlparse(self.uri)
        if parsed.scheme not in {"http", "https"}:
            return None
        host = parsed.hostname or ""
        if host not in {"127.0.0.1", "localhost", "0.0.0.0", "::1"}:
            return None
        return host, parsed.port or 19530

    @staticmethod
    def _is_port_open(host: str, port: int) -> bool:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            return False

    def _ensure_docker_ready(self) -> None:
        deadline = time.time() + self.timeout_seconds
        desktop_started = False
        last_error = "Docker is not ready."
        while time.time() < deadline:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
            if result.returncode == 0:
                return
            last_error = (result.stderr or result.stdout or last_error).strip()
            if sys.platform == "win32" and not desktop_started and self.docker_desktop_executable.is_file():
                subprocess.Popen(
                    [str(self.docker_desktop_executable)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                desktop_started = True
            time.sleep(2)
        raise RuntimeError(f"Docker Desktop 未就绪，无法自动启动 Milvus。{last_error}")

    def _container_exists(self) -> bool:
        result = subprocess.run(
            ["docker", "inspect", self.container_name],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        return result.returncode == 0

    def _container_running(self) -> bool:
        result = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", self.container_name],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        return result.returncode == 0 and result.stdout.strip().lower() == "true"

    def _write_runtime_files(self) -> None:
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        embed_config = self.runtime_dir / "embedEtcd.yaml"
        user_config = self.runtime_dir / "user.yaml"
        embed_config.write_text(
            "etcd:\n"
            "  use:\n"
            "    embed: true\n"
            "  data:\n"
            "    dir: /var/lib/milvus/etcd\n",
            encoding="utf-8",
        )
        user_config.write_text("common:\n  security:\n    authorizationEnabled: false\n", encoding="utf-8")
        (self.runtime_dir / "volumes" / "milvus").mkdir(parents=True, exist_ok=True)
        (self.runtime_dir / "volumes" / "etcd").mkdir(parents=True, exist_ok=True)

    def _build_run_command(self) -> list[str]:
        runtime_dir = self.runtime_dir.resolve()
        milvus_data = runtime_dir / "volumes" / "milvus"
        embed_config = runtime_dir / "embedEtcd.yaml"
        user_config = runtime_dir / "user.yaml"
        return [
            "docker",
            "run",
            "-d",
            "--name",
            self.container_name,
            "--security-opt",
            "seccomp:unconfined",
            "-e",
            "ETCD_USE_EMBED=true",
            "-e",
            "ETCD_DATA_DIR=/var/lib/milvus/etcd",
            "-e",
            "ETCD_CONFIG_PATH=/milvus/configs/embedEtcd.yaml",
            "-e",
            "COMMON_STORAGETYPE=local",
            "-e",
            "DEPLOY_MODE=STANDALONE",
            "-v",
            f"{milvus_data}:/var/lib/milvus",
            "-v",
            f"{embed_config}:/milvus/configs/embedEtcd.yaml",
            "-v",
            f"{user_config}:/milvus/configs/user.yaml",
            "-p",
            "19530:19530",
            "-p",
            "9091:9091",
            "-p",
            "2379:2379",
            "--health-cmd",
            "curl -f http://localhost:9091/healthz",
            "--health-interval",
            "30s",
            "--health-start-period",
            "90s",
            "--health-timeout",
            "20s",
            "--health-retries",
            "3",
            self.image,
            "milvus",
            "run",
            "standalone",
        ]

    def _wait_for_port(self, host: str, port: int) -> None:
        deadline = time.time() + self.timeout_seconds
        while time.time() < deadline:
            if self._is_port_open(host, port):
                return
            state = self._container_state()
            if state and not state.get("Running", False):
                exit_code = state.get("ExitCode")
                logs = self._tail_logs()
                detail = f"Milvus container exited with code {exit_code} before {host}:{port} became ready."
                if logs:
                    detail = f"{detail}\nLast Milvus logs:\n{logs}"
                raise RuntimeError(detail)
            time.sleep(2)
        raise RuntimeError(
            f"Milvus 容器已启动，但 {host}:{port} 在 {self.timeout_seconds} 秒内仍未就绪。"
        )

    def _container_state(self) -> dict | None:
        result = subprocess.run(
            ["docker", "inspect", self.container_name, "--format", "{{json .State}}"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError:
            return None

    def _tail_logs(self) -> str:
        result = subprocess.run(
            ["docker", "logs", "--tail", "80", self.container_name],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        output = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
        return self._compact_text(output, limit=3000)

    @staticmethod
    def _compact_text(text: str, *, limit: int) -> str:
        cleaned = "\n".join(line.rstrip() for line in text.splitlines() if line.strip())
        if len(cleaned) <= limit:
            return cleaned
        return cleaned[-limit:]

    @staticmethod
    def _run(command: list[str], action: str) -> None:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if result.returncode != 0:
            output = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(f"Failed to {action}: {output}")

    @staticmethod
    def _to_posix(path: Path) -> str:
        return path.resolve().as_posix()
