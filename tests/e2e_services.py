"""Lifecycle helpers for database-backed end-to-end tests."""

import os
import signal
import socket
import subprocess
import time
import uuid
from pathlib import Path
from urllib.parse import ParseResult, unquote, urlparse

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEST_DATABASE_ADMIN_URL = (
    "postgres://country_guess:country_guess_dev@127.0.0.1:5432/postgres"
    "?sslmode=disable"
)


def wait_for_service(url, timeout=30):
    for _ in range(timeout):
        try:
            if requests.get(url, timeout=1).status_code == 200:
                return
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)
    raise RuntimeError(f"Service at {url} is unhealthy")


def _free_local_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _database_admin_url() -> str:
    return os.environ.get(
        "TEST_DATABASE_ADMIN_URL", DEFAULT_TEST_DATABASE_ADMIN_URL
    )


def _postgres_connection_args(
    database_url: str,
) -> tuple[ParseResult, list[str], dict[str, str], str]:
    parsed = urlparse(database_url)
    if parsed.scheme not in {"postgres", "postgresql"} or not parsed.username:
        raise RuntimeError(
            "TEST_DATABASE_ADMIN_URL must be a PostgreSQL URL with a username"
        )

    args = ["--username", unquote(parsed.username)]
    if parsed.hostname:
        args.extend(["--host", parsed.hostname])
    if parsed.port:
        args.extend(["--port", str(parsed.port)])

    environment = os.environ.copy()
    if parsed.password is not None:
        environment["PGPASSWORD"] = unquote(parsed.password)

    maintenance_database = parsed.path.lstrip("/") or "postgres"
    return parsed, args, environment, maintenance_database


def _create_test_database() -> tuple[str, str]:
    admin_url = _database_admin_url()
    parsed, connection_args, environment, maintenance_database = (
        _postgres_connection_args(admin_url)
    )
    database_name = f"country_guess_e2e_{os.getpid()}_{uuid.uuid4().hex[:12]}"

    subprocess.run(
        [
            "createdb",
            *connection_args,
            "--maintenance-db",
            maintenance_database,
            database_name,
        ],
        cwd=REPO_ROOT,
        env=environment,
        check=True,
    )

    database_url = parsed._replace(
        path=f"/{database_name}", params="", fragment=""
    ).geturl()
    return database_name, database_url


def _drop_test_database(database_name):
    if not database_name:
        return

    _, connection_args, environment, maintenance_database = (
        _postgres_connection_args(_database_admin_url())
    )
    subprocess.run(
        [
            "dropdb",
            *connection_args,
            "--maintenance-db",
            maintenance_database,
            "--if-exists",
            "--force",
            database_name,
        ],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
    )


def stop_process(process):
    if process is None or process.poll() is not None:
        return

    if os.name == "posix":
        os.killpg(process.pid, signal.SIGTERM)
    else:
        process.terminate()

    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait()


def start_drawingstore():
    database_name = None
    drawingstore_port = _free_local_port()
    drawingstore_process = None

    try:
        database_name, database_url = _create_test_database()

        drawingstore_env: dict[str, str] = os.environ.copy()
        drawingstore_env.update(
            {
                "DATABASE_URL": database_url,
                "HTTP_ADDR": f":{drawingstore_port}",
            }
        )
        drawingstore_process = subprocess.Popen(
            ["go", "-C", "drawingstore", "run", "."],
            cwd=REPO_ROOT,
            env=drawingstore_env,
            start_new_session=True,
        )
        wait_for_service(f"http://127.0.0.1:{drawingstore_port}/health")
    except Exception:
        stop_drawingstore(database_name, drawingstore_process)
        raise

    return (
        database_name,
        drawingstore_process,
        f"http://127.0.0.1:{drawingstore_port}",
    )


def stop_drawingstore(database_name, drawingstore_process):
    stop_process(drawingstore_process)

    _drop_test_database(database_name)
