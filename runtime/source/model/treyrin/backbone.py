"""Local, shared GGUF backbone. Original weights and native chat template stay intact."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import secrets
import socket
import subprocess
import time
import urllib.error
import urllib.request


ENGINE = Path(r"D:\MAT\Marigold\Marigold_1\Main\llama\llama-server.exe")
MANIFEST = Path.home() / ".ollama/models/manifests/registry.ollama.ai/library/qwen3/4b"
ROOT = Path(__file__).resolve().parents[1]


class NativeBackbone:
    def __init__(self, url: str, key: str, model_digest: str):
        self.url, self.key, self.model_digest = url, key, model_digest
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(self, path: str, payload: dict | None = None, timeout: int = 120):
        request = urllib.request.Request(
            self.url + path,
            data=None if payload is None else json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + self.key},
        )
        with self.opener.open(request, timeout=timeout) as response:
            return json.load(response)

    def chat(self, messages: list[dict], **parameters) -> dict:
        """No decision prompt, hidden message, or altered template is inserted."""
        return self.request("/v1/chat/completions", {
            "model": "local", "messages": messages, "stream": False, **parameters,
        })

    def encode(self, texts: list[str]) -> list[list[float]]:
        response = self.request("/v1/embeddings", {"model": "local", "input": texts})
        ordered = sorted(response["data"], key=lambda row: row["index"])
        if len(ordered) != len(texts):
            raise ValueError("Backbone returned the wrong number of representations")
        return [row["embedding"] for row in ordered]


def installed_weights() -> tuple[Path, str]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    layer = next(x for x in manifest["layers"] if x["mediaType"] == "application/vnd.ollama.image.model")
    path = Path.home() / ".ollama/models/blobs" / layer["digest"].replace(":", "-")
    if path.stat().st_size != layer["size"]:
        raise ValueError("Local weight size changed")
    return path, layer["digest"]


@contextmanager
def local_backbone(log_dir: Path, embeddings: bool = True):
    """Own one temporary process, bind loopback, and always terminate it on exit."""
    log_dir.mkdir(parents=True, exist_ok=True)
    weights, digest = installed_weights()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    key = secrets.token_hex(24)
    command = [str(ENGINE), "-m", str(weights), "--host", "127.0.0.1", "--port", str(port),
               "-c", "8192", "-ngl", "99", "-np", "1", "--jinja", "--no-ui", "--api-key", key]
    if embeddings:
        command += ["--embeddings", "--pooling", "last"]
    native = NativeBackbone(f"http://127.0.0.1:{port}", key, digest)
    log_path = log_dir / f"engine-{time.time_ns()}.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ENGINE.parent, stdout=log, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"Local backbone exited: {log_path}")
                try:
                    if native.request("/health", timeout=2).get("status") == "ok":
                        break
                except (OSError, urllib.error.URLError):
                    pass
                time.sleep(0.25)
            else:
                raise TimeoutError("Backbone startup exceeded 120 seconds")
            yield native
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)


def verify_original_digest() -> bool:
    path, expected = installed_weights()
    with path.open("rb") as handle:
        actual = "sha256:" + hashlib.file_digest(handle, "sha256").hexdigest()
    return actual == expected
