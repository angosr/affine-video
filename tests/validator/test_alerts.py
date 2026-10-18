import json
import logging
import shutil
import ssl
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from validator import alerts


@pytest.mark.parametrize(
    "url",
    [
        "http://alerts.example/hook",
        "https://",
        "https:///missing-host",
        "https://user:password@alerts.example/hook",
        "https://alerts.example/hook#secret",
        "https://alerts.example:invalid",
        None,
    ],
)
def test_webhook_rejects_invalid_or_ambiguous_urls(url):
    with pytest.raises(ValueError, match="HTTPS"):
        alerts.webhook_alerter(url, service="veda-evaluator")


@pytest.mark.parametrize("failure", ["http_status", "transport", "unexpected"])
def test_webhook_failure_is_best_effort_without_logging_secrets(monkeypatch, caplog, failure):
    token = "private-webhook-token"
    url = f"https://alerts.example/{token}?key=another-secret"

    def post(*_args, **_kwargs):
        request = httpx.Request("POST", url)
        if failure == "http_status":
            return httpx.Response(503, request=request)
        if failure == "transport":
            raise httpx.ConnectError(f"offline: {url}", request=request)
        raise RuntimeError(f"internal failure at {url}")

    monkeypatch.setattr(alerts.httpx, "post", post)
    callback = alerts.webhook_alerter(url, service="veda-evaluator")
    with caplog.at_level(logging.ERROR, logger="validator.alerts"):
        callback("SCORING_RETRY_EXHAUSTED", {"challenge_id": "example", "attempts": 5})

    assert "push alert delivery failed" in caplog.text
    assert token not in caplog.text
    assert "another-secret" not in caplog.text
    assert caplog.records[-1].alert_event == "SCORING_RETRY_EXHAUSTED"
    assert caplog.records[-1].alert_service == "veda-evaluator"


def test_webhook_delivers_json_over_real_verified_https(tmp_path, monkeypatch, caplog):
    openssl = shutil.which("openssl")
    if openssl is None:
        pytest.skip("local HTTPS integration requires openssl")
    certificate = tmp_path / "certificate.pem"
    key = tmp_path / "key.pem"
    subprocess.run(
        [
            openssl,
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-days",
            "1",
            "-subj",
            "/CN=localhost",
            "-addext",
            "subjectAltName=IP:127.0.0.1",
            "-keyout",
            str(key),
            "-out",
            str(certificate),
        ],
        check=True,
        capture_output=True,
    )
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["content-length"]))
            received.append((self.path, self.headers["content-type"], json.loads(body)))
            self.send_response(204)
            self.end_headers()

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certificate, key)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    monkeypatch.setenv("SSL_CERT_FILE", str(certificate))
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    thread.start()
    try:
        callback = alerts.webhook_alerter(
            f"https://127.0.0.1:{server.server_port}/alerts", service="veda-evaluator"
        )
        callback("ADMISSION_RETRY_EXHAUSTED", {"challenge_id": "example", "attempts": 5})
        assert received == [
            (
                "/alerts",
                "application/json",
                {
                    "event": "ADMISSION_RETRY_EXHAUSTED",
                    "service": "veda-evaluator",
                    "details": {"challenge_id": "example", "attempts": 5},
                },
            )
        ]
        monkeypatch.delenv("SSL_CERT_FILE")
        with caplog.at_level(logging.ERROR, logger="validator.alerts"):
            callback("UNTRUSTED_CERTIFICATE_PROBE", {})
        assert len(received) == 1
        assert caplog.records[-1].alert_event == "UNTRUSTED_CERTIFICATE_PROBE"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()
