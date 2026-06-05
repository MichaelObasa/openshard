"""Tests for the pre-send secret guard at the provider boundary.

Covers ``openshard.providers.base.guard_prompt_before_send`` and its
integration into provider send points, ``ExecutionResult`` propagation, and
safe serialization for the receipt.

All secrets here are clearly fake, test-only values — never real credentials.
"""
from __future__ import annotations

import unittest
from dataclasses import asdict
from types import SimpleNamespace
from unittest import mock

from openshard.execution.generator import ExecutionGenerator, ExecutionResult
from openshard.providers.anthropic import AnthropicProvider
from openshard.providers.base import (
    BaseProvider,
    ChatResponse,
    ModelInfo,
    PreSendSecretScanError,
    UsageStats,
    guard_prompt_before_send,
)
from openshard.providers.openai import OpenAIProvider
from openshard.providers.openrouter import OpenRouterClient
from openshard.security.secret_scan import SecretScanResult

# Clearly fake, test-only values (not live credentials).
FAKE_AWS_KEY = "AKIA1234567890ABCDEF"  # AKIA + 16 chars — matches aws pattern
FAKE_PROMPT_WITH_SECRET = f"deploy with AWS_ACCESS_KEY_ID={FAKE_AWS_KEY} now"
CLEAN_PROMPT = "refactor the parser and add a unit test for empty input"


# ---------------------------------------------------------------------------
# guard_prompt_before_send — unit
# ---------------------------------------------------------------------------

class TestGuardPromptBeforeSend(unittest.TestCase):

    def test_clean_prompt_is_byte_identical_and_no_result(self):
        scrubbed, result = guard_prompt_before_send(CLEAN_PROMPT)
        self.assertEqual(scrubbed, CLEAN_PROMPT)
        self.assertIsNone(result)

    def test_secret_is_redacted_and_result_returned(self):
        scrubbed, result = guard_prompt_before_send(FAKE_PROMPT_WITH_SECRET)
        self.assertIsInstance(result, SecretScanResult)
        self.assertEqual(len(result.findings), 1)
        self.assertEqual(result.findings[0].kind, "aws_access_key_id")
        # Raw secret never survives in the scrubbed prompt.
        self.assertNotIn(FAKE_AWS_KEY, scrubbed)

    def test_oversize_fails_closed(self):
        with self.assertRaises(PreSendSecretScanError):
            guard_prompt_before_send(FAKE_PROMPT_WITH_SECRET, max_chars=10)

    def test_oversize_exception_has_no_prompt_content(self):
        try:
            guard_prompt_before_send(FAKE_PROMPT_WITH_SECRET, max_chars=10)
            self.fail("expected PreSendSecretScanError")
        except PreSendSecretScanError as exc:
            self.assertNotIn(FAKE_AWS_KEY, str(exc))
            self.assertNotIn(FAKE_AWS_KEY, repr(exc))

    def test_scan_failure_fails_closed(self):
        with mock.patch(
            "openshard.providers.base.scrub_text_for_secrets",
            side_effect=RuntimeError("boom"),
        ):
            with self.assertRaises(PreSendSecretScanError):
                guard_prompt_before_send(FAKE_PROMPT_WITH_SECRET)


# ---------------------------------------------------------------------------
# Provider boundary — real OpenRouter client with a mocked transport
# ---------------------------------------------------------------------------

_CANNED_RESPONSE = {
    "model": "z-ai/glm-5.1",
    "choices": [{"message": {"content": "ok"}}],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}


class TestProviderBoundary(unittest.TestCase):

    def _client(self) -> OpenRouterClient:
        # Constructing the client only builds an httpx.Client — no network.
        return OpenRouterClient(api_key="test-only-not-a-real-key")

    def test_secret_redacted_before_transport(self):
        client = self._client()
        with mock.patch.object(
            client, "_post", return_value=_CANNED_RESPONSE
        ) as posted:
            resp = client.execute("z-ai/glm-5.1", FAKE_PROMPT_WITH_SECRET)
        posted.assert_called_once()
        sent_payload = posted.call_args.args[1]
        sent_blob = repr(sent_payload)
        # The raw secret never reaches the outgoing payload; a redaction does.
        self.assertNotIn(FAKE_AWS_KEY, sent_blob)
        self.assertIn("AKIA...", sent_blob)
        # The scan result rides back on the response.
        self.assertIsInstance(resp.presend_secret_scan, SecretScanResult)
        self.assertEqual(len(resp.presend_secret_scan.findings), 1)

    def test_clean_prompt_sent_unchanged(self):
        client = self._client()
        with mock.patch.object(
            client, "_post", return_value=_CANNED_RESPONSE
        ) as posted:
            resp = client.execute("z-ai/glm-5.1", CLEAN_PROMPT)
        sent_payload = posted.call_args.args[1]
        sent_user_msg = sent_payload["messages"][-1]["content"]
        self.assertEqual(sent_user_msg, CLEAN_PROMPT)
        self.assertIsNone(resp.presend_secret_scan)

    def test_fail_closed_means_no_request_sent(self):
        client = self._client()
        with mock.patch.object(client, "_post") as posted:
            with mock.patch(
                "openshard.providers.base.scrub_text_for_secrets",
                side_effect=RuntimeError("boom"),
            ):
                with self.assertRaises(PreSendSecretScanError):
                    client.execute("z-ai/glm-5.1", FAKE_PROMPT_WITH_SECRET)
            posted.assert_not_called()

    def test_oversize_means_no_request_sent(self):
        client = self._client()
        omitted = ("[omitted]", SecretScanResult(scanned_files_count=0, omitted=True))
        with mock.patch.object(client, "_post") as posted:
            with mock.patch(
                "openshard.providers.base.scrub_text_for_secrets",
                return_value=omitted,
            ):
                with self.assertRaises(PreSendSecretScanError):
                    client.execute("z-ai/glm-5.1", FAKE_PROMPT_WITH_SECRET)
            posted.assert_not_called()

    def test_no_raw_secret_in_serialized_result(self):
        client = self._client()
        with mock.patch.object(client, "_post", return_value=_CANNED_RESPONSE):
            resp = client.execute("z-ai/glm-5.1", FAKE_PROMPT_WITH_SECRET)
        blob = repr(asdict(resp.presend_secret_scan))
        self.assertNotIn(FAKE_AWS_KEY, blob)
        # A stable fingerprint is recorded for accountability.
        self.assertTrue(resp.presend_secret_scan.findings[0].fingerprint)


# ---------------------------------------------------------------------------
# ExecutionResult propagation
# ---------------------------------------------------------------------------

class _FakeClient(BaseProvider):
    """Minimal provider that returns a canned, parseable response."""

    def __init__(self, scan: SecretScanResult | None) -> None:
        self._scan = scan
        self.seen_prompt: str | None = None

    def list_models(self) -> list[ModelInfo]:  # pragma: no cover - unused
        return []

    def get_model_info(self, model_id: str) -> ModelInfo | None:  # pragma: no cover
        return None

    def execute(self, model, prompt, system=None, max_tokens=None) -> ChatResponse:
        self.seen_prompt = prompt
        return ChatResponse(
            content='{"summary": "ok", "files": [], "notes": []}',
            model=model,
            usage=UsageStats(1, 1, 2),
            presend_secret_scan=self._scan,
        )


class TestExecutionResultPropagation(unittest.TestCase):

    def test_generate_copies_presend_scan(self):
        scan = SecretScanResult(scanned_files_count=0, summary="x")
        gen = ExecutionGenerator(provider=_FakeClient(scan))
        result = gen.generate("do a thing")
        self.assertIs(result.presend_secret_scan, scan)

    def test_generate_copies_none_when_clean(self):
        gen = ExecutionGenerator(provider=_FakeClient(None))
        result = gen.generate("do a thing")
        self.assertIsNone(result.presend_secret_scan)


# ---------------------------------------------------------------------------
# Receipt surfacing — the merged scan result serializes without raw secrets
# ---------------------------------------------------------------------------

class TestReceiptSurfacing(unittest.TestCase):

    def test_presend_findings_serialize_safely(self):
        # Mirrors what the pipeline records as ``secret_scan_result``.
        _, result = guard_prompt_before_send(FAKE_PROMPT_WITH_SECRET)
        merged = SecretScanResult(
            scanned_files_count=0,
            findings=list(result.findings),
            blocked=False,
            summary="1 potential secret detected (pre-send guard ...)",
        )
        blob = repr(asdict(merged))
        self.assertNotIn(FAKE_AWS_KEY, blob)
        self.assertIn("aws_access_key_id", blob)

    def test_execution_result_presend_serializes_safely(self):
        _, result = guard_prompt_before_send(FAKE_PROMPT_WITH_SECRET)
        exec_result = ExecutionResult(
            summary="ok", files=[], notes=[], presend_secret_scan=result
        )
        self.assertNotIn(FAKE_AWS_KEY, repr(asdict(exec_result)))


# ---------------------------------------------------------------------------
# Provider boundary — Anthropic (SDK client mocked, no network)
# ---------------------------------------------------------------------------

def _fake_anthropic_response():
    return SimpleNamespace(
        content=[SimpleNamespace(text="ok")],
        model="claude-sonnet-4-6",
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
    )


class TestAnthropicBoundary(unittest.TestCase):

    def _provider(self):
        # Bypass __init__ so no SDK install / API key / network is required;
        # inject a mock client to exercise the real execute() path.
        provider = AnthropicProvider.__new__(AnthropicProvider)
        provider._client = mock.MagicMock()
        return provider

    def test_secret_redacted_before_transport(self):
        provider = self._provider()
        provider._client.messages.create.return_value = _fake_anthropic_response()
        resp = provider.execute("anthropic/claude-sonnet-4.6", FAKE_PROMPT_WITH_SECRET)
        provider._client.messages.create.assert_called_once()
        sent = repr(provider._client.messages.create.call_args.kwargs["messages"])
        self.assertNotIn(FAKE_AWS_KEY, sent)
        self.assertIn("AKIA...", sent)
        self.assertIsInstance(resp.presend_secret_scan, SecretScanResult)
        self.assertEqual(len(resp.presend_secret_scan.findings), 1)

    def test_fail_closed_means_no_request_sent(self):
        provider = self._provider()
        with mock.patch(
            "openshard.providers.base.scrub_text_for_secrets",
            side_effect=RuntimeError("boom"),
        ):
            with self.assertRaises(PreSendSecretScanError):
                provider.execute("anthropic/claude-sonnet-4.6", FAKE_PROMPT_WITH_SECRET)
        provider._client.messages.create.assert_not_called()


# ---------------------------------------------------------------------------
# Provider boundary — OpenAI (SDK client mocked, no network)
# ---------------------------------------------------------------------------

def _fake_openai_response():
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
        model="gpt-4o-mini",
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )


class TestOpenAIBoundary(unittest.TestCase):

    def _provider(self):
        provider = OpenAIProvider.__new__(OpenAIProvider)
        provider._client = mock.MagicMock()
        return provider

    def test_secret_redacted_before_transport(self):
        provider = self._provider()
        provider._client.chat.completions.create.return_value = _fake_openai_response()
        resp = provider.execute("openai/gpt-4o-mini", FAKE_PROMPT_WITH_SECRET)
        provider._client.chat.completions.create.assert_called_once()
        sent = repr(
            provider._client.chat.completions.create.call_args.kwargs["messages"]
        )
        self.assertNotIn(FAKE_AWS_KEY, sent)
        self.assertIn("AKIA...", sent)
        self.assertIsInstance(resp.presend_secret_scan, SecretScanResult)
        self.assertEqual(len(resp.presend_secret_scan.findings), 1)

    def test_fail_closed_means_no_request_sent(self):
        provider = self._provider()
        with mock.patch(
            "openshard.providers.base.scrub_text_for_secrets",
            side_effect=RuntimeError("boom"),
        ):
            with self.assertRaises(PreSendSecretScanError):
                provider.execute("openai/gpt-4o-mini", FAKE_PROMPT_WITH_SECRET)
        provider._client.chat.completions.create.assert_not_called()


if __name__ == "__main__":
    unittest.main()
