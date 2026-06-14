from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from openshard.native.agent_loop_receipts import ReceiptEmitter, _serialise
from openshard.native.agent_loop_types import (
    AgentAction,
    AgentDecision,
    AgentLoopEvent,
    AgentObservation,
    ApprovalReceipt,
    ApprovalRequest,
    IterationCost,
    ReceiptIteration,
    ToolResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_receipt(idx: int = 1) -> ReceiptIteration:
    action = AgentAction(
        action_id=f"a{idx}",
        iteration_index=idx,
        kind="read_file",
        args={"path": "foo.py"},
        rationale="test",
        needs_approval=False,
    )
    return ReceiptIteration(
        iteration_index=idx,
        timestamp="2026-01-01T00:00:00+00:00",
        action=action,
        result=ToolResult(action_id=f"a{idx}", ok=True, output="content"),
        observation=AgentObservation(
            action_id=f"a{idx}",
            iteration_index=idx,
            status="success",
            summary="read ok",
        ),
        decision=AgentDecision(
            iteration_index=idx,
            decision="continue",
            reason="more to do",
        ),
        cost=IterationCost(input_tokens=10, output_tokens=5, estimated_usd=0.001),
    )


def _full_receipt() -> ReceiptIteration:
    """Receipt with approval fields populated."""
    action = AgentAction(
        action_id="aw",
        iteration_index=2,
        kind="write_file",
        args={"path": "out.py", "content": "secret"},
        rationale="write result",
        needs_approval=True,
    )
    req = ApprovalRequest(
        request_id="r1",
        iteration_index=2,
        action=action,
        risk_reason="mutates file",
        policy_gate="needs_approval",
    )
    rec = ApprovalReceipt(
        request_id="r1",
        granted=True,
        timestamp="2026-01-01T00:00:01+00:00",
    )
    return ReceiptIteration(
        iteration_index=2,
        timestamp="2026-01-01T00:00:01+00:00",
        action=action,
        approval_request=req,
        approval_receipt=rec,
        result=ToolResult(action_id="aw", ok=True, output="wrote 6 chars"),
        observation=AgentObservation(
            action_id="aw",
            iteration_index=2,
            status="success",
            summary="write ok",
        ),
        decision=AgentDecision(
            iteration_index=2,
            decision="completed",
            reason="done",
            stop_reason="completed",
        ),
    )


# ---------------------------------------------------------------------------
# _serialise
# ---------------------------------------------------------------------------

class TestSerialise(unittest.TestCase):
    def test_flat_dataclass(self):
        r = ToolResult(action_id="a1", ok=True, output="hello")
        d = _serialise(r)
        self.assertIsInstance(d, dict)
        self.assertEqual(d["action_id"], "a1")
        self.assertTrue(d["ok"])
        self.assertEqual(d["output"], "hello")

    def test_nested_dataclass(self):
        receipt = _minimal_receipt()
        d = _serialise(receipt)
        self.assertIsInstance(d["action"], dict)
        self.assertEqual(d["action"]["kind"], "read_file")

    def test_none_values_preserved(self):
        receipt = _minimal_receipt()
        d = _serialise(receipt)
        self.assertIsNone(d["approval_request"])
        self.assertIsNone(d["approval_receipt"])

    def test_list_field_serialised(self):
        obs = AgentObservation(
            action_id="a1",
            iteration_index=0,
            status="success",
            summary="ok",
            key_findings=["found X", "found Y"],
        )
        d = _serialise(obs)
        self.assertEqual(d["key_findings"], ["found X", "found Y"])

    def test_raw_content_stored_always_false(self):
        receipt = _minimal_receipt()
        # Manually set True to confirm serialiser overrides it.
        receipt.raw_content_stored = True
        receipt.result.raw_content_stored = True
        d = _serialise(receipt)
        self.assertFalse(d["raw_content_stored"])
        self.assertFalse(d["result"]["raw_content_stored"])

    def test_args_dict_preserved(self):
        receipt = _minimal_receipt()
        d = _serialise(receipt)
        self.assertEqual(d["action"]["args"], {"path": "foo.py"})

    def test_schema_version_preserved(self):
        receipt = _minimal_receipt()
        d = _serialise(receipt)
        self.assertEqual(d["schema_version"], 1)


# ---------------------------------------------------------------------------
# ReceiptEmitter.emit
# ---------------------------------------------------------------------------

class TestEmit(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.path = Path(self._td.name) / "receipts.jsonl"
        self.emitter = ReceiptEmitter(self.path)

    def tearDown(self):
        self._td.cleanup()

    def test_creates_file_on_first_emit(self):
        self.assertFalse(self.path.exists())
        self.emitter.emit(_minimal_receipt())
        self.assertTrue(self.path.exists())

    def test_each_line_is_valid_json(self):
        self.emitter.emit(_minimal_receipt(1))
        self.emitter.emit(_minimal_receipt(2))
        lines = self.path.read_text(encoding="utf-8").splitlines()
        for line in lines:
            obj = json.loads(line)
            self.assertIsInstance(obj, dict)

    def test_records_appended_in_order(self):
        self.emitter.emit(_minimal_receipt(1))
        self.emitter.emit(_minimal_receipt(2))
        records = self.emitter.read_all()
        self.assertEqual(records[0]["iteration_index"], 1)
        self.assertEqual(records[1]["iteration_index"], 2)

    def test_does_not_truncate_existing_file(self):
        self.emitter.emit(_minimal_receipt(1))
        emitter2 = ReceiptEmitter(self.path)
        emitter2.emit(_minimal_receipt(2))
        self.assertEqual(self.emitter.record_count(), 2)

    def test_full_receipt_roundtrips(self):
        self.emitter.emit(_full_receipt())
        records = self.emitter.read_all()
        r = records[0]
        self.assertIsNotNone(r["approval_request"])
        self.assertTrue(r["approval_receipt"]["granted"])
        self.assertEqual(r["decision"]["stop_reason"], "completed")

    def test_raw_content_stored_false_in_written_record(self):
        receipt = _minimal_receipt()
        receipt.raw_content_stored = True
        self.emitter.emit(receipt)
        record = self.emitter.read_all()[0]
        self.assertFalse(record["raw_content_stored"])
        self.assertFalse(record["result"]["raw_content_stored"])

    def test_schema_version_in_written_record(self):
        self.emitter.emit(_minimal_receipt())
        record = self.emitter.read_all()[0]
        self.assertEqual(record["schema_version"], 1)

    def test_no_trailing_whitespace_on_lines(self):
        self.emitter.emit(_minimal_receipt())
        raw = self.path.read_text(encoding="utf-8")
        for line in raw.splitlines():
            self.assertEqual(line, line.rstrip())


# ---------------------------------------------------------------------------
# ReceiptEmitter.emit_event
# ---------------------------------------------------------------------------

class TestEmitEvent(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.path = Path(self._td.name) / "events.jsonl"
        self.emitter = ReceiptEmitter(self.path)

    def tearDown(self):
        self._td.cleanup()

    def _event(self, event_type: str = "loop_started") -> AgentLoopEvent:
        return AgentLoopEvent(
            event_id="ev1",
            run_id="run1",
            timestamp="2026-01-01T00:00:00+00:00",
            event_type=event_type,
            iteration_index=0,
            state="received_task",
            summary="test event",
        )

    def test_event_written_as_json(self):
        self.emitter.emit_event(self._event())
        records = self.emitter.read_all()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["event_type"], "loop_started")

    def test_event_raw_content_stored_false(self):
        e = self._event()
        e.raw_content_stored = True
        self.emitter.emit_event(e)
        record = self.emitter.read_all()[0]
        self.assertFalse(record["raw_content_stored"])

    def test_mixed_receipt_and_event_in_same_file(self):
        self.emitter.emit(_minimal_receipt())
        self.emitter.emit_event(self._event("action_executed"))
        records = self.emitter.read_all()
        self.assertEqual(len(records), 2)
        self.assertIn("iteration_index", records[0])
        self.assertEqual(records[1]["event_type"], "action_executed")


# ---------------------------------------------------------------------------
# ReceiptEmitter.read_all
# ---------------------------------------------------------------------------

class TestReadAll(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.path = Path(self._td.name) / "r.jsonl"
        self.emitter = ReceiptEmitter(self.path)

    def tearDown(self):
        self._td.cleanup()

    def test_returns_empty_list_when_file_missing(self):
        self.assertEqual(self.emitter.read_all(), [])

    def test_returns_all_records(self):
        for i in range(1, 6):
            self.emitter.emit(_minimal_receipt(i))
        records = self.emitter.read_all()
        self.assertEqual(len(records), 5)

    def test_record_count(self):
        self.emitter.emit(_minimal_receipt(1))
        self.emitter.emit(_minimal_receipt(2))
        self.assertEqual(self.emitter.record_count(), 2)

    def test_record_count_zero_when_no_file(self):
        self.assertEqual(self.emitter.record_count(), 0)


if __name__ == "__main__":
    unittest.main()
