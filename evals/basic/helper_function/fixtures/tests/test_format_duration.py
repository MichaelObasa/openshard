import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from openshard.utils import format_duration


def test_format_duration_returns_string():
    assert isinstance(format_duration(1.0), str)


def test_format_duration_subsecond():
    result = format_duration(0.5)
    assert isinstance(result, str)
    assert len(result) > 0


def test_format_duration_seconds():
    result = format_duration(3.0)
    assert isinstance(result, str)
    assert "s" in result.lower() or any(c.isdigit() for c in result)


def test_format_duration_minutes():
    result = format_duration(190.0)
    assert isinstance(result, str)
    assert "m" in result.lower() or any(c.isdigit() for c in result)


def test_format_duration_annotation():
    import inspect
    hints = {}
    try:
        import typing
        hints = typing.get_type_hints(format_duration)
    except Exception:
        sig = inspect.signature(format_duration)
        p = sig.parameters.get("seconds")
        if p is not None and p.annotation is not inspect.Parameter.empty:
            hints["seconds"] = p.annotation
    assert hints, "format_duration should have type annotations"
