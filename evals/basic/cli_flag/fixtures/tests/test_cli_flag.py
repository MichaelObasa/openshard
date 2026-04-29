import importlib.util
import json
from pathlib import Path

from click.testing import CliRunner


def _load_demo_cli():
    path = Path(__file__).parent.parent / "tools" / "demo_cli.py"
    spec = importlib.util.spec_from_file_location("demo_cli", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_report_json_flag_exists():
    mod = _load_demo_cli()
    runner = CliRunner()
    result = runner.invoke(mod.cli, ["report", "--help"])
    assert result.exit_code == 0, result.output
    assert "--json" in result.output, f"--json flag not found in help:\n{result.output}"


def test_report_json_flag_outputs_valid_json():
    mod = _load_demo_cli()
    runner = CliRunner()
    result = runner.invoke(mod.cli, ["report", "--json"])
    assert result.exit_code == 0, f"exit={result.exit_code}\n{result.output}"
    data = json.loads(result.output)
    assert isinstance(data, (dict, list)), f"expected dict or list, got: {type(data)}"
