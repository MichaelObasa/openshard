from openshard.execution.gates import GateDecision, GateEvaluator, resolve_gate_decisions


def make_gate(mode, risky=None, threshold=0.10):
    return GateEvaluator(approval_mode=mode, risky_paths=risky, cost_threshold=threshold)


# auto: only stack mismatch triggers
def test_auto_no_prompt_for_file_write():
    g = make_gate("auto")
    assert not g.check_file_write(["src/auth/login.py"]).required

def test_auto_no_prompt_for_shell():
    g = make_gate("auto")
    assert not g.check_shell_command("rm -rf /").required

def test_auto_no_prompt_for_high_cost():
    g = make_gate("auto")
    assert not g.check_high_cost(999.0).required

def test_auto_prompts_for_stack_mismatch():
    g = make_gate("auto")
    assert g.check_stack_mismatch(["main.rs"]).required

def test_auto_no_prompt_for_empty_stack_mismatch():
    g = make_gate("auto")
    assert not g.check_stack_mismatch([]).required


# ask: prompts for everything
def test_ask_prompts_for_file_write():
    g = make_gate("ask")
    assert g.check_file_write(["src/main.py"]).required

def test_ask_prompts_for_shell_command():
    g = make_gate("ask")
    assert g.check_shell_command("python -m pytest").required


# smart: whitelist, risky paths, cost threshold
def test_smart_allows_whitelisted_pytest():
    g = make_gate("smart")
    assert not g.check_shell_command("python -m pytest").required

def test_smart_allows_npm_test():
    g = make_gate("smart")
    assert not g.check_shell_command("npm test").required

def test_smart_prompts_for_non_whitelisted_shell():
    g = make_gate("smart")
    assert g.check_shell_command("curl https://example.com").required

def test_smart_prompts_for_risky_path():
    g = make_gate("smart", risky=["auth"])
    assert g.check_file_write(["src/auth/login.py"]).required

def test_smart_no_prompt_safe_path():
    g = make_gate("smart", risky=["auth"])
    assert not g.check_file_write(["src/utils/helpers.py"]).required

def test_smart_high_cost_above_threshold():
    g = make_gate("smart", threshold=0.10)
    assert g.check_high_cost(0.15).required

def test_smart_high_cost_below_threshold():
    g = make_gate("smart", threshold=0.10)
    assert not g.check_high_cost(0.05).required


# risky path normalization
def test_risky_path_backslash_normalization():
    g = make_gate("smart", risky=["auth"])
    assert g.check_file_write(["src\\auth\\login.py"]).required


# verify gate: ask mode prompts even when no test command detected (uses "(verify)" sentinel)
def test_ask_prompts_for_verify_no_command_detected():
    g = make_gate("ask")
    assert g.check_shell_command("(verify)").required

def test_smart_prompts_for_verify_no_command_detected():
    g = make_gate("smart")
    # "(verify)" is not in the safe whitelist
    assert g.check_shell_command("(verify)").required

def test_auto_no_prompt_for_verify_no_command_detected():
    g = make_gate("auto")
    assert not g.check_shell_command("(verify)").required

# file write gate: ask mode prompts even when files list is empty
def test_ask_prompts_for_file_write_empty_list():
    g = make_gate("ask")
    dec = g.check_file_write([])
    assert dec.required


# resolve_gate_decisions: route combined gate decisions through the
# canonical deny > ask > allow policy resolver.
# Deny dominance is intentionally NOT tested here — GateDecision cannot express
# deny (only required: bool). That ordering stays covered by test_policy_decision.py.
def test_resolve_empty_returns_not_required():
    dec = resolve_gate_decisions([])
    assert not dec.required
    assert dec.reason == ""

def test_resolve_all_allow_not_required():
    dec = resolve_gate_decisions([GateDecision(False, ""), GateDecision(False, "")])
    assert not dec.required

def test_resolve_ask_beats_allow():
    dec = resolve_gate_decisions([GateDecision(False, ""), GateDecision(True, "risky path")])
    assert dec.required
    assert dec.reason == "risky path"

def test_resolve_multiple_asks_preserve_priority_order():
    # First (highest priority) ask wins the tie deterministically.
    dec = resolve_gate_decisions([GateDecision(True, "file write"), GateDecision(True, "risky path")])
    assert dec.required
    assert dec.reason == "file write"

def test_resolve_missing_reason_does_not_crash():
    dec = resolve_gate_decisions([GateDecision(True, "")])
    assert dec.required
    assert dec.reason == ""

def test_resolve_deterministic_output():
    inputs = [GateDecision(True, "file write"), GateDecision(True, "risky path")]
    results = {resolve_gate_decisions(inputs).reason for _ in range(5)}
    assert results == {"file write"}
