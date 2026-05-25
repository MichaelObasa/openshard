from __future__ import annotations

PLAN_FAST_PATHS: tuple[str, ...] = (
    "plan ",
    "how should i ",
    "what is the safest way to ",
    "how do i approach ",
)

_USAGE = (
    "Plan Mode needs a task. Example:\n"
    "  /plan refactor the auth module safely\n"
    "  /plan add tests for the payment module\n"
    "  /plan review this Terraform repo before changing anything"
)

_APPROACH = (
    "  Suggested approach\n"
    "    1. Inspect relevant files and repo shape\n"
    "    2. Identify risk areas\n"
    "    3. Propose a small change scope\n"
    "    4. Run targeted checks\n"
    "    5. Review receipt before applying anything risky"
)

_RISK_NOTES = (
    "  Risk notes\n"
    "    - Auth/payment/infra/security tasks should require stronger review\n"
    "    - Destructive writes should require approval"
)

_NEXT_STEP = (
    "  Next step\n"
    "    Run the task normally when ready, or use a workflow pack if available."
)


def answer_plan_mode(task: str) -> str:
    if not task.strip():
        return _USAGE
    return (
        "PLAN\n"
        "\n"
        "  Local plan only\n"
        "    No repo scan, no provider call, no files changed.\n"
        "\n"
        f"  Goal\n"
        f"    {task.strip()}\n"
        "\n"
        f"{_APPROACH}\n"
        "\n"
        f"{_RISK_NOTES}\n"
        "\n"
        f"{_NEXT_STEP}"
    )
