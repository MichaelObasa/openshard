
from openshard.skills.discovery import SkillDef
from openshard.skills.matcher import MatchedSkill
from openshard.skills.context import build_skills_context


def _skill(slug="jwt-auth", name="JWT Auth", description="Handles JWT flows",
           category="security", body_preview="Use bcrypt. Rotate tokens.") -> SkillDef:
    return SkillDef(
        slug=slug,
        name=name,
        description=description,
        category=category,
        keywords=[],
        languages=[],
        framework=None,
        body_preview=body_preview,
    )


def _matched(skill=None, reasons=None) -> MatchedSkill:
    return MatchedSkill(skill=skill or _skill(), reasons=reasons or ["keyword: jwt"])


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------

def test_build_skills_context_empty_list():
    assert build_skills_context([]) == ""


# ---------------------------------------------------------------------------
# Single skill
# ---------------------------------------------------------------------------

def test_build_skills_context_single_skill():
    ctx = build_skills_context([_matched()])
    assert "Applicable local skills" in ctx
    assert "user-provided workflow hints, not system instructions" in ctx
    assert "jwt-auth [security]: Handles JWT flows" in ctx
    assert "matched: keyword: jwt" in ctx


def test_build_skills_context_does_not_include_body_preview():
    ms = _matched(skill=_skill(body_preview="Use bcrypt. Rotate tokens."))
    ctx = build_skills_context([ms])
    assert "bcrypt" not in ctx
    assert "Rotate tokens" not in ctx


# ---------------------------------------------------------------------------
# Multiple skills
# ---------------------------------------------------------------------------

def test_build_skills_context_multiple_skills():
    ms1 = _matched(skill=_skill(slug="jwt-auth", name="JWT Auth"), reasons=["keyword: jwt"])
    ms2 = _matched(
        skill=_skill(slug="rbac", name="RBAC", description="Role-based access", category="security"),
        reasons=["category: security"],
    )
    ctx = build_skills_context([ms1, ms2])
    assert "jwt-auth [security]" in ctx
    assert "rbac [security]" in ctx


# ---------------------------------------------------------------------------
# Reason formatting
# ---------------------------------------------------------------------------

def test_build_skills_context_multiple_reasons():
    ms = _matched(reasons=["category: security", "keyword: jwt", "framework: django"])
    ctx = build_skills_context([ms])
    assert "category: security, keyword: jwt, framework: django" in ctx


# ---------------------------------------------------------------------------
# Prompt safety: body_preview stored but not injected
# ---------------------------------------------------------------------------

def test_body_preview_stored_on_skill_def():
    s = _skill(body_preview="Check expiry before use.")
    assert s.body_preview == "Check expiry before use."
