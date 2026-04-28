import pytest

from openshard.analysis.repo import RepoFacts
from openshard.skills.discovery import SkillDef
from openshard.skills.matcher import match_skills


def _skill(
    slug="test-skill",
    name="Test Skill",
    category="security",
    keywords=None,
    languages=None,
    framework=None,
) -> SkillDef:
    return SkillDef(
        slug=slug,
        name=name,
        description="",
        category=category,
        keywords=keywords or [],
        languages=languages or [],
        framework=framework,
    )


def _facts(languages=None, framework=None) -> RepoFacts:
    return RepoFacts(
        languages=languages or [],
        package_files=[],
        framework=framework,
        test_command=None,
        risky_paths=[],
        changed_files=[],
    )


# ---------------------------------------------------------------------------
# Single-signal matches
# ---------------------------------------------------------------------------

def test_category_match_qualifies():
    skills = [_skill(category="security")]
    result = match_skills(skills, "add a login form", "security", _facts())
    assert len(result) == 1
    assert result[0].reasons == ["category: security"]


def test_keyword_match_qualifies():
    skills = [_skill(category="boilerplate", keywords=["login"])]
    result = match_skills(skills, "add a login form", "security", _facts())
    assert len(result) == 1
    assert result[0].reasons == ["keyword: login"]


def test_framework_match_qualifies():
    skills = [_skill(category="boilerplate", framework="django")]
    result = match_skills(skills, "add a form", "security", _facts(framework="django"))
    assert len(result) == 1
    assert result[0].reasons == ["framework: django"]


# ---------------------------------------------------------------------------
# Language-only does NOT qualify
# ---------------------------------------------------------------------------

def test_language_alone_does_not_qualify():
    skills = [_skill(category="boilerplate", languages=["python"])]
    # category mismatch, no keywords, no framework
    result = match_skills(skills, "add a form", "security", _facts(languages=["python"]))
    assert result == []


# ---------------------------------------------------------------------------
# Language + category qualifies
# ---------------------------------------------------------------------------

def test_language_plus_category_qualifies():
    skills = [_skill(category="security", languages=["python"])]
    result = match_skills(skills, "add a form", "security", _facts(languages=["python"]))
    assert len(result) == 1
    reasons = result[0].reasons
    assert "category: security" in reasons
    assert "language: python" in reasons


# ---------------------------------------------------------------------------
# Multiple reasons — skill appears once with all reasons
# ---------------------------------------------------------------------------

def test_multiple_reasons_combined():
    skill = _skill(category="security", keywords=["auth"], framework="django", languages=["python"])
    result = match_skills([skill], "add auth endpoint", "security", _facts(languages=["python"], framework="django"))
    assert len(result) == 1
    reasons = result[0].reasons
    assert "category: security" in reasons
    assert "keyword: auth" in reasons
    assert "framework: django" in reasons
    assert "language: python" in reasons


# ---------------------------------------------------------------------------
# No match → excluded
# ---------------------------------------------------------------------------

def test_no_match_excluded():
    skills = [_skill(category="boilerplate", keywords=["validate"])]
    result = match_skills(skills, "implement auth flow", "security", _facts())
    assert result == []


# ---------------------------------------------------------------------------
# Results sorted by reason count descending
# ---------------------------------------------------------------------------

def test_sorted_by_reason_count():
    skill_many = _skill(slug="many", category="security", keywords=["auth"], framework="django")
    skill_one = _skill(slug="one", category="visual", keywords=["auth"])
    result = match_skills([skill_one, skill_many], "add auth", "security", _facts(framework="django"))
    assert result[0].skill.slug == "many"
    assert result[1].skill.slug == "one"


# ---------------------------------------------------------------------------
# Max 5 results returned
# ---------------------------------------------------------------------------

def test_max_five_results():
    skills = [_skill(slug=f"s{i}", category="security") for i in range(10)]
    result = match_skills(skills, "fix auth", "security", _facts())
    assert len(result) == 5


# ---------------------------------------------------------------------------
# Case-insensitive keyword match
# ---------------------------------------------------------------------------

def test_keyword_case_insensitive():
    skills = [_skill(category="standard", keywords=["Auth"])]
    result = match_skills(skills, "implement AUTH flow", "standard", _facts())
    assert len(result) == 1
    assert any("keyword" in r for r in result[0].reasons)


# ---------------------------------------------------------------------------
# Only first matching keyword contributes one reason
# ---------------------------------------------------------------------------

def test_only_first_keyword_reason():
    skills = [_skill(category="standard", keywords=["auth", "login", "session"])]
    result = match_skills(skills, "auth login session", "standard", _facts())
    keyword_reasons = [r for r in result[0].reasons if r.startswith("keyword:")]
    assert len(keyword_reasons) == 1
