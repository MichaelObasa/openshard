from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SkillDef:
    slug: str
    name: str
    description: str
    category: str
    keywords: list[str]
    languages: list[str]
    framework: str | None


def _parse_list(value: str) -> list[str]:
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_frontmatter(text: str) -> dict[str, str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    result: dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            break
        if ":" in line:
            key, _, value = line.partition(":")
            result[key.strip()] = value.strip()
    return result


def discover_skills(root: Path) -> list[SkillDef]:
    """Scan <root>/.openshard/skills/*/SKILL.md and return parsed skill defs."""
    skills_dir = root / ".openshard" / "skills"
    if not skills_dir.is_dir():
        return []
    skills: list[SkillDef] = []
    for skill_md in sorted(skills_dir.glob("*/SKILL.md")):
        try:
            text = skill_md.read_text(encoding="utf-8")
            fm = _parse_frontmatter(text)
            if not fm.get("name"):
                continue
            skills.append(SkillDef(
                slug=skill_md.parent.name,
                name=fm["name"],
                description=fm.get("description", ""),
                category=fm.get("category", "standard"),
                keywords=_parse_list(fm.get("keywords", "")),
                languages=_parse_list(fm.get("languages", "")),
                framework=fm.get("framework") or None,
            ))
        except Exception:
            continue
    return skills
