"""Versioned prompt templates stored as YAML (``prompts/<id>/v<version>.yaml``).

Prompts are code: they are reviewed in PRs, evaluated in CI, and every answer records the prompt
id, version and content hash, so any response can be traced to the exact text that produced it.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class PromptTemplate:
    id: str
    version: int
    system: str
    user: str
    sha256: str

    @property
    def ref(self) -> str:
        return f"{self.id}@v{self.version}"

    def render(self, **values: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.system.format(**values).strip()},
            {"role": "user", "content": self.user.format(**values).strip()},
        ]


def load_prompt(prompts_dir: Path, prompt_id: str, version: int | None = None) -> PromptTemplate:
    """Load ``version`` of a prompt, or the highest version when ``version`` is None."""
    folder = prompts_dir / prompt_id
    versions = sorted(int(p.stem.removeprefix("v")) for p in folder.glob("v*.yaml"))
    if not versions:
        raise FileNotFoundError(f"no versions of prompt {prompt_id!r} in {folder}")
    chosen = version if version is not None else versions[-1]
    raw = (folder / f"v{chosen}.yaml").read_bytes()
    doc = yaml.safe_load(raw)
    if doc["id"] != prompt_id or doc["version"] != chosen:
        raise ValueError(f"{folder / f'v{chosen}.yaml'}: id/version do not match the file name")
    return PromptTemplate(
        id=prompt_id,
        version=chosen,
        system=doc["system"],
        user=doc["user"],
        sha256=hashlib.sha256(raw).hexdigest(),
    )
