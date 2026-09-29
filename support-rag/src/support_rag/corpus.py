"""Help-center corpus: one YAML file per topic with the article in every supported language.

The format mirrors a CMS export, so every topic is guaranteed to exist in all languages, and
facts (fees, limits, deadlines) are maintained in one place per topic.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml

from support_rag.config import LANGUAGES


@dataclass(frozen=True)
class Article:
    topic: str
    lang: str
    title: str
    body: str
    category: str
    updated: str

    @property
    def id(self) -> str:
        return f"{self.topic}.{self.lang}"

    @property
    def url(self) -> str:
        return f"https://help.nordalp.example/{self.lang}/{self.topic}"


class CorpusError(ValueError):
    """Raised when the corpus is incomplete or inconsistent."""


def load_corpus(directory: Path) -> list[Article]:
    articles = []
    for path in sorted(directory.glob("*.yaml")):
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        topic = path.stem
        missing = set(LANGUAGES) - set(doc.get("translations", {}))
        if missing:
            raise CorpusError(f"{path.name}: missing languages {sorted(missing)}")
        for lang in LANGUAGES:
            t = doc["translations"][lang]
            if not t.get("title") or not t.get("body", "").strip():
                raise CorpusError(f"{path.name}: empty title or body for {lang}")
            articles.append(
                Article(
                    topic=topic,
                    lang=lang,
                    title=t["title"].strip(),
                    body=t["body"].strip(),
                    category=doc["category"],
                    updated=str(doc["updated"]),
                )
            )
    if not articles:
        raise CorpusError(f"no articles in {directory}")
    return articles
