"""Split articles into retrieval chunks along paragraph boundaries."""

from dataclasses import dataclass

from support_rag.corpus import Article


@dataclass(frozen=True)
class Chunk:
    article: Article
    index: int
    text: str  # what is embedded and shown to the LLM: title + paragraph(s)

    @property
    def id(self) -> str:
        return f"{self.article.id}#{self.index}"


def chunk_article(article: Article, max_words: int = 120) -> list[Chunk]:
    """Greedily pack whole paragraphs up to ``max_words``; the title prefixes every chunk.

    Help articles are short and each paragraph is one self-contained step or fact, so paragraph
    packing keeps facts intact. A single oversized paragraph becomes its own chunk.
    """
    paragraphs = [p.strip() for p in article.body.split("\n\n") if p.strip()]
    groups: list[list[str]] = []
    words = 0
    for p in paragraphs:
        n = len(p.split())
        if groups and words + n <= max_words:
            groups[-1].append(p)
            words += n
        else:
            groups.append([p])
            words = n
    return [
        Chunk(article=article, index=i, text=f"{article.title}\n\n" + "\n\n".join(g))
        for i, g in enumerate(groups)
    ]
