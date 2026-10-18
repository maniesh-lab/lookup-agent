"""Split a Markdown policy file into chunks, one per section.

Each policy file has a "# Title" and several "## Section" headings. A
section is usually one idea (for example "Who pays for return shipping"),
so it makes a good chunk. Sections longer than max_chars are split
further at paragraph breaks.
"""

from dataclasses import dataclass


@dataclass
class Chunk:
    heading: str  # "Returns Policy > Who pays for return shipping"
    content: str
    index: int

    @property
    def text_for_embedding(self) -> str:
        """The heading is added so a chunk like "It costs 6.95 USD." still
        says what it is about."""
        return f"{self.heading}\n{self.content}"


def parse_title(markdown: str, fallback: str) -> str:
    for line in markdown.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def split_paragraphs(text: str, max_chars: int) -> list[str]:
    """Group paragraphs into pieces of at most max_chars (a single longer
    paragraph is kept whole rather than cut mid-sentence)."""
    pieces, current = [], ""
    for para in [p.strip() for p in text.split("\n\n") if p.strip()]:
        if current and len(current) + len(para) + 2 > max_chars:
            pieces.append(current)
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current:
        pieces.append(current)
    return pieces


def chunk_markdown(markdown: str, title: str, max_chars: int = 900) -> list[Chunk]:
    sections: list[tuple[str, str]] = []
    heading, lines = "Overview", []

    for line in markdown.splitlines():
        if line.startswith("# "):
            continue
        if line.startswith("## "):
            sections.append((heading, "\n".join(lines)))
            heading, lines = line[3:].strip(), []
        else:
            lines.append(line)
    sections.append((heading, "\n".join(lines)))

    chunks: list[Chunk] = []
    for section_heading, body in sections:
        for piece in split_paragraphs(body.strip(), max_chars):
            chunks.append(Chunk(f"{title} > {section_heading}", piece, len(chunks)))
    return chunks
