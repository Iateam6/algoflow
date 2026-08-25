from __future__ import annotations

import re
from dataclasses import dataclass, field


SECTION_BREAK = re.compile(r"\n\s*\n+")
SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class EvidenceChunk:
    chunk_id: str
    text: str
    tenant_id: str
    case_id: str
    source_hash: str
    page_number: int
    chunk_index: int
    heading: str | None = None
    metadata: dict = field(default_factory=dict)


def _split_oversized(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    sentences = SENTENCE_BREAK.split(text)
    parts: list[str] = []
    current = ""
    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > max_chars:
            parts.append(current.strip())
            current = ""
        if len(sentence) > max_chars:
            for start in range(0, len(sentence), max_chars):
                piece = sentence[start : start + max_chars].strip()
                if piece:
                    parts.append(piece)
        else:
            current = f"{current} {sentence}".strip()
    if current:
        parts.append(current)
    return parts


def chunk_sources(
    sources: list[dict],
    *,
    tenant_id: str,
    case_id: str,
    max_chars: int = 1800,
) -> list[EvidenceChunk]:
    chunks: list[EvidenceChunk] = []
    for source in sources:
        index = 0
        for page_number, page_text in enumerate(source.get("pages", []), start=1):
            sections = [part.strip() for part in SECTION_BREAK.split(page_text) if part.strip()]
            heading: str | None = None
            for section in sections:
                first_line = section.splitlines()[0].strip()
                if len(first_line) <= 120 and (
                    first_line.endswith(":") or first_line.isupper()
                ):
                    heading = first_line
                for piece in _split_oversized(section, max_chars):
                    # Section-level splitting keeps form labels, address blocks,
                    # lists, and signatures together whenever they fit.
                    chunk_id = (
                        f"{tenant_id}:{case_id}:{source['file_hash']}:"
                        f"{page_number}:{index}"
                    )
                    chunks.append(
                        EvidenceChunk(
                            chunk_id=chunk_id,
                            text=piece,
                            tenant_id=tenant_id,
                            case_id=case_id,
                            source_hash=source["file_hash"],
                            page_number=page_number,
                            chunk_index=index,
                            heading=heading,
                        )
                    )
                    index += 1
    return chunks

