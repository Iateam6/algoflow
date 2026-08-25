from __future__ import annotations

import asyncio
import json
import re
from typing import Sequence

from django.conf import settings

from case_jobs.integrations.openai_client import get_openai_client


FENCED_OUTPUT_PATTERN = re.compile(r"^```(?:json|markdown)?\s*|\s*```$", re.IGNORECASE)
URL_PATTERN = re.compile(r"https?://\S+", re.IGNORECASE)


def _clean_model_output(value: str) -> str:
    return FENCED_OUTPUT_PATTERN.sub("", (value or "").strip()).strip()


def sanitize_exhibits(exhibits: Sequence[dict]) -> list[dict]:
    sanitized: list[dict] = []
    for exhibit in exhibits:
        if not isinstance(exhibit, dict):
            continue
        items: list[dict] = []
        for item in exhibit.get("items", []):
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type") or "").strip()
            name = str(item.get("name") or "").strip()
            if not item_type or not name:
                continue
            if isinstance(item.get("file_count"), int) and item["file_count"] > 0:
                file_count = item["file_count"]
            elif item_type == "file" and isinstance(item.get("files"), list):
                file_count = len(item["files"])
            else:
                file_count = 1
            items.append(
                {"type": item_type, "name": name, "file_count": file_count}
            )
        if items:
            sanitized.append(
                {
                    "number": exhibit.get("number"),
                    "title": str(exhibit.get("title") or "").strip(),
                    "items": items,
                }
            )
    return sanitized


def build_deterministic_exhibit_markdown(exhibits: Sequence[dict]) -> str:
    sanitized = sanitize_exhibits(exhibits)
    lines = [
        "# Submitted Exhibits",
        "Use this exhibit structure as supporting context. Do not include source URLs.",
    ]
    for exhibit in sanitized:
        number = exhibit["number"]
        title = exhibit["title"]
        heading = f"## Exhibit {number}"
        if title:
            heading += f": {title}"
        lines.extend(["", heading])
        for item_index, item in enumerate(exhibit["items"], start=1):
            count = item["file_count"]
            count_text = f", {count} file{'s' if count != 1 else ''}"
            lines.append(
                f"- {number}.{item_index} {item['name']} "
                f"({item['type']}{count_text})"
            )
    return "\n".join(lines).strip()


def _valid_exhibit_markdown(markdown_text: str, exhibits: Sequence[dict]) -> bool:
    if not markdown_text or URL_PATTERN.search(markdown_text):
        return False
    cursor = 0
    for exhibit in sanitize_exhibits(exhibits):
        for item_index, _ in enumerate(exhibit["items"], start=1):
            label = f"{exhibit['number']}.{item_index}"
            position = markdown_text.find(label, cursor)
            if position < 0:
                return False
            cursor = position + len(label)
    return True


class ExhibitMarkdownAgent:
    def __init__(self, client=None):
        self.client = client

    async def generate(self, exhibits: Sequence[dict]) -> str:
        sanitized = sanitize_exhibits(exhibits)
        fallback = build_deterministic_exhibit_markdown(sanitized)
        if not sanitized:
            return ""
        prompt = {
            "instruction": (
                "Create raw Markdown describing the submitted exhibit structure for use as "
                "context by another drafting model. Preserve exhibit order, item order, "
                "numbers, titles, names, types, and file counts exactly. Label items as "
                "<exhibit number>.<one-based item position>. Do not include URLs."
            ),
            "exhibits": sanitized,
        }
        try:
            client = self.client or get_openai_client()
            response = await asyncio.to_thread(
                client.responses.create,
                model=settings.AUXILIARY_CONTEXT_MODEL,
                input=json.dumps(prompt, separators=(",", ":")),
            )
            markdown_text = _clean_model_output(response.output_text or "")
            return markdown_text if _valid_exhibit_markdown(markdown_text, sanitized) else fallback
        except Exception:
            return fallback
