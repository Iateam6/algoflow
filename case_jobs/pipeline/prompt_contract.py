from __future__ import annotations


LEGAL_DRAFTING_CONTRACT_HEADING = "# Shared Legal Drafting Behavior Contract"

_CONFLICTING_TEMPLATE_RULES = (
    (
        "If the required information is not available, leave the placeholder blank.",
        "If the required information is not available, replace it with [MISSING: field name].",
    ),
    (
        "If any required information is missing, leave the corresponding placeholder blank;",
        "If any required information is missing, replace the corresponding placeholder with [MISSING: field name];",
    ),
)


def prepare_template_for_generation(template: str) -> str:
    """Remove legacy missing-data instructions that conflict with the contract."""
    prepared = template.strip()
    for old_rule, new_rule in _CONFLICTING_TEMPLATE_RULES:
        prepared = prepared.replace(old_rule, new_rule)
    return prepared


def build_legal_drafting_contract() -> str:
    """Return the behavior rules shared by every visa document generator."""
    return "\n".join(
        (
            LEGAL_DRAFTING_CONTRACT_HEADING,
            "Treat the supplied template as a structural and formatting guide, not as evidence.",
            "Use submitted request values as authoritative and use the Retrieved Case Record as the factual evidence source.",
            "Never copy example facts from a template and never infer or invent a client fact.",
            "Use a formal, precise, concise legal tone suitable for immigration filings.",
            "Resolve template fields only when a value is supported by submitted data or retrieved evidence.",
            "When a required value is unsupported, replace it with [MISSING: field name] using a concise field name.",
            "Do not leave raw template placeholders, empty brackets, or unresolved slash-separated alternatives in the output.",
            "Do not print chunk IDs, retrieval scores, cache details, or other internal evidence metadata in the final document.",
            "Return only the completed document as raw Markdown enclosed in triple backticks.",
        )
    )
