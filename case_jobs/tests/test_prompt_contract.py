from pathlib import Path

from django.test import SimpleTestCase

from case_jobs.pipeline.prompt_contract import (
    LEGAL_DRAFTING_CONTRACT_HEADING,
    build_legal_drafting_contract,
    prepare_template_for_generation,
)
from case_jobs.pipeline.verification import contains_unresolved_template_field


PROJECT_ROOT = Path(__file__).resolve().parents[2]
AGENT_PATHS = tuple(
    PROJECT_ROOT / module / "agent.py"
    for module in ("h_1b", "eb_1a", "eb_1c", "eb_2niw", "eb_5", "e_2", "l1", "o1", "tn")
)


class PromptContractTests(SimpleTestCase):
    def test_contract_requires_grounding_and_explicit_missing_markers(self):
        contract = build_legal_drafting_contract()
        self.assertIn(LEGAL_DRAFTING_CONTRACT_HEADING, contract)
        self.assertIn("never infer or invent", contract)
        self.assertIn("[MISSING: field name]", contract)
        self.assertIn("Do not print chunk IDs", contract)

    def test_every_agent_includes_shared_contract(self):
        for path in AGENT_PATHS:
            with self.subTest(path=path):
                source = path.read_text(encoding="utf-8")
                self.assertIn(
                    "from case_jobs.pipeline.prompt_contract import build_legal_drafting_contract",
                    source,
                )
                self.assertIn("build_legal_drafting_contract(),", source)

    def test_generation_prompts_send_file_only_chunk_metadata_without_manifest(self):
        for path in AGENT_PATHS:
            with self.subTest(path=path):
                source = path.read_text(encoding="utf-8")
                self.assertIn('f"- File: {metadata.get(\'source_name\', \'unknown\')}"', source)
                self.assertNotIn('"# Source Manifest"', source)
                self.assertNotIn("summarise_source_manifest(source_manifest),", source)

    def test_legacy_blank_instruction_is_normalized(self):
        prepared = prepare_template_for_generation(
            "If the required information is not available, leave the placeholder blank."
        )
        self.assertNotIn("leave the placeholder blank", prepared)
        self.assertIn("[MISSING: field name]", prepared)

    def test_missing_marker_is_allowed_but_raw_placeholder_is_rejected(self):
        self.assertFalse(contains_unresolved_template_field("[MISSING: attorney address]"))
        self.assertTrue(contains_unresolved_template_field("[Attorney Address]"))

    def test_placeholder_resolution_runtime_has_been_removed(self):
        enrichment_source = (
            PROJECT_ROOT / "case_jobs" / "pipeline" / "context_enrichment.py"
        ).read_text(encoding="utf-8")
        generation_source = (
            PROJECT_ROOT / "case_jobs" / "pipeline" / "legacy_generation.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("TemplateFieldResolver", enrichment_source)
        self.assertNotIn("TemplateFieldResolver", generation_source)
