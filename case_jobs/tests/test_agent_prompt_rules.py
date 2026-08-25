"""Cross-visa prompt invariants; prompt bodies remain visa-owned."""

from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
AGENT_PATHS = (
    PROJECT_ROOT / "h_1b" / "agent.py",
    PROJECT_ROOT / "eb_1a" / "agent.py",
    PROJECT_ROOT / "eb_1c" / "agent.py",
    PROJECT_ROOT / "eb_2niw" / "agent.py",
    PROJECT_ROOT / "eb_5" / "agent.py",
    PROJECT_ROOT / "e_2" / "agent.py",
    PROJECT_ROOT / "l1" / "agent.py",
    PROJECT_ROOT / "o1" / "agent.py",
    PROJECT_ROOT / "tn" / "agent.py",
)

OLD_RULE = '"Use only the retrieved case record and the source manifest."'
GROUNDING_RULES = (
    '"Act like a lawyer: analyze the retrieved case record and the source manifest, then draft a document grounded in those materials."',
    '"Use the retrieved case record as the primary source of factual support and the source manifest as supporting evidence."',
    '"When a template field or placeholder is not directly available, look for equivalent or related evidence in the retrieved case record/source manifest and use that to fill the section."',
    '"For example, if the template asks for \'petitioner\' or \'employer\' and the case record uses a different but equivalent term, use the correct party from the evidence."',
    '"Fill every relevant section only with supported facts; use [MISSING: field name] when required evidence is unavailable."',
    '"Never leave a required field blank and never invent a replacement value."',
    '"Return only the final document enclosed in triple backticks."',
)
STRUCTURAL_RULE_PREFIX = '"Treat the template as a structural guide'


class AgentPromptRuleTests(unittest.TestCase):
    def test_all_agents_retain_grounding_specific_output_rules(self):
        for agent_path in AGENT_PATHS:
            with self.subTest(agent_path=agent_path):
                source = agent_path.read_text(encoding="utf-8")

                self.assertNotIn(OLD_RULE, source)
                self.assertIn(STRUCTURAL_RULE_PREFIX, source)
                for rule in GROUNDING_RULES:
                    self.assertIn(rule, source)

    def test_tn_and_e2_labeled_blocks_use_explicit_line_breaks(self):
        tn_source = (PROJECT_ROOT / "tn" / "agent.py").read_text(encoding="utf-8")
        self.assertIn("**Employer:** [Company Name]<br>", tn_source)
        self.assertIn("**Beneficiary:** [Beneficiary Full Name]<br>", tn_source)
        self.assertIn("**Position:** [Job Title]<br>", tn_source)
        self.assertIn("**By:** [Beneficiary Full Name]<br>", tn_source)
        self.assertIn("**Title:** [Job Title]<br>", tn_source)
        self.assertIn("**Company:** [Company Name]<br>", tn_source)

        e2_source = (PROJECT_ROOT / "e_2" / "agent.py").read_text(encoding="utf-8")
        self.assertIn("**By:** [Beneficiary Full Name]<br>", e2_source)
        self.assertIn("**Title:** [Job Title]<br>", e2_source)
        self.assertIn("**Company:** [Company Name]<br>", e2_source)


if __name__ == "__main__":
    unittest.main()
