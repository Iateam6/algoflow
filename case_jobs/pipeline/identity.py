from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import asdict, dataclass

from django.conf import settings

from case_jobs.api.schemas import GenerationRequest
from case_jobs.exceptions import BeneficiaryNotFound, PetitionerMismatch


logger = logging.getLogger(__name__)


def _normalized(value: str) -> str:
    if not value:
        return ""
    normalized = unicodedata.normalize("NFKD", value)
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", " ", normalized.lower()).strip()


def _name_tokens(full_name: str) -> list[str]:
    tokens = [token for token in _normalized(full_name).split() if token]
    return [token for token in tokens if any(ch.isalnum() for ch in token)]


def _strong_name_match_tokens(tokens: list[str], page_tokens: set[str]) -> bool:
    if not tokens or not page_tokens:
        return False
    first = tokens[0]
    last = tokens[-1]
    if len(first) < 2 and len(tokens) >= 2:
        candidate = tokens[1]
        if len(candidate) >= 2:
            first = candidate
    if len(last) < 2 and len(tokens) >= 2:
        candidate = tokens[-2]
        if len(candidate) >= 2:
            last = candidate
    return first in page_tokens and last in page_tokens


@dataclass(frozen=True)
class IdentityRecord:
    case_id: str
    attorney_name: str
    attorney_address: dict
    beneficiary_name: str
    beneficiary_address: dict
    petitioner_name: str
    petitioner_address: dict
    service_center_name: str
    service_center_address: dict
    self_petition: bool
    supporting_sources: tuple[dict, ...]
    employer: str | None = None
    field_of_extraordinary_ability: str | None = None
    receipt_number: str | None = None
    conflicting_identities: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)


def build_identity_record(
    request: GenerationRequest,
    extracted_sources: list[dict],
) -> IdentityRecord:
    policy = (getattr(settings, "IDENTITY_EVIDENCE_POLICY", "warn") or "warn").strip().lower()
    if policy not in {"strict", "warn", "off"}:
        logger.warning("Unknown IDENTITY_EVIDENCE_POLICY=%r; defaulting to warn", policy)
        policy = "warn"

    self_petition = _normalized(request.beneficiary.full_name) == _normalized(
        request.petitioner.full_name
    )
    beneficiary_tokens = _name_tokens(request.beneficiary.full_name)
    petitioner_tokens = _name_tokens(request.petitioner.full_name)
    beneficiary_sources: list[dict] = []
    petitioner_found = self_petition
    if policy != "off":
        for source in extracted_sources:
            for page_number, page in enumerate(source.get("pages", []), start=1):
                normalized_page = _normalized(page)
                page_tokens = set(normalized_page.split())
                if _strong_name_match_tokens(beneficiary_tokens, page_tokens):
                    beneficiary_sources.append(
                        {
                            "file_hash": source["file_hash"],
                            "page_number": page_number,
                        }
                    )
                if _strong_name_match_tokens(petitioner_tokens, page_tokens):
                    petitioner_found = True

    conflicting: list[str] = []
    if policy == "strict":
        if not beneficiary_sources:
            raise BeneficiaryNotFound(
                "The beneficiary name was not found in the submitted evidence"
            )
        if not petitioner_found:
            raise PetitionerMismatch(
                "The petitioner name was not found in the submitted evidence"
            )
    elif policy == "warn":
        if not beneficiary_sources:
            conflicting.append("beneficiary_not_found")
            logger.warning(
                "beneficiary name not found in extracted evidence case_id=%s",
                request.case_id,
            )
        if not petitioner_found:
            conflicting.append("petitioner_not_found")
            logger.warning(
                "petitioner name not found in extracted evidence case_id=%s",
                request.case_id,
            )

    return IdentityRecord(
        case_id=request.case_id,
        attorney_name=request.preparer.full_name,
        attorney_address=asdict(request.preparer.address),
        beneficiary_name=request.beneficiary.full_name,
        beneficiary_address=asdict(request.beneficiary.address),
        petitioner_name=request.petitioner.full_name,
        petitioner_address=asdict(request.petitioner.address),
        service_center_name="",
        service_center_address={},
        self_petition=self_petition,
        supporting_sources=tuple(beneficiary_sources),
        conflicting_identities=tuple(conflicting),
    )
