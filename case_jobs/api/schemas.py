from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Address:
    line_1: str | None
    line_2: str | None
    city: str | None
    state: str | None
    province: str | None
    postal_code: str | None
    country: str | None


@dataclass(frozen=True)
class Party:
    full_name: str
    address: Address


@dataclass(frozen=True)
class Preparer:
    full_name: str
    firm_name: str
    address: Address


@dataclass(frozen=True)
class SourceFile:
    url: str


@dataclass(frozen=True)
class ExhibitItem:
    type: str
    name: str
    url: str | None = None
    files: tuple[SourceFile, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"type": self.type, "name": self.name}
        if self.type in {"form", "document"}:
            result["url"] = self.url
        else:
            result["files"] = [asdict(source) for source in self.files]
        return result


@dataclass(frozen=True)
class Exhibit:
    number: int
    title: str
    items: tuple[ExhibitItem, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "number": self.number,
            "title": self.title,
            "items": [item.to_dict() for item in self.items],
        }


@dataclass(frozen=True)
class GenerationRequest:
    case_id: str
    document_type: str
    document_slug: str
    preparer: Preparer
    beneficiary: Party
    petitioner: Party
    files: tuple[SourceFile, ...]
    exhibits: tuple[Exhibit, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        if (
            self.exhibits
            and self.document_type == "Exhibit List"
            and self.document_slug == "exhibit-list"
        ):
            return {
                "case_id": self.case_id,
                "document_type": self.document_type,
                "document_slug": self.document_slug,
                "preparer": {
                    "full_name": self.preparer.full_name,
                    "firm_name": self.preparer.firm_name,
                    "address": asdict(self.preparer.address),
                },
                "beneficiary": asdict(self.beneficiary),
                "petitioner": asdict(self.petitioner),
                "exhibits": [exhibit.to_dict() for exhibit in self.exhibits],
            }
        result = {
            "case_id": self.case_id,
            "document_type": self.document_type,
            "document_slug": self.document_slug,
            "preparer": asdict(self.preparer),
            "beneficiary": asdict(self.beneficiary),
            "petitioner": asdict(self.petitioner),
            "files": [asdict(source) for source in self.files],
        }
        if self.exhibits:
            result["exhibits"] = [exhibit.to_dict() for exhibit in self.exhibits]
        return result


@dataclass(frozen=True)
class TenantPrincipal:
    tenant_id: str
    subject: str
    claims: dict[str, Any]
