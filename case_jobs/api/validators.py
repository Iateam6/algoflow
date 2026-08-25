from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlsplit

from django.conf import settings

from case_jobs.api.schemas import (
    Address,
    Exhibit,
    ExhibitItem,
    GenerationRequest,
    Party,
    Preparer,
    SourceFile,
)
from case_jobs.exceptions import ValidationError
from case_jobs.integrations.webhook_client import resolve_webhook_url
from case_jobs.registry import get_adapter


CASE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
REQUEST_FIELDS = {
    "case_id",
    "document_type",
    "document_slug",
    "preparer",
    "beneficiary",
    "petitioner",
}
OPTIONAL_REQUEST_FIELDS = {"files", "exhibits"}
EXHIBIT_REQUEST_FIELDS = {
    "case_id",
    "document_type",
    "document_slug",
    "preparer",
    "beneficiary",
    "petitioner",
    "exhibits",
}


def _required_string(container: dict[str, Any], field: str, context: str) -> str:
    value = container.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{context}.{field} is required")
    return value.strip()


def _optional_string(container: dict[str, Any], field: str) -> str | None:
    value = container.get(field)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValidationError(f"{field} must be a string or null")
    return value.strip() or None


def _address(value: Any, context: str) -> Address:
    if not isinstance(value, dict):
        raise ValidationError(f"{context} is required")
    allowed = {"line_1", "line_2", "city", "state", "province", "postal_code", "country"}
    if set(value) - allowed:
        raise ValidationError(f"{context} contains unsupported fields")

    return Address(
        line_1=_optional_string(value, "line_1"),
        line_2=_optional_string(value, "line_2"),
        city=_optional_string(value, "city"),
        state=_optional_string(value, "state"),
        province=_optional_string(value, "province"),
        postal_code=_optional_string(value, "postal_code"),
        country=_optional_string(value, "country"),
    )


def _party(payload: dict[str, Any], field: str) -> Party:
    value = payload.get(field)
    if not isinstance(value, dict):
        raise ValidationError(f"{field} is required")
    if set(value) != {"full_name", "address"}:
        raise ValidationError(f"{field} must contain full_name and address only")
    return Party(
        full_name=_required_string(value, "full_name", field),
        address=_address(value.get("address"), f"{field}.address"),
    )


def _preparer(payload: dict[str, Any]) -> Preparer:
    value = payload.get("preparer")
    if not isinstance(value, dict):
        raise ValidationError("preparer is required")
    if set(value) != {"full_name", "firm_name", "address"}:
        raise ValidationError("preparer must contain full_name, firm_name, and address only")
    return Preparer(
        full_name=_required_string(value, "full_name", "preparer"),
        firm_name=_required_string(value, "firm_name", "preparer"),
        address=_address(value.get("address"), "preparer.address"),
    )


def _exhibit_preparer(payload: dict[str, Any]) -> Preparer:
    value = payload.get("preparer")
    if not isinstance(value, dict):
        raise ValidationError("preparer is required")
    if set(value) != {"full_name", "firm_name", "address"}:
        raise ValidationError(
            "preparer must contain full_name, firm_name, and address only"
        )
    return Preparer(
        full_name=_required_string(value, "full_name", "preparer"),
        firm_name=_required_string(value, "firm_name", "preparer"),
        address=_address(value.get("address"), "preparer.address"),
    )


def _source_url(value: Any, context: str) -> SourceFile:
    if not isinstance(value, dict) or set(value) != {"url"}:
        raise ValidationError(f"{context} must contain only url")
    url = _required_string(value, "url", context)
    parsed = urlsplit(url)
    if parsed.scheme.lower() != "https" or not parsed.hostname:
        raise ValidationError(f"{context}.url must be HTTPS")
    if parsed.username or parsed.password:
        raise ValidationError(f"{context}.url cannot contain credentials")
    return SourceFile(url=url)


def _files(value: Any) -> tuple[SourceFile, ...]:
    if not isinstance(value, list) or not value:
        raise ValidationError("at least one file is required")
    if len(value) > settings.MAX_FILES_PER_JOB:
        raise ValidationError("file count exceeds the configured limit")

    result: list[SourceFile] = []
    for index, item in enumerate(value):
        result.append(_source_url(item, f"files[{index}]"))
    return tuple(result)


def _optional_files(value: Any) -> tuple[SourceFile, ...]:
    if value is None or value == []:
        return ()
    return _files(value)


def _exhibits(value: Any) -> tuple[tuple[Exhibit, ...], tuple[SourceFile, ...]]:
    if not isinstance(value, list) or not value:
        raise ValidationError("at least one exhibit is required")

    exhibits: list[Exhibit] = []
    source_files: list[SourceFile] = []
    seen_numbers: set[int] = set()
    for exhibit_index, raw_exhibit in enumerate(value):
        context = f"exhibits[{exhibit_index}]"
        if not isinstance(raw_exhibit, dict) or set(raw_exhibit) != {
            "number",
            "title",
            "items",
        }:
            raise ValidationError(f"{context} must contain number, title, and items only")
        number = raw_exhibit.get("number")
        if isinstance(number, bool) or not isinstance(number, int) or number <= 0:
            raise ValidationError(f"{context}.number must be a positive integer")
        if number in seen_numbers:
            raise ValidationError(f"{context}.number must be unique")
        seen_numbers.add(number)

        title = raw_exhibit.get("title")
        if not isinstance(title, str):
            raise ValidationError(f"{context}.title must be a string")
        raw_items = raw_exhibit.get("items")
        if not isinstance(raw_items, list) or not raw_items:
            raise ValidationError(f"{context}.items must be a non-empty list")

        items: list[ExhibitItem] = []
        for item_index, raw_item in enumerate(raw_items):
            item_context = f"{context}.items[{item_index}]"
            if not isinstance(raw_item, dict):
                raise ValidationError(f"{item_context} must be an object")
            item_type = raw_item.get("type")
            if item_type not in {"form", "document", "file"}:
                raise ValidationError(
                    f"{item_context}.type must be form, document, or file"
                )
            name = _required_string(raw_item, "name", item_context)
            if item_type in {"form", "document"}:
                if set(raw_item) != {"type", "name", "url"}:
                    raise ValidationError(
                        f"{item_context} must contain type, name, and url only"
                    )
                source = _source_url({"url": raw_item.get("url")}, item_context)
                source_files.append(source)
                items.append(ExhibitItem(type=item_type, name=name, url=source.url))
                continue

            if set(raw_item) != {"type", "name", "files"}:
                raise ValidationError(
                    f"{item_context} must contain type, name, and files only"
                )
            raw_files = raw_item.get("files")
            if not isinstance(raw_files, list) or not raw_files:
                raise ValidationError(f"{item_context}.files must be a non-empty list")
            item_files = tuple(
                _source_url(raw_file, f"{item_context}.files[{file_index}]")
                for file_index, raw_file in enumerate(raw_files)
            )
            source_files.extend(item_files)
            items.append(
                ExhibitItem(type=item_type, name=name, files=item_files)
            )

        exhibits.append(
            Exhibit(number=number, title=title.strip(), items=tuple(items))
        )

    return tuple(exhibits), tuple(source_files)


def _optional_exhibits(value: Any) -> tuple[Exhibit, ...]:
    if not isinstance(value, list) or not value:
        return ()

    exhibits: list[Exhibit] = []
    seen_numbers: set[int] = set()
    for exhibit_index, raw_exhibit in enumerate(value):
        if not isinstance(raw_exhibit, dict):
            continue
        number = raw_exhibit.get("number")
        if (
            isinstance(number, bool)
            or not isinstance(number, int)
            or number <= 0
            or number in seen_numbers
        ):
            continue
        title = raw_exhibit.get("title")
        if title is None:
            title = ""
        if not isinstance(title, str):
            continue
        raw_items = raw_exhibit.get("items")
        if not isinstance(raw_items, list):
            continue

        items: list[ExhibitItem] = []
        for item_index, raw_item in enumerate(raw_items):
            if not isinstance(raw_item, dict):
                continue
            item_type = raw_item.get("type")
            name = raw_item.get("name")
            if item_type not in {"form", "document", "file"}:
                continue
            if not isinstance(name, str) or not name.strip():
                continue
            context = f"exhibits[{exhibit_index}].items[{item_index}]"
            if item_type in {"form", "document"}:
                if set(raw_item) != {"type", "name", "url"}:
                    continue
                try:
                    source = _source_url({"url": raw_item.get("url")}, context)
                except ValidationError:
                    continue
                items.append(
                    ExhibitItem(type=item_type, name=name.strip(), url=source.url)
                )
                continue

            if set(raw_item) != {"type", "name", "files"}:
                continue
            raw_files = raw_item.get("files")
            if not isinstance(raw_files, list):
                continue
            valid_files: list[SourceFile] = []
            for file_index, raw_file in enumerate(raw_files):
                try:
                    valid_files.append(
                        _source_url(raw_file, f"{context}.files[{file_index}]")
                    )
                except ValidationError:
                    continue
            if valid_files:
                items.append(
                    ExhibitItem(
                        type=item_type,
                        name=name.strip(),
                        files=tuple(valid_files),
                    )
                )

        if items:
            seen_numbers.add(number)
            exhibits.append(
                Exhibit(number=number, title=title.strip(), items=tuple(items))
            )

    return tuple(exhibits)


def _validate_tn_exhibit_request(payload: dict[str, Any], visa_type: str) -> GenerationRequest:
    if set(payload) != EXHIBIT_REQUEST_FIELDS:
        raise ValidationError("request contains missing or unsupported fields")
    case_id = _required_string(payload, "case_id", "request")
    if not CASE_ID_PATTERN.fullmatch(case_id):
        raise ValidationError("case_id contains unsupported characters")
    document_slug = _required_string(payload, "document_slug", "request")
    if document_slug != "exhibit-list":
        raise ValidationError("request.document_slug must be exhibit-list")
    adapter = get_adapter(visa_type)
    if "Exhibit List" not in adapter.supported_document_types:
        raise ValidationError(f"unsupported document_type for {visa_type}")
    if not settings.WEBHOOK_ROOT_URL:
        raise ValidationError("WEBHOOK_ROOT_URL is not configured")
    resolve_webhook_url(settings.WEBHOOK_ROOT_URL)
    exhibits, files = _exhibits(payload.get("exhibits"))
    return GenerationRequest(
        case_id=case_id,
        document_type="Exhibit List",
        document_slug=document_slug,
        preparer=_exhibit_preparer(payload),
        beneficiary=_party(payload, "beneficiary"),
        petitioner=_party(payload, "petitioner"),
        files=files,
        exhibits=exhibits,
    )


def validate_generation_request(payload: Any, visa_type: str) -> GenerationRequest:
    if not isinstance(payload, dict):
        raise ValidationError("request body must be a JSON object")
    if (
        visa_type.strip().lower() == "tn"
        and payload.get("document_type") == "Exhibit List"
        and "exhibits" in payload
    ):
        return _validate_tn_exhibit_request(payload, visa_type)
    payload_fields = set(payload)
    if not REQUEST_FIELDS.issubset(payload_fields) or payload_fields - (
        REQUEST_FIELDS | OPTIONAL_REQUEST_FIELDS
    ):
        raise ValidationError("request contains missing or unsupported fields")

    case_id = _required_string(payload, "case_id", "request")
    if not CASE_ID_PATTERN.fullmatch(case_id):
        raise ValidationError("case_id contains unsupported characters")

    document_type = _required_string(payload, "document_type", "request")
    try:
        adapter = get_adapter(visa_type)
    except LookupError as exc:
        raise ValidationError(str(exc)) from exc
    if document_type not in adapter.supported_document_types:
        raise ValidationError(f"unsupported document_type for {visa_type}")

    if not settings.WEBHOOK_ROOT_URL:
        raise ValidationError("WEBHOOK_ROOT_URL is not configured")
    resolve_webhook_url(settings.WEBHOOK_ROOT_URL)

    standard_exhibits = (
        ()
        if document_type == "Exhibit List"
        else _optional_exhibits(payload.get("exhibits"))
    )
    files = _optional_files(payload.get("files"))
    if not standard_exhibits and not files:
        raise ValidationError("at least one file or exhibit is required")
    return GenerationRequest(
        case_id=case_id,
        document_type=document_type,
        document_slug=_required_string(payload, "document_slug", "request"),
        preparer=_preparer(payload),
        beneficiary=_party(payload, "beneficiary"),
        petitioner=_party(payload, "petitioner"),
        files=files,
        exhibits=standard_exhibits,
    )
