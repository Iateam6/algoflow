INVALID_FILE_URLS_ERROR = "No valid file URLs provided"
DISALLOWED_FILE_FIELDS = {"name", "slug", "slag"}


def build_url_source_entries(files: object) -> list[dict[str, str]]:
    if not isinstance(files, list) or not files:
        return []

    source_entries = []
    for index, file_entry in enumerate(files, start=1):
        if not isinstance(file_entry, dict):
            return []

        if DISALLOWED_FILE_FIELDS.intersection(file_entry) or set(file_entry) != {"url"}:
            return []

        url = file_entry.get("url")
        if not isinstance(url, str) or not url.strip():
            return []

        source_entries.append(
            {
                "name": f"source-{index}",
                "url": url.strip(),
            }
        )

    return source_entries
