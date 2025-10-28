import os
import time
import hashlib
import aiohttp
import aiofiles
import asyncio
from PyPDF2 import PdfMerger
from typing import Tuple, Optional
import environ

# Load environment variables
env = environ.Env()
env.read_env()

API_KEY = env("API_KEY")
BASE_URL = "https://api.pdf.co/v1"


async def upload_to_pdfco(file_path: str) -> str:
    """
    Asynchronously uploads a local file to PDF.co and returns the uploaded file URL.
    """
    url = f"{BASE_URL}/file/upload"
    headers = {"x-api-key": API_KEY}

    async with aiohttp.ClientSession() as session:
        async with aiofiles.open(file_path, "rb") as f:
            file_data = await f.read()

        form = aiohttp.FormData()
        form.add_field("file", file_data, filename=os.path.basename(file_path))

        async with session.post(url, headers=headers, data=form) as resp:
            if resp.status != 200:
                raise Exception(f"Upload failed: {resp.status} {resp.reason}")
            data = await resp.json()
            if data.get("error"):
                raise Exception(f"Upload error: {data.get('message')}")
            return data["url"]


async def convert_to_pdf(file_url: str, file_ext: str, output_name: str) -> str:
    """
    Converts DOC/DOCX/IMAGE/XLS/HTML/TXT/CSV to PDF using PDF.co asynchronously.
    Skips files already in PDF format.
    Includes:
    - Fast I/O with aiohttp
    - Retry logic for transient network issues
    - Error-safe JSON parsing
    """
    ext = file_ext.lower().strip()
    print(f"[CONVERT] Checking file type '{ext}' for {output_name}...")

    # ✅ Strict skip for PDFs
    if ext == ".pdf":
        print(f"[SKIP] '{output_name}' already PDF, skipping conversion.")
        return file_url

    # ✅ Endpoint selection
    endpoint_map = {
        ".doc": "/pdf/convert/from/doc",
        ".docx": "/pdf/convert/from/doc",
        ".jpg": "/pdf/convert/from/image",
        ".jpeg": "/pdf/convert/from/image",
        ".png": "/pdf/convert/from/image",
        ".bmp": "/pdf/convert/from/image",
        ".tiff": "/pdf/convert/from/image",
        ".tif": "/pdf/convert/from/image",
        ".xls": "/pdf/convert/from/xls",
        ".xlsx": "/pdf/convert/from/xls",
        ".txt": "/pdf/convert/from/html",
        ".csv": "/pdf/convert/from/html",
        ".html": "/pdf/convert/from/html",
    }

    endpoint = endpoint_map.get(ext)
    if not endpoint:
        raise ValueError(f"Unsupported file type: {ext}")

    url = f"{BASE_URL}{endpoint}"
    headers = {"x-api-key": API_KEY}
    payload = {"name": f"{output_name}.pdf", "url": file_url}

    # ✅ Retry logic with exponential backoff
    for attempt in range(3):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, data=payload, timeout=60) as resp:
                    if resp.status != 200:
                        raise Exception(f"HTTP {resp.status}: {resp.reason}")

                    try:
                        data = await resp.json()
                    except Exception as e:
                        raise Exception(f"Invalid JSON response: {e}")

                    if not data.get("url"):
                        raise Exception(f"PDF.co failed: {data}")

                    print(f"[SUCCESS] Converted '{output_name}' → {data['url']}")
                    return data["url"]

        except Exception as e:
            print(f"[RETRY {attempt + 1}/3] Conversion failed for '{output_name}': {e}")
            await asyncio.sleep(2 ** attempt)  # exponential backoff

    raise Exception(f"[FAIL] Conversion permanently failed for '{output_name}' after retries.")


async def download_file(url: str, dest_path: str):
    """
    Downloads a file and saves it locally.
    """
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=60) as resp:
            if resp.status != 200:
                raise Exception(f"Failed to download {url}: {resp.status} {resp.reason}")
            async with aiofiles.open(dest_path, "wb") as f:
                async for chunk in resp.content.iter_chunked(8192):
                    await f.write(chunk)


async def _download_and_hash(url: str, dest_path: str) -> Tuple[str, Optional[str]]:
    """
    Downloads a file asynchronously and returns (path, md5_hash or None on error).
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=60) as resp:
                if resp.status != 200:
                    print(f"[ERROR] Failed to download {url}: {resp.status} {resp.reason}")
                    return dest_path, None

                md5 = hashlib.md5()
                async with aiofiles.open(dest_path, "wb") as f:
                    async for chunk in resp.content.iter_chunked(16384):
                        md5.update(chunk)
                        await f.write(chunk)

                return dest_path, md5.hexdigest()

    except Exception as e:
        print(f"[ERROR] Exception downloading {url}: {e}")
        return dest_path, None


async def merge_pdfs(pdf_urls: list, output_path: str):
    """
    Optimized async version:
    - Downloads all PDFs concurrently
    - Skips duplicates by URL and content hash
    - Uses efficient chunk-based hashing (no full file reads)
    - Cleans up temporary files automatically
    """
    merger = PdfMerger()
    seen_hashes = set()
    seen_urls = set()
    temp_files = []

    print(f"[MERGE] Starting optimized merge for {len(pdf_urls)} PDFs...")

    # Deduplicate URLs first
    unique_urls = [u for u in pdf_urls if u not in seen_urls and not seen_urls.add(u)]
    print(f"[DEBUG] Deduplicated {len(pdf_urls) - len(unique_urls)} duplicate URLs. {len(unique_urls)} unique remain.")

    # Prepare download tasks
    tasks = []
    for i, url in enumerate(unique_urls, start=1):
        local_temp = f"{output_path}_part_{i}.pdf"
        temp_files.append(local_temp)
        tasks.append(_download_and_hash(url, local_temp))

    # Download concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results
    for result, url in zip(results, unique_urls):
        if isinstance(result, Exception):
            print(f"[ERROR] Exception for {url}: {result}")
            continue

        path, file_hash = result
        if not file_hash:
            print(f"[SKIP] Skipping failed download: {url}")
            continue

        if file_hash in seen_hashes:
            print(f"[SKIP] Duplicate content skipped: {url}")
            try:
                os.remove(path)
            except OSError:
                pass
            continue

        seen_hashes.add(file_hash)
        try:
            merger.append(path)
            print(f"[APPEND] Added: {path}")
        except Exception as e:
            print(f"[ERROR] Failed to append {path}: {e}")

    # Write merged file
    print(f"\n[MERGE] Writing final merged PDF: {output_path}")
    merger.write(output_path)
    merger.close()
    print("[MERGE] Merge completed successfully!")

    # Cleanup
    for file in temp_files:
        if os.path.exists(file):
            try:
                os.remove(file)
            except Exception:
                pass
    print("[CLEANUP] Temporary files removed.")
