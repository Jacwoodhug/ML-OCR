"""Download top English fonts from Google Fonts for OCR training data generation.

Usage:
    python scripts/download_google_fonts.py [--top-n 200] [--output data/fonts/google]
                                            [--out-json data/fonts/google_fonts.json]

Fetches the most popular Latin-subset fonts from the Google Fonts API, downloads
the regular-weight .ttf files, runs the same render test as collect_fonts.py to
reject symbol/icon fonts, and saves an approved font list in the same JSON format
expected by SynthGenerator.

Requires a Google Fonts API key in .env (GOOGLE_FONTS_API_KEY).
Free key: https://console.cloud.google.com/ → APIs & Services → Credentials.
No billing required for this API.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from dotenv import load_dotenv

from scripts.collect_fonts import render_test

load_dotenv()

GOOGLE_FONTS_API = "https://www.googleapis.com/webfonts/v1/webfonts"
REQUEST_DELAY = 0.05  # seconds between font file downloads


def fetch_font_list(api_key: str, top_n: int) -> list[dict]:
    """Fetch font metadata sorted by popularity from the Google Fonts API."""
    resp = requests.get(
        GOOGLE_FONTS_API,
        params={"key": api_key, "sort": "popularity"},
        timeout=30,
    )
    resp.raise_for_status()
    items = resp.json().get("items", [])
    print(f"Total fonts returned by API: {len(items)}")

    # Keep only fonts with a Latin subset
    latin = [f for f in items if "latin" in f.get("subsets", [])]
    print(f"Latin-subset fonts: {len(latin)}")

    return latin[:top_n]


def download_font(font: dict, out_dir: Path) -> Path | None:
    """Download the regular-weight TTF for a font. Returns the saved path or None on failure."""
    files = font.get("files", {})
    # Prefer regular weight; fall back to the first available variant
    url = files.get("regular") or next(iter(files.values()), None)
    if not url:
        return None

    # Google Fonts API sometimes returns http — force https
    url = url.replace("http://", "https://")

    safe_name = font["family"].replace(" ", "_")
    dest = out_dir / f"{safe_name}.ttf"

    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        dest.write_bytes(r.content)
        return dest
    except Exception as e:
        print(f"  WARN: failed to download {font['family']}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Download Google Fonts for OCR training")
    parser.add_argument("--top-n", type=int, default=200, help="Number of top fonts to fetch (default: 200)")
    parser.add_argument("--output", default="data/fonts/google", help="Directory to save .ttf files")
    parser.add_argument("--out-json", default="data/fonts/google_fonts.json", help="Output JSON path")
    args = parser.parse_args()

    api_key = os.environ.get("GOOGLE_FONTS_API_KEY", "")
    if not api_key:
        print("Error: GOOGLE_FONTS_API_KEY not set. Add it to your .env file.", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching top {args.top_n} Latin fonts from Google Fonts API...")
    fonts = fetch_font_list(api_key, args.top_n)
    print(f"Processing {len(fonts)} fonts...\n")

    approved = []
    skipped_download = 0
    failed_render = 0

    for i, font in enumerate(fonts, 1):
        family = font["family"]
        print(f"[{i:3d}/{len(fonts)}] {family}", end=" ... ", flush=True)

        path = download_font(font, out_dir)
        if path is None:
            print("download failed")
            skipped_download += 1
            continue

        if not render_test(str(path)):
            print("failed render test")
            path.unlink(missing_ok=True)
            failed_render += 1
            continue

        approved.append({
            "path": str(path),
            "filename": path.name,
            "family": family,
        })
        print("OK")
        time.sleep(REQUEST_DELAY)

    print(f"\nResults:")
    print(f"  Download failures : {skipped_download}")
    print(f"  Failed render test: {failed_render}")
    print(f"  Approved          : {len(approved)}")

    if not approved:
        print("No fonts approved — check your API key and network.", file=sys.stderr)
        sys.exit(1)

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(approved, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved font list to {out_json}")


if __name__ == "__main__":
    main()
