"""Enumerate and filter Windows system fonts for OCR training data generation.

Usage:
    python scripts/collect_fonts.py [--output data/fonts/fonts.json]

Scans C:\\Windows\\Fonts for .ttf/.otf files, applies a blocklist of known
symbol/icon fonts, then runs an automated render test to reject fonts that
don't produce distinct Latin glyphs. Saves the approved font list to JSON.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Hard blocklist: known symbol / icon / dingbat font families
# Matched as case-insensitive substrings against font file names
# ---------------------------------------------------------------------------
BLOCKLIST_SUBSTRINGS = [
    "wingding",
    "webding",
    "symbol",
    "marlett",
    "segmdl2",
    "segoe mdl2",
    "hololens mdl2",
    "segoe fluent",
    "segoe ui emoji",
    "segoe chess",
    "bookshelf",
    "directions mt",
    "holidays mt",
    "keystrokes mt",
    "map symbols",
    "parties mt",
    "vacation mt",
    "transport mt",
    "refspecialty",
    "cariadings",
    "almanac mt",
    "mczee",
    "temp installer",
    "mtsymbol",
    "mt extra",
]


def is_blocklisted(font_path: str) -> bool:
    """Check if a font file name matches any blocklist substring."""
    name_lower = Path(font_path).stem.lower()
    return any(sub in name_lower for sub in BLOCKLIST_SUBSTRINGS)


# ---------------------------------------------------------------------------
# Automated render test
# ---------------------------------------------------------------------------
TEST_STRING = "ABCabc123!?@#"
RENDER_SIZE = 40  # font size for the test render
GLYPH_CANVAS = 60  # canvas size per glyph


def render_test(font_path: str) -> bool:
    """Render test characters and verify they produce distinct, non-blank glyphs.

    Returns True if the font passes (i.e. looks like a real text font).
    """
    try:
        font = ImageFont.truetype(font_path, RENDER_SIZE)
    except Exception:
        return False

    glyph_images = []
    for ch in TEST_STRING:
        img = Image.new("L", (GLYPH_CANVAS, GLYPH_CANVAS), 0)
        draw = ImageDraw.Draw(img)
        draw.text((5, 5), ch, fill=255, font=font)
        glyph_images.append(img)

    # Check 1: at least some glyphs are non-blank
    pixel_sums = [sum(img.getdata()) for img in glyph_images]
    non_blank = sum(1 for s in pixel_sums if s > 100)
    if non_blank < len(TEST_STRING) * 0.7:
        return False

    # Check 2: glyphs are diverse (not all identical like symbol fonts)
    # Compare each glyph to the first — at least 40% should differ meaningfully
    if len(glyph_images) < 2:
        return False

    ref_data = list(glyph_images[0].getdata())
    diff_count = 0
    for img in glyph_images[1:]:
        img_data = list(img.getdata())
        pixel_diff = sum(abs(a - b) for a, b in zip(ref_data, img_data))
        if pixel_diff > 500:  # threshold for "meaningfully different"
            diff_count += 1

    if diff_count < (len(glyph_images) - 1) * 0.4:
        return False

    return True


def collect_fonts(fonts_dir: str = r"C:\Windows\Fonts") -> list[dict]:
    """Scan fonts directory and return list of approved font metadata."""
    approved = []
    extensions = {".ttf", ".otf"}

    if not os.path.isdir(fonts_dir):
        print(f"Warning: fonts directory not found: {fonts_dir}")
        return approved

    font_files = [
        os.path.join(fonts_dir, f)
        for f in os.listdir(fonts_dir)
        if Path(f).suffix.lower() in extensions
    ]

    print(f"Found {len(font_files)} .ttf/.otf files in {fonts_dir}")

    blocked = 0
    failed_render = 0

    for font_path in sorted(font_files):
        fname = Path(font_path).name

        if is_blocklisted(font_path):
            blocked += 1
            continue

        if not render_test(font_path):
            failed_render += 1
            continue

        # Extract font family name from the font itself
        try:
            pil_font = ImageFont.truetype(font_path, 20)
            family = getattr(pil_font, "getname", lambda: (fname, ""))[0]
            if callable(family):
                family = pil_font.getname()[0]
        except Exception:
            family = Path(font_path).stem

        approved.append({
            "path": font_path,
            "filename": fname,
            "family": family,
        })

    print(f"Blocklisted: {blocked}")
    print(f"Failed render test: {failed_render}")
    print(f"Approved: {len(approved)}")

    return approved


def main():
    parser = argparse.ArgumentParser(description="Collect and filter system fonts for OCR training")
    parser.add_argument("--output", default="data/fonts/fonts.json", help="Output JSON path")
    parser.add_argument("--fonts-dir", default=r"C:\Windows\Fonts", help="System fonts directory")
    args = parser.parse_args()

    approved = collect_fonts(args.fonts_dir)

    if not approved:
        print("No fonts passed filtering!", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(approved, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(approved)} fonts to {out_path}")


if __name__ == "__main__":
    main()
