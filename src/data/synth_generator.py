"""On-the-fly synthetic OCR image generator.

Generates training images with random text rendered over varied backgrounds
(solid colors, gradients, texture crops) using random system fonts.
"""

import json
import math
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from src.data.alphabet import CHARS

# ---------------------------------------------------------------------------
# Vocabulary for word-mode text generation
# ---------------------------------------------------------------------------
# Common English words for realistic-looking training text
COMMON_WORDS = [
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
    "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
    "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
    "people", "into", "year", "your", "good", "some", "could", "them", "see",
    "other", "than", "then", "now", "look", "only", "come", "its", "over",
    "think", "also", "back", "after", "use", "two", "how", "our", "work",
    "first", "well", "way", "even", "new", "want", "because", "any", "these",
    "give", "day", "most", "us", "Hello", "World", "Error", "File", "Open",
    "Save", "Print", "Name", "Email", "Phone", "Address", "City", "State",
    "Price", "Total", "Date", "Order", "Item", "Code", "Number", "Page",
    "www", "http", "com", "org", "net", "info", "data", "test", "user",
    "2024", "2025", "2026", "$19.99", "#100", "50%", "@admin", "v2.0",
]


class SynthGenerator:
    """Generates synthetic OCR training images on the fly."""

    def __init__(
        self,
        fonts_json: str | None = None,
        backgrounds_dir: str | None = None,
        img_height: int = 32,
        img_min_width: int = 32,
        img_max_width: int = 800,
        min_text_len: int = 1,
        max_text_len: int = 50,
        word_mode_prob: float = 0.7,
        bg_solid_prob: float = 0.3,
        bg_gradient_prob: float = 0.3,
        bg_texture_prob: float = 0.4,
        bw: bool = False,
        font_paths: list[str] | None = None,
    ):
        # Load font list
        if font_paths:
            self.fonts = list(font_paths)
        elif fonts_json:
            with open(fonts_json, "r", encoding="utf-8") as f:
                self.fonts = json.load(f)
        else:
            raise ValueError("Must provide either fonts_json or font_paths")
        if not self.fonts:
            raise ValueError("No fonts loaded")

        # Load background image paths
        self.bg_images: list[str] = []
        if backgrounds_dir and os.path.isdir(backgrounds_dir):
            exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
            self.bg_images = [
                os.path.join(backgrounds_dir, f)
                for f in os.listdir(backgrounds_dir)
                if Path(f).suffix.lower() in exts
            ]
        self.has_textures = len(self.bg_images) > 0

        self.img_height = img_height
        self.img_min_width = img_min_width
        self.img_max_width = img_max_width
        self.min_text_len = min_text_len
        self.max_text_len = max_text_len
        self.word_mode_prob = word_mode_prob

        self.bw = bw

        # Normalize background probabilities
        total = bg_solid_prob + bg_gradient_prob + bg_texture_prob
        self.bg_solid_prob = bg_solid_prob / total
        self.bg_gradient_prob = bg_gradient_prob / total
        self.bg_texture_prob = bg_texture_prob / total

    # ------------------------------------------------------------------
    # Text generation
    # ------------------------------------------------------------------
    def _random_text(self) -> str:
        """Generate random text string."""
        length = random.randint(self.min_text_len, self.max_text_len)

        if random.random() < self.word_mode_prob:
            # Word mode: join random words until we reach target length
            words = []
            current_len = 0
            while current_len < length:
                word = random.choice(COMMON_WORDS)
                # Randomly apply case variations
                r = random.random()
                if r < 0.1:
                    word = word.upper()
                elif r < 0.2:
                    word = word.capitalize()
                words.append(word)
                current_len += len(word) + 1  # +1 for space
            text = " ".join(words)
            # Trim to target length range
            text = text[:length]
            # Optionally add trailing punctuation
            if random.random() < 0.2 and len(text) < self.max_text_len:
                text += random.choice([".", "!", "?", ",", ";", ":"])
        else:
            # Random char mode: sample directly from charset
            text = "".join(random.choice(CHARS) for _ in range(length))

        return text.strip() or "a"  # never return empty

    # ------------------------------------------------------------------
    # Background generation
    # ------------------------------------------------------------------
    def _random_color(self) -> tuple[int, int, int]:
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    def _luminance(self, color: tuple[int, int, int]) -> float:
        return (0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]) / 255.0

    def _pick_contrasting_color(self, bg_color: tuple[int, int, int]) -> tuple[int, int, int]:
        """Pick a random color with luminance at least 40% from bg_color's luminance."""
        bg_lum = self._luminance(bg_color)
        for _ in range(50):
            color = self._random_color()
            if abs(self._luminance(color) - bg_lum) >= 0.40:
                return color
        return (0, 0, 0) if bg_lum > 0.5 else (255, 255, 255)

    def _solid_background(self, width: int, height: int) -> Image.Image:
        return Image.new("RGB", (width, height), self._random_color())

    def _gradient_background(self, width: int, height: int) -> Image.Image:
        """Generate a linear gradient background."""
        img = Image.new("RGB", (width, height))
        c1 = self._random_color()
        c2 = self._random_color()
        angle = random.uniform(0, 2 * math.pi)

        pixels = np.zeros((height, width, 3), dtype=np.uint8)
        # Create coordinate grids
        y_coords, x_coords = np.mgrid[0:height, 0:width]
        # Project onto gradient direction
        proj = (x_coords * math.cos(angle) + y_coords * math.sin(angle))
        proj_min, proj_max = proj.min(), proj.max()
        if proj_max > proj_min:
            t = (proj - proj_min) / (proj_max - proj_min)
        else:
            t = np.zeros_like(proj)

        for i in range(3):
            pixels[:, :, i] = (c1[i] * (1 - t) + c2[i] * t).astype(np.uint8)

        img = Image.fromarray(pixels)
        return img

    def _texture_background(self, width: int, height: int) -> Image.Image:
        """Random crop from a DTD texture image, resized to target dimensions."""
        if not self.has_textures:
            return self._solid_background(width, height)

        bg_path = random.choice(self.bg_images)
        try:
            bg = Image.open(bg_path).convert("RGB")
        except Exception:
            return self._solid_background(width, height)

        bw, bh = bg.size
        # If background is too small, resize it up
        if bw < width or bh < height:
            scale = max(width / bw, height / bh) * 1.1
            bg = bg.resize((int(bw * scale), int(bh * scale)), Image.BILINEAR)
            bw, bh = bg.size

        # Random crop
        x = random.randint(0, max(0, bw - width))
        y = random.randint(0, max(0, bh - height))
        crop = bg.crop((x, y, x + width, y + height))
        return crop

    def _make_background(self, width: int, height: int) -> Image.Image:
        """Generate a random background image."""
        r = random.random()
        if r < self.bg_solid_prob:
            return self._solid_background(width, height)
        elif r < self.bg_solid_prob + self.bg_gradient_prob:
            return self._gradient_background(width, height)
        else:
            return self._texture_background(width, height)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def generate(self) -> tuple[Image.Image, str]:
        """Generate a single synthetic OCR sample.

        Returns:
            (image, text) — PIL Image (RGB, height=self.img_height) and the
            ground truth text string.
        """
        text = self._random_text()

        # Pick a random font and size
        font_info = random.choice(self.fonts)
        # Font size: random, scaled relative to image height
        # Use a larger canvas then resize down to img_height
        render_height = random.randint(48, 128)
        font_size = int(render_height * random.uniform(0.5, 0.9))

        try:
            font = ImageFont.truetype(font_info["path"], font_size)
        except Exception:
            # Fallback to default
            font = ImageFont.load_default()

        # Measure text bounding box
        dummy = Image.new("RGB", (1, 1))
        draw = ImageDraw.Draw(dummy)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        # Create canvas with padding
        pad_x = random.randint(2, max(3, text_h // 4))
        pad_y = random.randint(2, max(3, text_h // 6))
        canvas_w = text_w + 2 * pad_x
        canvas_h = text_h + 2 * pad_y

        # Generate background and pick text color
        bg = self._make_background(canvas_w, canvas_h)
        if self.bw:
            # Sample mean color of the text region to enforce contrast against whatever bg was generated
            x0 = max(0, pad_x)
            y0 = max(0, pad_y)
            x1 = min(canvas_w, pad_x + text_w)
            y1 = min(canvas_h, pad_y + text_h)
            region = np.array(bg.crop((x0, y0, x1, y1)))
            mean_color = tuple(int(v) for v in region.mean(axis=(0, 1)))
            text_color = self._pick_contrasting_color(mean_color)
        else:
            text_color = self._random_color()

        # Draw text
        draw = ImageDraw.Draw(bg)
        draw.text((pad_x - bbox[0], pad_y - bbox[1]), text, fill=text_color, font=font)

        # Resize to target height, preserving aspect ratio
        aspect = canvas_w / max(canvas_h, 1)
        target_w = int(self.img_height * aspect)
        target_w = max(self.img_min_width, min(self.img_max_width, target_w))

        img = bg.resize((target_w, self.img_height), Image.BILINEAR)

        if self.bw:
            img = img.convert("L").convert("RGB")

        return img, text
