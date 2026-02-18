#!/usr/bin/env python3
"""
Vitriol — Dataset Collector
Downloads art from Wikimedia Commons for GAN training.

Styles:
  caricature  — Honoré Daumier, Thomas Nast, James Gillray
  graffiti    — street art, graffiti, murals
  propaganda  — Soviet constructivist posters, Rodchenko, Lissitzky
  pixel       — pixel art, 8-bit game art
"""

import os
import json
import time
import hashlib
import argparse
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.parse import quote
from PIL import Image
from io import BytesIO

DATA_DIR = Path(__file__).parent / "data"

STYLES = {
    "caricature": {
        "daumier": [
            "Category:Prints by Honoré Daumier",
            "Category:Caricatures by Honoré Daumier",
            "Category:Lithographs by Honoré Daumier",
        ],
        "nast": [
            "Category:Thomas Nast",
            "Category:Cartoons by Thomas Nast",
        ],
        "gillray": [
            "Category:James Gillray",
            "Category:Prints by James Gillray",
        ],
    },
    "graffiti": {
        "street_art": [
            "Category:Graffiti in Berlin",
            "Category:Graffiti in London",
            "Category:Graffiti in New York City",
            "Category:Street art in Berlin",
            "Category:Street art in London",
        ],
        "murals": [
            "Category:Murals in Berlin",
            "Category:Murals in Los Angeles",
        ],
    },
    "propaganda": {
        "soviet_posters": [
            "Category:Soviet propaganda posters",
            "Category:Soviet World War II posters",
            "Category:Constructivism (art)",
        ],
        "rodchenko": [
            "Category:Alexander Rodchenko",
        ],
        "lissitzky": [
            "Category:El Lissitzky",
        ],
    },
    "pixel": {
        "pixel_art": [
            "Category:Pixel art",
            "Category:Pixel art characters",
        ],
        "game_sprites": [
            "Category:Video game sprites",
            "Category:8-bit art",
        ],
    },
}

# Flat view for backward compat
CATEGORIES = {}
for style_cats in STYLES.values():
    CATEGORIES.update(style_cats)

HEADERS = {"User-Agent": "Vitriol/1.0 (ariannamethod; dataset collection)"}
IMG_SIZE = 128
THUMB_WIDTH = 512  # download at this width, then resize


def fetch_json(url):
    req = Request(url, headers=HEADERS)
    with urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def get_category_images(category, limit=200):
    """Fetch image URLs from a Wikimedia Commons category."""
    base = "https://commons.wikimedia.org/w/api.php"
    images = []
    cont = None

    while len(images) < limit:
        params = (
            f"?action=query&generator=categorymembers"
            f"&gcmtitle={quote(category)}"
            f"&gcmtype=file&gcmlimit=50"
            f"&prop=imageinfo&iiprop=url|mime"
            f"&iiurlwidth={THUMB_WIDTH}"
            f"&format=json"
        )
        if cont:
            params += f"&gcmcontinue={cont}"

        url = base + params
        data = fetch_json(url)

        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            info = page.get("imageinfo", [{}])[0]
            mime = info.get("mime", "")
            if mime.startswith("image/") and mime != "image/svg+xml":
                thumb = info.get("thumburl") or info.get("url")
                if thumb:
                    images.append(thumb)

        cont = data.get("continue", {}).get("gcmcontinue")
        if not cont:
            break
        time.sleep(0.5)  # be polite

    return images[:limit]


def download_and_resize(url, out_path, size=IMG_SIZE):
    """Download image, resize to square, save as PNG."""
    try:
        req = Request(url, headers=HEADERS)
        with urlopen(req, timeout=30) as resp:
            img_data = resp.read()
        img = Image.open(BytesIO(img_data)).convert("RGB")
        # center crop to square
        w, h = img.size
        s = min(w, h)
        left = (w - s) // 2
        top = (h - s) // 2
        img = img.crop((left, top, left + s, top + s))
        img = img.resize((size, size), Image.LANCZOS)
        img.save(out_path, "PNG")
        return True
    except Exception as e:
        print(f"  SKIP {url[:60]}... ({e})")
        return False


def collect(style=None, sources=None, per_category=200):
    DATA_DIR.mkdir(exist_ok=True)

    # Determine which sources to collect
    if style:
        style_cats = STYLES.get(style, {})
        if sources:
            to_collect = {s: style_cats[s] for s in sources if s in style_cats}
        else:
            to_collect = style_cats
    elif sources:
        to_collect = {s: CATEGORIES[s] for s in sources if s in CATEGORIES}
    else:
        to_collect = CATEGORIES

    total = 0
    collected_dirs = []
    for source, cats in to_collect.items():
        print(f"\n=== {source.upper()} ({len(cats)} categories) ===")
        source_dir = DATA_DIR / source
        source_dir.mkdir(exist_ok=True)
        collected_dirs.append(source_dir)

        for cat in cats:
            print(f"\n  {cat}")
            urls = get_category_images(cat, limit=per_category)
            print(f"  Found {len(urls)} images")

            for i, url in enumerate(urls):
                name = hashlib.md5(url.encode()).hexdigest()[:12] + ".png"
                out = source_dir / name
                if out.exists():
                    continue
                if download_and_resize(url, out):
                    total += 1
                    if (i + 1) % 10 == 0:
                        print(f"  Downloaded {i+1}/{len(urls)}")
                time.sleep(0.3)

    # Merge into style directory for training
    style_name = style or "all"
    merge_dir = DATA_DIR / style_name
    merge_dir.mkdir(exist_ok=True)
    for d in collected_dirs:
        for f in d.glob("*.png"):
            dest = merge_dir / f"{d.name}_{f.name}"
            if not dest.exists():
                os.link(f, dest)

    final_count = len(list(merge_dir.glob("*.png")))
    print(f"\n{'='*40}")
    print(f"Total images collected: {total}")
    print(f"Combined dataset: {final_count} images in {merge_dir}")
    print(f"Ready for training!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vitriol dataset collector")
    parser.add_argument("--style", choices=list(STYLES.keys()),
                        help="Style to collect (caricature, graffiti)")
    parser.add_argument("--sources", nargs="+",
                        help="Specific sources within style")
    parser.add_argument("--per-category", type=int, default=200,
                        help="Max images per category (default: 200)")
    parser.add_argument("--size", type=int, default=128,
                        help="Output image size (default: 128)")
    args = parser.parse_args()

    if args.size != IMG_SIZE:
        IMG_SIZE = args.size

    collect(style=args.style, sources=args.sources, per_category=args.per_category)
