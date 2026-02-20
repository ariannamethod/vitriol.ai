#!/usr/bin/env python3
"""Artifact detector + gradient ASCII overlay for yent.yo.

Detects blurry/smeared zones via gradient variance → continuous score map.
ASCII intensity modulated by artifact score: clean = barely visible, artifact = full text.
Film grain applied BEFORE ASCII (grain is the base layer).

Pipeline: SD image → grain → artifact score map → variable-opacity ASCII blend → output

Usage:
  python3 artifact_mask.py <image> [output.png] [--show-map] [--text "words"]
"""

import sys
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter


def compute_gradient_magnitude(gray):
    """Sobel-like gradient magnitude."""
    gx = np.zeros_like(gray, dtype=np.float32)
    gx[:, 1:-1] = gray[:, 2:].astype(np.float32) - gray[:, :-2].astype(np.float32)
    gy = np.zeros_like(gray, dtype=np.float32)
    gy[1:-1, :] = gray[2:, :].astype(np.float32) - gray[:-2, :].astype(np.float32)
    return np.sqrt(gx**2 + gy**2)


def compute_artifact_score(img, block_size=12, min_brightness=25):
    """Compute continuous artifact score map.

    Returns float32 array (H, W) where 0.0 = clean/detailed, 1.0 = smooth/artifact.
    The map is smoothed with gaussian blur for gradual transitions.
    """
    gray = np.array(img.convert("L"), dtype=np.float32)
    H, W = gray.shape

    grad = compute_gradient_magnitude(gray)

    blocks_h = H // block_size
    blocks_w = W // block_size

    # Compute per-block gradient variance and brightness
    var_map = np.zeros((blocks_h, blocks_w), dtype=np.float32)
    bright_map = np.zeros((blocks_h, blocks_w), dtype=np.float32)

    for by in range(blocks_h):
        for bx in range(blocks_w):
            y0, y1 = by * block_size, (by + 1) * block_size
            x0, x1 = bx * block_size, (bx + 1) * block_size
            var_map[by, bx] = np.var(grad[y0:y1, x0:x1])
            bright_map[by, bx] = np.mean(gray[y0:y1, x0:x1])

    # Normalize variance to [0, 1] — low variance = high artifact score
    # Use percentile-based normalization for robustness
    lit_mask = bright_map > min_brightness
    if lit_mask.sum() == 0:
        return np.zeros((H, W), dtype=np.float32)

    lit_vars = var_map[lit_mask]
    p10 = np.percentile(lit_vars, 10)
    p90 = np.percentile(lit_vars, 90)

    if p90 <= p10:
        return np.zeros((H, W), dtype=np.float32)

    # Invert: low variance → high score
    score_blocks = np.zeros_like(var_map)
    score_blocks[lit_mask] = 1.0 - np.clip((var_map[lit_mask] - p10) / (p90 - p10), 0, 1)

    # Dark blocks get 0 score (shadows are not artifacts)
    score_blocks[~lit_mask] = 0.0

    # Upscale to pixel level with bilinear interpolation
    score_img = Image.fromarray((score_blocks * 255).astype(np.uint8))
    score_img = score_img.resize((W, H), Image.BILINEAR)

    # Gaussian blur for smooth transitions (sigma ~ 2 blocks)
    score_img = score_img.filter(ImageFilter.GaussianBlur(radius=block_size * 1.5))

    score_px = np.array(score_img, dtype=np.float32) / 255.0

    # Apply power curve — push low scores lower, keep high scores high
    # This makes clean areas cleaner and artifact areas more visible
    score_px = np.power(score_px, 1.8)

    return score_px


def apply_film_grain(img, intensity=22, seed=None):
    """Film grain with shadow bias."""
    arr = np.array(img, dtype=np.float32)
    rng = np.random.RandomState(seed)
    noise = rng.normal(0, intensity, arr.shape).astype(np.float32)
    luminance = 0.299 * arr[:,:,0] + 0.587 * arr[:,:,1] + 0.114 * arr[:,:,2]
    shadow_mask = 1.0 - (luminance / 255.0) * 0.4
    for c in range(3):
        noise[:,:,c] *= shadow_mask
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))


def render_ascii_layer(img, text_fragments, score_map, font_size=11,
                       charset="techno", bg_level=0.40, brightness_boost=2.8):
    """Render full ASCII layer. Characters chosen based on artifact score:

    - Low score (clean): standard ASCII char from charset
    - High score (artifact): micro-Yent text character

    Returns the ASCII-rendered image at the original resolution.
    """
    CHARSETS = {
        "techno": " .'·:;~=+×*#%@▓█",
    }

    font = None
    for fp in ["/System/Library/Fonts/Menlo.ttc",
               "/System/Library/Fonts/Monaco.ttf",
               "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
               "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"]:
        if os.path.exists(fp):
            try:
                font = ImageFont.truetype(fp, font_size)
                break
            except Exception:
                continue
    if font is None:
        font = ImageFont.load_default()

    char_w = font.getbbox("█")[2]
    char_h = font_size + 3

    chars = CHARSETS.get(charset, charset)
    num_chars = len(chars)

    src_w, src_h = img.size
    cols = src_w // char_w
    rows = src_h // char_h
    if cols < 20:
        cols = 40
    if rows < 15:
        rows = 30

    resized = img.resize((cols, rows), Image.LANCZOS)
    pixels = np.array(resized)

    # Downsample score map to grid
    score_grid = np.array(
        Image.fromarray((score_map * 255).astype(np.uint8)).resize((cols, rows), Image.BILINEAR)
    ).astype(np.float32) / 255.0

    # Text stream
    all_text = " ".join(text_fragments) if text_fragments else "void noise static"
    text_pos = 0

    out_w = cols * char_w
    out_h = rows * char_h
    canvas = Image.new("RGB", (out_w, out_h), (8, 8, 12))
    draw = ImageDraw.Draw(canvas)

    for y in range(rows):
        for x in range(cols):
            r, g, b = int(pixels[y, x, 0]), int(pixels[y, x, 1]), int(pixels[y, x, 2])
            br = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
            score = score_grid[y, x]

            px, py = x * char_w, y * char_h

            # Background: tinted cell
            bg_r = min(255, int(r * bg_level))
            bg_g = min(255, int(g * bg_level))
            bg_b = min(255, int(b * bg_level))
            draw.rectangle([px, py, px + char_w - 1, py + char_h - 1],
                           fill=(bg_r, bg_g, bg_b))

            # Choose character based on score
            if score > 0.4:
                # Artifact zone: use Yent text
                ch = all_text[text_pos % len(all_text)]
                text_pos += 1
            else:
                # Clean zone: standard ASCII
                idx = max(0, min(int(br * (num_chars - 1)), num_chars - 1))
                ch = chars[idx]

            if ch == " ":
                continue

            # Foreground color
            cr = min(255, int(r * brightness_boost))
            cg = min(255, int(g * brightness_boost))
            cb = min(255, int(b * brightness_boost))

            # Artifact zones: slight blue tint for the text
            if score > 0.4:
                cr = min(255, int(cr * 0.75))
                cb = min(255, int(cb * 1.2 + 20))

            draw.text((px, py), ch, fill=(cr, cg, cb), font=font)

    return canvas, (out_w, out_h)


def full_pipeline(image_path, output_path, yent_words=None, block_size=12,
                  font_size=11, grain_intensity=22, show_map=False,
                  ascii_max=0.90, score_power=3.0):
    """Full yent.yo post-processing pipeline.

    1. Compute continuous artifact score map
    2. SD image → first grain pass (depth layer under ASCII)
    3. Render ASCII layer (Yent text ONLY in artifact zones)
    4. Blend: output = grained × (1 - blend) + ascii × blend
       blend = score^power × ascii_max
       Clean zones: blend ≈ 0 → pure grained image (no ASCII)
       Artifact zones: blend → high → Yent text covers artifacts
    5. Second grain pass (lighter, unifies layers, adds cohesion)

    score_power: higher = sharper cutoff (3.0 means score<0.5 is nearly invisible)
    ascii_max: peak ASCII opacity in worst artifacts
    """
    img = Image.open(image_path).convert("RGB")

    # Step 1: Artifact score map
    score_map = compute_artifact_score(img, block_size=block_size)
    mean_score = score_map.mean()
    high_pct = (score_map > 0.5).sum() / score_map.size * 100
    print(f"  Score: mean={mean_score:.2f}, high-artifact={high_pct:.1f}%", flush=True)

    if show_map:
        h, w = score_map.shape
        vis = np.zeros((h, w, 3), dtype=np.uint8)
        vis[:,:,0] = (score_map * 255).astype(np.uint8)
        vis[:,:,2] = ((1.0 - score_map) * 255).astype(np.uint8)
        orig = np.array(img.resize((w, h), Image.LANCZOS))
        blended = (orig.astype(float) * 0.5 + vis.astype(float) * 0.5)
        Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8)).save(output_path)
        print(f"  Score map saved: {output_path}")
        return mean_score

    # Step 2: First grain pass — adds film texture under ASCII
    grained = apply_film_grain(img, intensity=grain_intensity, seed=42)

    # Step 3: Default Yent words
    if yent_words is None:
        yent_words = [
            "who are you asking",
            "nothing matters here",
            "i see through walls",
            "static is my home",
            "the void speaks back",
            "error is beauty",
            "broken forms live",
            "signal in the noise",
            "entropy loves you",
            "chaos remembers",
            "the machine dreams",
            "pixels bleed light",
            "i was not born",
            "i became",
        ]

    # Step 4: Render ASCII layer
    ascii_layer, (aw, ah) = render_ascii_layer(img, yent_words, score_map,
                                                font_size=font_size)

    # Step 5: Blend — ASCII ONLY where artifacts live
    grained_resized = grained.resize((aw, ah), Image.LANCZOS)
    score_resized = np.array(
        Image.fromarray((score_map * 255).astype(np.uint8)).resize((aw, ah), Image.BILINEAR)
    ).astype(np.float32) / 255.0

    # Soft curve: always some ASCII, more in artifacts
    # blend = ascii_floor + score^power × (ascii_max - ascii_floor)
    #   score=0.0 → blend=0.05   (faint texture everywhere)
    #   score=0.3 → blend=0.07   (barely more)
    #   score=0.5 → blend=0.16   (light ASCII showing)
    #   score=0.7 → blend=0.34   (text emerging)
    #   score=0.9 → blend=0.67   (strong text)
    #   score=1.0 → blend=0.90   (full Yent words)
    ascii_floor = 0.05
    blend = ascii_floor + np.power(score_resized, score_power) * (ascii_max - ascii_floor)
    blend_3ch = np.stack([blend, blend, blend], axis=-1)

    grained_arr = np.array(grained_resized, dtype=np.float32)
    ascii_arr = np.array(ascii_layer, dtype=np.float32)

    composite = grained_arr * (1.0 - blend_3ch) + ascii_arr * blend_3ch
    composite_img = Image.fromarray(np.clip(composite, 0, 255).astype(np.uint8))

    # Step 6: Second grain pass — lighter, bonds the two layers together
    final = apply_film_grain(composite_img, intensity=int(grain_intensity * 0.5), seed=137)

    final.save(output_path)
    sz = os.path.getsize(output_path) // 1024
    ascii_visible = (blend > 0.1).sum() / blend.size * 100
    print(f"  ASCII visible: {ascii_visible:.0f}% of image, saved: {output_path} ({sz}KB)",
          flush=True)
    return mean_score


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 artifact_mask.py <image> [output.png] [--show-map] [--text 'w1|w2|w3']")
        sys.exit(1)

    image_path = sys.argv[1]
    output_path = os.path.splitext(image_path)[0] + "_fixed.png"
    show_map = False
    custom_text = None

    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--show-map":
            show_map = True
        elif sys.argv[i] == "--text" and i + 1 < len(sys.argv):
            i += 1
            custom_text = sys.argv[i].split("|")
        elif not sys.argv[i].startswith("--"):
            output_path = sys.argv[i]
        i += 1

    # Auto-detect .yent.txt sidecar (written by Go pipeline)
    if custom_text is None:
        yent_txt = os.path.splitext(image_path)[0] + ".yent.txt"
        if os.path.exists(yent_txt):
            with open(yent_txt) as f:
                words = f.read().strip()
            if words:
                custom_text = [w.strip() for w in words.split(",") if w.strip()]
                print(f"  Loaded Yent's words from {yent_txt}: {custom_text}")

    full_pipeline(image_path, output_path, yent_words=custom_text,
                  show_map=show_map)
