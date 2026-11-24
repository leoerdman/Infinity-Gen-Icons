#!/usr/bin/env python3
"""Build svg/*.svg (vector trace) and compress icons/*.png to PNG8.

Icons are pure white; the shape lives only in alpha: a solid outline (255) and,
in most icons, a translucent fill (51 = 20 %). Each level is traced separately
with potrace so the SVG keeps the two-tone look.

Needs: potrace, pngquant, Python with Pillow and numpy.
Usage: python3 tools/build.py [--png8] [--check]
"""
import json, hashlib, re, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
SCALE = 4          # trace an upscaled alpha mask: sub-pixel accurate curves
FILL_ALPHA = 51    # the translucent fill level used across the pack


def trace(mask, w, h):
    """Binary mask (upscaled) -> SVG path data in original pixel units."""
    with tempfile.TemporaryDirectory() as d:
        pbm, svg = Path(d, "m.pbm"), Path(d, "m.svg")
        Image.fromarray(np.where(mask, 0, 255).astype(np.uint8)).convert("1").save(pbm)
        subprocess.run(["potrace", "-s", "--flat", "-t", str(4 * SCALE), "-a", "1.0",
                        "-O", "0.4", "-u", "2", "-o", svg, pbm], check=True)
        text = svg.read_text()
    # potrace writes paths in 1/2 px of the upscaled bitmap, y flipped
    paths = " ".join(re.findall(r'<path d="([^"]+)"', text, re.S))
    paths = re.sub(r"\s+", " ", paths).strip()
    k = 1 / (2 * SCALE)
    return paths, f"translate(0,{h}) scale({k:g},{-k:g})"


def build_svg(png):
    im = Image.open(png).convert("RGBA")
    w, h = im.size
    big = np.asarray(im.getchannel("A").resize((w * SCALE, h * SCALE), Image.LANCZOS))
    # Fill areas are flat alpha FILL_ALPHA; the anti-aliased rim of an outline only
    # passes through that level in a thin band, which the opening removes. So the
    # fill never reaches the outer edge (no grey halo), even in small dots.
    flat = (big >= FILL_ALPHA - 16) & (big <= FILL_ALPHA + 16)
    core = Image.fromarray(flat.astype(np.uint8) * 255)
    core = core.filter(ImageFilter.MinFilter(7)).filter(ImageFilter.MaxFilter(7))
    layers = []
    if np.asarray(core).any():
        # grow the fill 2 px under the outline (far less than its width): no seam
        grown = np.asarray(core.filter(ImageFilter.MaxFilter(4 * SCALE + 1))) > 0
        fill = (np.asarray(core) > 0) | (grown & (big > FILL_ALPHA // 2))
        layers.append((fill, f' fill-opacity="{FILL_ALPHA / 255:.2f}"'))
        layers.append((big > (FILL_ALPHA + 255) // 2, ""))
    else:
        layers.append((big > 127, ""))
    body = []
    for mask, attr in layers:
        d, tf = trace(mask, w, h)
        if d:
            body.append(f'<path transform="{tf}"{attr} d="{d}"/>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
            f'width="{w}" height="{h}" fill="#fff">' + "".join(body) + "</svg>\n")


def png8(png):
    subprocess.run(["pngquant", "--force", "--strip", "--speed", "1",
                    "--quality", "80-95", "--ext", ".png", png], check=False)


def check(png, svg):
    """Mean and worst-case alpha error of the SVG rendered back at original size."""
    import cairosvg, io
    ref = np.asarray(Image.open(png).convert("RGBA"))[..., 3].astype(int)
    out = cairosvg.svg2png(bytestring=svg.encode())
    got = np.asarray(Image.open(io.BytesIO(out)).convert("RGBA"))[..., 3].astype(int)
    diff = np.abs(ref - got)
    return diff.mean(), (diff > 96).sum() / max(1, (ref > 0).sum())


def main():
    args = set(sys.argv[1:])
    manifest = json.loads((ROOT / "icons.json").read_text())
    (ROOT / "svg").mkdir(exist_ok=True)
    worst = []
    for icon in manifest["icons"]:
        png = ROOT / icon["file"]
        svg = build_svg(png)  # trace the original before any lossy step
        out = ROOT / "svg" / f"{icon['id']}.svg"
        out.write_text(svg)
        if "--check" in args:
            worst.append((*check(png, svg), icon["id"]))
        if "--png8" in args:
            png8(png)
        icon["svg"] = f"svg/{icon['id']}.svg"
        icon["sha256"] = hashlib.sha256(png.read_bytes()).hexdigest()
        icon["svg_sha256"] = hashlib.sha256(out.read_bytes()).hexdigest()
    (ROOT / "icons.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n")
    if worst:
        worst.sort(reverse=True)
        print("mean alpha error (0-255), share of badly-off pixels — worst 10:")
        for m, bad, i in worst[:10]:
            print(f"  {i:20} {m:5.2f}  {bad:.4f}")
        print(f"pack mean: {np.mean([w[0] for w in worst]):.2f}")


if __name__ == "__main__":
    main()
