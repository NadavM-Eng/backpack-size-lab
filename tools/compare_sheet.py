#!/usr/bin/env python3
"""
Put reference photos next to renders of your model, one row per view, so differences are easy to see.
Plain Python with Pillow (and numpy for --blend).

    python tools/compare_sheet.py --out work/compare/<id>/round1.png \\
        photo_front.jpg work/renders/<id>/<id>_front.png front \\
        photo_side.jpg  work/renders/<id>/<id>_side.png  side

Arguments after --out come in threes: photo, render, label. Each pair is scaled to the same height.
With --blend two more columns show the two laid over each other and their outlines compared (photo
and render each cropped to the bag and scaled to the same height, never stretched): orange where only
the photo has bag, blue where only the render has. Any orange or blue band wider than about 1 cm is a
silhouette error to fix first. Crop photos so that only the bag is in them (no person, no other bags).
"""
import argparse
import os

from PIL import Image, ImageDraw


def bag_mask(im, tol=150):
    """Boolean mask of the pixels that differ from the background (estimated row by row from the left and
    right edges, so a studio gradient doesn't count), with a thin floor shadow left out."""
    import numpy as np
    a = np.asarray(im.convert("RGB")).astype(int)
    bg = np.median(np.concatenate([a[:, :6], a[:, -6:]], axis=1), axis=1)[:, None, :]
    m = (np.abs(a - bg).sum(-1) > tol)
    m[:, :6] = m[:, -6:] = False
    rows = m.sum(1) > 0
    if not rows.any():
        return m
    # a column belongs to the bag when a fair share of the bag's rows are set in it (a floor shadow is a thin band)
    cols = m.sum(0) > 0.06 * rows.sum()
    rows = m[:, cols].sum(1) > 0.06 * max(1, cols.sum()) if cols.any() else rows
    out = np.zeros_like(m)
    out[np.ix_(rows, cols)] = m[np.ix_(rows, cols)]
    return out


def crop_to_bag(im, tol=150):
    """Crop to the bag (see bag_mask)."""
    import numpy as np
    m = bag_mask(im, tol)
    ys, xs = np.where(m)
    if not len(xs):
        return im
    return im.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))


def centred(im, w, fill=(255, 255, 255)):
    """Paste an image in the middle of a canvas w wide, without stretching it."""
    c = Image.new("RGB", (w, im.height), fill)
    c.paste(im, ((w - im.width) // 2, 0))
    return c


def outline_diff(p, r):
    """Photo-only pixels orange, render-only pixels blue, both grey: where the two outlines disagree."""
    import numpy as np
    mp, mr = bag_mask(centred(p, max(p.width, r.width), p.getpixel((2, 2)))), bag_mask(centred(r, max(p.width, r.width), r.getpixel((2, 2))))
    out = np.full(mp.shape + (3,), 250, np.uint8)
    out[mp & mr] = (150, 150, 150)
    out[mp & ~mr] = (230, 120, 40)
    out[~mp & mr] = (40, 110, 220)
    return Image.fromarray(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--height", type=int, default=520)
    ap.add_argument("--blend", action="store_true", help="add columns with photo and render overlaid and their outlines compared")
    ap.add_argument("items", nargs="+", help="photo render label, photo render label, ...")
    a = ap.parse_args()
    if len(a.items) % 3:
        raise SystemExit("give photo, render and label in threes")
    rows = []
    for i in range(0, len(a.items), 3):
        photo, render, label = a.items[i:i + 3]
        p, r = Image.open(photo).convert("RGB"), Image.open(render).convert("RGB")
        H = a.height
        if a.blend:
            p, r = crop_to_bag(p), crop_to_bag(r)
        sc = lambda im: im.resize((max(1, round(im.width * H / im.height)), H))
        cells = [sc(p), sc(r)]
        if a.blend:
            # both cropped to the bag and scaled to the same height, then laid over each other unstretched
            w = max(c.width for c in cells)
            pa, ra = centred(cells[0], w, cells[0].getpixel((2, 2))), centred(cells[1], w, cells[1].getpixel((2, 2)))
            cells.append(Image.blend(pa, ra, 0.5))
            cells.append(outline_diff(cells[0], cells[1]))
        rows.append((label, cells))
    W = max(sum(c.width for c in cells) + 16 * (len(cells) + 1) for _, cells in rows)
    Ht = sum(a.height + 44 for _ in rows) + 16
    sheet = Image.new("RGB", (W, Ht), (236, 236, 233))
    d = ImageDraw.Draw(sheet)
    y = 16
    for label, cells in rows:
        heads = ["photo", "render"] + (["overlaid", "outlines: orange photo only, blue render only"] if a.blend else [])
        d.text((16, y), label, fill=(20, 20, 20))
        x = 16
        for c, h in zip(cells, heads):
            d.text((x, y + 14), h, fill=(90, 90, 90))
            sheet.paste(c, (x, y + 28))
            x += c.width + 16
        y += a.height + 44
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    sheet.save(a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
