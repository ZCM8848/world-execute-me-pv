"""Rasterise Natural Earth 110m land polygons into a compact ASCII land mask.

Source: Natural Earth, 1:110m physical vectors (public domain).
    https://www.naturalearthdata.com/
    ne_110m_land.geojson

The mask is a simple equirectangular bitmap used by the globe emblem in
art.py: row 0 is +90 deg latitude, column 0 is -180 deg longitude.

Usage:
    python tools/landmask.py <ne_110m_land.geojson> <out.npz> [W H]
"""
import json
import math
import sys

import numpy as np


def load_rings(path):
    with open(path, encoding="utf-8") as f:
        gj = json.load(f)
    rings = []
    for feat in gj["features"]:
        g = feat["geometry"]
        if g is None:
            continue
        if g["type"] == "Polygon":
            polys = [g["coordinates"]]
        elif g["type"] == "MultiPolygon":
            polys = g["coordinates"]
        else:
            continue
        for poly in polys:
            for ring in poly:
                lon = np.asarray([p[0] for p in ring], dtype=np.float64)
                lat = np.asarray([p[1] for p in ring], dtype=np.float64)
                if len(lon) < 4:
                    continue
                d = np.diff(lon)
                d = (d + 180.0) % 360.0 - 180.0
                lon = np.concatenate(([lon[0]], lon[0] + np.cumsum(d)))
                rings.append((lon, lat))
    return rings


def rasterise(rings, W, H):
    mask = np.zeros((H, W), dtype=np.uint8)
    for j in range(H):
        latc = 90.0 - (j + 0.5) * 180.0 / H
        xs = []
        for lon, lat in rings:
            y1 = lat[:-1]
            y2 = lat[1:]
            cross = (y1 > latc) != (y2 > latc)
            if not cross.any():
                continue
            idx = np.nonzero(cross)[0]
            x1 = lon[idx]
            x2 = lon[idx + 1]
            ys1 = y1[idx]
            ys2 = y2[idx]
            xs.extend(x1 + (latc - ys1) * (x2 - x1) / (ys2 - ys1))
        if not xs:
            continue
        xs.sort()
        for k in range(0, len(xs) - 1, 2):
            a, b = xs[k], xs[k + 1]
            c0 = int(math.floor((a + 180.0) / 360.0 * W))
            c1 = int(math.ceil((b + 180.0) / 360.0 * W))
            for c in range(c0, c1):
                mask[j, c % W] = 1
    return mask


def preview(mask):
    H, W = mask.shape
    step_y, step_x = 2, 4
    rows = []
    for j in range(0, H, step_y):
        line = ""
        for i in range(0, W, step_x):
            blk = mask[j:j + step_y, i:i + step_x]
            line += "#" if blk.mean() > 0.35 else ("." if blk.mean() > 0.1 else " ")
        rows.append(line)
    return "\n".join(rows)


def main():
    src = sys.argv[1]
    dst = sys.argv[2]
    W = int(sys.argv[3]) if len(sys.argv) > 3 else 360
    H = int(sys.argv[4]) if len(sys.argv) > 4 else 180
    rings = load_rings(src)
    mask = rasterise(rings, W, H)
    np.savez_compressed(dst, mask=mask)
    print(f"rings={len(rings)} shape={mask.shape} land={mask.mean() * 100:.1f}% -> {dst}")
    if "--preview" in sys.argv:
        print(preview(mask))


if __name__ == "__main__":
    main()
