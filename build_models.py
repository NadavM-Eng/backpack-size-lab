#!/usr/bin/env python3
"""
Parametric backpack generator (Blender bpy, headless).

    python3 build_models.py                 # build every bag in bags.json (+ previews)
    python3 build_models.py --only <id>     # build a single bag
    python3 build_models.py --no-render     # skip preview renders
    python3 build_models.py --render-only <id>   # (internal) render previews of an existing GLB

Everything is driven by each bag's `shape` object (see DEFAULTS for every field and its
fallback).  Geometry is computed in centimetres in Blender's Z-up frame with the bag FRONT
facing -Y; the glTF exporter (export_yup) turns that into glTF Y-up / front +Z, and vertices
are scaled to metres on mesh creation.  Origin = bottom-centre of the body footprint.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree

HERE = os.path.dirname(os.path.abspath(__file__))
BAGS_JSON = os.path.join(HERE, "docs", "bags.json")
MODELS_DIR = os.path.join(HERE, "docs", "models")
PREVIEW_DIR = os.path.join(HERE, "previews")
CM = 0.01  # cm -> m

# --------------------------------------------------------------------------------------
# Spec defaults (missing keys fall back to these; explicit null disables a feature)
# --------------------------------------------------------------------------------------
DEFAULTS = {
    "height": 45.0, "width": 30.0, "body_depth": 15.0, "depth_external": None,
    "corner_radius": 5.0, "top_round": 0.4, "front_bulge": 1.0, "taper_top": 0.96,
    "fabric": {"color": "#1b1c1e", "roughness": 0.75, "weave": "nylon"},
    "trim_color": None, "zip_color": "#111112", "pull_style": "cord", "pull_color": "#2b2b2d",
    "hardware_color": None, "logo_color": None,
    "front_pocket": None, "lower_pocket": None, "top_pocket": False,
    "clamshell_zip": True, "side_bottle_pocket": "none", "side_zip_pocket": False,
    "side_handle": False, "d_rings": False,
    "top_handle": {"type": "webbing", "width": 11.0},
    "straps": {"width": 7.0, "thickness": 2.0, "length": 58.0, "sternum": True, "load_lifters": False},
    "back_panel": {"pads": 2, "thickness": 1.8, "top_laptop_zip": False},
    "luggage_strap": False, "logo": "none",
}
SUB_DEFAULTS = {
    "front_pocket": {"type": "panel", "top": 0.8, "bottom": 0.35, "inset": 2.4, "zip": "top-arc"},
    "lower_pocket": {"top": 0.38, "bottom": 0.08, "inset": 2.6, "zip": "top-straight"},
}


def merge_shape(shape):
    out = {}
    for k, dv in DEFAULTS.items():
        v = shape.get(k, dv)
        if isinstance(dv, dict) and isinstance(v, dict):
            v = {**dv, **v}
        if k in SUB_DEFAULTS:
            if v is True:
                v = dict(SUB_DEFAULTS[k])
            elif isinstance(v, dict):
                v = {**SUB_DEFAULTS[k], **v}
            elif not v:
                v = None
        out[k] = v
    for k, v in shape.items():  # keep unknown extras
        out.setdefault(k, v)
    if out["depth_external"] is None:
        out["depth_external"] = out["body_depth"] + out["back_panel"]["thickness"] + out["straps"]["thickness"] + 1
    if out["top_handle"] is True:
        out["top_handle"] = dict(DEFAULTS["top_handle"])
    if out["straps"] is True or out["straps"] is None:
        out["straps"] = dict(DEFAULTS["straps"])
    if out["back_panel"] in (None, False):
        out["back_panel"] = {"pads": 0, "thickness": 0.0}
    sb = str(out["side_bottle_pocket"] or "none").lower()
    out["side_bottle_pocket"] = sb if sb in ("left", "right", "both") else "none"
    return out


# --------------------------------------------------------------------------------------
# small math helpers
# --------------------------------------------------------------------------------------
def nrm(v):
    v = np.asarray(v, float)
    return v / np.clip(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12, None)


def sstep(e0, e1, x):
    t = np.clip((np.asarray(x, float) - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def hex_rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])


def srgb2lin(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def shade(hexcol, f):
    return "#" + "".join("%02x" % int(round(np.clip(x * f, 0, 1) * 255)) for x in hex_rgb(hexcol))


def resample(P, ds, closed=False, extra=()):
    P = np.asarray(P, float)
    ex = [np.asarray(e, float) for e in extra]
    if closed:
        P = np.vstack([P, P[:1]])
        ex = [np.vstack([e, e[:1]]) for e in ex]
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    L = s[-1]
    n = max(2, int(round(L / ds)) + 1)
    t = np.linspace(0, L, n + 1)[:-1] if closed else np.linspace(0, L, n)
    res = lambda A: np.stack([np.interp(t, s, A[:, i]) for i in range(A.shape[1])], 1)
    return [res(P)] + [res(e) for e in ex]


def smooth_rows(A, k=3, closed=False, iters=2):
    A = np.asarray(A, float).copy()
    if len(A) < 3:
        return A
    for _ in range(iters):
        if closed:
            A = sum(np.roll(A, i, 0) for i in range(-k, k + 1)) / (2 * k + 1)
        else:
            pad = np.concatenate([np.repeat(A[:1], k, 0), A, np.repeat(A[-1:], k, 0)])
            A = sum(pad[i:i + len(A)] for i in range(2 * k + 1)) / (2 * k + 1)
    return A


def arclen(P, closed=False):
    P = np.asarray(P, float)
    if closed:
        P = np.vstack([P, P[:1]])
    return np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])


# profiles (closed, CCW in (u, v))
def prof_rrect(w, h, r, seg=3):
    r = max(1e-3, min(r, w / 2 * 0.999, h / 2 * 0.999))
    pts = []
    for cx, cy, a0 in ((w / 2 - r, h / 2 - r, 0), (-w / 2 + r, h / 2 - r, 90),
                       (-w / 2 + r, -h / 2 + r, 180), (w / 2 - r, -h / 2 + r, 270)):
        for i in range(seg + 1):
            a = math.radians(a0 + 90 * i / seg)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return np.array(pts)


def prof_circle(r, n=8):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([r * np.cos(a), r * np.sin(a)], 1)


def prof_strap(w, h, n=32):
    """Padded strap section: domed outer face (+v), flatter body-facing face, soft rounded edges."""
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    c, s = np.cos(a), np.sin(a)
    u = w / 2 * np.sign(c) * np.abs(c) ** 0.55
    v = np.where(s >= 0, 0.62 * h * np.abs(s) ** 0.9, -0.38 * h * np.abs(s) ** 0.5)
    return np.stack([u, v], 1)


def prof_pillow(w, h, n=28, p=3.2):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    c, s = np.cos(a), np.sin(a)
    return np.stack([w / 2 * np.sign(c) * np.abs(c) ** (2 / p), h / 2 * np.sign(s) * np.abs(s) ** (2 / 2.6)], 1)


# --------------------------------------------------------------------------------------
# mesh accumulation
# --------------------------------------------------------------------------------------
def grid_faces(R, C, wrap_c, wrap_r):
    rr = R if wrap_r else R - 1
    cc = C if wrap_c else C - 1
    r, c = np.meshgrid(np.arange(rr), np.arange(cc), indexing="ij")
    r, c = r.ravel(), c.ravel()
    corners = [(r, c), (r, c + 1), (r + 1, c + 1), (r + 1, c)]
    F = np.stack([(a % R) * C + (b % C) for a, b in corners], 1)
    return F, corners


def box_uv(V, F, tile):
    P = V[F]
    n = np.abs(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]))
    if F.shape[1] == 4:
        n = n + np.abs(np.cross(P[:, 2] - P[:, 0], P[:, 3] - P[:, 0]))
    ax = n.argmax(1)
    U = np.where(ax[:, None] == 0, P[:, :, 1], P[:, :, 0])
    W = np.where(ax[:, None] == 2, P[:, :, 1], P[:, :, 2])
    return np.stack([U, W], -1) / tile


class Model:
    def __init__(self, bid):
        self.bid = bid
        self.objs = {}  # name -> dict(mat, V list, groups list, nv)

    def add(self, mat, V, groups, orient="closed", ref=None, obj=None):
        V = np.asarray(V, float)
        groups = [(np.asarray(F, np.int64), np.asarray(L, float)) for F, L in groups if len(F)]
        if orient == "closed":
            vol = 0.0
            for F, _ in groups:
                P = V[F]
                for k in range(1, F.shape[1] - 1):
                    vol += np.einsum("ij,ij->i", P[:, 0], np.cross(P[:, k], P[:, k + 1])).sum()
            if vol < 0:
                groups = [(F[:, ::-1], L[:, ::-1]) for F, L in groups]
        elif orient == "ref" and ref is not None:
            ref = np.asarray(ref, float)
            tot = 0.0
            for F, _ in groups:
                P = V[F]
                fn = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
                tot += np.einsum("ij,ij->i", fn, ref[F].mean(1)).sum()
            if tot < 0:
                groups = [(F[:, ::-1], L[:, ::-1]) for F, L in groups]
        name = obj or mat
        o = self.objs.setdefault(name, {"mat": mat, "V": [], "G": [], "nv": 0})
        o["V"].append(V)
        o["G"].extend((F + o["nv"], L) for F, L in groups)
        o["nv"] += len(V)

    def all_vertices(self, exclude=()):
        vs = [np.concatenate(o["V"]) for k, o in self.objs.items() if k not in exclude and o["V"]]
        return np.concatenate(vs) if vs else np.zeros((0, 3))


def sweep(M, mat, P, ups, prof, closed=False, caps=True, scale=None, tile=2.0, obj=None):
    P = np.asarray(P, float)
    ups = np.broadcast_to(np.asarray(ups, float), P.shape)
    n = len(P)
    if closed:
        T = np.roll(P, -1, 0) - np.roll(P, 1, 0)
    else:
        T = np.gradient(P, axis=0)
    T = nrm(T)
    N = nrm(ups - np.einsum("ij,ij->i", ups, T)[:, None] * T)
    B = np.cross(N, T)
    prof = np.asarray(prof, float)
    m = len(prof)
    sx = np.ones(n) if scale is None else np.asarray(scale)[:, 0]
    sy = np.ones(n) if scale is None else np.asarray(scale)[:, 1]
    G = (P[:, None, :] + B[:, None, :] * (prof[None, :, 0, None] * sx[:, None, None])
         + N[:, None, :] * (prof[None, :, 1, None] * sy[:, None, None]))
    s = arclen(P, closed)
    if not closed:
        s = s[:n]
    pp = arclen(prof, True)
    UV = np.stack(np.meshgrid(s, pp, indexing="ij"), -1) / tile
    V = G.reshape(-1, 3)
    F, corners = grid_faces(n, m, True, closed)
    L = np.stack([UV[a, b] for a, b in corners], 1)
    groups = [(F, L)]
    if caps and not closed:
        Vc = [V]
        base = len(V)
        for end, idx in ((0, 0), (1, n - 1)):
            ring = np.arange(m) + idx * m
            c = G[idx].mean(0)
            Vc.append(c[None])
            ci = base
            base += 1
            j = np.arange(m)
            if end == 0:
                Ft = np.stack([np.full(m, ci), ring[(j + 1) % m], ring[j]], 1)
            else:
                Ft = np.stack([np.full(m, ci), ring[j], ring[(j + 1) % m]], 1)
            Pu = np.vstack([prof, [[0, 0]]])
            uvl = np.stack([Pu[np.full(m, m)], Pu[Ft[:, 1] - idx * m], Pu[Ft[:, 2] - idx * m]], 1) / tile
            groups.append((Ft, uvl))
        V = np.concatenate(Vc)
    M.add(mat, V, groups, orient="closed" if (closed or caps) else None, obj=obj)
    return G, T, N, B


def rbox(M, mat, center, T, N, L, w, h, r=0.12, seg=2, obj=None, prof=None):
    T = nrm(T)
    P = np.array([center - T * L / 2, center + T * L / 2])
    pr = prof if prof is not None else prof_rrect(w, h, r, seg)
    sweep(M, mat, P, [N, N], pr, caps=True, tile=1.0, obj=obj)


def tube(M, mat, P, ups, r, n=8, closed=False, obj=None, tile=1.0):
    return sweep(M, mat, P, ups, prof_circle(r, n), closed=closed, caps=not closed, tile=tile, obj=obj)


# --------------------------------------------------------------------------------------
# raycast surface
# --------------------------------------------------------------------------------------
class Surf:
    def __init__(self):
        self.chunks = []
        self._bvh = None

    def add(self, V, F_list):
        self.chunks.append((np.asarray(V, float), [np.asarray(F) for F in F_list]))
        self._bvh = None

    def bvh(self):
        if self._bvh is None:
            verts, polys, off = [], [], 0
            for V, Fl in self.chunks:
                verts.extend(V.tolist())
                for F in Fl:
                    polys.extend((F + off).tolist())
                off += len(V)
            self._bvh = BVHTree.FromPolygons(verts, polys, all_triangles=False)
        return self._bvh

    def ray(self, o, d):
        b = self.bvh()
        loc, n, _, _ = b.ray_cast(Vector(o), Vector(d))
        if loc is None:
            loc, n, _, _ = b.find_nearest(Vector(np.asarray(o) + np.asarray(d) * 500.0))
        n = np.array(n)
        if np.dot(n, d) > 0:
            n = -n
        return np.array(loc), n


def view_ray(view, q):
    a, b = float(q[0]), float(q[1])
    return {
        "front": ((a, -500, b), (0, 1, 0)), "back": ((a, 500, b), (0, -1, 0)),
        "left": ((-500, a, b), (1, 0, 0)), "right": ((500, a, b), (-1, 0, 0)),
        "top": ((a, b, 500), (0, 0, -1)),
    }[view]


def project(surf, view, Q):
    out = [surf.ray(*view_ray(view, q)) for q in np.asarray(Q, float)]
    return np.array([o[0] for o in out]), np.array([o[1] for o in out])


def section(surf, axis, val, c2, angles):
    """Points where the body meets plane <axis>=val, by radial rays toward centre c2 (in-plane)."""
    P, N = [], []
    for a in angles:
        ca, sa = math.cos(a), math.sin(a)
        if axis == "y":
            o = (c2[0] + 300 * ca, val, c2[1] + 300 * sa); d = (-ca, 0, -sa)
        else:
            o = (val, c2[0] + 300 * ca, c2[1] + 300 * sa); d = (0, -ca, -sa)
        p, n = surf.ray(o, d)
        P.append(p); N.append(n)
    return np.array(P), np.array(N)


# --------------------------------------------------------------------------------------
# the bag builder
# --------------------------------------------------------------------------------------
class Bag:
    def __init__(self, bag):
        self.bid = bag["id"]
        self.S = S = merge_shape(bag.get("shape", {}))
        self.H, self.W, self.D = float(S["height"]), float(S["width"]), float(S["body_depth"])
        self.bulge = float(S["front_bulge"] or 0)
        self.A0 = self.W / 2
        self.B0 = (self.D - self.bulge) / 2
        self.rc = min(float(S["corner_radius"]), 0.95 * self.B0, 0.95 * self.A0)
        self.rb = min(0.6 * self.rc + 1.0, 0.55 * self.B0, 3.5)
        tr = float(S["top_round"] or 0)
        self.Rf = float(np.clip(tr * self.A0, 1.0, 0.95 * self.A0))
        self.Rs = float(np.clip(tr * 1.6 * self.B0, 0.8 * min(self.rc, self.B0), 0.92 * self.B0))
        self.taper = float(S["taper_top"] or 1)
        self.M = Model(self.bid)
        self.body_surf = Surf()
        self.surf = Surf()
        self.y_cz = -self.B0 + 0.42 * 2 * self.B0
        self.notes = []

    # ---- body silhouette functions
    def Aw(self, z):
        return self.A0 * (1 + (self.taper - 1) * z / self.H)

    def Bd(self, z):
        return self.B0 * (1 + (self.taper - 1) * 0.3 * z / self.H)

    def AB(self, z):
        H, Rf, Rs, rb = self.H, self.Rf, self.Rs, self.rb
        a, b = self.Aw(z), self.Bd(z)
        if z > H - Rf:
            a = a - Rf + math.sqrt(max(0.0, Rf * Rf - (z - (H - Rf)) ** 2))
        if z > H - Rs:
            b = b - Rs + math.sqrt(max(0.0, Rs * Rs - (z - (H - Rs)) ** 2))
        if z < rb:
            dz = rb - z
            k = rb - math.sqrt(max(0.0, rb * rb - dz * dz))
            a, b = a - k, b - k
        return a, b

    def x_seam(self, z):
        return self.Aw(z) - self.rc * (1 - math.sin(math.pi / 4))

    # ---- body
    def ring(self, a, b, r, nfh=12, ns=10, nc=10):
        pts, ns_ = [], []
        def add(p, n):
            pts.append(p); ns_.append(n)
        ar, br = a - r, b - r
        for i in range(nfh):
            add((ar * i / nfh, -b), (0, -1))
        # sequence: FR corner, right side, BR corner, back, BL corner, left side, FL corner, front-left
        def corner(cx, cy, a0):
            for i in range(nc):
                t = math.radians(a0 + 90 * i / nc)
                add((cx + r * math.cos(t), cy + r * math.sin(t)), (math.cos(t), math.sin(t)))
        corner(ar, -br, -90)
        for i in range(ns):
            add((a, -br + 2 * br * i / ns), (1, 0))
        corner(ar, br, 0)
        for i in range(2 * nfh):
            add((ar - 2 * ar * i / (2 * nfh), b), (0, 1))
        corner(-ar, br, 90)
        for i in range(ns):
            add((-a, br - 2 * br * i / ns), (-1, 0))
        corner(-ar, -br, 180)
        for i in range(nfh):
            add((-ar + ar * i / nfh, -b), (0, -1))
        return np.array(pts), np.array(ns_)

    def build_body(self):
        H, rb = self.H, self.rb
        Rm = max(self.Rf, self.Rs)
        rings = []  # (z, a, b, s)
        a0, b0 = self.AB(0.0)
        for s in (0.06, 0.2, 0.38, 0.56, 0.72, 0.86, 0.95):
            rings.append((0.0, a0 * s, b0 * s, s))
        for th in np.linspace(0, np.pi / 2, 8):
            rings.append((rb * (1 - math.cos(th)), None, None, 1))
        zw0, zw1 = rb, H - Rm
        nw = max(2, int((zw1 - zw0) / 0.9))
        for z in np.linspace(zw0, zw1, nw)[1:]:
            rings.append((z, None, None, 1))
        for ph in np.linspace(0, np.pi / 2, 20)[1:]:
            rings.append((H - Rm + Rm * math.sin(ph), None, None, 1))
        aT, bT = self.AB(H)
        for s in (0.95, 0.86, 0.72, 0.56, 0.38, 0.2, 0.06):
            rings.append((H, aT * s, bT * s, s))
        grid = []
        for z, a, b, s in rings:
            if a is None:
                a, b = self.AB(z)
            r = min(self.rc, 0.985 * min(a, b)) if s == 1 else min(self.rc, 0.985 * min(a, b))
            r = max(r, 1e-3)
            P2, N2 = self.ring(a, b, r)
            x, y = P2[:, 0].copy(), P2[:, 1].copy()
            if self.bulge:
                wf = np.clip(-N2[:, 1], 0, 1) ** 2
                wx = np.cos(0.5 * np.pi * np.clip(x / max(a, 1e-6), -1, 1)) ** 0.7
                wz = math.sin(math.pi * min(max(z / H, 0), 1)) ** 0.9
                y = y - self.bulge * wf * wx * wz
            grid.append(np.stack([x, y, np.full_like(x, z)], 1))
        G = np.array(grid)
        R, C, _ = G.shape
        V = G.reshape(-1, 3)
        F, _ = grid_faces(R, C, True, False)
        cb = len(V); ct = cb + 1
        V = np.vstack([V, [[0, 0, 0]], [[0, 0, H]]])
        j = np.arange(C)
        Fb = np.stack([np.full(C, cb), (j + 1) % C, j], 1)
        top = (R - 1) * C
        Ft = np.stack([np.full(C, ct), top + j, top + (j + 1) % C], 1)
        self.M.add("fabric", V, [(F, box_uv(V, F, 2.0)), (Fb, box_uv(V, Fb, 2.0)), (Ft, box_uv(V, Ft, 2.0))],
                   orient="closed", obj="body")
        self.body_surf.add(V, [F, Fb, Ft])
        self.surf.add(V, [F, Fb, Ft])
        self.body_V = V

    # ---- generic surface patch
    def patch(self, view, outline, thick, edge_w, mat, crown=0.0, obj=None, add=True, ds=0.45, tile=2.0):
        O = resample(outline, ds, closed=True)[0]
        c = O.mean(0)
        Lc = np.linalg.norm(O - c, axis=1).mean()
        dl = np.array([0, 0.015, 0.06, 0.15, 0.3, 0.5, 0.75, 1.0]) * edge_w
        fs = list(1 - dl / Lc)
        fs += list(np.linspace(fs[-1], 0.04, 7)[1:])
        rows, refs = [], []
        for k, f in enumerate(fs):
            q = c + (O - c) * f
            loc, nn = project(self.surf, view, q)
            d = (1 - f) * Lc
            if k == 0:
                h = -0.12
            else:
                x = min(d / edge_w, 1.0)
                h = thick * math.sqrt(max(0.0, 1 - (1 - x) ** 2)) * (1 + crown * (1 - f))
            rows.append(loc + nn * h)
            refs.append(nn)
        G = np.array(rows)
        Rn = np.array(refs)
        R, C, _ = G.shape
        V = G.reshape(-1, 3)
        ref = Rn.reshape(-1, 3)
        F, _ = grid_faces(R, C, True, False)
        ci = len(V)
        cen_loc, cen_n = project(self.surf, view, [c])
        V = np.vstack([V, cen_loc + cen_n * thick * (1 + crown)])
        ref = np.vstack([ref, cen_n])
        j = np.arange(C)
        last = (R - 1) * C
        Ft = np.stack([np.full(C, ci), last + j, last + (j + 1) % C], 1)
        self.M.add(mat, V, [(F, box_uv(V, F, tile)), (Ft, box_uv(V, Ft, tile))], orient="ref", ref=ref, obj=obj)
        if add:
            self.surf.add(V, [F, Ft])
        return O, c

    # ---- zipper
    def zipper(self, P, N, tape_w=2.0, teeth_w=0.62, closed=False, stops=True):
        P, N = resample(P, 0.2, closed=closed, extra=[N])
        N = nrm(smooth_rows(N, 3, closed))
        M = self.M
        sweep(M, "zip_tape", P + N * 0.02, N, prof_rrect(tape_w, 0.12, 0.05, 2), closed=closed, tile=1.0)
        sy = np.where(np.arange(len(P)) % 2 == 0, 1.0, 0.55)
        pt = prof_rrect(teeth_w, 0.26, 0.06, 1) + np.array([0, 0.13])
        sweep(M, "zip_teeth", P + N * 0.06, N, pt, closed=closed, scale=np.stack([np.ones(len(P)), sy], 1), tile=1.0)
        if stops and not closed:
            for i, j in ((0, 1), (-1, -2)):
                T = nrm(P[i] - P[j])
                rbox(M, "zip_teeth", P[i] + N[i] * 0.2, T, N[i], 0.7, teeth_w + 0.25, 0.32, 0.08)
        T = nrm(np.gradient(P, axis=0))
        return P, N, T

    def slider(self, P, N, T, frac, pref=(0, 0, -1)):
        i = int(np.clip(round(frac * (len(P) - 1)), 1, len(P) - 2))
        p, t = P[i], nrm(T[i])
        # orient the pull with the underlying body normal (pocket rims are sloped)
        _, nb, _, _ = self.body_surf.bvh().find_nearest(Vector(p))
        n = nrm(np.array(nb))
        if np.dot(n, N[i]) < 0:
            n = -n
        t = nrm(t - np.dot(t, n) * n)
        pref = np.asarray(pref, float)
        d = pref - np.dot(pref, n) * n
        if np.linalg.norm(d) < 0.25:
            d = t if np.dot(t, pref) >= 0 else -t
        d = nrm(d)
        M = self.M
        style = self.S["pull_style"]
        sm = "metal" if style == "carabiner" else "slider"
        rbox(M, sm, p + n * 0.52, t, n, 1.35, 1.0, 0.46, 0.16)
        b = nrm(np.cross(n, d))
        base = p + n * 0.62 + d * 0.45
        if style == "carabiner":
            L, w, rr = 3.4, 1.35, 0.17
            pts = stadium(base, d, b, L, w, 28)
            tube(M, "metal", pts, n, rr, 8, closed=True)
            # gate spine bar to read as a carabiner
            tube(M, "metal", np.array([base + d * 0.7 + b * w * 0.18, base + d * (L - 0.7) + b * w * 0.18]), n, 0.08, 6)
        else:
            L, w = 2.2, 0.55
            pts = stadium(base, d, b, L, w, 24)
            tube(M, "cord", pts, n, 0.16, 8, closed=True)
            rbox(M, "pull", base + d * (L + 0.75) , d, n, 1.9, 0.95, 0.5, 0.2)

    # ---- body detail sections
    def piping_loops(self):
        tc = self.rc * (1 - math.sin(math.pi / 4))
        ang = np.linspace(0, 2 * np.pi, 900, endpoint=False)
        for yv in (-self.B0 + tc, self.B0 - tc):
            P, N = section(self.body_surf, "y", yv, (0, self.H * 0.5), ang)
            P, N = resample(P, 0.35, closed=True, extra=[N])
            N = nrm(smooth_rows(N, 3, True))
            tube(self.M, "trim", P + N * 0.06, N, 0.22, 8, closed=True)

    def clamshell(self):
        H, A0 = self.H, self.A0
        al = math.atan2(0.5 * H - 0.1 * H, A0)
        ang = np.linspace(math.pi + al, -al, 900)
        P, N = section(self.body_surf, "y", self.y_cz, (0, H * 0.5), ang)
        P, N, T = self.zipper(P, N, 2.2, 0.66)
        self.slider(P, N, T, 0.455, pref=-T[int(0.455 * (len(P) - 1))])
        self.slider(P, N, T, 0.545, pref=T[int(0.545 * (len(P) - 1))])

    # ---- front pockets
    def pocket(self, spec, kind_default="panel"):
        H = self.H
        zb, zt = spec["bottom"] * H, spec["top"] * H
        zm = 0.5 * (zb + zt)
        xm = self.x_seam(zm) - float(spec.get("inset", 2.4))
        zipk = spec.get("zip", "top-straight")
        r = min(3.0, 0.25 * (zt - zb), 0.25 * xm)
        segs = []  # list of (pts, is_zip)
        def arc(cx, cz, rr, a0, a1, n=10):
            t = np.radians(np.linspace(a0, a1, n))
            return np.stack([cx + rr * np.cos(t), cz + rr * np.sin(t)], 1)
        def line(p0, p1, n=None):
            p0, p1 = np.array(p0, float), np.array(p1, float)
            n = n or max(2, int(np.linalg.norm(p1 - p0) / 0.5))
            return p0 + (p1 - p0) * np.linspace(0, 1, n)[:, None]
        x0, x1 = -xm, xm
        if zipk == "top-arc":
            ar = min(0.18 * 2 * xm, 0.35 * (zt - zb))
            zs = zt - ar
            xs = np.linspace(-1, 1, 60)
            top = np.stack([xs * xm, zs + ar * np.clip(1 - xs ** 2, 0, 1) ** 0.6], 1)
            segs.append((top, True))
            segs.append((line((x1, zs), (x1, zb + r))[1:], False))
            segs.append((arc(x1 - r, zb + r, r, 0, -90)[1:], False))
            segs.append((line((x1 - r, zb), (x0 + r, zb))[1:], False))
            segs.append((arc(x0 + r, zb + r, r, -90, -180)[1:], False))
            segs.append((line((x0, zb + r), (x0, zs))[1:-1], False))
            self.fp_zip_top = zt
            self.fp_arc = ar
        elif zipk == "u-shape":
            segs.append((line((x0, zt - r - 0.8), (x0, zb + r)), True))
            segs.append((arc(x0 + r, zb + r, r, 180, 270)[1:], True))
            segs.append((line((x0 + r, zb), (x1 - r, zb))[1:], True))
            segs.append((arc(x1 - r, zb + r, r, 270, 360)[1:], True))
            segs.append((line((x1, zb + r), (x1, zt - r - 0.8))[1:], True))
            segs.append((line((x1, zt - r - 0.8), (x1, zt - r))[1:], False))
            segs.append((arc(x1 - r, zt - r, r, 0, 90)[1:], False))
            segs.append((line((x1 - r, zt), (x0 + r, zt))[1:], False))
            segs.append((arc(x0 + r, zt - r, r, 90, 180)[1:], False))
            segs.append((line((x0, zt - r), (x0, zt - r - 0.8))[1:-1], False))
            self.fp_arc = 0
        else:  # top-straight
            segs.append((line((x0 + r * 0.6, zt), (x1 - r * 0.6, zt)), True))
            segs.append((line((x1 - r * 0.6, zt), (x1 - r, zt))[1:], False))
            segs.append((arc(x1 - r, zt - r, r, 90, 0)[1:], False))
            segs.append((line((x1, zt - r), (x1, zb + r))[1:], False))
            segs.append((arc(x1 - r, zb + r, r, 0, -90)[1:], False))
            segs.append((line((x1 - r, zb), (x0 + r, zb))[1:], False))
            segs.append((arc(x0 + r, zb + r, r, -90, -180)[1:], False))
            segs.append((line((x0, zb + r), (x0, zt - r))[1:], False))
            segs.append((arc(x0 + r, zt - r, r, 180, 90)[1:], False))
            segs.append((line((x0 + r, zt), (x0 + r * 0.6, zt))[1:-1], False))
            self.fp_arc = 0
        outline = np.concatenate([s for s, _ in segs])
        mask = np.concatenate([np.full(len(s), z) for s, z in segs])
        caddy = spec.get("type") == "caddy"
        thick = 0.75 if caddy else 0.5
        O, c = self.patch("front", outline, thick, 0.9 if caddy else 0.7, "fabric", crown=0.5 if caddy else 0.3, obj="pockets")
        # zipper on the rim
        zp = c + (outline[mask] - c) * 0.985
        P, N = project(self.surf, "front", zp)
        P, N, T = self.zipper(P, N, 1.7, 0.52)
        if zipk == "u-shape":
            self.slider(P, N, T, 0.04, pref=(0, 0, -1))
        else:
            self.slider(P, N, T, 0.9, pref=(0, 0, -1))
        # piping on the remaining edge
        rest = outline[~mask]
        if len(rest) > 2:
            # pad with neighbouring zip points so the piping tucks under the zip ends
            pp = c + (rest - c) * 0.995
            Pp, Np = project(self.surf, "front", pp)
            Pp, Np = resample(Pp, 0.35, extra=[Np])
            Np = nrm(smooth_rows(Np, 2))
            tube(self.M, "trim", Pp + Np * 0.04, Np, 0.2, 8)
        return zb, zt, xm

    # ---- side features
    def side_x(self, side):
        return "left" if side < 0 else "right"

    def bottle_sleeve(self, side):
        H, B0 = self.H, self.B0
        view = self.side_x(side)
        nu, nv = 30, 20
        u = np.linspace(0, 1, nu)
        v = np.linspace(0, 1, nv)
        y = -(B0 - 1.4) + u * 2 * (B0 - 1.4)
        z0, z1 = 1.0, 0.34 * H
        z = z0 + v * (z1 - z0)
        bul = float(self.S.get("bottle_pocket_bulge", 0.85))
        rows, refs = [], []
        for j in range(nv):
            q = np.stack([y, np.full(nu, z[j])], 1)
            loc, nn = project(self.body_surf, view, q)
            h = 0.12 + bul * np.sin(np.pi * u) ** 0.45 * (0.55 + 0.45 * sstep(0, 0.7, v[j]))
            h = np.where((u == 0) | (u == 1), -0.1, h)
            if j == 0:
                h = np.full(nu, -0.1)
            rows.append(loc + nn * h[:, None]); refs.append(nn)
        G = np.array(rows).transpose(1, 0, 2)  # (nu, nv): rows along y
        Rf = np.array(refs).transpose(1, 0, 2)
        V = G.reshape(-1, 3)
        F, corners = grid_faces(nu, nv, False, False)
        UVg = np.stack(np.meshgrid(u * 2 * (B0 - 1.4), z, indexing="ij"), -1) / 2.0
        L = np.stack([UVg[a, b] for a, b in corners], 1)
        self.M.add("sleeve", V, [(F, L)], orient="ref", ref=Rf.reshape(-1, 3))
        self.surf.add(V, [F])
        # elastic hem along the open top
        top = G[:, -1]
        nt = nrm(Rf[:, -1])
        tube(self.M, "trim", top + nt * 0.02, nt, 0.3, 8)
        # side seams of the sleeve
        for k in (0, nu - 1):
            tube(self.M, "trim", G[k] + Rf[k] * 0.08, Rf[k], 0.18, 6)

    def side_zip(self):
        H = self.H
        left_bottle = self.S["side_bottle_pocket"] in ("left", "both")
        z0 = 0.47 * H if left_bottle else 0.36 * H
        z1 = 0.74 * H if self.S["side_handle"] else 0.76 * H
        ys = 0.5 * (self.y_cz + (self.B0 - 1.5))
        q = np.stack([np.full(40, ys), np.linspace(z0, z1, 40)], 1)
        P, N = project(self.body_surf, "left", q)
        P, N, T = self.zipper(P, N, 1.6, 0.5)
        self.slider(P, N, T, 0.92, pref=(0, 0, -1))

    def side_handle(self):
        H = self.H
        z = 0.82 * H
        L = min(9.0, 2 * self.B0 - 2.0)
        ys = np.linspace(-L / 2, L / 2, 50)
        base, nn = project(self.body_surf, "left", np.stack([ys, np.full(50, z)], 1))
        nn = nrm(smooth_rows(nn, 3))
        s = ys / (L / 2)
        q = np.clip(np.abs(s) / 0.62, 0, 1)
        lift = 2.0 * (1 - q ** 2) ** 0.8
        th = 0.7
        C = base + nn * (th * 0.5 + lift[:, None])
        sc = np.stack([np.ones(50), np.where(q >= 1, 0.6, 1.0)], 1)
        sweep(self.M, "webbing", C, nn, prof_rrect(2.6, th, 0.25, 3), scale=sc, tile=1.0)
        for e in (0, -1):
            rbox(self.M, "trim", base[e] + nn[e] * (th * 0.6 + 0.02), nrm(base[1] - base[0]), nn[e], 1.4, 2.8, 0.12, 0.05)

    def d_ring(self, side):
        H, B0 = self.H, self.B0
        yd, zd = -B0 + self.rc * 0.55, 0.70 * H
        loc, n = self.body_surf.ray(*view_ray(self.side_x(side), (yd, zd)))
        down = np.array([0, 0, -1.0]); down = nrm(down - np.dot(down, n) * n)
        rbox(self.M, "webbing", loc + n * 0.12 + down * 0.3, down, n, 2.6, 1.7, 0.24, 0.08)
        b = nrm(np.cross(n, down))
        base = loc + n * 0.36 + down * 1.4
        pts = [base + b * x for x in np.linspace(-0.9, 0.9, 6)]
        for t in np.linspace(0, np.pi, 16)[1:-1]:
            pts.append(base + b * 0.9 * math.cos(t) + down * 1.5 * math.sin(t))
        pts = np.array(pts)
        tube(self.M, "metal", pts, n, 0.14, 8, closed=True)

    # ---- top
    def top_handle(self):
        spec = self.S["top_handle"]
        if not spec:
            return
        w = float(spec.get("width", 11.0))
        riv = spec.get("type", "webbing") == "riveted"
        yh = self.B0 * 0.3
        xs = np.linspace(-w / 2, w / 2, 70)
        base, nn = project(self.body_surf, "top", np.stack([xs, np.full(70, yh)], 1))
        nn = nrm(smooth_rows(nn, 3))
        s = xs / (w / 2)
        tab = 3.2 / (w / 2)
        q = np.clip(np.abs(s) / (1 - tab), 0, 1)
        hgt = 3.2 if riv else 2.6
        lift = hgt * (1 - q ** 2) ** 0.75
        th = 1.3 if riv else 0.32
        wid = 3.0 if riv else 2.5
        up = np.tile([0, 0, 1.0], (70, 1))
        thick_s = np.where(q >= 1, 0.5 if riv else 1.0, 1.0)
        C = base + nn * (th * 0.5 * thick_s[:, None] - 0.05) + up * lift[:, None]
        prof = prof_pillow(wid, th, 24, 4) if riv else prof_rrect(wid, th, 0.1, 2)
        sweep(self.M, "fabric" if riv else "webbing", C, up * 0.5 + nn * 0.5, prof,
              scale=np.stack([np.ones(70), thick_s], 1), tile=1.0)
        for sgn in (-1, 1):
            if riv:
                for xr in (w / 2 - 1.0, w / 2 - 2.3):
                    i = int(np.argmin(np.abs(xs - sgn * xr)))
                    p = base[i] + nn[i] * (th * 0.5 + 0.02)
                    rbox(self.M, "metal", p, nn[i], np.array([1.0, 0, 0]), 0.22, 0, 0, prof=prof_circle(0.5, 14))
            else:
                i = int(np.argmin(np.abs(xs - sgn * (w / 2 - 1.4))))
                rbox(self.M, "trim", base[i] + nn[i] * (th + 0.03), np.array([1.0, 0, 0]), nn[i], 1.2, wid * 0.9, 0.08, 0.04)

    def top_pocket(self):
        y = 0.5 * (self.y_cz - self.B0) + 0.4
        xm = 0.28 * self.W
        q = np.stack([np.linspace(-xm, xm, 60), np.full(60, y)], 1)
        P, N = project(self.body_surf, "top", q)
        P, N, T = self.zipper(P, N, 1.7, 0.5)
        self.slider(P, N, T, 0.88, pref=(0, -1, 0))

    def laptop_zip(self):
        H = self.H
        zl = H - self.Rs * 0.45
        xm = 0.34 * self.W
        xs = np.linspace(-xm, xm, 60)
        q = np.stack([xs, zl + 0.5 * (1 - (xs / xm) ** 2)], 1)
        P, N = project(self.body_surf, "back", q)
        P, N, T = self.zipper(P, N, 1.8, 0.55)
        self.slider(P, N, T, 0.9, pref=(1, 0, 0))

    # ---- logo
    def logo(self, fp):
        kind = self.S["logo"]
        if not kind or kind == "none":
            return
        H = self.H
        if fp is not None:
            zb, zt, xm = fp
            if zt <= 0.7 * H:
                z = min(zt + 0.12 * H, self.H - self.Rf - 3)
            else:
                z = zt - getattr(self, "fp_arc", 0) - 4.0
        else:
            z = 0.72 * H
        if kind == "patch":
            w, h, r = 5.0, 3.2, 0.5
        else:
            w, h, r = 6.0, 1.8, 0.35
        o = rr_outline(0, z, w, h, r)
        self.patch("front", o, 0.22, 0.12, "logo", obj="logo")
        if kind == "patch":
            Pq = rr_outline(0, z, w - 0.6, h - 0.6, 0.3)
            P, N = project(self.surf, "front", Pq)
            P, N = resample(P, 0.3, closed=True, extra=[N])
            tube(self.M, "trim", P + N * 0.02, N, 0.06, 5, closed=True)
            self.patch("front", rr_outline(0, z, 2.6, 1.2, 0.3), 0.08, 0.05, "logo_mark", obj="logo_mark", add=False)
        else:
            self.patch("front", rr_outline(-1.1, z, 2.2, 0.55, 0.2), 0.08, 0.04, "logo_mark", obj="logo_mark", add=False)
            self.patch("front", rr_outline(1.6, z, 1.2, 0.55, 0.2), 0.08, 0.04, "logo_mark", obj="logo_mark", add=False)

    # ---- back
    def back_panel(self):
        H = self.H
        bp = self.S["back_panel"] or {}
        pads = int(bp.get("pads", 0) or 0)
        th = float(bp.get("thickness", 1.8) or 0)
        self.z0 = H - self.Rs - 1.0
        zm = 0.5 * H
        xf = self.Aw(zm) - self.rc * 0.85
        ztop = self.z0 - 0.5
        outl = []
        if pads >= 3:
            outl += [rect_outline(0.9, xf, 0.53 * H, ztop, 3.5), rect_outline(-xf, -0.9, 0.53 * H, ztop, 3.5),
                     rect_outline(-xf, xf, 0.10 * H, 0.34 * H, 3.5)]
            lug = (0.375 * H, 0.49 * H)
        elif pads == 2:
            outl += [rect_outline(1.0, xf, 0.30 * H, ztop, 3.5), rect_outline(-xf, -1.0, 0.30 * H, ztop, 3.5)]
            lug = (0.10 * H, 0.235 * H)
        elif pads == 1:
            outl += [rect_outline(-xf, xf, 0.30 * H, ztop, 4.0)]
            lug = (0.10 * H, 0.235 * H)
        else:
            lug = (0.40 * H, 0.52 * H)
        if th > 0:
            for o in outl:
                self.patch("back", o, th, min(1.6, th * 0.85), "pad", crown=0.15, obj="pads", ds=0.5, tile=3.0)
        if self.S["luggage_strap"]:
            xl = xf + 0.4
            o = rect_outline(-xl, xl, lug[0], lug[1], 0.6)
            self.patch("back", o, 0.4, 0.25, "webbing", obj="luggage", tile=1.0)
            for sgn in (-1, 1):
                loc, n = self.surf.ray(*view_ray("back", (sgn * (xl - 0.8), 0.5 * (lug[0] + lug[1]))))
                rbox(self.M, "trim", loc + n * 0.06, np.array([0, 0, 1.0]), n, lug[1] - lug[0] - 0.6, 1.0, 0.12, 0.05)
        if bp.get("top_laptop_zip"):
            self.laptop_zip()

    # ---- straps
    def straps(self, front_min):
        st = self.S["straps"]
        if not st:
            return
        H, W, A0 = self.H, self.W, self.A0
        w, T = float(st.get("width", 7)), float(st.get("thickness", 2))
        Lp = 0.55 * float(st.get("length", 58))
        z0 = self.z0
        x0 = 0.12 * W + w / 2
        x1 = min(A0 - 0.85 * w, A0 - 5.0)
        y_out = float(self.S["depth_external"]) + front_min - (0.9 if st.get("sternum", True) else 0.0)  # strap outer surface target (y)
        n = 140
        t = np.linspace(0, 1, n)
        sx = np.where(t < 0.3, 0.84 + 0.16 * sstep(0, 0.3, t), 1.0 - 0.42 * sstep(0.3, 1.0, t))
        sy = 0.55 + 0.45 * np.sin(np.pi * t) ** 0.4
        bump = np.sin(np.pi * t ** 0.75)
        ramp = 0.35 + 0.65 * sstep(0, 0.1, t)

        def path(z1, sgn):
            x = sgn * (x0 + (x1 - x0) * (0.5 - 0.5 * np.cos(np.pi * t)) + (W / 30.0) * 0.7 * np.sin(2 * np.pi * t))
            z = z0 - (z0 - z1) * t
            loc, _ = project(self.surf, "back", np.stack([x, z], 1))
            yf = loc[:, 1]
            yd = np.array([yf[max(0, i - 12):i + 13].max() for i in range(n)])
            yd = smooth_rows(yd[:, None], 6, iters=4)[:, 0]
            yd = np.maximum(yd, yf)
            base = yd + (T * sy / 2 + 0.2) * ramp
            outer0 = base + T * sy / 2
            s_amp = max(0.0, (y_out - outer0.max()) / 1.0)
            # solve amplitude so that max(outer) == y_out
            lo, hi = 0.0, 40.0
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                if (outer0 + mid * bump).max() > y_out:
                    hi = mid
                else:
                    lo = mid
            s_amp = lo if (outer0.max() < y_out) else 0.0
            # always keep a visible S-bow (bounded so total depth stays within ~1.2 cm of spec)
            s_amp = max(s_amp, min(1.4, max(0.0, y_out + 1.2 - outer0.max())))
            y = base + s_amp * bump
            return np.stack([x, y, z], 1), s_amp

        lo, hi = 0.08 * H, 0.5 * H
        for _ in range(18):
            z1 = 0.5 * (lo + hi)
            P, _ = path(z1, 1)
            if arclen(P)[-1] > Lp:
                lo = z1
            else:
                hi = z1
        z1 = 0.5 * (lo + hi)
        self.strap_info = {"padded_len": round(Lp, 1), "z_end": round(z1, 1)}
        M = self.M
        ends, frames = {}, {}
        for sgn in (-1, 1):
            P, amp = path(z1, sgn)
            self.strap_info["standoff"] = round(amp, 2)
            ups = np.tile([0, 1.0, 0], (n, 1))
            G, Tn, Nn, Bn = sweep(M, "strap", P, ups, prof_strap(w, T, 32), scale=np.stack([sx, sy], 1), tile=2.0)
            frames[sgn] = (P, Tn, Nn, Bn)
            # edge binding
            for e in (1, -1):
                Ep = P + Bn * (e * w / 2 * sx * 0.99)[:, None] + Nn * (-0.05 * T * sy)[:, None]
                tube(M, "trim", Ep, Nn, min(0.28, T * 0.14), 6)
            # lower webbing + ladder lock + anchor
            ax = sgn * (A0 - 2.8)
            anc, an = self.surf.ray(*view_ray("back", (ax, 2.6)))
            E = P[-6]
            dirv = nrm(anc - E)
            k = 30
            pts = E + (anc - E) * np.linspace(0, 1, k)[:, None]
            fl, _ = project(self.surf, "back", pts[:, [0, 2]])
            pts[:, 1] = np.maximum(pts[:, 1], fl[:, 1] + 0.22)
            pts[-1] = anc + an * 0.1
            pts = smooth_rows(pts, 2, iters=1); pts[-1] = anc + an * 0.1
            upw = nrm(np.tile([0, 1.0, 0], (k, 1)) + 0.0)
            sweep(M, "webbing", pts, upw, prof_rrect(2.5, 0.16, 0.05, 1), tile=1.0)
            bk = P[-1] + dirv * 1.6
            rbox(M, "plastic", bk + np.array([0, 0.25, 0]), dirv, np.array([0, 1.0, 0]), 1.5, 3.1, 0.55, 0.15)
            # webbing tail hanging from the buckle
            tl = float(np.clip(bk[2] - 1.5, 2.0, 9.0))
            tail = bk + np.array([0, 0.45, 0]) + np.array([sgn * 0.3, 0, -1.0]) * np.linspace(0.5, tl, 14)[:, None]
            fl, _ = project(self.surf, "back", tail[:, [0, 2]])
            tail[:, 1] = np.maximum(tail[:, 1], fl[:, 1] + 0.5)
            sweep(M, "webbing", tail, np.tile([0, 1.0, 0], (14, 1)), prof_rrect(2.5, 0.16, 0.05, 1), tile=1.0)
            rbox(M, "webbing", anc + an * 0.15, np.array([0, 0, 1.0]), an, 2.4, 3.4, 0.3, 0.1)
            ends[sgn] = P
        # sternum strap
        if st.get("sternum", True):
            ts = 0.36
            i = int(ts * (n - 1))
            outs = {}
            for sgn in (-1, 1):
                P, Tn, Nn, Bn = frames[sgn]
                # rail
                r0, r1 = int(0.2 * (n - 1)), int(0.5 * (n - 1))
                rail = P[r0:r1] + Nn[r0:r1] * (T * sy[r0:r1] / 2 + 0.03)[:, None]
                sweep(M, "webbing", rail, Nn[r0:r1], prof_rrect(1.0, 0.12, 0.04, 1), tile=1.0)
                outs[sgn] = (P[i] + Nn[i] * (T * sy[i] / 2 + 0.22), Nn[i], Tn[i])
            pa, pb = outs[-1][0], outs[1][0]
            yy = max(pa[1], pb[1])
            pts = pa + (pb - pa) * np.linspace(0, 1, 30)[:, None]
            pts[:, 1] = yy + 0.25 * np.sin(np.pi * np.linspace(0, 1, 30))
            pts[0, 1] = pa[1]; pts[-1, 1] = pb[1]
            sweep(M, "webbing", pts, np.tile([0, 1.0, 0], (30, 1)), prof_rrect(1.9, 0.15, 0.05, 1), tile=1.0)
            mid = pts[15] + np.array([0, 0.35, 0])
            X = np.array([1.0, 0, 0]); Y = np.array([0, 1.0, 0])
            rbox(M, "plastic", mid - X * 1.25, X, Y, 2.6, 2.4, 0.85, 0.3)
            rbox(M, "plastic", mid + X * 1.15, X, Y, 2.3, 2.1, 0.65, 0.25)
            for sgn in (-1, 1):
                p, nn_, tt = outs[sgn]
                rbox(M, "plastic", p - nn_ * 0.05, X, nn_, 1.4, 2.3, 0.45, 0.15)
        # load lifters
        if st.get("load_lifters"):
            for sgn in (-1, 1):
                P, Tn, Nn, Bn = frames[sgn]
                i = int(0.12 * (n - 1))
                ps = P[i] + Nn[i] * (T * sy[i] / 2 * 0.6) + Bn[i] * 0
                anc, an = self.surf.ray(*view_ray("back", (sgn * (self.Aw(self.z0) - 1.6), self.z0 + 0.8)))
                k = 16
                pts = ps + (anc - ps) * np.linspace(0, 1, k)[:, None]
                pts[-1] = anc + an * 0.1
                sweep(M, "webbing", pts, np.tile([0, 1.0, 0], (k, 1)), prof_rrect(2.0, 0.15, 0.05, 1), tile=1.0)
                dv = nrm(anc - ps)
                rbox(M, "plastic", ps + (anc - ps) * 0.45 + np.array([0, 0.25, 0]), dv, np.array([0, 1.0, 0]), 1.3, 2.5, 0.5, 0.12)
                rbox(M, "webbing", anc + an * 0.12, dv, an, 2.2, 2.6, 0.25, 0.08)

    # ---- assemble
    def build(self):
        S = self.S
        self.build_body()
        self.piping_loops()
        if S["clamshell_zip"]:
            self.clamshell()
        fp = None
        if S["front_pocket"]:
            fp = self.pocket(S["front_pocket"])
        if S["lower_pocket"]:
            self.pocket({**S["lower_pocket"], "zip": "top-straight", "type": "panel"})
        self.logo(fp)
        sb = S["side_bottle_pocket"]
        if sb in ("left", "both"):
            self.bottle_sleeve(-1)
        if sb in ("right", "both"):
            self.bottle_sleeve(1)
        if S["side_zip_pocket"]:
            self.side_zip()
        if S["side_handle"]:
            self.side_handle()
        if S["d_rings"]:
            self.d_ring(-1); self.d_ring(1)
        if S["top_pocket"]:
            self.top_pocket()
        self.top_handle()
        self.back_panel()
        front_min = self.M.all_vertices()[:, 1].min()
        self.front_min = front_min
        self.straps(front_min)


def stadium(base, d, b, L, w, n):
    r = w / 2
    pts = []
    for t in np.linspace(-np.pi / 2, np.pi / 2, n // 2):
        pts.append(base + d * (L - r + r * math.cos(t)) + b * r * math.sin(t))
    for t in np.linspace(np.pi / 2, 3 * np.pi / 2, n // 2):
        pts.append(base + d * (r + r * math.cos(t)) + b * r * math.sin(t))
    return np.array(pts)


def rr_outline(cx, cz, w, h, r, seg=8):
    p = prof_rrect(w, h, r, seg)
    return p + np.array([cx, cz])


def rect_outline(x0, x1, z0, z1, r):
    return rr_outline(0.5 * (x0 + x1), 0.5 * (z0 + z1), abs(x1 - x0), abs(z1 - z0), r, 8)


# --------------------------------------------------------------------------------------
# procedural textures
# --------------------------------------------------------------------------------------
def periodic_noise(N, cutoff, seed):
    rng = np.random.default_rng(seed)
    f = np.fft.fftfreq(N) * N
    fx, fy = np.meshgrid(f, f)
    amp = np.exp(-(fx ** 2 + fy ** 2) / (2 * cutoff ** 2))
    spec = (rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))) * amp
    n = np.real(np.fft.ifft2(spec))
    return (n - n.mean()) / (n.std() + 1e-9)


def height_field(kind, N, seed=1):
    u, v = np.meshgrid(np.arange(N) / N, np.arange(N) / N)
    if kind in ("nylon", "ripstop"):
        f = 24
        fu, fv = u * f, v * f
        i, j = np.floor(fu), np.floor(fv)
        a, b = fu - i, fv - j
        warp = np.sin(np.pi * b) ** 0.6 * (0.55 + 0.45 * np.sin(np.pi * a))
        weft = np.sin(np.pi * a) ** 0.6 * (0.55 + 0.45 * np.sin(np.pi * b))
        h = np.where((i + j) % 2 == 0, warp, weft)
        rip = ((i % 6) == 0) * np.sin(np.pi * a) ** 0.5 + ((j % 6) == 0) * np.sin(np.pi * b) ** 0.5
        h = 0.75 * h + 0.45 * rip
    elif kind in ("ballistic", "cordura"):
        f = 16
        fu, fv = u * f, v * f
        i, j = np.floor(fu), np.floor(fv)
        a, b = fu - i, fv - j
        warp = np.sin(np.pi * b) ** 0.5 * (0.6 + 0.4 * np.sin(np.pi * (fu / 2 % 1)))
        weft = np.sin(np.pi * a) ** 0.5 * (0.6 + 0.4 * np.sin(np.pi * (fv / 2 % 1)))
        h = np.where((np.floor(i / 2) + np.floor(j / 2)) % 2 == 0, warp, weft)
    elif kind == "webbing":
        f = 10
        b = (v * f) % 1
        h = np.sin(np.pi * b) ** 0.7 * (0.8 + 0.2 * np.sin(2 * np.pi * u * 30 + np.floor(v * f) * 1.7))
    elif kind in ("spacer", "mesh"):
        f = 6 if kind == "spacer" else 8
        fu, fv = u * f, v * f
        d1 = np.hypot(fu - np.round(fu), fv - np.round(fv))
        d2 = np.hypot(fu - (np.floor(fu) + 0.5), fv - (np.floor(fv) + 0.5))
        d = np.minimum(d1, d2)
        h = sstep(0.16, 0.30, d) if kind == "spacer" else sstep(0.20, 0.30, d)
    else:
        h = np.zeros((N, N))
    h = h + 0.08 * periodic_noise(N, N / 24, seed)
    return h


def normal_from_height(h, strength):
    dx = (np.roll(h, -1, 1) - np.roll(h, 1, 1)) * 0.5 * strength
    dy = (np.roll(h, -1, 0) - np.roll(h, 1, 0)) * 0.5 * strength
    n = nrm(np.stack([-dx, -dy, np.ones_like(h)], -1))
    return n * 0.5 + 0.5


def make_image(name, rgb, nonc=True):
    N = rgb.shape[0]
    img = bpy.data.images.new(name, N, N, alpha=False)
    rgba = np.concatenate([rgb, np.ones(rgb.shape[:2] + (1,))], -1).astype(np.float32)
    img.pixels.foreach_set(rgba.ravel())
    # generated images pack as blank buffers -> write a real PNG, reload, pack
    tmp = os.path.join(tempfile.gettempdir(), "bag_tex_%d" % os.getpid())
    os.makedirs(tmp, exist_ok=True)
    fp = os.path.join(tmp, name + ".png")
    img.filepath_raw = fp
    img.file_format = "PNG"
    img.save()
    bpy.data.images.remove(img)
    img = bpy.data.images.load(fp)
    img.name = name
    if nonc:
        img.colorspace_settings.name = "Non-Color"
    img.pack()
    return img


# --------------------------------------------------------------------------------------
# materials
# --------------------------------------------------------------------------------------
class Mats:
    def __init__(self, bag):
        self.bid = bag.bid
        S = bag.S
        fab = S["fabric"] or {}
        self.S = S
        self.fab_col = fab.get("color", "#1b1c1e")
        self.fab_r = float(fab.get("roughness", 0.75))
        weave = fab.get("weave", "nylon")
        trim = S["trim_color"] or shade(self.fab_col, 1.25)
        zipc = S["zip_color"]
        pullc = S["pull_color"]
        hw = S["hardware_color"] or (pullc if S["pull_style"] == "carabiner" else "#55575b")
        logoc = S["logo_color"] or ("#3c3d41" if S["logo"] != "patch" else "#2f3034")
        self.tex = {}
        self.defs = {
            "fabric": dict(col=self.fab_col, rough=self.fab_r, tex=weave, ns=0.9),
            "strap": dict(col=self.fab_col, rough=self.fab_r, tex=weave, ns=0.9),
            "trim": dict(col=trim, rough=min(1, self.fab_r * 0.85), tex="webbing", ns=0.6),
            "webbing": dict(col=shade(self.fab_col, 1.08), rough=0.78, tex="webbing", ns=1.0),
            "pad": dict(col=shade(self.fab_col, 1.18), rough=0.9, tex="spacer", ns=1.2),
            "sleeve": dict(col=shade(self.fab_col, 0.8), rough=0.85, tex="mesh", ns=1.3, double=True),
            "zip_tape": dict(col=zipc, rough=0.55, tex="webbing", ns=0.5),
            "zip_teeth": dict(col=shade(zipc, 2.2), rough=0.38, metal=0.65),
            "slider": dict(col="#2c2d30", rough=0.35, metal=0.9),
            "metal": dict(col=hw, rough=0.28, metal=1.0),
            "cord": dict(col=pullc, rough=0.8, tex="webbing", ns=0.8),
            "pull": dict(col=pullc, rough=0.45),
            "plastic": dict(col="#131314", rough=0.42),
            "logo": dict(col=logoc, rough=0.5 if S["logo"] != "patch" else 0.85),
            "logo_mark": dict(col=shade(logoc, 1.9), rough=0.55),
        }
        self.cache = {}

    def texture(self, kind, rough):
        key = (kind, round(rough, 3))
        if key in self.tex:
            return self.tex[key]
        N = {"nylon": 512, "ripstop": 512, "ballistic": 512, "cordura": 512}.get(kind, 256)
        h = height_field(kind, N, seed=hash(kind) % 1000)
        nmap = normal_from_height(h, {"spacer": 6.0, "mesh": 6.0}.get(kind, 4.0) * N / 256)
        hn = (h - h.min()) / (np.ptp(h) + 1e-9)
        rmap = np.clip(rough + 0.12 * (0.5 - hn), 0.02, 1.0)
        rgb_r = np.stack([np.ones_like(rmap), rmap, np.zeros_like(rmap)], -1)
        ni = make_image(f"{self.bid}_{kind}_normal", nmap)
        ri = make_image(f"{self.bid}_{kind}_rough_{int(rough*100)}", rgb_r)
        self.tex[key] = (ni, ri)
        return self.tex[key]

    def get(self, key):
        if key in self.cache:
            return self.cache[key]
        d = self.defs[key]
        m = bpy.data.materials.new(f"{self.bid}_{key}")
        try:
            m.use_nodes = True
        except Exception:
            pass
        nt = m.node_tree
        bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
        col = srgb2lin(hex_rgb(d["col"]))
        bsdf.inputs["Base Color"].default_value = (*col, 1.0)
        bsdf.inputs["Roughness"].default_value = d["rough"]
        bsdf.inputs["Metallic"].default_value = d.get("metal", 0.0)
        if d.get("tex"):
            ni, ri = self.texture(d["tex"], d["rough"])
            tn = nt.nodes.new("ShaderNodeTexImage"); tn.image = ni; tn.location = (-700, -300)
            nm = nt.nodes.new("ShaderNodeNormalMap"); nm.inputs["Strength"].default_value = d.get("ns", 1.0)
            nm.location = (-300, -300)
            nt.links.new(tn.outputs["Color"], nm.inputs["Color"])
            nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
            tr = nt.nodes.new("ShaderNodeTexImage"); tr.image = ri; tr.location = (-700, 0)
            sep = nt.nodes.new("ShaderNodeSeparateColor"); sep.location = (-350, 0)
            nt.links.new(tr.outputs["Color"], sep.inputs["Color"])
            nt.links.new(sep.outputs["Green"], bsdf.inputs["Roughness"])
        m.use_backface_culling = not d.get("double", False)
        m.diffuse_color = (*col, 1.0)
        self.cache[key] = m
        return m


# --------------------------------------------------------------------------------------
# blender scene / export
# --------------------------------------------------------------------------------------
def make_objects(bag, mats):
    total_tris = 0
    for name, o in bag.M.objs.items():
        V = np.concatenate(o["V"])
        faces, uvs = [], []
        for F, L in o["G"]:
            faces.extend(F.tolist())
            uvs.append(L.reshape(-1, 2))
            total_tris += len(F) * (F.shape[1] - 2)
        me = bpy.data.meshes.new(f"{bag.bid}_{name}")
        me.from_pydata((V * CM).tolist(), [], faces)
        uvl = me.uv_layers.new(name="UVMap")
        uvl.data.foreach_set("uv", np.concatenate(uvs).astype(np.float32).ravel())
        me.materials.append(mats.get(o["mat"]))
        me.shade_smooth()
        hard = o["mat"] in ("metal", "plastic", "slider", "pull", "zip_teeth", "logo", "logo_mark")
        me.set_sharp_from_angle(angle=math.radians(38 if hard else 62))
        me.update()
        ob = bpy.data.objects.new(f"{bag.bid}_{name}", me)
        bpy.context.scene.collection.objects.link(ob)
    return total_tris


def export_glb(path):
    bpy.ops.export_scene.gltf(
        filepath=path, export_format="GLB", export_yup=True, export_apply=True,
        export_texcoords=True, export_normals=True, export_materials="EXPORT",
        export_image_format="AUTO", export_cameras=False, export_lights=False, use_selection=False,
    )


def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def measure_glb(path):
    reset_scene()
    bpy.ops.import_scene.gltf(filepath=path)
    mn, mx = np.full(3, 1e9), np.full(3, -1e9)
    tris = 0
    body = None
    for ob in bpy.context.scene.objects:
        if ob.type != "MESH":
            continue
        me = ob.data
        co = np.empty(len(me.vertices) * 3); me.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        Mw = np.array(ob.matrix_world)
        co = co @ Mw[:3, :3].T + Mw[:3, 3]
        mn = np.minimum(mn, co.min(0)); mx = np.maximum(mx, co.max(0))
        me.calc_loop_triangles()
        tris += len(me.loop_triangles)
        if ob.name.endswith("_body"):
            body = (co.min(0), co.max(0))
    return mn / CM, mx / CM, tris, (body[0] / CM, body[1] / CM) if body else None


def build_bag(bag):
    t0 = time.time()
    reset_scene()
    # Per-brand builder: builders/<bag["builder"]>.py may define `class Builder(Bag)`
    # overriding any part of the build. Without one, the generic Bag is used.
    cls = Bag
    bname = bag.get("builder")
    if bname:
        import importlib.util
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "builders", f"{bname}.py")
        if os.path.exists(p):
            spec = importlib.util.spec_from_file_location(f"builders_{bname}", p)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            # builder modules get the generator's helpers (Bag, sweep, rbox, tube,
            # patch outlines, numpy as np, ...) injected, so they never re-import it
            mod.__dict__.update({k: v for k, v in globals().items() if not k.startswith("__")})
            spec.loader.exec_module(mod)
            cls = getattr(mod, "Builder", Bag)
    B = cls(bag)
    B.build()
    mats = Mats(B)
    tris = make_objects(B, mats)
    os.makedirs(MODELS_DIR, exist_ok=True)
    out = os.path.join(MODELS_DIR, f"{B.bid}.glb")
    export_glb(out)
    mn, mx, tris2, body = measure_glb(out)
    size = os.path.getsize(out)
    S = B.S
    rep = {
        "id": B.bid, "file": out, "size_mb": round(size / 1e6, 2), "tris": tris2,
        "bbox_cm_blender": {"min": mn.round(2).tolist(), "max": mx.round(2).tolist()},
        "overall_cm": {"width_x": round(mx[0] - mn[0], 2), "height_y": round(mx[2] - mn[2], 2),
                       "depth_z": round(mx[1] - mn[1], 2)},
        "body_cm": {"width": round(body[1][0] - body[0][0], 2), "height": round(body[1][2] - body[0][2], 2),
                    "depth": round(body[1][1] - body[0][1], 2), "min_z": round(body[0][2], 3)} if body else None,
        "spec": {"height": S["height"], "width": S["width"], "depth_external": S["depth_external"],
                 "body_depth": S["body_depth"]},
        "straps": getattr(B, "strap_info", None),
        "seconds": round(time.time() - t0, 1),
    }
    dz = rep["overall_cm"]["depth_z"] - float(S["depth_external"])
    rep["checks"] = {
        "body_height_ok": bool(abs(rep["body_cm"]["height"] - S["height"]) < 0.3),
        "body_width_ok": bool(abs(rep["body_cm"]["width"] - S["width"]) < 0.5),
        "depth_ok": bool(abs(dz) <= 1.5), "depth_delta": round(float(dz), 2),
        "size_ok": bool(size <= 6e6), "tris_ok": bool(tris2 <= 150000),
    }
    rep["overlaps"] = same_plane_overlaps(out)
    return rep


def same_plane_overlaps(path):
    """Surfaces of different materials lying in the same plane flicker on the site as the camera turns
    (tools/check_model.py). Returns the list, largest first, or None if the check could not run."""
    try:
        sys.path.insert(0, os.path.join(HERE, "tools"))
        import check_model
        with open(path, "rb") as f:
            js, bn = check_model.parse_glb(f.read())
        return check_model.overlaps(check_model.glb_prims(js, bn))
    except Exception as e:  # the check is advice; never fail a build over it
        print("  (same-plane overlap check skipped: %s)" % e)
        return None


# --------------------------------------------------------------------------------------
# preview rendering (run in a subprocess for robustness)
# --------------------------------------------------------------------------------------
def render_previews(bid, size=900, engine="CYCLES", samples=48):
    path = os.path.join(MODELS_DIR, f"{bid}.glb")
    reset_scene()
    bpy.ops.import_scene.gltf(filepath=path)
    sc = bpy.context.scene
    mn, mx = np.full(3, 1e9), np.full(3, -1e9)
    for ob in sc.objects:
        if ob.type == "MESH":
            co = np.array([ob.matrix_world @ v.co for v in ob.data.vertices])
            mn = np.minimum(mn, co.min(0)); mx = np.maximum(mx, co.max(0))
    c = (mn + mx) / 2
    ext = (mx - mn).max()
    # floor
    bpy.ops.mesh.primitive_plane_add(size=6, location=(0, 0, mn[2]))
    fl = bpy.context.active_object
    fm = bpy.data.materials.new("floor"); fm.use_nodes = True
    fb = next(n for n in fm.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    fb.inputs["Base Color"].default_value = (0.55, 0.56, 0.58, 1); fb.inputs["Roughness"].default_value = 0.8
    fl.data.materials.append(fm)
    world = bpy.data.worlds.new("w"); sc.world = world; world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    bg.inputs["Color"].default_value = (0.62, 0.64, 0.67, 1); bg.inputs["Strength"].default_value = 0.35
    def light(name, typ, loc, energy, size_=1.0):
        ld = bpy.data.lights.new(name, typ); ld.energy = energy
        if typ == "AREA":
            ld.size = size_
        ob = bpy.data.objects.new(name, ld); sc.collection.objects.link(ob); ob.location = loc
        con = ob.constraints.new("TRACK_TO"); con.target = tgt; con.track_axis = "TRACK_NEGATIVE_Z"; con.up_axis = "UP_Y"
        return ob
    tgt = bpy.data.objects.new("tgt", None); sc.collection.objects.link(tgt); tgt.location = c
    light("key", "AREA", (1.2, -1.3, 1.5), 260, 1.2)
    light("fill", "AREA", (-1.5, -0.6, 0.8), 90, 1.5)
    light("rim", "AREA", (-0.6, 1.6, 1.4), 200, 1.0)
    light("rim2", "AREA", (1.4, 1.2, 0.6), 80, 1.0)
    cd = bpy.data.cameras.new("cam"); cd.lens = 50
    cam = bpy.data.objects.new("cam", cd); sc.collection.objects.link(cam); sc.camera = cam
    con = cam.constraints.new("TRACK_TO"); con.target = tgt; con.track_axis = "TRACK_NEGATIVE_Z"; con.up_axis = "UP_Y"
    r = sc.render
    r.resolution_x = r.resolution_y = size
    try:
        r.engine = engine
    except Exception:
        r.engine = "CYCLES"
    if r.engine == "CYCLES":
        sc.cycles.device = "CPU"; sc.cycles.samples = samples
        try:
            sc.cycles.use_denoising = True
        except Exception:
            pass
    try:
        sc.view_settings.view_transform = "AgX"; sc.view_settings.look = "AgX - Medium High Contrast"
    except Exception:
        pass
    sc.view_settings.exposure = 0.0
    dist = ext * 1.9
    os.makedirs(PREVIEW_DIR, exist_ok=True)
    for tag, dvec in (("front", (0.62, -1.0, 0.42)), ("back", (-0.55, 1.0, 0.38))):
        d = nrm(np.array(dvec))
        cam.location = c + d * dist
        r.filepath = os.path.join(PREVIEW_DIR, f"{bid}_{tag}.png")
        bpy.ops.render.render(write_still=True)
    return r.engine


def main():
    argv = sys.argv[1:]
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", action="append")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--render-only")
    ap.add_argument("--engine", default=os.environ.get("BAG_RENDER_ENGINE", "CYCLES"))
    ap.add_argument("--samples", type=int, default=48)
    ap.add_argument("--size", type=int, default=900)
    a = ap.parse_args(argv)
    if a.render_only:
        eng = render_previews(a.render_only, a.size, a.engine, a.samples)
        print("RENDERED", a.render_only, eng)
        return
    with open(BAGS_JSON, encoding="utf-8") as f:
        data = json.load(f)
    bags = data["bags"]
    if a.only:
        bags = [b for b in bags if b["id"] in a.only]
        if not bags:
            sys.exit(f"no bag with id {a.only}")
    reports = []
    for bag in bags:
        rep = build_bag(bag)
        reports.append(rep)
        print("BUILT", json.dumps({k: v for k, v in rep.items() if k != "overlaps"}))
        if rep.get("overlaps") is not None:
            import check_model
            print("same-plane overlaps in %s (they flicker on the site; fix the visible ones):" % bag["id"])
            check_model.print_overlaps(rep["overlaps"], "gltf")
        if not a.no_render:
            for eng in (a.engine, "CYCLES", "BLENDER_WORKBENCH"):
                res = subprocess.run([sys.executable, os.path.abspath(__file__), "--render-only", bag["id"],
                                      "--engine", eng, "--samples", str(a.samples), "--size", str(a.size)],
                                     capture_output=True, text=True)
                if res.returncode == 0 and "RENDERED" in res.stdout:
                    print("PREVIEW", bag["id"], eng)
                    break
                print("render failed with", eng, res.stderr[-800:])
    print("\nSUMMARY")
    for r in reports:
        print(f"{r['id']:20s} {r['size_mb']:5.2f}MB {r['tris']:7d} tris  overall W{r['overall_cm']['width_x']} "
              f"H{r['overall_cm']['height_y']} D{r['overall_cm']['depth_z']}  body W{r['body_cm']['width']} "
              f"H{r['body_cm']['height']} D{r['body_cm']['depth']}  checks {r['checks']}")


if __name__ == "__main__":
    main()
