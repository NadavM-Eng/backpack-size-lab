"""
The Original LTT Backpack (LTTStore) builder, modelled from the reference photos in
references/ltt/ (every number used here is in references/ltt/MEASUREMENTS.md).

build_models.py injects its globals into this module before running it: Bag, sweep, rbox,
tube, project, section, resample, smooth_rows, nrm, sstep, prof_*, rr_outline, rect_outline,
box_uv, grid_faces, Mats, Vector, bpy, np, math, and the rest.

Frame (Blender, cm): Z up, FRONT = -Y, BACK = +Y, width X. +X is the viewer's right seen
from the front, so the zipped side pocket is on +X and the webbing loops and carabiner are on -X.
"""
import bmesh

# front silhouette (front.jpg): (z cm, half-width cm), body H 47.6, W 33
_SIL = np.array([
    (0.0, 13.6), (0.15, 14.9), (0.42, 15.5), (0.99, 16.3), (1.55, 16.6), (2.1, 16.7), (5.0, 16.7),
    (5.6, 16.62), (7.2, 16.55), (12.0, 16.5), (30.0, 16.5), (37.0, 16.45), (39.4, 16.3), (40.5, 16.2),
    (41.7, 15.95), (42.8, 15.7), (43.9, 15.3), (45.06, 14.45), (45.62, 13.75), (46.19, 12.8),
    (46.75, 11.8), (47.03, 10.9), (47.32, 9.6), (47.6, 8.2),
])
_H0, _W0 = 47.6, 33.0
Z_BOT = 5.2          # bottom-panel seam height


# ------------------------------------------------------------------------------------------
# material extension: a hexagonal air-mesh colour texture (kind="hex")
# ------------------------------------------------------------------------------------------
def _hex_material(mats, key, d):
    N = 256
    u, v = np.meshgrid(np.arange(N) / N, np.arange(N) / N)
    f = 8.0
    fv = v * f * 1.1547
    row = np.floor(fv)
    fu = u * f + 0.5 * (row % 2)
    du = np.abs(fu - np.round(fu))
    dv = np.abs(fv - np.floor(fv) - 0.5)
    # hexagon-ish cell distance
    dist = np.maximum(du * 1.0, du * 0.5 + dv * 0.866)
    yarn = sstep(0.30, 0.40, dist)          # 1 on the yarn ridges, 0 in the holes
    c_hole = hex_rgb(d.get("hole", "#0c0c0d"))
    c_yarn = hex_rgb(d["col"])
    rgb = c_hole[None, None, :] * (1 - yarn[..., None]) + c_yarn[None, None, :] * yarn[..., None]
    rgba = np.concatenate([rgb, np.ones((N, N, 1))], -1).astype(np.float32)
    img = bpy.data.images.new(f"{mats.bid}_{key}_hex", N, N, alpha=False)
    img.pixels.foreach_set(rgba.ravel())
    import tempfile as _tf, os as _os
    fp = _os.path.join(_tf.gettempdir(), f"{mats.bid}_{key}_hex_{_os.getpid()}.png")
    img.filepath_raw = fp
    img.file_format = "PNG"
    img.save()
    bpy.data.images.remove(img)
    img = bpy.data.images.load(fp)
    img.name = f"{mats.bid}_{key}_hex"
    img.pack()
    m = bpy.data.materials.new(f"{mats.bid}_{key}")
    try:
        m.use_nodes = True
    except Exception:
        pass
    nt = m.node_tree
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Roughness"].default_value = d["rough"]
    tn = nt.nodes.new("ShaderNodeTexImage"); tn.image = img; tn.location = (-600, 100)
    nt.links.new(tn.outputs["Color"], bsdf.inputs["Base Color"])
    m.diffuse_color = (*srgb2lin(c_yarn), 1.0)
    return m


if not getattr(Mats, "_ltt_patched", False):
    _l_init = Mats.__init__
    _l_get = Mats.get

    def _init(self, bag):
        _l_init(self, bag)
        ex = getattr(bag, "extra_mats", None)
        if ex:
            self.defs.update(ex)

    def _get(self, key):
        if key in self.cache:
            return self.cache[key]
        d = self.defs.get(key, {})
        if d.get("kind") == "hex":
            m = _hex_material(self, key, d)
            self.cache[key] = m
            return m
        return _l_get(self, key)

    Mats.__init__ = _init
    Mats.get = _get
    Mats._ltt_patched = True


def text_mesh(s, height=1.0, depth=0.04):
    cu = bpy.data.curves.new("tmp_txt", "FONT")
    cu.body = s
    cu.size = 1.0
    cu.extrude = depth / 2
    cu.align_x = "CENTER"
    cu.align_y = "CENTER"
    cu.resolution_u = 4
    ob = bpy.data.objects.new("tmp_txt", cu)
    bpy.context.scene.collection.objects.link(ob)
    dg = bpy.context.evaluated_depsgraph_get()
    oe = ob.evaluated_get(dg)
    me = oe.to_mesh()
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    bm.verts.ensure_lookup_table()
    V = np.array([v.co[:] for v in bm.verts], float)
    F = np.array([[v.index for v in f.verts] for f in bm.faces], np.int64)
    bm.free()
    oe.to_mesh_clear()
    bpy.data.objects.remove(ob)
    bpy.data.curves.remove(cu)
    ext = V[:, 1].max() - V[:, 1].min()
    V[:, :2] *= height / max(ext, 1e-6)
    V[:, :2] -= (V[:, :2].max(0) + V[:, :2].min(0)) / 2
    V[:, 2] = V[:, 2] - V[:, 2].min()
    return V, F


def hull2d(P):
    P = sorted(map(tuple, np.round(P, 6)))
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, up = [], []
    for p in P:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(P):
        while len(up) >= 2 and cross(up[-2], up[-1], p) <= 0:
            up.pop()
        up.append(p)
    return np.array(lo[:-1] + up[:-1])


def rr2_outline(cx, x_half, z0, z1, r_top, r_bot, seg=10):
    """rounded rectangle (x, z) with different top / bottom corner radii (CCW)."""
    pts = []
    for (ccx, ccz, r, a0) in ((cx + x_half - r_bot, z0 + r_bot, r_bot, 270), (cx + x_half - r_top, z1 - r_top, r_top, 0),
                              (cx - x_half + r_top, z1 - r_top, r_top, 90), (cx - x_half + r_bot, z0 + r_bot, r_bot, 180)):
        for i in range(seg + 1):
            a = math.radians(a0 + 90 * i / seg)
            pts.append((ccx + r * math.cos(a), ccz + r * math.sin(a)))
    return np.array(pts)


def u_path(cx, x_half, z0, z1, r_top, z_end, seg=14):
    """open path: bottom-left (z_end) -> up -> over the top corners -> down to bottom-right."""
    pts = [(cx - x_half, z) for z in np.linspace(z_end, z1 - r_top, 20)]
    for i in range(seg + 1):
        a = math.radians(180 - 90 * i / seg)
        pts.append((cx - x_half + r_top + r_top * math.cos(a), z1 - r_top + r_top * math.sin(a)))
    pts += [(x, z1) for x in np.linspace(cx - x_half + r_top, cx + x_half - r_top, 30)[1:-1]]
    for i in range(seg + 1):
        a = math.radians(90 - 90 * i / seg)
        pts.append((cx + x_half - r_top + r_top * math.cos(a), z1 - r_top + r_top * math.sin(a)))
    pts += [(cx + x_half, z) for z in np.linspace(z1 - r_top, z_end, 20)]
    return np.array(pts)


class Builder(Bag):
    def __init__(self, bag):
        super().__init__(bag)
        S = self.S
        self.fH = self.H / _H0
        self.fW = self.W / _W0
        self.bulge = 0.5
        self.B0 = self.D / 2                    # 10.5: front face -10.5, back face +10.5
        self.rc = 3.0                           # plan-section corner radius
        self.Rsf = 2.6                          # side-profile top-front radius
        self.Rsb = 2.6
        self.rb = 1.6
        D = self.D
        self.y_front = -self.B0
        self.y_seam = self.B0 - 0.345 * D       # tech block / front block seam
        self.y_tech = self.B0 - 0.23 * D        # tech-compartment zip
        self.y_main = -self.B0 + 0.36 * D       # main (clamshell) zip
        fab = S["fabric"]["color"]
        self.extra_mats = {
            "fabric": dict(col="#1b1c1e", rough=0.72, tex="webbing", ns=0.8),
            "strap": dict(col="#1b1c1e", rough=0.72, tex="webbing", ns=0.6),
            "bottom": dict(col="#3a3b3f", rough=0.4, tex="nylon", ns=0.25),
            "handle": dict(col="#1b1c1e", rough=0.6, tex="webbing", ns=0.6),
            "trim": dict(col="#202124", rough=0.55),
            "groove": dict(col="#09090a", rough=0.7),
            "hexmesh": dict(kind="hex", col="#6f7175", hole="#0d0d0e", rough=0.75),
            "webbing": dict(col="#161718", rough=0.8, tex="webbing", ns=0.9),
            "zip_tape": dict(col="#121213", rough=0.28),
            "zip_teeth": dict(col="#0a0a0b", rough=0.35),
            "slider": dict(col="#3a3b3e", rough=0.35, metal=0.8),
            "pull": dict(col="#a7a9ac", rough=0.33, metal=1.0),
            "tabpull": dict(col="#8e9094", rough=0.3, metal=1.0),
            "metal": dict(col="#a7a9ac", rough=0.33, metal=1.0),
            "plastic": dict(col="#131314", rough=0.45),
            "logo": dict(col="#202124", rough=0.6),
            "logo_mark": dict(col="#2c2d30", rough=0.5),
        }

    # ---- silhouette
    def Aw(self, z):
        zn = np.clip(z / self.fH, 0, _H0)
        return float(np.interp(zn, _SIL[:, 0], _SIL[:, 1])) * self.fW

    def ABB(self, z):
        H, rb = self.H, self.rb
        a = self.Aw(z)
        out = []
        for Rs in (self.Rsf, self.Rsb):
            b = self.B0
            if z > H - Rs:
                b = b - Rs + math.sqrt(max(0.0, Rs * Rs - (z - (H - Rs)) ** 2))
            if z < rb:
                dz = rb - z
                b = b - (rb - math.sqrt(max(0.0, rb * rb - dz * dz)))
            out.append(max(b, 0.6))
        return a, out[0], out[1]

    def AB(self, z):
        a, bf, bb = self.ABB(z)
        return a, 0.5 * (bf + bb)

    # ---- body
    def build_body(self):
        H = self.H
        rings = []
        a0, bf0, bb0 = self.ABB(0.0)
        for s in (0.08, 0.25, 0.45, 0.62, 0.76, 0.87, 0.95):
            rings.append((0.0, a0 * s, (bf0 * s, bb0 * s), s))
        zs = np.concatenate([np.linspace(0, 1.6, 8), np.linspace(1.9, Z_BOT - 0.15, 7), [Z_BOT + 0.15],
                             np.linspace(Z_BOT + 0.6, H - 3.2, 30), np.linspace(H - 3.0, H, 16)])
        for z in zs:
            rings.append((float(z), None, None, 1))
        aT, bfT, bbT = self.ABB(H)
        for s in (0.95, 0.86, 0.72, 0.55, 0.36, 0.17):
            rings.append((H, aT * s, (bfT * s, bbT * s), s))
        grid = []
        for z, a, bb_, s in rings:
            if a is None:
                a, bf, bk = self.ABB(z)
            else:
                bf, bk = bb_
            b = 0.5 * (bf + bk)
            r = max(min(self.rc, 0.985 * min(a, b)), 1e-3)
            P2, N2 = self.ring(a, b, r, nfh=24, ns=20, nc=10)
            x, y = P2[:, 0].copy(), P2[:, 1].copy() + 0.5 * (bk - bf)
            if self.bulge:
                wf = np.clip(-N2[:, 1], 0, 1) ** 2
                wx = np.cos(0.5 * np.pi * np.clip(x / max(a, 1e-6), -1, 1)) ** 0.8
                wz = sstep(0, 0.2 * H, z) * sstep(0, 0.15 * H, H - z)
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
        vol = 0.0
        for FF in (F, Fb, Ft):
            P = V[FF]
            for k in range(1, FF.shape[1] - 1):
                vol += np.einsum("ij,ij->i", P[:, 0], np.cross(P[:, k], P[:, k + 1])).sum()
        if vol < 0:
            F, Fb, Ft = F[:, ::-1], Fb[:, ::-1], Ft[:, ::-1]
        self.M.add("fabric", V, [(FF, box_uv(V, FF, 3.0)) for FF in (F, Fb, Ft)], orient=None, obj="body")
        # bottom panel: a thin coated shell over the body below the seam
        VN = np.zeros_like(V)
        for FF in (F, Fb, Ft):
            P = V[FF]
            fn = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
            if FF.shape[1] == 4:
                fn = fn + np.cross(P[:, 2] - P[:, 0], P[:, 3] - P[:, 0])
            for k in range(FF.shape[1]):
                np.add.at(VN, FF[:, k], fn)
        VN = nrm(VN)
        cen = V[F].mean(1)
        band = cen[:, 2] < Z_BOT
        faces = [F[band], Fb]
        used = np.unique(np.concatenate([f.ravel() for f in faces]))
        remap = np.full(len(V), -1); remap[used] = np.arange(len(used))
        Vs = V[used] + VN[used] * 0.06
        gs = [(remap[f], box_uv(Vs, remap[f], 2.0)) for f in faces]
        self.M.add("bottom", Vs, gs, orient=None, obj="bottom")
        self.body_surf.add(V, [F, Fb, Ft])
        self.surf.add(V, [F, Fb, Ft])

    # ---- zips and pulls
    def aqua_zip(self, P, N, w=1.1, closed=False, stops=True):
        P, N = resample(P, 0.5, closed=closed, extra=[N])
        N = nrm(smooth_rows(N, 3, closed))
        M = self.M
        sweep(M, "zip_tape", P + N * 0.04, N, prof_rrect(w, 0.16, 0.07, 1), closed=closed, tile=1.0)
        sweep(M, "zip_teeth", P + N * 0.12, N, np.array([[0.08, 0.0], [0.04, 0.05], [-0.04, 0.05], [-0.08, 0.0]]),
              closed=closed, tile=1.0)
        if stops and not closed:
            for i, j in ((0, 1), (-1, -2)):
                T = nrm(P[i] - P[j])
                rbox(M, "zip_tape", P[i] + N[i] * 0.12, T, N[i], 0.6, w * 0.75, 0.26, 0.08)
        T = nrm(np.gradient(P, axis=0))
        return P, N, T

    def _frame_at(self, P, N, T, frac, pref, surf=None):
        i = int(np.clip(round(frac * (len(P) - 1)), 1, len(P) - 2))
        p, t = P[i], nrm(T[i])
        _, nb, _, _ = (surf or self.surf).bvh().find_nearest(Vector(p))
        n = nrm(np.array(nb))
        if np.dot(n, N[i]) < 0:
            n = -n
        t = nrm(t - np.dot(t, n) * n)
        pref = np.asarray(pref, float)
        d = pref - np.dot(pref, n) * n
        if np.linalg.norm(d) < 0.25:
            d = t if np.dot(t, pref) >= 0 else -t
        d = nrm(d)
        return p, n, t, d

    def carabiner_pull(self, P, N, T, frac, pref=(0, 0, -1), L=4.4, w=1.65, surf=None):
        """titanium carabiner zip pull: slider + link ring + pear-shaped carabiner lying on the surface."""
        p, n, t, d = self._frame_at(P, N, T, frac, pref, surf)
        M = self.M
        rbox(M, "slider", p + n * 0.36, t, n, 1.35, 1.0, 0.45, 0.17)
        b = nrm(np.cross(n, d))
        # link ring (in the n-d plane) from the slider tab to the carabiner
        rc = p + n * 0.5 + d * 0.85
        ang = np.linspace(0, 2 * np.pi, 14, endpoint=False)
        ring = rc[None] + (np.cos(ang)[:, None] * d * 0.42 + np.sin(ang)[:, None] * n * 0.3)
        tube(M, "pull", ring, b, 0.07, 5, closed=True, obj="carabiners")
        # pear outline in (s along d, q along b)
        rt, rb_ = 0.38, w / 2
        s0 = 1.3
        pts = []
        for cx, r in ((s0 + rt, rt), (s0 + L - rb_, rb_)):
            for a in np.linspace(0, 2 * np.pi, 40, endpoint=False):
                pts.append((cx + r * math.cos(a), 0.18 * (1 if cx > s0 + 1 else 0) + r * math.sin(a)))
        O = hull2d(np.array(pts))
        O = resample(O, 0.12, closed=True)[0]
        O = smooth_rows(O, 1, True, iters=1)
        lift = 0.32 + 0.1 * (O[:, 0] - s0) / L
        C3 = p[None] + d[None] * O[:, :1] + b[None] * O[:, 1:2] + n[None] * lift[:, None]
        tube(M, "pull", C3, np.tile(n, (len(C3), 1)), 0.16, 7, closed=True, obj="carabiners")
        # gate wire along one long side
        g0 = np.array([s0 + 0.55, -rt * 0.55]); g1 = np.array([s0 + L - rb_ * 0.9, -rb_ * 0.62])
        gs = g0 + (g1 - g0) * np.linspace(0, 1, 8)[:, None]
        G3 = p[None] + d[None] * gs[:, :1] + b[None] * gs[:, 1:2] + n[None] * 0.42
        tube(M, "pull", G3, np.tile(n, (8, 1)), 0.07, 5, obj="carabiners")

    def tab_pull(self, P, N, T, frac, pref=(0, 0, -1), L=2.6, w=0.85, surf=None):
        """plain YKK metal tab pull."""
        p, n, t, d = self._frame_at(P, N, T, frac, pref, surf)
        M = self.M
        rbox(M, "slider", p + n * 0.34, t, n, 1.2, 0.9, 0.42, 0.15)
        c = p + n * 0.42 + d * (0.5 + L / 2)
        sweep(M, "tabpull", np.array([p + n * 0.45 + d * 0.45, c + d * L / 2]), np.tile(n, (2, 1)),
              prof_rrect(w, 0.18, 0.08, 2), tile=1.0, obj="tabpulls")

    def loop_zip(self, yv, z_end_l, z_end_r, w=1.1, surf=None):
        H = self.H
        c = (0.0, H * 0.5)
        a_r = math.atan2(z_end_r - c[1], self.A0)
        ang = np.linspace(a_r, math.pi - math.atan2(z_end_l - c[1], self.A0), 700)
        P, N = section(surf or self.body_surf, "y", yv, c, ang)
        return self.aqua_zip(P, N, w)

    def piping_loop(self, yv, r=0.2, mat="trim", sink=0.04):
        ang = np.linspace(0, 2 * np.pi, 800, endpoint=False)
        P, N = section(self.body_surf, "y", yv, (0, self.H * 0.5), ang)
        P, N = resample(P, 0.5, closed=True, extra=[N])
        N = nrm(smooth_rows(N, 3, True))
        tube(self.M, mat, P + N * sink, N, r, 6, closed=True)

    def bottom_seam(self):
        ang = np.linspace(0, 2 * np.pi, 600, endpoint=False)
        pts, nrms = [], []
        for a_ in ang:
            ca, sa = math.cos(a_), math.sin(a_)
            p, n = self.body_surf.ray((300 * ca, 300 * sa, Z_BOT), (-ca, -sa, 0))
            pts.append(p); nrms.append(n)
        P, N = resample(np.array(pts), 0.4, closed=True, extra=[np.array(nrms)])
        N = nrm(smooth_rows(N, 3, True))
        tube(self.M, "trim", P + N * 0.05, N, 0.2, 6, closed=True)

    # ---- front: sunglasses zip, organizer pocket, logo patch
    def sunglasses_zip(self):
        z = 41.1 * self.fH
        xs = np.linspace(-10.7, 12.0, 80)
        P, N = project(self.body_surf, "front", np.stack([xs, np.full(80, z)], 1))
        # welt: raised lips above and below the coated zip
        for dz in (0.78, -0.78):
            Q, Nq = project(self.body_surf, "front", np.stack([np.linspace(-11.0, 12.3, 80), np.full(80, z + dz)], 1))
            sweep(self.M, "trim", Q + Nq * 0.02, Nq, prof_rrect(0.45, 0.32, 0.12, 2), tile=1.0, obj="welt")
        P, N, T = self.aqua_zip(P, N, 1.05)
        self.tab_pull(P, N, T, 0.06, pref=(0, 0, -1), L=2.3, w=0.8)

    def front_pocket(self):
        xh, z0, z1, rt, rb = 12.6, 7.5, 36.9, 2.6, 3.2
        O = rr2_outline(0.0, xh, z0, z1, rt, rb)
        self.patch("front", O, 3.4, 1.7, "fabric", crown=0.04, obj="front_pocket", ds=0.4, tile=3.0)
        # zip around the top and down both side walls (on the rounded wall, ~0.25 cm inside the outline)
        U = u_path(0.0, xh - 0.22, z0, z1 - 0.22, rt - 0.22, z0 + 2.4)
        P, N = project(self.surf, "front", U)
        P, N, T = self.aqua_zip(P, N, 1.0)
        self.carabiner_pull(P, N, T, 0.30, pref=nrm(np.array([-0.8, -0.2, -1.0])), L=3.8, w=1.45)
        # bottom seam piping along the pocket's lower edge
        bot = np.stack([np.linspace(-xh + rb, xh - rb, 40), np.full(40, z0 + 0.25)], 1)
        Pb, Nb = project(self.surf, "front", bot)
        tube(self.M, "trim", Pb + Nb * 0.03, Nb, 0.16, 6)

    def logo(self, fp=None):
        cx, cz, s = 7.9, 31.4, 5.8
        o = rr_outline(cx, cz, s, s + 0.1, 0.55)
        self.patch("front", o, 0.28, 0.18, "logo", obj="logo")
        # embossed square frame
        fr = resample(rr_outline(cx, cz, s - 0.9, s - 0.8, 0.25), 0.2, closed=True)[0]
        P, N = project(self.surf, "front", fr)
        sweep(self.M, "logo_mark", P + N * 0.0, N, prof_rrect(0.28, 0.16, 0.06, 1), closed=True, tile=1.0, obj="logo_mark")
        self.text_on_surface("LTT", "front", cx, cz - 0.05, 3.5, "logo_mark", "logo_mark", depth=0.12, lift=-0.03)

    def text_on_surface(self, s, view, cx, cz, width, mat, obj, udir=(1, 0), depth=0.035, lift=0.02):
        V, F = text_mesh(s, 1.0, depth)
        span = V[:, 0].max() - V[:, 0].min()
        V[:, :2] *= width / span
        u = np.asarray(udir, float); u = u / np.linalg.norm(u)
        v = np.array([-u[1], u[0]])
        q = np.array([cx, cz])[None, :] + V[:, :1] * u[None, :] + V[:, 1:2] * v[None, :]
        loc, nn = project(self.surf, view, q)
        Vw = loc + nn * (lift + V[:, 2:3])
        self.M.add(mat, Vw, [(F, np.zeros((len(F), 3, 2)))], orient="ref", ref=nn, obj=obj)

    # ---- sides
    def side_zips(self):
        H = self.H
        # tech compartment (back block): over the top, down to ~5 cm both sides
        P, N, T = self.loop_zip(self.y_tech, 5.6, 5.6, 1.5)
        self.carabiner_pull(P, N, T, 0.74, pref=(0, 0, -1))
        # main compartment (front block): -X down to the stash seam, +X into the side pocket
        P, N, T = self.loop_zip(self.y_main, 18.9 * self.fH, 30.0 * self.fH, 1.45)
        # two carabiners parked on the top, meeting near the middle (top_carabiner_pulls_open.jpg)
        top = [i for i in range(len(P)) if P[i][2] > self.H - 0.3]
        ia = top[int(len(top) * 0.38)] / (len(P) - 1); ib = top[int(len(top) * 0.62)] / (len(P) - 1)
        self.carabiner_pull(P, N, T, ia, pref=(1, 0, 0), L=3.8, w=1.45)
        self.carabiner_pull(P, N, T, ib, pref=(-1, 0, 0), L=3.8, w=1.45)
        # seam / groove between the tech block and the front block
        self.piping_loop(self.y_seam - 0.35, r=0.17)
        self.piping_loop(self.y_seam + 0.35, r=0.17)
        self.piping_loop(self.y_seam, r=0.3, mat="groove", sink=-0.12)
        # panel-edge piping at the front and back vertical edges
        tcb = self.rc * (1 - math.sin(math.pi / 4))
        self.piping_loop(-self.B0 + tcb, r=0.16)
        self.piping_loop(self.B0 - tcb, r=0.16)

    def side_pocket(self):
        """+X zipped side pocket on the front block (side_with_front_side_pocket.jpg)."""
        y0, y1 = -self.B0 + 2.0, -self.B0 + 13.0
        z0, z1 = 7.1 * self.fH, 30.4 * self.fH
        # outline in (y, z) seen from +X: build with x->y mapping
        O = rr2_outline(0.5 * (y0 + y1), 0.5 * (y1 - y0), z0, z1, 1.8, 2.4)
        self.patch("right", O, 2.2, 1.1, "fabric", crown=0.05, obj="side_pocket", ds=0.4, tile=3.0)
        U = u_path(0.5 * (y0 + y1), 0.5 * (y1 - y0) - 0.15, z0, z1 - 0.15, 1.65, z1 - 7.0)
        P, N = project(self.surf, "right", U)
        P, N, T = self.aqua_zip(P, N, 0.95)
        # two YKK pulls at the top front corner (front = -Y)
        self.tab_pull(P, N, T, 0.30, pref=(0, 0, -1), L=2.2, w=0.75)
        self.tab_pull(P, N, T, 0.35, pref=(0, 0.3, -1), L=2.2, w=0.75)
        # flap seam
        q = np.stack([np.linspace(y0 + 0.6, y1 - 0.6, 40), np.full(40, 22.8 * self.fH)], 1)
        Pq, Nq = project(self.surf, "right", q)
        tube(self.M, "trim", Pq + Nq * 0.02, Nq, 0.13, 6)

    def left_side(self):
        """-X side: stash-pocket seam, two webbing straps and the clipped titanium carabiner."""
        y_b, y_f = self.B0 - 0.335 * self.D, self.B0 - 0.88 * self.D
        q = np.stack([np.linspace(y_f, y_b, 40), np.full(40, 18.9 * self.fH)], 1)
        P, N = project(self.body_surf, "left", q)
        tube(self.M, "trim", P + N * 0.04, N, 0.18, 6)
        for zc in (13.5, 10.7):
            o = rect_outline(y_f, y_b, zc - 1.25, zc + 1.25, 0.15)
            self.patch("left", o, 0.25, 0.12, "webbing", obj="side_webbing", tile=1.0)
            for yy in (y_f + 0.6, y_b - 0.6, 0.5 * (y_f + y_b)):
                loc, n = self.surf.ray(*view_ray("left", (yy, zc)))
                rbox(self.M, "trim", loc + n * 0.02, np.array([0, 0, 1.0]), n, 2.3, 0.6, 0.12, 0.05)
        # carabiner clipped to the lower webbing, hanging down
        yc = self.B0 - 0.62 * self.D
        loc, n = self.surf.ray(*view_ray("left", (yc, 10.7 * self.fH)))
        d = np.array([0, 0, -1.0]); b = nrm(np.cross(n, d))
        L, w, rt = 7.4, 3.0, 0.75
        pts = []
        for cx, r in ((rt, rt), (L - w / 2, w / 2)):
            for a in np.linspace(0, 2 * np.pi, 48, endpoint=False):
                pts.append((cx + r * math.cos(a), r * math.sin(a) + (0.35 if r > 1 else 0)))
        O = resample(hull2d(np.array(pts)), 0.18, closed=True)[0]
        s0 = -0.9
        lift = 0.45 + 0.25 * np.clip((O[:, 0]) / L, 0, 1)
        C3 = loc[None] + d[None] * (O[:, :1] + s0) + b[None] * O[:, 1:2] + n[None] * lift[:, None]
        tube(self.M, "metal", C3, np.tile(n, (len(C3), 1)), 0.22, 8, closed=True, obj="carabiner_big")
        g = np.array([[s0 + 1.2, -rt * 0.8], [s0 + L - w * 0.75, -w * 0.45]])
        gs = g[0] + (g[1] - g[0]) * np.linspace(0, 1, 10)[:, None]
        G3 = loc[None] + d[None] * gs[:, :1] + b[None] * gs[:, 1:2] + n[None] * 0.62
        tube(self.M, "metal", G3, np.tile(n, (10, 1)), 0.13, 6, obj="carabiner_big")

    # ---- top handle: flat padded webbing handle over the tech block
    def top_handle(self):
        span, rise = 5.0, 3.2
        yh = -self.B0 + 0.84 * self.D
        k = 60
        t = np.linspace(-1, 1, k)
        xs = span * np.sign(t) * np.abs(t) ** 0.85
        base, nn = project(self.body_surf, "top", np.stack([xs, np.full(k, yh)], 1))
        lift = (rise - 0.6) * (1 - np.abs(t) ** 2.4) ** 0.6 + 0.3
        C = base.copy()
        C[:, 2] = base[:, 2] + lift
        C = smooth_rows(C, 2, iters=2)
        C[:, 0] = xs
        C[[0, -1], 2] = base[[0, -1], 2] + 0.25
        sweep(self.M, "handle", C, np.tile([0, 1.0, 0], (k, 1)), prof_pillow(1.15, 2.6, 22, 3.0), tile=2.0, obj="handle")
        for sgn in (-1, 1):
            loc, n = self.body_surf.ray(*view_ray("top", (sgn * (span + 0.6), yh)))
            rbox(self.M, "handle", loc + n * 0.12, np.array([1.0, 0, 0]), n, 3.4, 3.0, 0.35, 0.12)

    # ---- back panel: hex air mesh pads, spine channel, band
    def back_panel(self):
        H = self.H
        # band across the back (z 25-28)
        o = rect_outline(-13.6, 13.6, 25.0, 28.0, 0.4)
        self.patch("back", o, 0.3, 0.2, "fabric", obj="back_band", tile=2.0)
        # upper mesh pads either side of the spine channel
        for sgn in (-1, 1):
            o = rect_outline(sgn * 1.7, sgn * 13.6, 28.6, 45.4, 1.6)
            self.patch("back", o, 1.0, 0.7, "hexmesh", crown=0.05, obj="mesh", tile=2.6)
        # lumbar pad
        o = rect_outline(-8.8, 8.8, 1.4, 10.6, 1.6)
        self.patch("back", o, 1.3, 0.8, "hexmesh", crown=0.06, obj="mesh", tile=2.6)
        # spine channel strips (dark webbing) over the lumbar pad
        o = rect_outline(-1.4, 1.4, 1.6, 10.4, 0.3)
        self.patch("back", o, 0.18, 0.1, "webbing", obj="spine", tile=1.0)
        # spine channel at the top (between the mesh strips)
        o = rect_outline(-1.6, 1.6, 28.6, 45.6, 0.4)
        self.patch("back", o, 0.25, 0.15, "webbing", obj="spine", tile=1.0)
        # arched top yoke between the strap roots
        ang = np.linspace(0, np.pi, 30)
        q = np.stack([4.2 * np.cos(ang), 44.6 + 1.6 * np.sin(ang)], 1)
        P, N = project(self.surf, "back", q)
        tube(self.M, "trim", P + N * 0.15, N, 0.3, 6)

    # ---- shoulder straps (bag hanging: upper part vertical, lower part angles to the corners)
    def straps(self, front_min=None):
        H = self.H
        w0, T = 7.3, 2.4
        n = 110
        t = np.linspace(0, 1, n)
        B0 = self.B0
        # side-view centre line (y, z): over the top-back edge, bowing out ~4.6 cm behind the back
        ctrl = np.array([(B0 - 2.6, H + 0.75), (B0 + 0.6, H - 0.7), (B0 + 2.4, 42.5), (B0 + 3.2, 35.0),
                         (B0 + 3.45, 26.0), (B0 + 3.2, 17.0), (B0 + 2.7, 9.5), (B0 + 2.3, 5.6)])
        C4 = np.vstack([2 * ctrl[0] - ctrl[1], ctrl, 2 * ctrl[-1] - ctrl[-2]])
        dense = []
        for k in range(1, len(C4) - 2):
            p0, p1, p2, p3 = C4[k - 1], C4[k], C4[k + 1], C4[k + 2]
            for u in np.linspace(0, 1, 40, endpoint=False):
                dense.append(0.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u * u
                                    + (-p0 + 3 * p1 - 3 * p2 + p3) * u ** 3))
        dense.append(ctrl[-1])
        YZ = resample(np.array(dense), 0.05)[0]
        s_ = arclen(YZ); s_ /= s_[-1]
        yz = np.stack([np.interp(t, s_, YZ[:, 0]), np.interp(t, s_, YZ[:, 1])], 1)
        # back-view x of the centre line vs z (back.jpg)
        xz = np.array([(0.0, 7.9), (5.6, 12.2), (9.5, 11.6), (17.0, 9.6), (25.0, 7.4), (32.0, 7.3),
                       (40.0, 7.6), (49.0, 7.9)])
        sx = 1.0 - 0.22 * sstep(0.45, 1.0, t)
        sy = 0.8 + 0.2 * sstep(0, 0.1, t)
        sy = np.where(t > 0.9, sy * (1 - 0.35 * sstep(0.9, 1.0, t)), sy)
        M = self.M
        self.strap_info = {"padded_end_z": 5.6, "max_standoff": round(float((yz[:, 0] - B0).max() + 0.62 * T), 2)}
        for sgn in (-1, 1):
            x = sgn * np.interp(yz[:, 1], xz[:, 0], xz[:, 1])
            P = np.stack([x, yz[:, 0], yz[:, 1]], 1)
            P = smooth_rows(P, 2, iters=2)
            ups = np.tile([0, 1.0, 0], (n, 1))
            G, Tn, Nn, Bn = sweep(M, "strap", P, ups, prof_strap(w0, T, 28), scale=np.stack([sx, sy], 1),
                                  tile=2.0, obj="straps")
            for e in (1, -1):
                Ep = P + Bn * (e * w0 / 2 * sx * 0.985)[:, None] + Nn * (0.02 * T * sy)[:, None]
                tube(M, "trim", Ep, Nn, 0.2, 6, obj="strap_trim")
            # webbing from the padded end through the silver ladder buckle to the bottom corner
            Pend = P[-1]
            dv = nrm(P[-1] - P[-4])
            bk = Pend + dv * 1.6
            Yv = np.array([0, 1.0, 0])
            bx = nrm(np.cross(Yv, dv))
            rbox(M, "metal", bk + Yv * 0.0, dv, Yv, 2.4, 4.0, 0.3, 0.12, obj="buckles")
            rbox(M, "strap", bk - dv * 0.2 + Yv * 0.05, dv, Yv, 1.5, 3.4, 0.36, 0.12, obj="straps")
            anc, an = self.surf.ray(*view_ray("back", (sgn * 13.6, 1.4)))
            seg = Pend + (anc + an * 0.15 - Pend) * np.linspace(0, 1, 16)[:, None]
            sweep(M, "webbing", seg, np.tile(Yv, (16, 1)), prof_rrect(2.5, 0.16, 0.05, 1), tile=1.0, obj="webbing")
            rbox(M, "bottom", anc + an * 0.2, np.array([0, 0, 1.0]), an, 2.8, 3.2, 0.45, 0.15)
            # tail hanging down from the buckle
            tail = bk + Yv * 0.45 + np.array([sgn * 0.2, 0.15, -1.0]) * np.linspace(0.3, 3.0, 12)[:, None]
            tail[:, 2] = np.maximum(tail[:, 2], 0.2)
            sweep(M, "webbing", tail, np.tile(Yv, (12, 1)), prof_rrect(2.5, 0.16, 0.05, 1), tile=1.0, obj="webbing")

    # ---- assemble
    def build(self):
        self.build_body()
        self.bottom_seam()
        self.sunglasses_zip()
        self.front_pocket()
        self.logo()
        self.side_zips()
        self.side_pocket()
        self.left_side()
        self.top_handle()
        self.back_panel()
        self.front_min = self.M.all_vertices()[:, 1].min()
        self.straps()
