"""
Aer Travel Pack 4 28L builder, modelled from the reference photos in references/aer-tp4/
(see references/aer-tp4/MEASUREMENTS.md for every number used here).

Loaded by build_models.py, which injects its globals (Bag, sweep, rbox, tube, project, section,
resample, smooth_rows, nrm, sstep, prof_*, rr_outline, rect_outline, box_uv, grid_faces, Mats,
Vector, bpy, np, math, ...) into this module before executing it.  Techniques follow
builders/aer.py (City Pack Pro 2: same Aer design language); nothing is imported from it.

Frame (Blender, cm): Z up, FRONT = -Y, BACK = +Y, width X.  Viewer's left (seen from the front)
is -X.  +X ("side_right" photo) carries the bottle pocket and the "Aer" logo tab; the strap with
the tri-ring is the +X strap, the strap with the small "Aer" tab is the -X strap.
"""
import bmesh

# ------------------------------------------------------------------------------------------
# measured profiles (MEASUREMENTS.md).  z is a fraction of the body height H (handle excluded)
# ------------------------------------------------------------------------------------------
_SIL = np.array([(0.0, 0.66), (0.01, 0.70), (0.04, 0.725), (0.064, 0.75), (0.118, 0.814), (0.171, 0.88),
                 (0.225, 0.92), (0.278, 0.95), (0.33, 0.973), (0.385, 0.99), (0.44, 1.0), (0.60, 1.0),
                 (0.73, 0.995), (0.78, 0.978), (0.82, 0.955), (0.86, 0.93), (0.90, 0.895), (0.93, 0.855),
                 (0.96, 0.80), (0.98, 0.72), (0.993, 0.60), (1.0, 0.48)])
_DEP = np.array([(0.0, 0.30), (0.012, 0.335), (0.028, 0.349), (0.062, 0.370), (0.096, 0.387), (0.128, 0.398),
                 (0.162, 0.406), (0.228, 0.423), (0.296, 0.43), (0.362, 0.432), (0.43, 0.438), (0.53, 0.443),
                 (0.596, 0.443), (0.664, 0.447), (0.73, 0.451), (0.80, 0.455), (0.864, 0.455), (0.898, 0.443),
                 (0.932, 0.415), (0.964, 0.385), (0.985, 0.345), (1.0, 0.29)])
_KD = 0.944         # studio side shot reads 21.4 cm at the fullest vs the published 20.3


def _mats_patch():
    if getattr(Mats, "_aer_patched", False):
        return
    _orig_init = Mats.__init__

    def _init(self, bag):
        _orig_init(self, bag)
        ex = getattr(bag, "extra_mats", None)
        if ex:
            self.defs.update(ex)

    Mats.__init__ = _init
    Mats._aer_patched = True


_mats_patch()


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
    sc = height / max(ext, 1e-6)
    V[:, :2] *= sc
    V[:, :2] -= (V[:, :2].max(0) + V[:, :2].min(0)) / 2
    V[:, 2] = V[:, 2] - V[:, 2].min()
    return V, F


def catmull(ctrl, per=30):
    """Catmull-Rom through the control points (any dimension)."""
    ctrl = np.asarray(ctrl, float)
    C4 = np.vstack([2 * ctrl[0] - ctrl[1], ctrl, 2 * ctrl[-1] - ctrl[-2]])
    out = []
    for k in range(1, len(C4) - 2):
        p0, p1, p2, p3 = C4[k - 1], C4[k], C4[k + 1], C4[k + 2]
        for u in np.linspace(0, 1, per, endpoint=False):
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u * u
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * u ** 3))
    out.append(ctrl[-1])
    return np.array(out)


def seg_dist(Q, a, b):
    """distance of 2-D points Q to segment a-b, and the segment parameter"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ab = b - a
    t = np.clip(((Q - a) @ ab) / max(ab @ ab, 1e-9), 0, 1)
    d = np.linalg.norm(Q - (a + t[:, None] * ab), axis=1)
    return d, t


def prof_strap_q(w, h, n=48, groove=0.22, gdepth=0.14):
    """padded strap section with two stitched channels on the domed outer face"""
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    c, s = np.cos(a), np.sin(a)
    u = w / 2 * np.sign(c) * np.abs(c) ** 0.5
    v = np.where(s >= 0, 0.6 * h * np.abs(s) ** 0.75, -0.4 * h * np.abs(s) ** 0.45)
    g = np.exp(-((np.abs(u) - groove * w) / (0.035 * w)) ** 2) * (s > 0)
    v = v - gdepth * h * g
    return np.stack([u, v], 1)


class Builder(Bag):
    def __init__(self, bag):
        super().__init__(bag)
        self.H = 47.0                     # body height; the padded top handle adds 1.9 -> 48.9
        self.H_total = 48.9
        self.W = 32.4
        self.A0 = self.W / 2
        self.sil = _SIL
        self.dep = _DEP
        self.Dmax = float(self.dep[:, 1].max()) * self.H * _KD
        self.Yb = 0.5 * self.Dmax
        self.pad_t = 1.0
        self.bulge = 0.9
        self.rc = 4.6
        self.Rsb = 2.5
        self.rb = 2.4
        D = self.Dmax
        self.y_cz = self.Yb - 0.71 * D       # clamshell zip
        self.y_lap = self.Yb - 0.21 * D      # laptop loop zip
        self.y_band = self.Yb - 0.32 * D     # coated strip in front of it
        self.y_band2 = self.Yb - 0.15 * D    # short coated strip behind it
        fab = "#18191b"
        self.extra_mats = {
            "fabric": dict(col=fab, rough=0.6, tex="ballistic", ns=0.85),
            "strap": dict(col=fab, rough=0.62, tex="ballistic", ns=0.45),
            "strapmesh": dict(col="#2b2c2f", rough=0.85, tex="mesh", ns=1.5),
            "crease": dict(col="#0b0b0c", rough=0.7),
            "coated": dict(col="#121314", rough=0.3),
            "trim": dict(col="#1b1c1e", rough=0.5, tex="webbing", ns=0.4),
            "pad": dict(col="#28292c", rough=0.85, tex="spacer", ns=1.3),
            "mesh": dict(col="#2a2b2e", rough=0.82, tex="mesh", ns=1.6),
            "webbing": dict(col="#151617", rough=0.78, tex="webbing", ns=0.9),
            "zip_tape": dict(col="#101112", rough=0.4),
            "zip_teeth": dict(col="#0a0a0b", rough=0.4),
            "slider": dict(col="#1a1b1d", rough=0.3, metal=0.4),
            "pull": dict(col="#1a1b1d", rough=0.42),
            "plastic": dict(col="#141415", rough=0.42),
            "metal": dict(col="#46474b", rough=0.3, metal=1.0),
            "reflect": dict(col="#4a4c50", rough=0.38, metal=0.15),
            "label": dict(col="#0e0e0f", rough=0.35),
            "logo_white": dict(col="#e9e9e9", rough=0.5),
        }

    # ---------------------------------------------------------------- silhouette
    def Aw(self, z):
        return float(np.interp(np.clip(z / self.H, 0, 1), self.sil[:, 0], self.sil[:, 1])) * self.A0

    def Dz(self, z):
        return float(np.interp(np.clip(z / self.H, 0, 1), self.dep[:, 0], self.dep[:, 1])) * self.H * _KD

    def Yback(self, z):
        H, R, rb = self.H, self.Rsb, self.rb
        y = self.Yb
        if z > H - R:
            y -= R - math.sqrt(max(0.0, R * R - (z - (H - R)) ** 2))
        if z < rb:
            y -= rb - math.sqrt(max(0.0, rb * rb - (rb - z) ** 2))
        return y

    def ABB(self, z):
        yb = self.Yback(z)
        D = self.Dz(z) - self.pad_t - self.bulge * sstep(0.0, 0.2, z / self.H)
        return self.Aw(z), self.Yb - D, yb

    # ---------------------------------------------------------------- body
    def build_body(self):
        H = self.H
        rings = []
        for s in (0.08, 0.25, 0.45, 0.62, 0.76, 0.87, 0.95):
            rings.append((0.0, s))
        zs = np.concatenate([np.linspace(0, 1.4, 7), np.linspace(1.7, H - 6.0, 62), np.linspace(H - 5.6, H, 24)])
        for z in zs:
            rings.append((float(z), 1.0))
        for s in (0.95, 0.86, 0.72, 0.55, 0.36, 0.17):
            rings.append((H, s))
        grid = []
        for z, s in rings:
            a, yf, yb = self.ABB(z)
            yc = 0.5 * (yf + yb)
            b = 0.5 * (yb - yf)
            a, b = a * s, b * s
            r = max(min(self.rc, 0.985 * min(a, b)), 1e-3)
            P2, N2 = self.ring(a, b, r, nfh=30, ns=16, nc=11)
            x, y = P2[:, 0].copy(), P2[:, 1].copy() + yc
            if self.bulge and s == 1.0:
                wf = np.clip(-N2[:, 1], 0, 1) ** 2
                wx = np.cos(0.5 * np.pi * np.clip(x / max(a, 1e-6), -1, 1)) ** 0.8
                wz = sstep(0, 0.2, z / H) * sstep(0, 0.12, 1 - z / H)
                y = y - self.bulge * wf * wx * wz
            grid.append(np.stack([x, y, np.full_like(x, z)], 1))
        G = np.array(grid)
        R, C, _ = G.shape
        V = G.reshape(-1, 3)
        self.soften(V)
        F, _ = grid_faces(R, C, True, False)
        cb = len(V); ct = cb + 1
        a0, yf0, yb0 = self.ABB(0.0)
        aT, yfT, ybT = self.ABB(H)
        V = np.vstack([V, [[0, 0.5 * (yf0 + yb0), 0]], [[0, 0.5 * (yfT + ybT), H]]])
        j = np.arange(C)
        Fb = np.stack([np.full(C, cb), (j + 1) % C, j], 1)
        top = (R - 1) * C
        Ft = np.stack([np.full(C, ct), top + j, top + (j + 1) % C], 1)
        self.M.add("fabric", V, [(F, box_uv(V, F, 2.0)), (Fb, box_uv(V, Fb, 2.0)), (Ft, box_uv(V, Ft, 2.0))],
                   orient="closed", obj="body")
        self.body_surf.add(V, [F, Fb, Ft])
        self.surf.add(V, [F, Fb, Ft])

    def soften(self, V):
        """Soft-goods look: the front panel puffs up inside its seam, pinched creases radiate from the
        darts, a soft sag fold lies under the front-pocket zip, the lower corners slump, and the side
        walls carry diagonal compression creases."""
        from scipy.spatial import cKDTree
        O = resample(self.front_outline(), 0.15, closed=True)[0]
        tr = cKDTree(O)
        H = self.H
        x, z = V[:, 0], V[:, 2]
        abb = np.array([self.ABB(zz) for zz in z])
        yc = 0.5 * (abb[:, 1] + abb[:, 2])
        aw = abb[:, 0]
        front = (V[:, 1] < yc) & (z > 0.3) & (z < H - 0.05)
        d, _ = tr.query(np.stack([x, z], 1))
        wf = np.clip((yc - V[:, 1]) / 6.0, 0, 1)
        fm = front * wf
        V[:, 1] += fm * 0.45 * np.exp(-(d / 0.75) ** 2)          # sink along the panel seam
        Q = np.stack([x, z], 1)
        inner = sstep(0.4, 2.5, d)
        dent = np.zeros(len(V))
        zf = 0.60 * H
        # creases: (start, end, width, depth)
        cr = []
        for sg in (-1, 1):
            cr += [((sg * 11.6, 0.94 * H), (sg * 5.0, 0.865 * H), 0.55, 0.30),     # upper dart, long fade
                   ((sg * 9.0, 0.90 * H), (sg * 3.0, 0.80 * H), 0.9, 0.12),        # radiating from the dart
                   ((sg * 12.5, 0.82 * H), (sg * 7.5, 0.72 * H), 0.9, 0.10),
                   ((sg * 12.6, 0.13 * H), (sg * 8.0, 0.05 * H), 0.6, 0.22),       # lower corner darts
                   ((sg * 11.5, 0.30 * H), (sg * 7.0, 0.16 * H), 1.0, 0.12),       # lower slump
                   ((sg * 4.0, 0.45 * H), (sg * 9.5, 0.38 * H), 1.2, 0.08)]
        cr += [((-11.0, zf - 1.6), (11.0, zf - 1.6), 0.9, 0.16),                  # sag under the pocket zip
               ((-9.0, zf - 4.2), (8.0, zf - 3.6), 1.6, 0.07)]
        for a, b, wdt, dep in cr:
            dd, t = seg_dist(Q, a, b)
            fade = np.sin(np.pi * np.clip(t, 0, 1)) ** 0.5
            dent += dep * fade * np.exp(-(dd / wdt) ** 2)
        V[:, 1] += fm * inner * dent * 2.0
        # puffiness between the creases + low-frequency wrinkles
        wr = (0.12 * np.sin(0.55 * x + 0.31 * z) * np.sin(0.23 * z + 0.4)
              + 0.08 * np.sin(0.9 * x - 0.6 * z + 1.3) + 0.05 * np.sin(1.4 * z + 0.7 * x)
              + 0.035 * np.sin(2.3 * x + 1.7 * z))
        V[:, 1] -= fm * inner * wr * 1.4
        # side walls: diagonal compression creases + low-frequency waviness
        side = np.clip((np.abs(x) - 0.82 * aw) / (0.18 * aw + 1e-6), 0, 1)
        yy = V[:, 1]
        ws = (0.18 * np.sin(0.42 * yy + 0.36 * z + 0.6) * np.sin(0.21 * z)
              + 0.08 * np.sin(0.9 * yy - 0.5 * z + 2.0) + 0.04 * np.sin(1.9 * z + 0.8 * yy))
        V[:, 0] += side * np.sign(x) * ws * sstep(0.03 * H, 0.12 * H, z) * sstep(0.0, 0.08 * H, H - z)
        # top surface: gentle undulation
        topw = sstep(H - 1.5, H - 0.2, z)
        V[:, 2] += topw * 0.12 * np.sin(0.35 * x + 0.4 * yy) * np.cos(0.3 * x)

    # ---------------------------------------------------------------- zips / pulls
    def aqua_zip(self, P, N, w=1.1, closed=False, stops=True):
        P, N = resample(P, 0.4, closed=closed, extra=[N])
        N = nrm(smooth_rows(N, 3, closed))
        M = self.M
        sweep(M, "zip_tape", P + N * 0.04, N, prof_rrect(w, 0.14, 0.06, 1), closed=closed, tile=1.0)
        sweep(M, "zip_teeth", P + N * 0.1, N, np.array([[0.08, 0.0], [0.05, 0.05], [-0.05, 0.05], [-0.08, 0.0]]),
              closed=closed, tile=1.0)
        if stops and not closed:
            for i, j in ((0, 1), (-1, -2)):
                T = nrm(P[i] - P[j])
                rbox(M, "zip_tape", P[i] + N[i] * 0.12, T, N[i], 0.5, w * 0.7, 0.22, 0.08)
        T = nrm(np.gradient(P, axis=0))
        return P, N, T

    def tab_pull(self, P, N, T, frac, pref=(0, 0, -1), L=6.0, w=1.15, garage=False, surf=None):
        surf = surf or self.body_surf
        i = int(np.clip(round(frac * (len(P) - 1)), 1, len(P) - 2))
        p, t = P[i], nrm(T[i])
        _, nb, _, _ = surf.bvh().find_nearest(Vector(p))
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
        rbox(M, "slider", p + n * 0.38, t, n, 1.3, 0.95, 0.45, 0.17)
        k = 12
        s = np.linspace(0, 1, k)
        base = p + n * 0.55 + d[None, :] * (0.5 + s[:, None] * L)
        for q in range(k):
            loc, nn, _, _ = surf.bvh().find_nearest(Vector(base[q]))
            loc = np.array(loc); nn = nrm(np.array(nn))
            if np.dot(nn, n) < 0:
                nn = -nn
            base[q] = loc + nn * (0.5 - 0.15 * s[q])
        prof = prof_rrect(w, 0.26, 0.11, 2)
        sc = np.stack([np.clip(np.where(s < 0.15, 0.7 + 2 * s, np.where(s > 0.85, 1.0 - 0.3 * (s - 0.85) / 0.15, 1.0)),
                               0.6, 1.0), np.ones(k)], 1)
        sweep(M, "pull", base, np.tile(n, (k, 1)), prof, scale=sc, tile=1.0, obj="pulls")
        if garage:
            rbox(M, "coated", p - t * 1.1 + n * 0.3, t, n, 2.2, 1.9, 0.6, 0.25)

    def loop_zip(self, yv, z_end_l, z_end_r, w=1.1):
        H = self.H
        c = (0.0, H * 0.5)
        a_r = math.atan2(z_end_r - c[1], self.A0)
        ang = np.linspace(a_r, math.pi - math.atan2(z_end_l - c[1], self.A0), 700)
        P, N = section(self.body_surf, "y", yv, c, ang)
        return self.aqua_zip(P, N, w)

    def coated_strip(self, yv, z_end, w=2.6):
        c = (0.0, self.H * 0.5)
        ang = np.linspace(math.atan2(z_end - c[1], self.A0), math.pi - math.atan2(z_end - c[1], self.A0), 600)
        Pb, Nb = section(self.body_surf, "y", yv, c, ang)
        Pb, Nb = resample(Pb, 0.4, extra=[Nb])
        Nb = nrm(smooth_rows(Nb, 3))
        sweep(self.M, "coated", Pb + Nb * 0.05, Nb, prof_rrect(w, 0.14, 0.06, 1), tile=1.0, obj="coated_strip")

    # ---------------------------------------------------------------- front panel
    def front_outline(self, inset_side=2.8, inset_top=2.4, z_bot=1.3):
        H = self.H
        zz = np.linspace(z_bot + 1.5, H, 300)
        a = np.array([self.Aw(z) for z in zz])
        Pr = np.stack([a, zz], 1)
        T = nrm(np.gradient(Pr, axis=0))
        Nn = np.stack([T[:, 1], -T[:, 0]], 1)
        d = inset_top + (inset_side - inset_top) * np.abs(Nn[:, 0]) ** 2
        Q = Pr - Nn * d[:, None]
        Q = Q[(Q[:, 1] <= H - inset_top) & (Q[:, 0] > 0.05)]
        Q = np.vstack([[[Q[0, 0] - 1.2, z_bot]], Q, [[0.0, Q[-1, 1]]]])
        Q = smooth_rows(Q, 3, iters=2)
        Q[0] = (self.Aw(z_bot) - inset_side - 1.6, z_bot)
        right = Q
        left = Q[::-1].copy(); left[:, 0] *= -1
        bot = np.stack([np.linspace(-Q[0, 0], Q[0, 0], 30)[1:-1], np.full(28, z_bot)], 1)
        O = np.vstack([right, left[1:], bot])
        return resample(O, 0.35, closed=True)[0]

    def front_panel(self):
        H = self.H
        O = self.front_outline()
        P, N = project(self.body_surf, "front", O)
        P, N = resample(P, 0.4, closed=True, extra=[N])
        N = nrm(smooth_rows(N, 3, True))
        tube(self.M, "trim", P + N * 0.04, N, 0.2, 6, closed=True, obj="piping")
        zf = 0.60 * H
        xl, xr = -self.Aw(zf) + 4.0, self.Aw(zf) - 4.0
        xs = np.linspace(xl, xr, 80)
        zq = zf + 0.5 * ((xs - 0.5 * (xl + xr)) / (0.5 * (xr - xl))) ** 2
        P, N = project(self.body_surf, "front", np.stack([xs, zq], 1))
        P, N, T = self.aqua_zip(P, N, 1.25)
        sweep(self.M, "coated", P + N * 0.02, N, prof_rrect(2.0, 0.1, 0.04, 1), tile=1.0)
        self.tab_pull(P, N, T, 0.03, pref=(0, 0, -1), L=7.5, w=1.2, garage=True)
        # darts: dark pinch line + soft ridges either side
        darts = [((11.2, 0.935 * H), (6.8, 0.885 * H)), ((12.4, 0.12 * H), (9.2, 0.06 * H))]
        for (x0, z0), (x1, z1) in darts:
            for sg in (-1, 1):
                q = np.stack([np.linspace(sg * x0, sg * x1, 24), np.linspace(z0, z1, 24)], 1)
                Pd, Nd = project(self.body_surf, "front", q)
                Nd = nrm(smooth_rows(Nd, 2))
                s = np.linspace(0, 1, len(Pd))
                fade = np.sin(np.pi * np.clip(s * 1.05, 0, 1)) ** 0.4
                sweep(self.M, "crease", Pd - Nd * 0.02, Nd, prof_rrect(0.16, 0.12, 0.05, 1),
                      scale=np.stack([0.4 + 0.6 * fade, np.ones(len(s))], 1), tile=1.0, obj="creases")
                Tn = nrm(np.gradient(Pd, axis=0))
                Bn = nrm(np.cross(Nd, Tn))
                for side in (-1, 1):
                    sweep(self.M, "fabric", Pd + Bn * side * 0.3 - Nd * 0.05, Nd,
                          prof_pillow(0.55, 0.36, 10, 2.4),
                          scale=np.stack([fade, fade], 1), tile=2.0, obj="creases_ridge")

    def top_zip(self):
        aT, yfT, ybT = self.ABB(self.H)
        yq = self.y_cz + 3.0
        xm = 9.5
        xs = np.linspace(-xm, xm, 60)
        q = np.stack([xs, np.full(60, yq) - 0.7 * (1 - (xs / xm) ** 2)], 1)
        P, N = project(self.body_surf, "top", q)
        P, N, T = self.aqua_zip(P, N, 1.15)
        sweep(self.M, "coated", P + N * 0.02, N, prof_rrect(1.9, 0.1, 0.04, 1), tile=1.0)
        self.tab_pull(P, N, T, 0.93, pref=(1, 0, -0.2), L=4.5, w=1.1)

    # ---------------------------------------------------------------- sides
    def side_zips(self):
        H = self.H
        P, N, T = self.loop_zip(self.y_cz, 0.04 * H, 0.04 * H, 1.15)
        self.tab_pull(P, N, T, 0.285, pref=(0, 0, -1), L=6.5)   # +X pulls parked at ~0.78 H (side_right)
        self.tab_pull(P, N, T, 0.80, pref=(0, 0, -1), L=6.5)    # -X pulls at ~0.55 H (34hero)
        P, N, T = self.loop_zip(self.y_lap, 0.60 * H, 0.60 * H, 1.1)
        self.tab_pull(P, N, T, 0.10, pref=(0, 0, -1), L=6.0)
        self.tab_pull(P, N, T, 0.90, pref=(0, 0, -1), L=6.0)
        self.coated_strip(self.y_band, 0.60 * H, 2.5)
        self.coated_strip(self.y_band2, 0.74 * H, 1.6)
        ang = np.linspace(0, 2 * np.pi, 800, endpoint=False)
        Pp, Np = section(self.body_surf, "y", self.Yb - 1.2, (0, H * 0.5), ang)
        Pp, Np = resample(Pp, 0.5, closed=True, extra=[Np])
        Np = nrm(smooth_rows(Np, 3, True))
        tube(self.M, "trim", Pp + Np * 0.04, Np, 0.18, 6, closed=True, obj="piping")

    def side_x(self, side):
        return "right" if side > 0 else "left"

    def side_pocket(self, side, zf, zb):
        """Stretch side pocket from the clamshell zip to the back, diagonal top edge, vertical pleats."""
        view = self.side_x(side)
        y0, y1 = self.y_cz + 0.8, self.Yb - 0.6
        nu, nv = 40, 24
        u = np.linspace(0, 1, nu)
        rows, refs = [], []
        for jv, v in enumerate(np.linspace(0, 1, nv)):
            zt = zf + (zb - zf) * u
            zz = 1.0 + v * (zt - 1.0)
            loc, nn = project(self.body_surf, view, np.stack([y0 + u * (y1 - y0), zz], 1))
            # bulge + two soft vertical pleats (stretch panel gathered at the bottom)
            pleat = 0.22 * (np.exp(-((u - 0.40) / 0.05) ** 2) + np.exp(-((u - 0.58) / 0.05) ** 2)) * (1 - 0.6 * v)
            fold = -0.15 * np.exp(-((u - 0.49) / 0.04) ** 2) * (1 - v)
            h = 0.08 + 0.5 * np.sin(np.pi * u) ** 0.6 * sstep(0, 0.4, v) * sstep(0, 0.1, 1 - v + 0.05)
            h = h + (pleat + fold) * sstep(0.0, 0.15, v)
            if jv == 0:
                h = np.full(nu, 0.05)
            rows.append(loc + nn * h[:, None]); refs.append(nn)
        G = np.array(rows).transpose(1, 0, 2)
        Rf = np.array(refs).transpose(1, 0, 2)
        V = G.reshape(-1, 3)
        F, _ = grid_faces(nu, nv, False, False)
        self.M.add("fabric", V, [(F, box_uv(V, F, 2.0))], orient="ref", ref=Rf.reshape(-1, 3), obj="side_pockets")
        self.surf.add(V, [F])
        top = G[:, -1]; nt = nrm(Rf[:, -1])
        sweep(self.M, "trim", top + nt * 0.1, nt, prof_rrect(0.9, 0.35, 0.14, 2), tile=1.0, obj="piping")

    def d_ring(self, side):
        view = self.side_x(side)
        yd, zd = self.Yb - 0.54 * self.Dmax, 0.84 * self.H
        loc, n = self.body_surf.ray(*view_ray(view, (yd, zd)))
        down = np.array([0, 0, -1.0]); down = nrm(down - np.dot(down, n) * n)
        rbox(self.M, "webbing", loc + n * 0.12 + down * 0.3, down, n, 2.4, 2.4, 0.24, 0.08)
        b = nrm(np.cross(n, down))
        base = loc + n * 0.42 + down * 1.3
        pts = [base + b * x for x in np.linspace(-1.05, 1.05, 6)]
        for t in np.linspace(0, np.pi, 18)[1:-1]:
            pts.append(base + b * 1.05 * math.cos(t) + down * 1.9 * math.sin(t))
        tube(self.M, "metal", np.array(pts), n, 0.17, 8, closed=True, obj="hardware")

    def side_handle(self, side):
        """padded coated grab handle on the front part of each side at 0.60 H"""
        view = self.side_x(side)
        z = 0.60 * self.H
        a, yf, yb = self.ABB(z)
        ya, yb_ = yf + 1.2, self.y_cz - 0.6
        k = 30
        ys = np.linspace(ya, yb_, k)
        loc, nn = project(self.body_surf, view, np.stack([ys, np.full(k, z)], 1))
        nn = nrm(smooth_rows(nn, 2))
        s = np.linspace(0, 1, k)
        C = loc + nn * (0.2 + 0.75 * np.sin(np.pi * s) ** 0.7)[:, None]
        up = np.tile([0, 0, 1.0], (k, 1))
        sweep(self.M, "coated", C, nn, prof_pillow(2.8, 0.55, 18, 3.0), tile=1.0, obj="side_handles")
        for i in (0, k - 1):
            rbox(self.M, "trim", loc[i] + nn[i] * 0.15, up[i], nn[i], 3.2, 1.6, 0.3, 0.1)

    def logo_tab(self):
        H = self.H
        yl = self.y_cz + 1.6
        zc = 0.23 * H
        o = rr_outline(yl, zc, 1.5, 4.2, 0.3)
        self.patch("right", o, 0.12, 0.08, "label", obj="label")
        V, F = text_mesh("Aer", 1.0, 0.02)
        span = V[:, 0].max() - V[:, 0].min()
        V[:, :2] *= 2.0 / span
        q = np.stack([yl - V[:, 1], zc + V[:, 0]], 1)
        loc, nn = project(self.surf, "right", q)
        Vw = loc + nn * (0.02 + V[:, 2:3])
        self.M.add("logo_white", Vw, [(F, np.zeros((len(F), 3, 2)))], orient="ref", ref=nn, obj="logo")

    def bottom_tabs(self):
        """flush webbing tabs at the bottom-front corners and a flat bottom handle"""
        H = self.H
        for sg in (-1, 1):
            x = sg * 12.4
            q = np.stack([np.full(10, x), np.linspace(0.06 * H, 0.2, 10)], 1)
            P, N = project(self.surf, "front", q)
            sweep(self.M, "webbing", P + N * 0.25, N, prof_rrect(2.4, 0.3, 0.1, 1), tile=1.0, obj="webbing")
        a0, yf0, yb0 = self.ABB(0.0)
        yh = yb0 - 4.0
        xs = np.linspace(-6, 6, 20)
        loc, nn = project(self.body_surf, "top", np.stack([xs, np.full(20, yh)], 1))   # top ray -> flip below
        # bottom surface: build directly under the body
        Pb = np.stack([xs, np.full(20, yh), np.full(20, -0.18)], 1)
        sweep(self.M, "webbing", Pb, np.tile([0, 0, -1.0], (20, 1)), prof_rrect(3.0, 0.35, 0.12, 1), tile=1.0,
              obj="webbing")

    # ---------------------------------------------------------------- top handle
    def top_handle(self):
        H = self.H
        span = 9.3
        yh = self.Yb - 1.3
        k = 70
        t = np.linspace(-1, 1, k)
        xs = span * t
        base, nn = project(self.body_surf, "top", np.stack([xs, np.full(k, yh)], 1))
        lift = (self.H_total - H - 0.9) * (1 - np.abs(t) ** 2.2) ** 0.6
        C = base.copy()
        C[:, 2] += lift + 0.45
        C[:, 1] += 0.3 * (1 - np.abs(t) ** 2)
        C = smooth_rows(C, 2, iters=2)
        ups = np.tile([0, 0.3, 1.0], (k, 1))
        sc = np.stack([1 - 0.25 * np.abs(t) ** 4, 1 - 0.3 * np.abs(t) ** 4], 1)
        sweep(self.M, "strap", C, ups, prof_pillow(3.4, 1.0, 22, 3.6), scale=sc, tile=2.0, obj="handle")
        self.anchor = {}
        for sg in (-1, 1):
            i = 0 if sg < 0 else k - 1
            rbox(self.M, "trim", base[i] + nn[i] * 0.2, np.array([0, 1.0, 0]), nn[i], 3.4, 2.6, 0.35, 0.12)
            self.anchor[sg] = (base[i] + nn[i] * 0.4, nn[i])

    # ---------------------------------------------------------------- back
    def back_panel(self):
        H = self.H
        th = self.pad_t
        xf = 13.2
        g = 1.4
        blocks = [(g, xf, 0.50 * H, 0.95 * H, 3.2, "mesh", th + 0.4),
                  (g, xf, 0.10 * H, 0.44 * H, 3.0, "pad", th + 1.4)]
        for x0, x1, z0, z1, rr, mat, tk in blocks:
            for sgn in (-1, 1):
                o = rect_outline(min(sgn * x0, sgn * x1), max(sgn * x0, sgn * x1), z0, z1, rr)
                self.patch("back", o, tk, 1.2 if mat == "pad" else 0.7, mat, crown=0.08, obj="back_" + mat,
                           ds=0.6, tile=4.0)
        o = rect_outline(-g, g, 0.10 * H, 0.95 * H, 0.6)
        self.patch("back", o, 0.5, 0.3, "mesh", obj="back_mesh", tile=4.0)
        # luggage pass-through: lay-flat band across the back
        xl = 15.0
        lz0, lz1 = 0.45 * H, 0.50 * H
        o = rect_outline(-xl, xl, lz0, lz1, 0.5)
        self.patch("back", o, 0.45, 0.25, "webbing", obj="luggage", tile=1.0)
        for sgn in (-1, 1):
            loc, n = self.surf.ray(*view_ray("back", (sgn * (xl - 0.5), 0.5 * (lz0 + lz1))))
            rbox(self.M, "trim", loc + n * 0.05, np.array([0, 0, 1.0]), n, lz1 - lz0 + 0.5, 1.2, 0.3, 0.1)
        # stitched seam line of the bottom base panel (shallow V)
        xs = np.linspace(-13.5, 13.5, 40)
        zs = 0.10 * H - 0.6 + 1.2 * np.abs(xs / 13.5)
        P, N = project(self.surf, "back", np.stack([xs, zs], 1))
        tube(self.M, "trim", P + N * 0.03, N, 0.16, 6, obj="piping")

    # ---------------------------------------------------------------- harness
    def strap_path(self, sgn, yp, n):
        """centre line of one padded shoulder strap (x, y, z), sampled uniformly by arc length"""
        ctrl = np.array([(4.8, -0.8, 44.6), (5.1, 0.8, 46.3), (5.7, 3.0, 46.9), (6.8, 5.5, 46.0),
                         (8.1, 7.4, 43.5), (9.2, 9.1, 40.1), (10.0, 11.0, 35.5), (10.3, 11.8, 30.8),
                         (10.6, 11.8, 26.2), (11.2, 10.9, 22.0), (12.0, 9.4, 18.4), (12.8, 7.8, 15.2),
                         (13.3, 6.4, 13.0)])
        ctrl[:, 0] *= sgn
        ctrl[:, 1] += yp
        D = catmull(ctrl, 40)
        D = resample(D, 0.05)[0]
        s_ = arclen(D); s_ /= s_[-1]
        t = np.linspace(0, 1, n)
        return np.stack([np.interp(t, s_, D[:, k]) for k in range(3)], 1), t

    def straps(self):
        H = self.H
        w, T = 7.5, 2.7
        n = 112
        lp, _ = self.surf.ray(*view_ray("back", (0.0, 0.7 * H)))
        yp = float(lp[1])
        M = self.M
        frames = {}
        for sgn in (-1, 1):
            P, t = self.strap_path(sgn, yp, n)
            # width: narrow root, full 7.5 cm over the shoulder, taper to the padded end
            sx = (0.62 + 0.38 * sstep(0.0, 0.18, t)) * (1 - 0.22 * sstep(0.62, 0.97, t))
            sy = (0.55 + 0.45 * sstep(0.0, 0.1, t)) * (1 - 0.35 * sstep(0.7, 1.0, t))
            sy = np.where(t > 0.965, sy * (1 - 0.5 * sstep(0.965, 1.0, t)), sy)
            # soft transverse puckers where the strap bends over the shoulder
            puck = np.exp(-((t - 0.16) / 0.09) ** 2)
            sy = sy * (1 + 0.07 * puck * np.sin(2 * np.pi * t * 34))
            sx = sx * (1 + 0.015 * np.sin(2 * np.pi * t * 9 + sgn))
            # outward direction: away from the wearer's shoulder region
            ctr = np.array([sgn * 7.0, yp + 2.0, 30.0])
            ups = nrm(P - ctr[None, :] * np.array([0, 1, 1]) - np.stack([P[:, 0], np.zeros(n), np.zeros(n)], 1))
            ups = nrm(ups + np.array([sgn * 0.12, 0, 0]))
            G, Tn, Nn, Bn = sweep(M, "strapmesh", P, ups, prof_strap_q(w, T, 40), scale=np.stack([sx, sy], 1),
                                  tile=4.0, obj="straps")
            frames[sgn] = (P, Tn, Nn, Bn, sx, sy, t)
            # bound edges
            for e in (1, -1):
                Ep = P + Bn * (e * w / 2 * sx * 0.985)[:, None] + Nn * (0.08 * T * sy)[:, None]
                tube(M, "trim", Ep, Nn, 0.22, 6, obj="strap_trim")
            # stitched channels (thin dark seams in the grooves)
            for e in (1, -1):
                Sp = P + Bn * (e * 0.22 * w * sx)[:, None] + Nn * (0.6 * T * sy * 0.86 - 0.02)[:, None]
                ii = slice(int(0.03 * n), int(0.95 * n))
                tube(M, "crease", Sp[ii], Nn[ii], 0.06, 4, obj="creases")
            outer = lambda i: Bn[i] if np.dot(Bn[i], [sgn, 0, 0]) > 0 else -Bn[i]
            face = lambda i: P[i] + Nn[i] * (0.6 * T * sy[i] * 0.98)
            iz = lambda zz, lo=0.25: int(np.argmin(np.abs(P[:, 2] - zz) + 100 * (t < lo)))
            # grey/reflective band across the strap at z ~ 35.6, slightly diagonal (outer edge higher),
            # following the domed outer face (analytic height of prof_strap_q at each u)
            i0, i1 = iz(36.4), iz(34.8)
            pts, nns = [], []
            for u in np.linspace(-0.97, 0.97, 24):
                fi = i0 + (i1 - i0) * (0.5 - 0.5 * u)
                ii = int(fi); fr = fi - ii
                ob = outer(ii)
                cc_ = min(abs(u) ** 2, 1.0)
                vv = 0.6 * T * math.sqrt(max(1 - cc_ * cc_, 0)) ** 0.75
                vv -= 0.14 * T * math.exp(-((abs(u) * w / 2 - 0.22 * w) / (0.035 * w)) ** 2)
                pc_ = (1 - fr) * P[ii] + fr * P[ii + 1]
                pts.append(pc_ + ob * u * (w / 2 * sx[ii]) + Nn[ii] * (vv * sy[ii] + 0.05))
                nns.append(Nn[ii])
            pts = np.array(pts); nns = np.array(nns)
            sweep(M, "reflect", pts, nns, prof_rrect(2.8, 0.08, 0.03, 1), tile=1.0, obj="reflect")
            if sgn > 0:
                # plastic tri-ring on the +X strap face (z ~ 31)
                i = iz(31.0)
                pc = face(i) + Nn[i] * 0.1
                rbox(M, "webbing", pc, Tn[i], Nn[i], 2.6, 2.4, 0.2, 0.06)
                base = pc + Nn[i] * 0.32 - Tn[i] * 1.2
                tri = []
                for a_ in np.linspace(0, 2 * np.pi, 24, endpoint=False):
                    rr_ = 1.25 / max(abs(math.cos(a_)) ** 3 + abs(math.sin(a_)) ** 3, 1e-3) ** (1 / 3)
                    tri.append(base + outer(i) * rr_ * math.cos(a_) - Tn[i] * 0.9 * rr_ * math.sin(a_) * 0.9)
                tube(M, "plastic", np.array(tri), Nn[i], 0.2, 6, closed=True, obj="hardware_plastic")
            else:
                # small "Aer" tab on the -X strap face (z ~ 32)
                i = iz(32.0)
                pc = face(i) + Nn[i] * 0.06
                rbox(M, "label", pc, Tn[i], Nn[i], 2.0, 1.1, 0.12, 0.05, obj="label")
            # padded end -> ladder lock -> webbing to the side-bottom anchor (+ tail)
            Pend = P[-1] - Tn[-1] * 0.4
            anc, an = self.surf.ray(*view_ray("right" if sgn > 0 else "left", (self.Yb - 1.5, 10.5)))
            anc = anc + an * 0.3
            bk = Pend + Tn[-1] * 2.0
            seg1 = Pend + (bk - Pend) * np.linspace(0, 1, 6)[:, None]
            mid = 0.5 * (bk + anc) + np.array([0, 1.0, 0])
            seg2 = catmull([bk, mid, anc], 8)
            web = np.vstack([seg1, seg2[1:]])
            web_up = nrm(np.cross(nrm(np.gradient(web, axis=0)), np.array([sgn, 0.0, 0.0])) * -1)
            web_up = np.where((web_up[:, 1:2] < 0), -web_up, web_up)
            sweep(M, "webbing", web, web_up, prof_rrect(2.5, 0.16, 0.05, 1), tile=1.0, obj="webbing")
            dv = nrm(bk - Pend)
            nb = Nn[-1]
            rbox(M, "plastic", bk + nb * 0.3, dv, nb, 2.8, 3.6, 0.55, 0.2, obj="hardware_plastic")
            bx = nrm(np.cross(nb, dv))
            for off_ in (-0.8, 0.0, 0.8):
                rbox(M, "plastic", bk + nb * 0.62 + dv * off_, bx, nb, 3.4, 0.42, 0.32, 0.12, obj="hardware_plastic")
            tail = bk + nb * 0.75 + (dv * 0.8 + nb * 0.55)[None, :] * np.linspace(0.3, 3.5, 12)[:, None]
            sweep(M, "webbing", tail, np.tile(nb, (12, 1)), prof_rrect(2.5, 0.16, 0.05, 1), tile=1.0, obj="webbing")
            rbox(M, "trim", anc, np.array([0, 0, 1.0]), an, 3.4, 3.0, 0.5, 0.15)
            self.load_lifter(sgn, P, Tn, Nn, Bn, sx, sy, t, w, T, outer, iz)
        self.sternum(frames, w, T)

    def load_lifter(self, sgn, P, Tn, Nn, Bn, sx, sy, t, w, T, outer, iz):
        """anchor at the top-handle end on the top-back edge -> ladder lock just past the anchor ->
        webbing running down to the strap's outer edge, sewn in at z ~ 40.5 (35-40 deg in side view)"""
        M = self.M
        A, an = self.anchor[sgn]
        i = iz(40.2, lo=0.1)
        S = P[i] + outer(i) * (w / 2 * sx[i] * 0.8) + Nn[i] * (0.3 * T * sy[i])
        # path: leave the anchor backward/up over the edge, then straight down to the strap edge
        A1 = A + np.array([0, 1.4, 0.5])
        Bk = A + (S - A) * 0.30 + np.array([0, 0.3, 0.55])
        path = catmull([A, A1, Bk, A + (S - A) * 0.65 + np.array([0, 0.1, 0.25]), S], 10)
        Tw = nrm(np.gradient(path, axis=0))
        ref = np.array([0, -0.2, 1.0]) + np.array([sgn * 0.6, 0, 0])
        upw = nrm(ref[None, :] - np.einsum("ij,j->i", Tw, ref)[:, None] * Tw)
        sweep(M, "webbing", path, upw, prof_rrect(2.0, 0.17, 0.05, 1), tile=1.0, obj="load_lifters")
        # ladder-lock buckle (body + 3 bars) on the webbing
        j = int(np.argmin(np.linalg.norm(path - Bk, axis=1)))
        tb, nb = Tw[j], upw[j]
        rbox(M, "plastic", path[j] + nb * 0.18, tb, nb, 3.0, 3.0, 0.5, 0.2, obj="hardware_plastic")
        bx = nrm(np.cross(nb, tb))
        for off_ in (-0.9, 0.0, 0.9):
            rbox(M, "plastic", path[j] + nb * 0.5 + tb * off_, bx, nb, 3.0, 0.4, 0.3, 0.12, obj="hardware_plastic")
        # tail: from the buckle back up over the anchor, flipping up (the tab above the top-back corner)
        tail = catmull([path[j] + nb * 0.6, path[j] + nb * 0.9 - tb * 2.0 + np.array([0, 0, 0.6]),
                        A + np.array([sgn * 0.3, 0.9, 1.2])], 8)
        Tt = nrm(np.gradient(tail, axis=0))
        upt = nrm(np.tile(nb, (len(tail), 1)) - np.einsum("ij,j->i", Tt, nb)[:, None] * Tt)
        sweep(M, "webbing", tail, upt, prof_rrect(2.0, 0.17, 0.05, 1), tile=1.0, obj="load_lifters")
        # box-stitched end where it joins the strap edge
        rbox(M, "trim", S + Nn[i] * 0.1, Tn[i], Nn[i], 2.4, 2.2, 0.3, 0.1)
        ang = math.degrees(math.atan2(A[2] - S[2], S[1] - A[1]))
        self.lifter_info = getattr(self, "lifter_info", {})
        self.lifter_info[sgn] = dict(anchor=np.round(A, 1).tolist(), strap=np.round(S, 1).tolist(),
                                     side_angle_deg=round(ang, 1))

    def sternum(self, frames, w, T):
        M = self.M
        outs = {}
        for sgn in (-1, 1):
            P, Tn, Nn, Bn, sx, sy, t = frames[sgn]
            inner = lambda i: -Bn[i] if np.dot(Bn[i], [sgn, 0, 0]) > 0 else Bn[i]
            ia = int(np.argmin(np.abs(P[:, 2] - 29.0) + 100 * (t < 0.3)))
            ib = int(np.argmin(np.abs(P[:, 2] - 16.6) + 100 * (t < 0.3)))
            idx = np.arange(ia, ib + 1)
            rail = np.array([P[i] + inner(i) * (w / 2 * sx[i] * 0.62) + Nn[i] * (0.6 * T * sy[i] * 0.72 + 0.35)
                             for i in idx])
            tube(M, "plastic", rail, Nn[idx], 0.22, 6, obj="hardware_plastic")
            for i in (idx[0], idx[-1]):
                rbox(M, "plastic", P[i] + inner(i) * (w / 2 * sx[i] * 0.62) + Nn[i] * (0.6 * T * sy[i] * 0.72 + 0.1),
                     Tn[i], Nn[i], 1.0, 1.4, 0.5, 0.15, obj="hardware_plastic")
            i = int(np.argmin(np.abs(P[:, 2] - 21.0) + 100 * (t < 0.3)))
            pr = P[i] + inner(i) * (w / 2 * sx[i] * 0.62) + Nn[i] * (0.6 * T * sy[i] * 0.72 + 0.35)
            rbox(M, "plastic", pr, Tn[i], Nn[i], 2.0, 1.4, 0.8, 0.25, obj="hardware_plastic")
            outs[sgn] = (pr + Nn[i] * 0.2, Nn[i])
        pa, pb = outs[-1][0], outs[1][0]
        k = 40
        s = np.linspace(0, 1, k)
        pts = pa + (pb - pa) * s[:, None]
        pts[:, 2] -= 1.6 * np.sin(np.pi * s)
        pts[:, 1] += 0.8 * np.sin(np.pi * s)
        sweep(M, "webbing", pts, np.tile([0, 1.0, 0.15], (k, 1)), prof_rrect(2.0, 0.15, 0.05, 1), tile=1.0,
              obj="webbing")
        c = pts[k // 2]
        Xa = nrm(pb - pa); Ya = np.array([0, 1.0, 0.0])
        rbox(M, "plastic", c - Xa * 1.5 + Ya * 0.25, Xa, Ya, 3.0, 2.6, 0.95, 0.3, obj="hardware_plastic")
        rbox(M, "plastic", c + Xa * 1.3 + Ya * 0.25, Xa, Ya, 2.4, 2.3, 0.75, 0.25, obj="hardware_plastic")
        for f in (0.22, 0.78):
            j = int(f * (k - 1))
            rbox(M, "plastic", pts[j] + Ya * 0.2, Xa, Ya, 1.6, 2.4, 0.4, 0.12, obj="hardware_plastic")

    # ---------------------------------------------------------------- assemble
    def build(self):
        H = self.H
        self.build_body()
        self.front_panel()
        self.top_zip()
        self.side_zips()
        self.side_pocket(1, 0.66 * H, 0.50 * H)
        self.side_pocket(-1, 0.64 * H, 0.56 * H)
        for sd in (-1, 1):
            self.d_ring(sd)
            self.side_handle(sd)
        self.logo_tab()
        self.bottom_tabs()
        self.top_handle()
        self.back_panel()
        self.straps()
        self.strap_info = {"width_cm": 7.5, "load_lifters": getattr(self, "lifter_info", None)}
