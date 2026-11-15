"""
JanSport SuperBreak (black, current version with the -X side bottle pocket), modelled from
references/jansport/ (numbers: references/jansport/MEASUREMENTS.md).

Loaded by build_models.py, which injects its globals (Bag, sweep, rbox, tube, project, section,
resample, smooth_rows, nrm, sstep, prof_*, rr_outline, rect_outline, box_uv, grid_faces, Mats,
Vector, bpy, np, math, ...) before executing this module.  Techniques follow builders/sympl.py
and builders/aer_tp4.py (nothing imported from them).

Frame (Blender, cm): Z up, FRONT = -Y, BACK = +Y, width X.  Viewer's left (seen from the front)
is -X: the stretch bottle pocket is on -X; the pulls are parked on +X.
"""
import bmesh

# front silhouette, simons_front_studio (z cm from the base, half-width cm), W 31.5 / H 42.5
_SIL = np.array([(0.0, 10.6), (0.25, 11.9), (0.6, 13.1), (1.2, 14.3), (2.0, 15.1), (2.8, 15.5), (4.0, 15.75),
                 (12.0, 15.75), (17.0, 15.7), (19.0, 15.55), (22.6, 15.0), (26.3, 14.6), (30.0, 14.15),
                 (31.8, 14.0), (33.6, 13.8), (35.5, 13.5), (37.3, 13.05), (38.4, 12.7), (39.1, 12.35),
                 (40.0, 11.75), (41.0, 10.9), (41.6, 10.1), (42.0, 9.1), (42.3, 7.6), (42.5, 5.4)])
_H0, _W0 = 42.5, 31.5
# main-body depth (z, depth cm): ~15 deep low down, front rolls back over the dome
_DEP = np.array([(0.0, 11.5), (0.6, 13.4), (1.5, 14.6), (3.0, 15.2), (12.0, 15.2), (20.0, 14.8), (26.0, 14.0),
                 (31.0, 12.9), (35.0, 11.7), (38.0, 10.4), (40.0, 9.2), (41.5, 7.9), (42.2, 6.6), (42.5, 5.2)])


def _mats_patch():
    if getattr(Mats, "_jansport_patched", False):
        return
    _orig_init = Mats.__init__

    def _init(self, bag):
        _orig_init(self, bag)
        ex = getattr(bag, "extra_mats", None)
        if ex:
            self.defs.update(ex)

    Mats.__init__ = _init
    Mats._jansport_patched = True


_mats_patch()


def text_mesh(s, height=1.0, depth=0.04, bold=True):
    cu = bpy.data.curves.new("tmp_txt", "FONT")
    cu.body = s
    cu.size = 1.0
    cu.extrude = depth / 2
    cu.align_x = "CENTER"
    cu.align_y = "CENTER"
    cu.resolution_u = 3
    if bold:
        try:
            cu.offset = 0.035
        except Exception:
            pass
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
    a, b = np.asarray(a, float), np.asarray(b, float)
    ab = b - a
    t = np.clip(((Q - a) @ ab) / max(ab @ ab, 1e-9), 0, 1)
    d = np.linalg.norm(Q - (a + t[:, None] * ab), axis=1)
    return d, t


def poly_dist(Q, O):
    """distance of 2-D points Q to an open polyline O"""
    best = np.full(len(Q), 1e9)
    for i in range(len(O) - 1):
        d, _ = seg_dist(Q, O[i], O[i + 1])
        best = np.minimum(best, d)
    return best


class Builder(Bag):
    def __init__(self, bag):
        super().__init__(bag)
        S = self.S
        self.H = 42.5
        self.W = 31.5                 # + gusset waviness -> ~32 cm measured
        self.A0 = self.W / 2
        self.fW = self.W / _W0
        self.Yb = 7.2                 # flat back plane (main body centred near y = 0)
        self.rc = 5.0                 # plan-section corner radius (soft, rounded gusset seams)
        self.Rsb = 3.2                # back-top radius in side view
        self.rb = 2.2                 # bottom radius
        self.bulge = 1.0
        # report against what the photos give (MEASUREMENTS.md)
        S["height"] = self.H
        S["width"] = 32.0
        S["body_depth"] = 15.2
        S["depth_external"] = 23.5
        fab = "#1a1a1c"
        self.extra_mats = {
            "fabric": dict(col=fab, rough=0.78, tex="nylon", ns=0.55),
            "fabric_pocket": dict(col=fab, rough=0.78, tex="nylon", ns=0.55),
            "strap": dict(col=fab, rough=0.78, tex="nylon", ns=0.45),
            "stretch": dict(col="#19191b", rough=0.62, tex="nylon", ns=0.35),
            "back": dict(col="#1c1c1e", rough=0.8, tex="nylon", ns=0.5),
            "trim": dict(col="#151516", rough=0.7, tex="webbing", ns=0.4),
            "webbing": dict(col="#151516", rough=0.8, tex="webbing", ns=0.9),
            "zip_tape": dict(col="#141415", rough=0.7, tex="webbing", ns=0.4),
            "zip_teeth": dict(col="#0c0c0d", rough=0.4),
            "slider": dict(col="#1c1c1e", rough=0.35, metal=0.5),
            "cord": dict(col="#121213", rough=0.85, tex="webbing", ns=0.6),
            "plastic": dict(col="#121213", rough=0.45),
            "label": dict(col="#1f2b5c", rough=0.75, tex="webbing", ns=0.3),
            "label_red": dict(col="#c22a31", rough=0.6),
            "label_white": dict(col="#eeeeee", rough=0.6),
        }

    # ------------------------------------------------------------------ silhouette
    def Aw(self, z):
        return float(np.interp(np.clip(z, 0, self.H), _SIL[:, 0], _SIL[:, 1])) * self.fW

    def Dz(self, z):
        return float(np.interp(np.clip(z, 0, self.H), _DEP[:, 0], _DEP[:, 1]))

    def Yback(self, z):
        H, R, rb = self.H, self.Rsb, self.rb
        y = self.Yb
        if z > H - R:
            y -= R - math.sqrt(max(0.0, R * R - (z - (H - R)) ** 2))
        if z < rb:
            y -= 0.6 * (rb - math.sqrt(max(0.0, rb * rb - (rb - z) ** 2)))
        return y

    def ABB(self, z):
        yb = self.Yback(z)
        return self.Aw(z), self.Yb - self.Dz(z), yb

    # ------------------------------------------------------------------ body
    def build_body(self):
        H = self.H
        rings = []
        for s in (0.08, 0.25, 0.45, 0.62, 0.76, 0.87, 0.95):
            rings.append((0.0, s))
        zs = np.concatenate([np.linspace(0, 2.6, 9), np.linspace(3.0, H - 7.0, 56), np.linspace(H - 6.6, H, 26)])
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
            if s == 1.0:
                wf = np.clip(-N2[:, 1], 0, 1) ** 2
                wx = np.cos(0.5 * np.pi * np.clip(x / max(a, 1e-6), -1, 1)) ** 0.8
                wz = sstep(0, 0.15, z / H) * sstep(0, 0.12, 1 - z / H)
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
        """Soft unstructured polyester: slumped lower front, wrinkles radiating over the dome,
        sagging base, wavy side gussets."""
        H = self.H
        x, y, z = V[:, 0], V[:, 1], V[:, 2]
        abb = np.array([self.ABB(zz) for zz in z])
        yc = 0.5 * (abb[:, 1] + abb[:, 2])
        aw = abb[:, 0]
        front = np.clip((yc - y) / 5.0, 0, 1) * sstep(0.2, 1.5, z) * sstep(0.0, 0.6, H - z)
        Q = np.stack([x, z], 1)
        dent = np.zeros(len(V))
        cr = []
        for sg in (-1, 1):
            cr += [((sg * 12.5, 38.0), (sg * 5.5, 34.5), 0.7, 0.30),     # dome wrinkles toward the label
                   ((sg * 14.0, 33.5), (sg * 8.5, 30.0), 0.8, 0.22),
                   ((sg * 10.0, 40.2), (sg * 4.0, 38.6), 0.6, 0.16),
                   ((sg * 15.0, 27.5), (sg * 11.5, 25.8), 0.8, 0.18),    # beside the pocket top
                   ((sg * 15.2, 4.5), (sg * 11.5, 1.2), 0.8, 0.25)]      # lower corner slump
        cr += [((-11.0, 28.6), (10.5, 28.2), 1.1, 0.10)]                # soft fold above the pocket
        for a, b, wdt, dep in cr:
            dd, t = seg_dist(Q, a, b)
            fade = np.sin(np.pi * np.clip(t, 0, 1)) ** 0.5
            dent += dep * fade * np.exp(-(dd / wdt) ** 2)
        V[:, 1] += front * dent * 2.0
        wr = (0.14 * np.sin(0.5 * x + 0.33 * z) * np.sin(0.21 * z + 0.4)
              + 0.08 * np.sin(0.85 * x - 0.55 * z + 1.3) + 0.04 * np.sin(1.5 * z + 0.7 * x))
        V[:, 1] -= front * wr * 1.2
        # side gussets: soft waviness + compression creases, stronger low down
        side = np.clip((np.abs(x) - 0.8 * aw) / (0.2 * aw + 1e-6), 0, 1)
        ws = (0.22 * np.sin(0.45 * y + 0.38 * z + 0.6) * np.sin(0.2 * z)
              + 0.1 * np.sin(0.9 * y - 0.5 * z + 2.0) + 0.05 * np.sin(1.9 * z + 0.8 * y))
        V[:, 0] += side * np.sign(x) * ws * sstep(0.03 * H, 0.1 * H, z) * sstep(0.0, 0.08 * H, H - z)
        # base sags a little in the middle; top dome undulates
        bot = 1 - sstep(0.0, 1.2, z)
        V[:, 2] += bot * 0.5 - bot * 0.5 * np.cos(0.5 * np.pi * np.clip(x / 15.0, -1, 1)) ** 2 * np.cos(
            0.5 * np.pi * np.clip(y / 8.0, -1, 1)) ** 2
        topw = sstep(H - 3.0, H - 0.3, z)
        V[:, 2] += topw * 0.18 * np.sin(0.45 * x + 0.5 * y) * np.cos(0.25 * x)

    # ------------------------------------------------------------------ zips / pulls
    def coil_zip(self, P, N, w=1.0, closed=False, stops=True, surf_off=0.04):
        P, N = resample(P, 0.3, closed=closed, extra=[N])
        N = nrm(smooth_rows(N, 3, closed))
        M = self.M
        sweep(M, "zip_tape", P + N * surf_off, N, prof_rrect(w, 0.12, 0.05, 1), closed=closed, tile=1.0, obj="zips")
        sy = np.where(np.arange(len(P)) % 2 == 0, 1.0, 0.6)
        pt = np.array([[0.22, 0.0], [0.16, 0.12], [-0.16, 0.12], [-0.22, 0.0]])
        sweep(M, "zip_teeth", P + N * (surf_off + 0.05), N, pt, closed=closed,
              scale=np.stack([np.ones(len(P)), sy], 1), tile=1.0, obj="zips")
        if stops and not closed:
            for i, j in ((0, 1), (-1, -2)):
                T = nrm(P[i] - P[j])
                rbox(M, "zip_tape", P[i] + N[i] * (surf_off + 0.08), T, N[i], 0.6, w * 0.8, 0.25, 0.08, obj="zips")
        T = nrm(np.gradient(P, axis=0))
        return P, N, T

    def cord_pull(self, p, n, t, d, L=6.0, surf=None):
        """slider on the zip + black cord loop hanging in direction d (lies on the surface)"""
        surf = surf or self.surf
        M = self.M
        n = nrm(n)
        t = nrm(t - np.dot(t, n) * n)
        d = nrm(d - np.dot(d, n) * n)
        rbox(M, "slider", p + n * 0.3, t, n, 1.5, 1.0, 0.5, 0.18, obj="pulls")
        b = nrm(np.cross(n, d))
        k = 18
        s = np.linspace(0, 1, k)
        ctr = p + n * 0.6 + d[None, :] * (0.6 + s[:, None] * L)
        for q in range(k):
            loc, nn, _, _ = surf.bvh().find_nearest(Vector(ctr[q]))
            loc = np.array(loc); nn = nrm(np.array(nn))
            if np.dot(nn, n) < 0:
                nn = -nn
            ctr[q] = loc + nn * (0.55 + 0.1 * s[q])
        half = 0.32 + 0.08 * s
        left = ctr + b * half[:, None]
        right = ctr - b * half[:, None]
        ang = np.linspace(0, np.pi, 8)[1:-1]
        bot = [ctr[-1] + d * 0.3 * math.sin(a_) + b * half[-1] * math.cos(a_) for a_ in ang]
        loop = np.vstack([left, np.array(bot), right[::-1]])
        tube(M, "cord", loop, np.tile(n, (len(loop), 1)), 0.16, 6, obj="pulls")
        # crimp / stopper where the cord meets the slider
        rbox(M, "plastic", ctr[1] + n * 0.05, d, n, 1.0, 0.95, 0.5, 0.18, obj="pulls")

    # ------------------------------------------------------------------ front pocket
    def pocket_outline(self):
        x0, x1 = -12.0, 12.0
        z0, z1 = 4.3, 25.3
        pts = []

        def arc(cx, cz, rr, a0, a1, n=12):
            for a_ in np.linspace(math.radians(a0), math.radians(a1), n):
                pts.append((cx + rr * math.cos(a_), cz + rr * math.sin(a_)))
        rt, rbt = 4.2, 3.0
        arc(x1 - rbt, z0 + rbt, rbt, -90, 0)
        arc(x1 - rt, z1 - rt, rt, 0, 90)
        arc(x0 + rt, z1 - rt, rt, 90, 180)
        arc(x0 + rbt, z0 + rbt, rbt, 180, 270)
        O = np.array(pts)
        # slight slump: the lower edge sags in the middle, the sides belly out a little
        return resample(O, 0.35, closed=True)[0]

    def zip_path(self):
        """inverted-U zip of the front pocket (2-D x, z)"""
        xz, zz, zl, r = 10.5, 20.3, 14.5, 2.6
        pts = [(-xz, zl + t) for t in np.linspace(0, zz - r - zl, 10)]
        for a_ in np.linspace(math.pi, math.pi / 2, 10):
            pts.append((-xz + r + r * math.cos(a_), zz - r + r * math.sin(a_)))
        for x in np.linspace(-xz + r, xz - r, 30)[1:-1]:
            pts.append((x, zz + 0.25 * (1 - (x / xz) ** 2)))
        for a_ in np.linspace(math.pi / 2, 0, 10):
            pts.append((xz - r + r * math.cos(a_), zz - r + r * math.sin(a_)))
        pts += [(xz, zz - r - t) for t in np.linspace(0, zz - r - zl, 10)]
        return resample(np.array(pts), 0.25)[0]

    def front_pocket(self):
        O = self.pocket_outline()
        Zp = self.zip_path()
        c = np.array([0.0, 14.0])
        Lc = np.linalg.norm(O - c, axis=1)
        edge_w = 2.2
        dl = np.array([0, 0.03, 0.1, 0.2, 0.33, 0.48, 0.65, 0.82, 1.0]) * edge_w
        rows, refs, Qs = [], [], []
        fs = [1 - dd / Lc for dd in dl]
        fin = np.linspace(0, 1, 20)[1:]
        last = fs[-1]
        for f in fin:
            fs.append(last * (1 - f) + 0.03 * f)
        for k, f in enumerate(fs):
            q = c + (O - c) * (f[:, None] if np.ndim(f) else f)
            Qs.append(q)
        Qall = np.concatenate(Qs)
        dz = poly_dist(Qall, Zp)
        # thickness field: lid band puffs most, the pocket body sags/thins toward the bottom
        xq, zq = Qall[:, 0], Qall[:, 1]
        T = (3.0 + 1.1 * sstep(4.0, 16.0, zq) + 0.6 * sstep(19.5, 23.0, zq) * (1 - sstep(24.0, 25.6, zq)))
        T *= (1 - 0.18 * (np.abs(xq) / 12.0) ** 3)
        groove = 0.75 * np.exp(-(dz / 0.75) ** 2)
        # slump creases on the lower pocket body + puckers at the lid
        Q = np.stack([xq, zq], 1)
        dent = np.zeros(len(Q))
        cr = [((-9.5, 9.5), (-2.0, 7.8), 1.0, 0.35), ((2.5, 8.2), (10.0, 10.5), 1.0, 0.3),
              ((-8.0, 13.5), (6.0, 12.8), 1.4, 0.18), ((-10.5, 6.2), (-4.5, 5.0), 0.8, 0.25),
              ((4.0, 5.2), (10.5, 6.5), 0.8, 0.22), ((-6.0, 23.4), (-2.0, 21.8), 0.6, 0.18),
              ((3.0, 22.0), (7.5, 23.6), 0.6, 0.15), ((-11.0, 17.5), (-7.5, 15.8), 0.7, 0.2),
              ((8.0, 16.2), (11.2, 18.0), 0.7, 0.2)]
        for a, b, wdt, dep in cr:
            dd, t = seg_dist(Q, a, b)
            fade = np.sin(np.pi * np.clip(t, 0, 1)) ** 0.5
            dent += 0.5 * dep * fade * np.exp(-(dd / wdt) ** 2)
        wr = 0.1 * np.sin(0.7 * xq + 0.4 * zq) * np.sin(0.5 * zq) + 0.05 * np.sin(1.3 * xq - 0.9 * zq)
        n_o = len(O)
        hrow = []
        for k, f in enumerate(fs):
            i0, i1 = k * n_o, (k + 1) * n_o
            if k < len(dl):
                x = min(dl[k] / edge_w, 1.0)
                prof = math.sqrt(max(0.0, 1 - (1 - x) ** 2)) if k else 0.0
                h = T[i0:i1] * prof
                if k == 0:
                    h = np.full(n_o, -0.1)
            else:
                h = T[i0:i1]
            inner = sstep(0.4, 1.0, (1 - f) if np.ndim(f) == 0 else 1 - f) if False else 1.0
            h = h - (groove[i0:i1] + dent[i0:i1] + wr[i0:i1]) * (k >= 4)
            loc, nn = project(self.body_surf, "front", Qs[k])
            rows.append(loc + nn * h[:, None]); refs.append(nn)
        G = np.array(rows)
        Rn = np.array(refs)
        R, C, _ = G.shape
        V = G.reshape(-1, 3)
        ref = Rn.reshape(-1, 3)
        F, _ = grid_faces(R, C, True, False)
        ci = len(V)
        cl, cn = project(self.body_surf, "front", [c])
        V = np.vstack([V, cl + cn * 4.0])
        ref = np.vstack([ref, cn])
        j = np.arange(C)
        lastr = (R - 1) * C
        Ft = np.stack([np.full(C, ci), lastr + j, lastr + (j + 1) % C], 1)
        self.M.add("fabric_pocket", V, [(F, box_uv(V, F, 2.0)), (Ft, box_uv(V, Ft, 2.0))], orient="ref", ref=ref,
                   obj="front_pocket")
        self.pocket_surf = Surf()
        self.pocket_surf.add(V, [F, Ft])
        self.surf.add(V, [F, Ft])
        # binding seam around the pocket base
        P, N = project(self.body_surf, "front", O)
        P, N = resample(P, 0.4, closed=True, extra=[N])
        N = nrm(smooth_rows(N, 3, True))
        tube(self.M, "trim", P + N * 0.15, N, 0.22, 6, closed=True, obj="seams")
        # the zip, sitting in its groove
        P, N = project(self.pocket_surf, "front", Zp)
        P, N, T = self.coil_zip(P, N, 0.95, surf_off=0.02)
        i = int(0.86 * (len(P) - 1))
        self.cord_pull(P[i], N[i], T[i], np.array([0.15, -0.3, -1.0]), L=5.5, surf=self.pocket_surf)
        # top-stitch seam along the lid edge (gusset fold)
        Ol = O[O[:, 1] > 20.5]
        Ol = Ol[np.argsort(np.arctan2(Ol[:, 1] - 14.0, Ol[:, 0]))]
        q = c + (Ol - c) * (1 - 1.0 / np.linalg.norm(Ol - c, axis=1))[:, None]
        Ps, Ns = project(self.pocket_surf, "front", q)
        Ps, Ns = resample(Ps, 0.4, extra=[Ns])
        Ns = nrm(smooth_rows(Ns, 3))
        tube(self.M, "trim", Ps - Ns * 0.02, Ns, 0.1, 5, obj="seams")

    # ------------------------------------------------------------------ main zip over the top
    def main_zip(self):
        H = self.H
        yv = self.Yb - 0.6 * self.Dz(H - 2) - 0.6
        c = (0.0, H * 0.5)
        zl = 16.0
        a_r = math.atan2(zl - c[1], self.A0)
        ang = np.linspace(a_r, math.pi - a_r, 700)
        P, N = section(self.body_surf, "y", yv, c, ang)
        P, N, T = self.coil_zip(P, N, 1.0)
        # two cord pulls parked on the +X upper side
        for fr in (0.22, 0.27):
            i = int(fr * (len(P) - 1))
            self.cord_pull(P[i], N[i], T[i], np.array([0.0, 0, -1.0]), L=5.5, surf=self.body_surf)

    # ------------------------------------------------------------------ label patch
    def label(self):
        zc, w, h = 33.4, 9.5, 5.0
        o = rr_outline(0.0, zc, w, h, 0.35)
        self.patch("front", o, 0.12, 0.12, "label", obj="label")
        # thin red border
        ob = rr_outline(0.0, zc, w - 0.5, h - 0.5, 0.25)
        P, N = project(self.surf, "front", ob)
        P, N = resample(P, 0.25, closed=True, extra=[N])
        N = nrm(smooth_rows(N, 2, True))
        tube(self.M, "label_red", P + N * 0.02, N, 0.07, 5, closed=True, obj="label_border")
        self.text_on_surface("JANSPORT", zc, 7.6, "label_white", "label_text_outline", lift=0.0, depth=0.04,
                             yscale=1.25, shear=0.0, grow=0.06)
        self.text_on_surface("JANSPORT", zc, 7.6, "label_red", "label_text", lift=0.03, depth=0.04,
                             yscale=1.25, shear=0.0)

    def text_on_surface(self, s, cz, width, mat, obj, lift=0.02, depth=0.035, yscale=1.0, shear=0.0, grow=0.0):
        V, F = text_mesh(s, 1.0, depth)
        span = V[:, 0].max() - V[:, 0].min()
        V[:, :2] *= width / span
        V[:, 1] *= yscale
        V[:, 0] += shear * V[:, 1]
        if grow:
            # crude outline: scale about each glyph-ish neighbourhood -> overall scale up
            V[:, 0] *= 1 + grow * 0.6
            V[:, 1] *= 1 + grow * 3.0
        q = np.stack([V[:, 0], cz + V[:, 1]], 1)
        loc, nn = project(self.surf, "front", q)
        Vw = loc + nn * (lift + V[:, 2:3])
        self.M.add(mat, Vw, [(F, np.zeros((len(F), 3, 2)))], orient="ref", ref=nn, obj=obj)

    # ------------------------------------------------------------------ haul loop
    def haul_loop(self):
        H = self.H
        yh = self.Yb - 0.32 * self.Dz(H - 1)
        top, n = self.body_surf.ray((0, yh, 200), (0, 0, -1))
        hw, ht = 3.0, 7.8          # outer width 7.5 -> centre-line half-span 2.5
        k = 40
        ang = np.linspace(0, np.pi, k)
        xs = -hw * np.cos(ang)
        zs = ht - 1.25 - 0.0
        Pz = np.where(np.abs(xs) > 0.0, top[2] - 0.3 + (ht - 1.25 + 0.3) * np.sin(ang) ** 0.35, 0)
        P = np.stack([xs * (1 - 0.05 * np.sin(ang) ** 2), np.full(k, yh) - 0.4 * np.sin(ang), Pz], 1)
        # tilt the loop back a little (it flops toward the back in the photos)
        P[:, 1] += 0.9 * np.sin(ang) ** 1.5
        P = smooth_rows(P, 2, iters=1)
        ups = np.tile([0, 1.0, 0], (k, 1))
        sweep(self.M, "webbing", P, ups, prof_rrect(2.0, 0.2, 0.08, 1), tile=1.0, obj="haul_loop")
        # stitched reinforcement box at the base
        for sx_ in (-hw, hw):   # bar-tack boxes under each leg
            p_, n_ = self.body_surf.ray((sx_, yh, 200), (0, 0, -1))
            rbox(self.M, "trim", p_ + n_ * 0.05, np.array([0, 1.0, 0]), n_, 2.6, 2.4, 0.1, 0.04, obj="haul_loop")

    # ------------------------------------------------------------------ back
    def back_panel(self):
        H = self.H
        o = rect_outline(-self.X(12.2), self.X(12.2), 3.0, 37.0, 5.5)
        self.patch("back", o, 0.3, 0.6, "back", obj="back_pad", ds=0.5)

    def X(self, x):
        return x * self.fW

    # ------------------------------------------------------------------ straps
    def straps(self):
        H = self.H
        w, T = 6.0, 1.5
        n = 90
        t = np.linspace(0, 1, n)
        lp, _ = self.surf.ray(*view_ray("back", (0.0, 0.5 * H)))
        yp = float(lp[1])
        z_end = 12.5
        ctrl = np.array([(yp - 4.6, H - 0.4), (yp - 2.2, H - 1.5), (yp + 0.6, 0.90 * H), (yp + 2.0, 0.78 * H),
                         (yp + 2.7, 0.62 * H), (yp + 2.6, 0.47 * H), (yp + 2.0, 0.36 * H), (yp + 1.8, z_end)])
        YZ = resample(catmull(ctrl, 40), 0.05)[0]
        s_ = arclen(YZ); s_ /= s_[-1]
        yz = np.stack([np.interp(t, s_, YZ[:, 0]), np.interp(t, s_, YZ[:, 1])], 1)
        M = self.M
        xa, xe = 4.3, self.X(10.8)
        sx = 1.0 - 0.06 * sstep(0.1, 0.8, t)
        sx = np.where(t > 0.9, sx * (1 - 0.5 * sstep(0.9, 1.0, t)), sx)
        sy = 0.55 + 0.45 * sstep(0, 0.1, t)
        sy = np.where(t > 0.88, sy * (1 - 0.5 * sstep(0.88, 1.0, t)), sy)
        for sgn in (-1, 1):
            x = sgn * (xa + (xe - xa) * (0.35 * t + 0.65 * sstep(0, 1, t)))
            P = np.stack([x, yz[:, 0] + T * sy * 0.4, yz[:, 1]], 1)
            ups = np.tile([0, 1.0, 0], (n, 1))
            # orient: near the top the strap lies on the top surface -> up = +z there
            ups[:12] = nrm(np.array([0, 0.3, 1.0]))[None] * (1 - t[:12, None] / t[12]) + ups[:12] * (t[:12, None] / t[12])
            ups = nrm(ups)
            G, Tn, Nn, Bn = sweep(M, "strap", P, ups, prof_strap(w, T, 28), scale=np.stack([sx, sy], 1),
                                  tile=2.0, obj="straps")
            for e in (1, -1):
                Ep = P + Bn * (e * w / 2 * sx * 0.975)[:, None] + Nn * (0.02 * T * sy)[:, None]
                tube(M, "trim", Ep, Nn, 0.16, 5, obj="strap_trim")
            # soft creases across the foam (it is thin and bends)
            for tc in (0.3, 0.45, 0.58):
                i = int(tc * (n - 1))
                q = P[i] + Nn[i] * (0.5 * T * sy[i])
                tube(M, "trim", np.stack([q - Bn[i] * w * 0.42, q + Bn[i] * w * 0.42]), Nn[i:i + 2], 0.07, 4,
                     obj="strap_trim")
            # webbing: padded end -> ladder lock -> bottom-corner tab; tail dangles
            Pend = P[-1]
            anc, an = self.surf.ray(*view_ray("back", (sgn * self.X(12.8), 1.6)))
            dv0 = nrm(P[-1] - P[-4])
            bk = Pend + dv0 * 3.0
            seg1 = Pend + (bk - Pend) * np.linspace(0, 1, 10)[:, None]
            seg2 = bk + (anc + an * 0.25 - bk) * np.linspace(0, 1, 10)[:, None]
            web = np.vstack([seg1, seg2[1:]])
            web = smooth_rows(web, 1, iters=1); web[0] = Pend; web[-1] = anc + an * 0.25
            sweep(M, "webbing", web, np.tile([0, 1.0, 0], (len(web), 1)), prof_rrect(2.5, 0.15, 0.05, 1),
                  tile=1.0, obj="webbing")
            dv = nrm(bk - Pend)
            Yv = np.array([0, 1.0, 0])
            rbox(M, "plastic", bk + Yv * 0.25, dv, Yv, 2.4, 3.3, 0.5, 0.18, obj="hardware")
            bx = nrm(np.cross(Yv, dv))
            for off_ in (-0.6, 0.6):
                rbox(M, "plastic", bk + Yv * 0.55 + dv * off_, bx, Yv, 3.2, 0.4, 0.3, 0.1, obj="hardware")
            # tail: hangs from the ladder lock and flops slightly outward/back
            kk = 16
            s2 = np.linspace(0, 1, kk)
            tl = 12.0
            tail = bk + Yv * 0.75 + np.stack([sgn * 1.2 * s2 ** 1.5, 0.6 * s2 + 0.8 * s2 ** 2,
                                              -tl * s2], 1)
            tail[:, 2] = np.maximum(tail[:, 2], 0.4 + 2.0 * s2)
            sweep(M, "webbing", tail, np.tile([0, 1.0, 0], (kk, 1)), prof_rrect(2.5, 0.15, 0.05, 1),
                  tile=1.0, obj="webbing")
            rbox(M, "trim", anc + an * 0.2, np.array([0, 0, 1.0]), an, 3.2, 3.0, 0.4, 0.12, obj="webbing")
        # top attachment box (straps sewn into the top-back seam)
        self.strap_info = {"padded_end_z": z_end, "width": w}

    # ------------------------------------------------------------------ bottle pocket (-X)
    def bottle_pocket(self):
        H = self.H
        nu, nv = 34, 24
        u = np.linspace(0, 1, nu)
        v = np.linspace(0, 1, nv)
        a0, yf0, yb0 = self.ABB(10.0)
        y0, y1 = yf0 + 2.4, yf0 + 0.72 * (yb0 - yf0)
        y = y0 + u * (y1 - y0)
        z0, z1 = 0.5, 19.5
        rows, refs = [], []
        for jv in range(nv):
            zt = z1 - 0.8 * np.sin(np.pi * u) ** 1.5
            zz = z0 + v[jv] * (zt - z0)
            q = np.stack([y, zz], 1)
            loc, nn = project(self.body_surf, "left", q)
            hb = 0.15 + 3.3 * np.sin(np.pi * u) ** 1.1 * (0.5 + 0.5 * sstep(0, 0.4, v[jv])) * (1 - 0.35 * v[jv] ** 2)
            # gathered (ruched) elastic: vertical ripples, stronger toward the top hem
            hb += (0.12 + 0.18 * v[jv]) * np.sin(u * np.pi * 9) ** 2 * np.sin(np.pi * u)
            hb = np.where((u == 0) | (u == 1), 0.06, hb)
            if jv == 0:
                hb = np.full(nu, 0.05)
            rows.append(loc + nn * hb[:, None]); refs.append(nn)
        G = np.array(rows).transpose(1, 0, 2)
        Rf = np.array(refs).transpose(1, 0, 2)
        V = G.reshape(-1, 3)
        F, corners = grid_faces(nu, nv, False, False)
        L_ = arclen(G[:, nv // 2])
        Zg = G[0, :, 2]
        UVg = np.stack(np.meshgrid(L_ / 2.0, (Zg - Zg[0]) / 2.0, indexing="ij"), -1)
        L = np.stack([UVg[a, b] for a, b in corners], 1)
        self.M.add("stretch", V, [(F, L)], orient="ref", ref=Rf.reshape(-1, 3), obj="bottle_pocket")
        self.surf.add(V, [F])
        top = G[:, -1]; ntp = nrm(Rf[:, -1])
        sweep(self.M, "trim", top + ntp * 0.08, ntp, prof_rrect(1.2, 0.3, 0.12, 2), tile=1.0, obj="bottle_pocket_trim")
        for kk in (0, nu - 1):
            sweep(self.M, "trim", G[kk] + Rf[kk] * 0.05, Rf[kk], prof_rrect(0.8, 0.22, 0.08, 2), tile=1.0,
                  obj="bottle_pocket_trim")

    # ------------------------------------------------------------------ gusset seams
    def seams(self):
        """soft piping-free seams where the front/back panels meet the side gusset"""
        H = self.H
        for side in ("front", "back"):
            pts = []
            zz = np.linspace(0.6, H, 160)
            for z in zz:
                a, yf, yb = self.ABB(z)
                pts.append(None)
            # trace via rays from the front/back at the gusset edge (x = a - rc * 0.3)
            Q = []
            for z in zz:
                a = self.Aw(z)
                Q.append((a - 0.35 * self.rc, z))
            Q = np.array(Q)
            Qt = []
            for z in np.linspace(H - 0.2, 0.6, 1):
                pass
            for sgn in (-1, 1):
                q = Q.copy(); q[:, 0] *= sgn
                q = q[q[:, 0] * sgn > 0.3]
                P, N = project(self.body_surf, side, q)
                ok = np.abs(P[:, 0]) > 1.0
                P, N = P[ok], N[ok]
                P, N = resample(P, 0.45, extra=[N])
                N = nrm(smooth_rows(N, 3))
                tube(self.M, "trim", P + N * 0.02, N, 0.13, 5, obj="seams")

    # ------------------------------------------------------------------ assemble
    def build(self):
        self.build_body()
        self.seams()
        self.front_pocket()
        self.label()
        self.main_zip()
        self.haul_loop()
        self.back_panel()
        self.bottle_pocket()
        self.straps()
