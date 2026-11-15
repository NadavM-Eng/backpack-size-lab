"""
Aer City Pack Pro 2 builder (20 L and 24 L), modelled from the reference photos in
references/aer/ (see references/aer/MEASUREMENTS.md for every number used here).

Loaded by build_models.py, which injects its globals (Bag, sweep, rbox, tube, project,
section, resample, smooth_rows, nrm, sstep, prof_*, rr_outline, rect_outline, box_uv,
grid_faces, Mats, Vector, bpy, np, math, ...) into this module before executing it.

Frame (Blender, cm): Z up, FRONT = -Y, BACK = +Y, width X.  Viewer's left (seen from the
front) is -X.  The +X side ("sideright" photos) carries the bottle pocket and the "Aer"
logo tab; on the 20 L the -X side has the vertical side quick-access zip pocket instead,
the 24 L has bottle pockets on both sides.
"""
import bmesh

# ------------------------------------------------------------------------------------------
# measured profiles (MEASUREMENTS.md).  z is a fraction of the body height H (handle excluded)
# ------------------------------------------------------------------------------------------
# front silhouette: half-width as a fraction of W/2
_SIL = {
    "20": np.array([(0.0, 0.60), (0.006, 0.68), (0.015, 0.715), (0.03, 0.735), (0.05, 0.75), (0.07, 0.78),
                    (0.10, 0.83), (0.15, 0.88), (0.20, 0.91), (0.30, 0.95), (0.40, 0.99), (0.50, 1.0),
                    (0.60, 1.0), (0.70, 0.995), (0.75, 0.975), (0.80, 0.95), (0.85, 0.91), (0.90, 0.855),
                    (0.93, 0.81), (0.96, 0.735), (0.98, 0.64), (0.993, 0.52), (1.0, 0.40)]),
    "24": np.array([(0.0, 0.60), (0.006, 0.68), (0.015, 0.715), (0.03, 0.735), (0.05, 0.75), (0.07, 0.775),
                    (0.10, 0.825), (0.15, 0.873), (0.20, 0.90), (0.30, 0.952), (0.40, 0.98), (0.50, 1.0),
                    (0.60, 1.0), (0.70, 0.996), (0.80, 0.969), (0.85, 0.934), (0.90, 0.886), (0.93, 0.84),
                    (0.96, 0.785), (0.98, 0.71), (0.993, 0.58), (1.0, 0.46)]),
}
# side profile: total depth (front face -> back face, back padding included) as a fraction of H
_DEP = {
    "20": np.array([(0.0, 0.33), (0.01, 0.37), (0.023, 0.385), (0.083, 0.405), (0.144, 0.42), (0.204, 0.438),
                    (0.264, 0.448), (0.355, 0.448), (0.416, 0.45), (0.476, 0.458), (0.537, 0.466),
                    (0.597, 0.471), (0.657, 0.481), (0.718, 0.491), (0.778, 0.496), (0.81, 0.496),
                    (0.84, 0.489), (0.87, 0.468), (0.90, 0.445), (0.93, 0.42), (0.96, 0.385),
                    (0.985, 0.33), (1.0, 0.28)]),
    "24": np.array([(0.0, 0.33), (0.01, 0.36), (0.03, 0.40), (0.07, 0.424), (0.15, 0.432), (0.26, 0.443),
                    (0.54, 0.44), (0.65, 0.445), (0.73, 0.452), (0.80, 0.46), (0.85, 0.458),
                    (0.88, 0.44), (0.91, 0.41), (0.94, 0.375), (0.97, 0.33), (0.99, 0.29), (1.0, 0.26)]),
}
_KD = 0.90          # studio side shots read ~10 % deeper than the published depth (perspective / stuffing)


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


class Builder(Bag):
    def __init__(self, bag):
        super().__init__(bag)
        self.size = "24" if "24" in self.bid else "20"
        big = self.size == "24"
        # body height without the top handle (the published height includes the ~2.5 cm handle)
        self.H = 43.5 if big else 42.0
        self.W = 30.5
        self.A0 = self.W / 2
        self.sil = _SIL[self.size]
        self.dep = _DEP[self.size]
        self.Dmax = float(self.dep[:, 1].max()) * self.H * _KD
        self.Yb = 0.5 * self.Dmax            # back face plane (before the back padding)
        self.pad_t = 1.0                     # mesh back-panel padding thickness
        self.bulge = 1.0                     # front pocket dome (plan view)
        self.rc = 4.2                        # plan-section corner radius
        self.Rsb = 3.0                       # back-top radius in side view
        self.rb = 2.2                        # bottom radius in side view
        D = self.Dmax
        self.y_cz = self.Yb - 0.66 * D       # clamshell zip plane
        self.y_lap = self.Yb - 0.17 * D      # laptop-compartment loop zip (with coated garage strip)
        self.y_band = self.Yb - 0.29 * D     # centre of the glossy coated strip in front of it
        fab = "#18191b"
        self.extra_mats = {
            "fabric": dict(col=fab, rough=0.62, tex="ballistic", ns=0.55),
            "strap": dict(col=fab, rough=0.62, tex="ballistic", ns=0.4),
            "crease": dict(col="#0b0b0c", rough=0.7),
            "coated": dict(col="#121314", rough=0.38),
            "trim": dict(col="#1b1c1e", rough=0.5, tex="webbing", ns=0.4),
            "pad": dict(col="#202124", rough=0.85, tex="spacer", ns=1.3),
            "mesh": dict(col="#1d1e20", rough=0.8, tex="mesh", ns=1.4),
            "webbing": dict(col="#141516", rough=0.8, tex="webbing", ns=0.9),
            "zip_tape": dict(col="#101112", rough=0.4),
            "zip_teeth": dict(col="#0a0a0b", rough=0.4),
            "slider": dict(col="#1a1b1d", rough=0.3, metal=0.4),
            "pull": dict(col="#1a1b1d", rough=0.42),
            "plastic": dict(col="#121213", rough=0.45),
            "metal": dict(col="#46474b", rough=0.3, metal=1.0),
            "reflect": dict(col="#55575b", rough=0.4, metal=0.15),
            "label": dict(col="#0e0e0f", rough=0.35),
            "logo_white": dict(col="#e9e9e9", rough=0.5),
        }

    # ---------------------------------------------------------------- silhouette
    def Aw(self, z):
        return float(np.interp(np.clip(z / self.H, 0, 1), self.sil[:, 0], self.sil[:, 1])) * self.A0

    def Dz(self, z):
        return float(np.interp(np.clip(z / self.H, 0, 1), self.dep[:, 0], self.dep[:, 1])) * self.H * _KD

    def Yback(self, z):
        """back face y (of the shell, the mesh padding sits on top of it)"""
        H, R, rb = self.H, self.Rsb, self.rb
        y = self.Yb
        if z > H - R:
            y -= R - math.sqrt(max(0.0, R * R - (z - (H - R)) ** 2))
        if z < rb:
            y -= rb - math.sqrt(max(0.0, rb * rb - (rb - z) ** 2))
        return y

    def ABB(self, z):
        """half-width, y of the front face, y of the back face (shell) at height z"""
        yb = self.Yback(z)
        D = self.Dz(z) - self.pad_t - self.bulge * sstep(0.0, 0.2, z / self.H)
        yf = self.Yb - D
        return self.Aw(z), yf, yb

    # ---------------------------------------------------------------- body
    def build_body(self):
        H = self.H
        rings = []
        a0, yf0, yb0 = self.ABB(0.0)
        for s in (0.08, 0.25, 0.45, 0.62, 0.76, 0.87, 0.95):
            rings.append((0.0, s))
        zs = np.concatenate([np.linspace(0, 1.4, 7), np.linspace(1.7, H - 6.0, 44), np.linspace(H - 5.6, H, 24)])
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
            P2, N2 = self.ring(a, b, r, nfh=26, ns=14, nc=12)
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
        yc0 = 0.5 * (yf0 + yb0)
        aT, yfT, ybT = self.ABB(H)
        V = np.vstack([V, [[0, yc0, 0]], [[0, 0.5 * (yfT + ybT), H]]])
        j = np.arange(C)
        Fb = np.stack([np.full(C, cb), (j + 1) % C, j], 1)
        top = (R - 1) * C
        Ft = np.stack([np.full(C, ct), top + j, top + (j + 1) % C], 1)
        self.M.add("fabric", V, [(F, box_uv(V, F, 2.0)), (Fb, box_uv(V, Fb, 2.0)), (Ft, box_uv(V, Ft, 2.0))],
                   orient="closed", obj="body")
        self.body_surf.add(V, [F, Fb, Ft])
        self.surf.add(V, [F, Fb, Ft])
        self.body_V = V

    def soften(self, V):
        """Soft-goods look: sink the body along the front-panel seam (the panel puffs up inside it)
        and add gentle low-frequency wrinkles on the front and sides."""
        from scipy.spatial import cKDTree
        O = self.front_outline()
        O = resample(O, 0.15, closed=True)[0]
        tr = cKDTree(O)
        H = self.H
        z = V[:, 2]
        yc = np.array([0.5 * (self.ABB(zz)[1] + self.ABB(zz)[2]) for zz in z])
        front = (V[:, 1] < yc) & (z > 0.3) & (z < H - 0.05)
        d, _ = tr.query(np.stack([V[:, 0], z], 1))
        wf = np.clip((yc - V[:, 1]) / 6.0, 0, 1)
        V[:, 1] += front * wf * 0.42 * np.exp(-(d / 0.75) ** 2)
        # wrinkles (front panel and side walls)
        x = V[:, 0]
        wr = (0.10 * np.sin(0.55 * x + 0.31 * z) * np.sin(0.23 * z + 0.4)
              + 0.07 * np.sin(0.9 * x - 0.6 * z + 1.3) + 0.05 * np.sin(1.3 * z + 0.7 * x))
        V[:, 1] -= front * wf * wr * sstep(0.5, 3.0, d)
        side = np.abs(x) > 0.8 * np.array([self.Aw(zz) for zz in z])
        V[:, 0] += side * np.sign(x) * 0.6 * wr * sstep(0.03 * H, 0.15 * H, z) * sstep(0.0, 0.1 * H, H - z)

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
        """Aer pull: slider + long flat rubbery tab hanging along the body."""
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
        sc = np.stack([np.where(s < 0.15, 0.7 + 2 * s, np.where(s > 0.85, 1.0 - 0.3 * (s - 0.85) / 0.15, 1.0)),
                       np.ones(k)], 1)
        sc[:, 0] = np.clip(sc[:, 0], 0.6, 1.0)
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

    # ---------------------------------------------------------------- front panel
    def front_outline(self, inset_side=3.0, inset_top=2.4, z_bot=1.3):
        """closed (x, z) outline of the front panel seam, inset from the silhouette"""
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
        right = Q                                   # bottom-right -> top centre
        left = Q[::-1].copy(); left[:, 0] *= -1    # top centre -> bottom-left
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
        # horizontal front-pocket zip, coated, slight smile, pull + garage on the viewer's left
        zf = 0.585 * H
        xl, xr = -self.Aw(zf) + 3.4, self.Aw(zf) - 3.4
        xs = np.linspace(xl, xr, 80)
        zq = zf + 0.7 * ((xs - 0.5 * (xl + xr)) / (0.5 * (xr - xl))) ** 2
        P, N = project(self.body_surf, "front", np.stack([xs, zq], 1))
        P, N, T = self.aqua_zip(P, N, 1.25)
        # coated welt under the zip (glossy, as in the photos)
        sweep(self.M, "coated", P + N * 0.02, N, prof_rrect(2.0, 0.1, 0.04, 1), tile=1.0)
        self.tab_pull(P, N, T, 0.03, pref=(0, 0, -1), L=7.0, w=1.2, garage=True)
        self.front_zip_z = zf
        # darts: upper corners (pinched, diagonal inward-down) and the lower V-darts
        sx = self.W / 30.5
        darts = [((10.6, 0.873 * H), (8.3, 0.836 * H)), ((8.9, 0.148 * H), (4.6, 0.205 * H))]
        for (x0, z0), (x1, z1) in darts:
            for sg in (-1, 1):
                q = np.stack([np.linspace(sg * x0 * sx, sg * x1 * sx, 24), np.linspace(z0, z1, 24)], 1)
                Pd, Nd = project(self.body_surf, "front", q)
                Nd = nrm(smooth_rows(Nd, 2))
                s = np.linspace(0, 1, len(Pd))
                fade = np.sin(np.pi * np.clip(s * 1.05, 0, 1)) ** 0.4
                sweep(self.M, "crease", Pd + Nd * 0.0, Nd, prof_rrect(0.16, 0.12, 0.05, 1),
                      scale=np.stack([0.4 + 0.6 * fade, np.ones(len(s))], 1), tile=1.0, obj="creases")
                # soft ridge either side of the pinch
                Tn = nrm(np.gradient(Pd, axis=0))
                Bn = nrm(np.cross(Nd, Tn))
                for side in (-1, 1):
                    sweep(self.M, "fabric", Pd + Bn * side * 0.28 - Nd * 0.05, Nd,
                          prof_pillow(0.5, 0.36, 10, 2.4),
                          scale=np.stack([fade, fade], 1), tile=2.0, obj="creases_ridge")

    def top_zip(self):
        """quick-access zip across the top, just behind the front panel"""
        H = self.H
        aT, yfT, ybT = self.ABB(H)
        yq = self.y_cz + 3.2
        xm = 9.0
        xs = np.linspace(-xm, xm, 60)
        q = np.stack([xs, np.full(60, yq) - 0.6 * (1 - (xs / xm) ** 2)], 1)
        P, N = project(self.body_surf, "top", q)
        P, N, T = self.aqua_zip(P, N, 1.15)
        sweep(self.M, "coated", P + N * 0.02, N, prof_rrect(1.9, 0.1, 0.04, 1), tile=1.0)
        self.tab_pull(P, N, T, 0.93, pref=(1, 0, -0.2), L=4.5, w=1.1)

    # ---------------------------------------------------------------- sides
    def side_zips(self):
        H = self.H
        # clamshell: over the top, down both sides to near the base
        P, N, T = self.loop_zip(self.y_cz, 0.04 * H, 0.04 * H, 1.15)
        self.tab_pull(P, N, T, 0.30, pref=(0, 0, -1), L=6.5)
        self.tab_pull(P, N, T, 0.70, pref=(0, 0, -1), L=6.5)
        # laptop loop zip with its coated garage strip, down to ~0.48 H (+X) / 0.25 H on -X (20 L
        # side quick-access pocket continues on the -X side)
        zl = 0.48 * H
        zr = 0.48 * H
        P, N, T = self.loop_zip(self.y_lap, zl, zr, 1.1)
        self.tab_pull(P, N, T, 0.13, pref=(0, 0, -1), L=6.0)
        self.tab_pull(P, N, T, 0.87, pref=(0, 0, -1), L=6.0)
        c = (0.0, H * 0.5)
        ang = np.linspace(math.atan2(0.47 * H - c[1], self.A0), math.pi - math.atan2(0.47 * H - c[1], self.A0), 600)
        Pb, Nb = section(self.body_surf, "y", self.y_band, c, ang)
        Pb, Nb = resample(Pb, 0.4, extra=[Nb])
        Nb = nrm(smooth_rows(Nb, 3))
        sweep(self.M, "coated", Pb + Nb * 0.05, Nb, prof_rrect(2.8, 0.14, 0.06, 1), tile=1.0, obj="coated_strip")
        # back-panel border piping
        ang = np.linspace(0, 2 * np.pi, 800, endpoint=False)
        Pp, Np = section(self.body_surf, "y", self.Yb - 1.2, (0, H * 0.5), ang)
        Pp, Np = resample(Pp, 0.5, closed=True, extra=[Np])
        Np = nrm(smooth_rows(Np, 3, True))
        tube(self.M, "trim", Pp + Np * 0.04, Np, 0.18, 6, closed=True, obj="piping")

    def side_x(self, side):
        return "right" if side > 0 else "left"

    def side_pocket(self, side, bottle=True):
        """Diagonal-top stretch pocket covering the side from the clamshell zip to the back."""
        H = self.H
        view = self.side_x(side)
        y0, y1 = self.y_cz + 0.8, self.Yb - 0.6
        zf, zb = 0.585 * H, 0.45 * H          # top edge: high at the front, low at the back
        nu, nv = 30, 20
        u = np.linspace(0, 1, nu)
        rows, refs = [], []
        for jv, v in enumerate(np.linspace(0, 1, nv)):
            zt = zf + (zb - zf) * u
            zz = 1.0 + v * (zt - 1.0)
            loc, nn = project(self.body_surf, view, np.stack([y0 + u * (y1 - y0), zz], 1))
            h = 0.08 + (0.45 if bottle else 0.15) * np.sin(np.pi * u) ** 0.6 * sstep(0, 0.4, v) * sstep(0, 0.1, 1 - v + 0.05)
            if jv == 0:
                h = np.full(nu, 0.05)
            rows.append(loc + nn * h[:, None]); refs.append(nn)
        G = np.array(rows).transpose(1, 0, 2)
        Rf = np.array(refs).transpose(1, 0, 2)
        V = G.reshape(-1, 3)
        F, corners = grid_faces(nu, nv, False, False)
        self.M.add("fabric", V, [(F, box_uv(V, F, 2.0))], orient="ref", ref=Rf.reshape(-1, 3), obj="side_pockets")
        self.surf.add(V, [F])
        top = G[:, -1]; nt = nrm(Rf[:, -1])
        sweep(self.M, "trim", top + nt * 0.08, nt, prof_rrect(0.8, 0.3, 0.12, 2), tile=1.0, obj="piping")
        if not bottle:
            # vertical side quick-access zip from the pocket top down to ~0.12 H, mid-depth
            yz = self.Yb - 0.47 * self.Dmax
            zt = zf + (zb - zf) * (yz - y0) / (y1 - y0)
            q = np.stack([np.full(40, yz), np.linspace(zt - 0.6, 0.12 * H, 40)], 1)
            P, N = project(self.surf, view, q)
            P, N, T = self.aqua_zip(P, N, 1.1)
            self.tab_pull(P, N, T, 0.03, pref=(0, 0, -1), L=6.0, surf=self.surf)

    def d_ring(self, side):
        H = self.H
        view = self.side_x(side)
        yd, zd = self.Yb - 0.54 * self.Dmax, 0.80 * H
        loc, n = self.body_surf.ray(*view_ray(view, (yd, zd)))
        down = np.array([0, 0, -1.0]); down = nrm(down - np.dot(down, n) * n)
        rbox(self.M, "webbing", loc + n * 0.12 + down * 0.3, down, n, 2.4, 2.4, 0.24, 0.08)
        b = nrm(np.cross(n, down))
        base = loc + n * 0.42 + down * 1.3
        pts = [base + b * x for x in np.linspace(-1.05, 1.05, 6)]
        for t in np.linspace(0, np.pi, 18)[1:-1]:
            pts.append(base + b * 1.05 * math.cos(t) + down * 1.9 * math.sin(t))
        tube(self.M, "metal", np.array(pts), n, 0.17, 8, closed=True, obj="hardware")

    def side_loop(self, side):
        """short webbing grab loop with a reflective stripe at front-pocket-zip height"""
        H = self.H
        view = self.side_x(side)
        z = 0.565 * H
        ya, yb_ = self.y_cz - 3.0, self.y_cz - 0.7
        ys = np.linspace(ya, yb_, 30)
        loc, nn = project(self.body_surf, view, np.stack([ys, np.full(30, z)], 1))
        nn = nrm(smooth_rows(nn, 2))
        s = np.linspace(0, 1, 30)
        C = loc + nn * (0.12 + 0.35 * np.sin(np.pi * s) ** 0.6)[:, None]
        sweep(self.M, "webbing", C, nn, prof_rrect(1.6, 0.2, 0.06, 1), tile=1.0, obj="webbing")
        sweep(self.M, "reflect", C + nn * 0.11, nn, prof_rrect(0.6, 0.04, 0.015, 1), tile=1.0, obj="reflect")

    def logo_tab(self):
        """small black tab with a white 'Aer' just behind the clamshell zip, low on the +X side"""
        H = self.H
        yl = self.y_cz + 1.6
        zc = 0.19 * H
        o = rr_outline(yl, zc, 1.5, 4.2, 0.3)
        self.patch("right", o, 0.12, 0.08, "label", obj="label")
        V, F = text_mesh("Aer", 1.0, 0.02)
        span = V[:, 0].max() - V[:, 0].min()
        V[:, :2] *= 2.0 / span
        q = np.stack([yl - V[:, 1], zc + V[:, 0]], 1)    # reads bottom -> top
        loc, nn = project(self.surf, "right", q)
        Vw = loc + nn * (0.02 + V[:, 2:3])
        self.M.add("logo_white", Vw, [(F, np.zeros((len(F), 3, 2)))], orient="ref", ref=nn, obj="logo")

    # ---------------------------------------------------------------- top handle
    def top_handle(self):
        H = self.H
        span = 6.0
        yh = self.Yb - 1.6
        k = 70
        t = np.linspace(-1, 1, k)
        xs = span * t
        base, nn = project(self.body_surf, "top", np.stack([xs, np.full(k, yh)], 1))
        lift = 1.7 * (1 - np.abs(t) ** 2.0) ** 0.7
        C = base.copy()
        C[:, 2] += lift + 0.35
        C[:, 1] += 0.4 * (1 - np.abs(t) ** 2)
        C = smooth_rows(C, 2, iters=2)
        ups = np.tile([0, 0.35, 1.0], (k, 1))
        sweep(self.M, "strap", C, ups, prof_pillow(3.2, 0.9, 20, 3.6), tile=2.0, obj="handle")
        for sg in (-1, 1):
            i = 0 if sg < 0 else k - 1
            rbox(self.M, "trim", base[i] + nn[i] * 0.2, np.array([1.0, 0, 0]), nn[i], 1.8, 3.3, 0.35, 0.12)

    # ---------------------------------------------------------------- back
    def back_panel(self):
        H = self.H
        th = self.pad_t
        xf = 12.0 * self.W / 30.5
        g = 0.9
        blocks = [(g, xf, 0.53 * H, 0.89 * H, 3.0, "mesh"), (g, xf, 0.07 * H, 0.455 * H, 2.4, "pad")]
        thk = {"mesh": th + 0.2, "pad": th - 0.1}
        for x0, x1, z0, z1, rr, mat in blocks:
            for sgn in (-1, 1):
                o = rect_outline(min(sgn * x0, sgn * x1), max(sgn * x0, sgn * x1), z0, z1, rr)
                self.patch("back", o, thk[mat], 0.6, mat, crown=0.03, obj="back_" + mat, ds=0.6, tile=1.5)
        # central channel strip
        o = rect_outline(-g, g, 0.07 * H, 0.89 * H, 0.5)
        self.patch("back", o, 0.45, 0.3, "mesh", obj="back_mesh", tile=1.5)
        # luggage pass-through
        xl = self.W / 2 - 1.8
        lz0, lz1 = 0.465 * H, 0.515 * H
        o = rect_outline(-xl, xl, lz0, lz1, 0.4)
        self.patch("back", o, th + 0.5, 0.25, "webbing", obj="luggage", tile=1.0)
        for sgn in (-1, 1):
            loc, n = self.surf.ray(*view_ray("back", (sgn * (xl - 0.4), 0.5 * (lz0 + lz1))))
            rbox(self.M, "trim", loc + n * 0.05, np.array([0, 0, 1.0]), n, lz1 - lz0 + 0.4, 1.0, 0.3, 0.1)

    # ---------------------------------------------------------------- harness
    def straps(self, front_min=None):
        H = self.H
        w, T = 6.8, 1.9
        n = 100
        t = np.linspace(0, 1, n)
        z_end = 0.42 * H
        xa, xe = 4.8, 12.3 * self.W / 30.5
        sx = 1.0 - 0.18 * sstep(0.1, 0.8, t)
        sx = np.where(t > 0.88, sx * (1 - 0.5 * sstep(0.88, 1.0, t)), sx)
        sy = 0.75 + 0.25 * sstep(0, 0.12, t)
        sy = np.where(t > 0.86, sy * (1 - 0.4 * sstep(0.86, 1.0, t)), sy)
        lp, _ = self.surf.ray(*view_ray("back", (0.0, 0.3 * H)))
        yp = float(lp[1])
        ctrl = np.array([(yp - 2.0, 0.985 * H), (yp + 1.0, 0.955 * H), (yp + 4.4, 0.87 * H),
                         (yp + 6.4, 0.75 * H), (yp + 6.9, 0.62 * H), (yp + 6.0, 0.51 * H), (yp + 4.6, z_end)])
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
        off = yz[:, 0] - yp
        M = self.M
        frames = {}
        for sgn in (-1, 1):
            x = sgn * (xa + (xe - xa) * (0.25 * t + 0.75 * sstep(0.05, 0.9, t)))
            P = np.stack([x, yz[:, 0] + T * sy * 0.4, yz[:, 1]], 1)
            ups = np.tile([0, 1.0, 0], (n, 1))
            G, Tn, Nn, Bn = sweep(M, "strap", P, ups, prof_strap(w, T, 28), scale=np.stack([sx, sy], 1),
                                  tile=1.5, obj="straps")
            frames[sgn] = (P, Tn, Nn, Bn)
            for e in (1, -1):
                Ep = P + Bn * (e * w / 2 * sx * 0.985)[:, None] + Nn * (0.02 * T * sy)[:, None]
                tube(M, "trim", Ep, Nn, 0.2, 6, obj="strap_trim")
            face = lambda i: P[i] + Nn[i] * (0.62 * T * sy[i] * 0.97)
            # reflective diagonal band near the top: inner edge higher than the outer edge
            i0, i1 = int(0.20 * (n - 1)), int(0.31 * (n - 1))
            pts, nns = [], []
            for kk, u in enumerate(np.linspace(-1, 1, 14)):
                # outer side (u=+1 toward |x| larger) sits lower along the strap
                bdir = Bn[i0] if np.dot(Bn[i0], [sgn, 0, 0]) > 0 else -Bn[i0]
                ii = int(round(i0 + (i1 - i0) * (0.5 + 0.5 * u)))
                pts.append(face(ii) + bdir * u * (w / 2 * sx[ii] * 0.92) + Nn[ii] * 0.06)
                nns.append(Nn[ii])
            pts = np.array(pts); nns = np.array(nns)
            sweep(M, "reflect", pts, nns, prof_rrect(1.9, 0.06, 0.02, 1), tile=1.0, obj="reflect")
            # D-ring on the strap face
            i = int(0.50 * (n - 1))
            pc = face(i) + Nn[i] * 0.1
            rbox(M, "webbing", pc, Tn[i], Nn[i], 2.6, 2.2, 0.2, 0.06)
            base = pc + Nn[i] * 0.3 + Tn[i] * 1.6
            pts = [base + Bn[i] * xx for xx in np.linspace(-1.0, 1.0, 6)]
            for tt in np.linspace(0, np.pi, 16)[1:-1]:
                pts.append(base + Bn[i] * 1.0 * math.cos(tt) + Tn[i] * 1.7 * math.sin(tt))
            tube(M, "metal", np.array(pts), Nn[i], 0.15, 8, closed=True, obj="hardware")
            # webbing: padded end -> ladder lock -> bottom-corner anchor, plus tail
            Pend = P[-1]
            anc, an = self.surf.ray(*view_ray("back", (sgn * (self.A0 - 2.6), 2.2)))
            bk = Pend + (anc - Pend) * 0.45
            seg1 = Pend + (bk - Pend) * np.linspace(0, 1, 14)[:, None]
            seg2 = bk + (anc + an * 0.2 - bk) * np.linspace(0, 1, 12)[:, None]
            web = np.vstack([seg1, seg2[1:]])
            web = smooth_rows(web, 1, iters=1); web[0] = Pend; web[-1] = anc + an * 0.2
            sweep(M, "webbing", web, np.tile([0, 1.0, 0], (len(web), 1)), prof_rrect(2.5, 0.16, 0.05, 1),
                  tile=1.0, obj="webbing")
            Yv = np.array([0, 1.0, 0])
            dv = nrm(bk - Pend)
            rbox(M, "plastic", bk + Yv * 0.25, dv, Yv, 2.6, 3.6, 0.55, 0.2)
            bx = nrm(np.cross(Yv, dv))
            for off_ in (-0.7, 0.0, 0.7):
                rbox(M, "plastic", bk + Yv * 0.62 + dv * off_, bx, Yv, 3.4, 0.4, 0.32, 0.12)
            tail = bk + Yv * 0.7 + np.array([sgn * 0.3, 0.3, -1.0]) * np.linspace(0.3, 5.0, 14)[:, None]
            sweep(M, "webbing", tail, np.tile([0, 1.0, 0], (14, 1)), prof_rrect(2.5, 0.16, 0.05, 1),
                  tile=1.0, obj="webbing")
            rbox(M, "trim", anc + an * 0.2, np.array([0, 0, 1.0]), an, 3.0, 3.4, 0.45, 0.15)
        self.strap_info = {"padded_end_z": round(z_end, 1), "max_standoff": round(float(off.max()), 2)}
        # sternum strap on rails
        i = int(0.86 * (n - 1))
        outs = {}
        for sgn in (-1, 1):
            P, Tn, Nn, Bn = frames[sgn]
            r0, r1 = int(0.62 * (n - 1)), int(0.93 * (n - 1))
            rail = P[r0:r1] + Nn[r0:r1] * (0.62 * T * sy[r0:r1] * 0.97)[:, None]
            sweep(M, "webbing", rail, Nn[r0:r1], prof_rrect(1.6, 0.12, 0.04, 1), tile=1.0, obj="webbing")
            outs[sgn] = (P[i] + Nn[i] * (0.62 * T * sy[i] + 0.3), Nn[i], Tn[i], Bn[i])
        pa, pb = outs[-1][0], outs[1][0]
        k = 30
        pts = pa + (pb - pa) * np.linspace(0, 1, k)[:, None]
        pts[:, 1] += 0.5 * np.sin(np.pi * np.linspace(0, 1, k))
        sweep(M, "webbing", pts, np.tile([0, 1.0, 0], (k, 1)), prof_rrect(1.9, 0.15, 0.05, 1), tile=1.0, obj="webbing")
        mid = pts[k // 2] + np.array([0, 0.35, 0])
        Xa = np.array([1.0, 0, 0]); Ya = np.array([0, 1.0, 0])
        rbox(M, "plastic", mid - Xa * 1.3, Xa, Ya, 2.7, 2.5, 0.9, 0.3)
        rbox(M, "plastic", mid + Xa * 1.2, Xa, Ya, 2.4, 2.2, 0.7, 0.25)
        for sgn in (-1, 1):
            p, nn_, tt, bb = outs[sgn]
            rbox(M, "plastic", p - nn_ * 0.08, tt, nn_, 1.6, 2.6, 0.5, 0.15)

    # ---------------------------------------------------------------- assemble
    def build(self):
        self.build_body()
        self.front_panel()
        self.top_zip()
        self.side_zips()
        big = self.size == "24"
        self.side_pocket(1, bottle=True)
        self.side_pocket(-1, bottle=big)
        for sd in (-1, 1):
            self.d_ring(sd)
            self.side_loop(sd)
        self.logo_tab()
        self.top_handle()
        self.back_panel()
        self.straps()
