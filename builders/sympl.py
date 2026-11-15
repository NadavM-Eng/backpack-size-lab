"""
Sympl Commuter Pack (V2.2) builder, modelled from the reference photos in
references/sympl/ (see references/sympl/MEASUREMENTS.md for every number used here).

Loaded by build_models.py, which injects its globals (Bag, sweep, rbox, tube, project,
section, resample, smooth_rows, nrm, sstep, prof_*, rr_outline, rect_outline, box_uv,
grid_faces, Mats, Vector, bpy, np, math, ...) into this module before executing it.

Frame (Blender, cm): Z up, FRONT = -Y, BACK = +Y, width X.  Viewer's left (seen from the
front) is -X: front vertical zip + side zips/pocket are on -X, mesh bottle pocket on +X.
"""
import bmesh

# Measured front silhouette of the 20 L (photo 04): (z cm, half-width cm), body H 42.8, W 30.
_SIL = np.array([
    (0.00, 8.2), (0.20, 9.0), (0.50, 10.3), (0.85, 11.2), (1.6, 11.6), (2.75, 12.1), (4.0, 12.75),
    (5.3, 13.3), (7.2, 13.95), (9.8, 14.4), (12.3, 14.7), (14.8, 14.85), (18.0, 15.0), (26.0, 15.0),
    (28.8, 14.85), (31.4, 14.7), (33.9, 14.45), (35.2, 14.25), (36.5, 13.9), (37.8, 13.4),
    (39.0, 12.7), (39.7, 12.2), (40.3, 11.6), (40.9, 10.85), (41.6, 9.8), (42.2, 8.4),
    (42.55, 7.6), (42.8, 6.6),
])
_H0, _W0 = 42.8, 30.0


# ------------------------------------------------------------------------------------------
# material extensions (Mats is the generator's class; patched once, only acts on bags that
# expose `extra_mats`)
# ------------------------------------------------------------------------------------------
def _mesh_material(mats, key, d):
    N = 256
    u, v = np.meshgrid(np.arange(N) / N, np.arange(N) / N)
    f = 8.0  # cells per texture tile
    # offset (brick) hexagonal-ish open weave
    fu = u * f + 0.5 * (np.floor(v * f * 1.15) % 2)
    fv = v * f * 1.15
    du = np.abs(fu - np.round(fu))
    dv = np.abs(fv - np.round(fv))
    hole = (du < 0.30) & (dv < 0.30) & ((du + dv) < 0.46)
    yarn = 0.85 + 0.15 * np.sin(2 * np.pi * u * f * 6) ** 2
    col = srgb2lin(hex_rgb(d["col"]))
    # store sRGB in the PNG (image colourspace sRGB)
    c_s = hex_rgb(d["col"])
    rgba = np.zeros((N, N, 4), np.float32)
    rgba[..., :3] = np.clip(c_s[None, None, :] * yarn[..., None], 0, 1)
    rgba[..., 3] = np.where(hole, 0.0, 1.0)
    img = bpy.data.images.new(f"{mats.bid}_{key}_weave", N, N, alpha=True)
    img.pixels.foreach_set(rgba.ravel())
    import tempfile as _tf, os as _os
    fp = _os.path.join(_tf.gettempdir(), f"{mats.bid}_{key}_weave_{_os.getpid()}.png")
    img.filepath_raw = fp
    img.file_format = "PNG"
    img.save()
    bpy.data.images.remove(img)
    img = bpy.data.images.load(fp)
    img.name = f"{mats.bid}_{key}_weave"
    img.alpha_mode = "STRAIGHT"
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
    rd = nt.nodes.new("ShaderNodeMath"); rd.operation = "ROUND"; rd.location = (-300, -100)
    nt.links.new(tn.outputs["Alpha"], rd.inputs[0])
    nt.links.new(rd.outputs[0], bsdf.inputs["Alpha"])
    for attr, val in (("blend_method", "CLIP"), ("surface_render_method", "DITHERED")):
        try:
            setattr(m, attr, val)
        except Exception:
            pass
    m.use_backface_culling = False
    m.diffuse_color = (*col, 1.0)
    return m


if not getattr(Mats, "_sympl_patched", False):
    _orig_init = Mats.__init__
    _orig_get = Mats.get

    def _init(self, bag):
        _orig_init(self, bag)
        ex = getattr(bag, "extra_mats", None)
        if ex:
            self.defs.update(ex)

    def _get(self, key):
        if key in self.cache:
            return self.cache[key]
        d = self.defs.get(key, {})
        if d.get("kind") == "alpha_mesh":
            m = _mesh_material(self, key, d)
            self.cache[key] = m
            return m
        return _orig_get(self, key)

    Mats.__init__ = _init
    Mats.get = _get
    Mats._sympl_patched = True


# ------------------------------------------------------------------------------------------
# text -> mesh (Blender font object, triangulated)
# ------------------------------------------------------------------------------------------
def text_mesh(s, height=1.0, depth=0.04):
    """Returns (V (n,3), F (m,3)) of text `s`, centred, x-height scaled so cap height ~ height,
    extruded `depth` along +z (from 0 to depth)."""
    cu = bpy.data.curves.new("tmp_txt", "FONT")
    cu.body = s
    cu.size = 1.0
    cu.extrude = depth / 2
    cu.align_x = "CENTER"
    cu.align_y = "CENTER"
    cu.resolution_u = 4
    try:
        cu.bevel_depth = 0.0
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
    # scale: lowercase text -> use full glyph extent in y
    ext = V[:, 1].max() - V[:, 1].min()
    sc = height / max(ext, 1e-6)
    V[:, :2] *= sc
    V[:, :2] -= (V[:, :2].max(0) + V[:, :2].min(0)) / 2
    V[:, 2] = V[:, 2] - V[:, 2].min()
    return V, F


class Builder(Bag):
    def __init__(self, bag):
        super().__init__(bag)
        S = self.S
        self.fH = self.H / _H0
        self.fW = self.W / _W0
        self.B0 = (self.D - self.bulge) / 2
        self.rc = min(float(S["corner_radius"]), 0.9 * self.B0)
        self.Rs = 5.0 * self.fH          # side-profile top radius (mean)
        self.Rsf = 6.5 * self.fH         # front-top radius in side view (photos 15 / 20)
        self.Rsb = 3.8 * self.fH         # back-top radius in side view
        self.rb = 2.6                     # side-profile bottom radius
        self.y_cz = -0.26 * self.B0       # clamshell plane (0.63 of depth from the back, photo 20)
        self.y_lap = 0.62 * self.B0       # laptop zip plane (0.19 from the back)
        self.y_spz = -0.66 * self.B0      # -X side-pocket zip (0.83 from the back)
        fab = S["fabric"]["color"]
        self.extra_mats = {
            "fabric": dict(col=fab, rough=0.5, tex="ballistic", ns=0.28),
            "strap": dict(col=fab, rough=0.5, tex="ballistic", ns=0.2),
            "band": dict(col=shade(fab, 0.94), rough=0.3),
            "handle": dict(col=fab, rough=0.28),
            "trim": dict(col=S["trim_color"] or fab, rough=0.42),
            "pad": dict(col=shade(fab, 1.04), rough=0.62),
            "webbing": dict(col="#141516", rough=0.8, tex="webbing", ns=0.9),
            "stripe": dict(col="#55575c", rough=0.6),
            "zip_tape": dict(col=S["zip_color"], rough=0.3),
            "zip_teeth": dict(col="#0b0b0c", rough=0.45),
            "slider": dict(col="#17181a", rough=0.32, metal=0.25),
            "pull": dict(col=S["pull_color"], rough=0.35),
            "plastic": dict(col="#121213", rough=0.45),
            "logo": dict(col="#0f1011", rough=0.28),
            "label": dict(col=shade(fab, 1.12), rough=0.55),
            "logo_mark": dict(col="#3a3c40", rough=0.5),
            "print": dict(col="#7a7c80", rough=0.5),
            "sleeve": dict(kind="alpha_mesh", col="#121314", rough=0.7),
        }

    # ---- scale helpers
    def Z(self, z):
        return z * self.fH

    def X(self, x):
        return x * self.fW

    # ---- silhouette
    def Aw(self, z):
        zn = np.clip(z / self.fH, 0, _H0)
        return float(np.interp(zn, _SIL[:, 0], _SIL[:, 1])) * self.fW

    def Bd(self, z):
        return self.B0

    def AB(self, z):
        a, bf, bb = self.ABB(z)
        return a, 0.5 * (bf + bb)

    def ABB(self, z):
        """half-width, front half-depth, back half-depth at height z"""
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

    # ---- lower band seam height as a function of body position
    def seam_front(self, ax):
        xc = self.X(10.4)
        return np.where(ax < xc, self.Z(7.0), self.Z(7.0) - (ax - xc) * 1.25)

    def seam_z(self, x, y):
        x = np.asarray(x, float); y = np.asarray(y, float)
        yf = -(self.B0 - self.rc)
        t = np.clip((y - yf) / (self.B0 - yf), 0, 1)
        side = self.Z(4.3) - self.Z(1.3) * t
        front = np.maximum(self.seam_front(np.abs(x)), side)
        return np.where(y < yf, front, side)

    # ---- body (custom ring stack, split into fabric / glossy lower band)
    def build_body(self):
        H = self.H
        rings = []
        a0, bf0, bb0 = self.ABB(0.0)
        for s in (0.08, 0.25, 0.45, 0.62, 0.76, 0.87, 0.95):
            rings.append((0.0, a0 * s, (bf0 * s, bb0 * s), s))
        zs = np.concatenate([np.linspace(0, 1.2, 7), np.linspace(1.45, self.Z(8.5), 26),
                             np.linspace(self.Z(9.3), H - 7.5, 26), np.linspace(H - 7.0, H, 26)])
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
            P2, N2 = self.ring(a, b, r, nfh=26, ns=14, nc=12)
            x, y = P2[:, 0].copy(), P2[:, 1].copy() + 0.5 * (bk - bf)
            if self.bulge:
                wf = np.clip(-N2[:, 1], 0, 1) ** 2
                wx = np.cos(0.5 * np.pi * np.clip(x / max(a, 1e-6), -1, 1)) ** 0.8
                wz = sstep(0, 0.22 * H, z) * sstep(0, 0.2 * H, H - z)
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
        # orientation of the full closed surface (so both halves stay outward-facing)
        vol = 0.0
        for FF in (F, Fb, Ft):
            P = V[FF]
            for k in range(1, FF.shape[1] - 1):
                vol += np.einsum("ij,ij->i", P[:, 0], np.cross(P[:, k], P[:, k + 1])).sum()
        if vol < 0:
            F, Fb, Ft = F[:, ::-1], Fb[:, ::-1], Ft[:, ::-1]
        cen = V[F].mean(1)
        band = cen[:, 2] < self.seam_z(cen[:, 0], cen[:, 1])
        for name, mat, faces in (("body", "fabric", [F[~band], Ft]), ("band", "band", [F[band], Fb])):
            used = np.unique(np.concatenate([f.ravel() for f in faces]))
            remap = np.full(len(V), -1); remap[used] = np.arange(len(used))
            Vs = V[used]
            gs = [(remap[f], box_uv(Vs, remap[f], 2.0)) for f in faces]
            self.M.add(mat, Vs, gs, orient=None, obj=name)
        self.body_surf.add(V, [F, Fb, Ft])
        self.surf.add(V, [F, Fb, Ft])
        self.body_V = V

    # ---- zips / pulls in Sympl style (AquaGuard coated coil, flat tab pulls)
    def aqua_zip(self, P, N, w=1.15, closed=False, stops=True):
        P, N = resample(P, 0.4, closed=closed, extra=[N])
        N = nrm(smooth_rows(N, 3, closed))
        M = self.M
        sweep(M, "zip_tape", P + N * 0.04, N, prof_rrect(w, 0.14, 0.06, 1), closed=closed, tile=1.0)
        sweep(M, "zip_teeth", P + N * 0.1, N, np.array([[0.09, 0.0], [0.05, 0.05], [-0.05, 0.05], [-0.09, 0.0]]),
              closed=closed, tile=1.0)
        if stops and not closed:
            for i, j in ((0, 1), (-1, -2)):
                T = nrm(P[i] - P[j])
                rbox(M, "zip_tape", P[i] + N[i] * 0.12, T, N[i], 0.5, w * 0.7, 0.22, 0.08)
        T = nrm(np.gradient(P, axis=0))
        return P, N, T

    def tab_pull(self, P, N, T, frac, pref=(0, 0, -1), L=4.2, w=1.05, garage=False):
        i = int(np.clip(round(frac * (len(P) - 1)), 1, len(P) - 2))
        p, t = P[i], nrm(T[i])
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
        rbox(M, "slider", p + n * 0.38, t, n, 1.2, 0.95, 0.42, 0.17)
        b = nrm(np.cross(n, d))
        # flat coated tab: slight lift at the slider end, lying on the body further down
        k = 10
        s = np.linspace(0, 1, k)
        base = p + n * 0.55 + d[None, :] * (0.4 + s[:, None] * L)
        # follow the body surface under the tab
        for q in range(k):
            loc, nn, _, _ = self.body_surf.bvh().find_nearest(Vector(base[q]))
            loc = np.array(loc); nn = nrm(np.array(nn))
            if np.dot(nn, n) < 0:
                nn = -nn
            base[q] = loc + nn * (0.42 - 0.12 * s[q])
        prof = prof_rrect(w, 0.2, 0.09, 2)
        sc = np.stack([np.where(s > 0.82, 1.0 - 0.25 * (s - 0.82) / 0.18, 1.0), np.ones(k)], 1)
        sweep(M, "pull", base, np.tile(n, (k, 1)), prof, scale=sc, tile=1.0)
        if garage:
            # small fabric zip garage above the slider
            rbox(M, "fabric", p - d * 0.9 + n * 0.3, d, n, 1.6, 1.6, 0.55, 0.25)

    def loop_zip(self, yv, z_end_l, z_end_r, w=1.2):
        """Zip on the body section y=yv running from z_end_l on -X over the top to z_end_r on +X."""
        H = self.H
        c = (0.0, H * 0.5)
        a_r = math.atan2(z_end_r - c[1], self.A0)
        ang = np.linspace(a_r, math.pi - math.atan2(z_end_l - c[1], self.A0), 700)
        P, N = section(self.body_surf, "y", yv, c, ang)
        return self.aqua_zip(P, N, w)

    def piping_loop(self, yv, r=0.2):
        ang = np.linspace(0, 2 * np.pi, 800, endpoint=False)
        P, N = section(self.body_surf, "y", yv, (0, self.H * 0.5), ang)
        P, N = resample(P, 0.5, closed=True, extra=[N])
        N = nrm(smooth_rows(N, 3, True))
        tube(self.M, "trim", P + N * 0.04, N, r, 6, closed=True)

    # ---- front panel: piping, quick-access vertical zip, caddy top zip, lower band seam
    def panel_outline(self):
        """2D (x, z) outline of the front panel's upper edge (inset of the silhouette),
        returned as an open polyline from the bottom-left corner over the top to the
        bottom-right corner."""
        H = self.H
        zc = self.Z(7.0)
        zz = np.linspace(zc, H, 400)
        a = np.array([self.Aw(z) for z in zz])
        Pr = np.stack([a, zz], 1)
        T = nrm(np.gradient(Pr, axis=0))
        Nn = np.stack([T[:, 1], -T[:, 0]], 1)  # outward normal for the right half
        d = 1.3 + (3.75 - 1.3) * np.abs(Nn[:, 0]) ** 2.2
        Q = Pr - Nn * d[:, None]
        Q = Q[Q[:, 1] <= H - 1.25]
        # close the top at x = 0 with a smooth arc
        Q = Q[Q[:, 0] > 0.05]
        Q = np.vstack([Q, [[0.0, Q[-1, 1]]]])
        Q[:, 0] = np.maximum(Q[:, 0], 0)
        Q = smooth_rows(Q, 4, iters=3)
        Q[0] = (self.Aw(zc) - 3.75 * 0.98, zc)
        Ql = Q.copy(); Ql[:, 0] *= -1                 # bottom-left -> top centre
        O = np.vstack([Ql, Q[::-1][1:]])              # ... -> bottom-right
        O = resample(O, 0.3)[0]
        return O

    def front_panel(self):
        H = self.H
        O = self.panel_outline()
        zc = self.Z(7.0)
        z_qz = self.Z(37.8)    # top of the quick-access zip on the left edge
        z_cr = self.Z(36.0)    # end of the caddy zip on the right edge
        # O runs bottom-left -> up -> over the top -> down -> bottom-right
        iL = int(np.argmax(O[:, 1] > z_qz))                      # first point above the quick zip top
        iR = len(O) - 1 - int(np.argmax(O[::-1, 1] > z_cr))      # last point above the caddy end (right)
        left = O[:iL + 1][::-1]                                   # top -> bottom
        top = O[iL:iR + 1]                                        # left -> right
        right = O[iR:]                                            # top -> bottom
        # quick-access zip (viewer's left edge)
        P, N = project(self.body_surf, "front", left)
        P, N, T = self.aqua_zip(P, N, 1.15)
        self.tab_pull(P, N, T, 0.035, pref=(0, 0, -1), L=5.0, w=1.3, garage=True)
        # caddy zip across the top edge
        tq = top
        P, N = project(self.body_surf, "front", tq)
        P, N, T = self.aqua_zip(P, N, 1.05)
        # caddy pull parked at the top-left corner, beside the quick-zip garage
        self.tab_pull(P, N, T, 0.06, pref=nrm(np.array([-1.0, 0, 0.35])), L=2.6, w=0.9)
        # piping on the right edge down to the band corner
        Pp, Np = project(self.body_surf, "front", right)
        Pp, Np = resample(Pp, 0.45, extra=[Np])
        Np = nrm(smooth_rows(Np, 2))
        tube(self.M, "trim", Pp + Np * 0.05, Np, 0.15, 6)
        # piping alongside the zips (outer side)
        self.panel_O = O

    def band_seam(self):
        ang = np.linspace(0, 2 * np.pi, 720, endpoint=False)
        pts, nrms = [], []
        bvh = self.body_surf
        for a_ in ang:
            ca, sa = math.cos(a_), math.sin(a_)
            z = self.Z(5.0)
            for _ in range(4):
                p, n = bvh.ray((300 * ca, 300 * sa, z), (-ca, -sa, 0))
                z = float(self.seam_z(p[0], p[1]))
            p, n = bvh.ray((300 * ca, 300 * sa, z), (-ca, -sa, 0))
            pts.append(p); nrms.append(n)
        P, N = resample(np.array(pts), 0.4, closed=True, extra=[np.array(nrms)])
        N = nrm(smooth_rows(N, 3, True))
        P = smooth_rows(P, 1, True, iters=1)
        tube(self.M, "trim", P + N * 0.05, N, 0.17, 6, closed=True)

    # ---- text helpers
    def text_on_surface(self, s, view, cx, cz, width, mat, obj, udir=(1, 0), depth=0.035, lift=0.02,
                        surf=None):
        V, F = text_mesh(s, 1.0, depth)
        span = V[:, 0].max() - V[:, 0].min()
        V[:, :2] *= width / span
        u = np.asarray(udir, float); u = u / np.linalg.norm(u)
        v = np.array([-u[1], u[0]])
        if view in ("back", "left"):
            v = -v  # viewer frames are mirrored relative to the (q0, q1) projection coords
        q = np.array([cx, cz])[None, :] + V[:, :1] * u[None, :] + V[:, 1:2] * v[None, :]
        loc, nn = project(surf or self.surf, view, q)
        Vw = loc + nn * (lift + V[:, 2:3])
        self.M.add(mat, Vw, [(F, np.zeros((len(F), 3, 2)))], orient="ref", ref=nn, obj=obj)

    def text_frame(self, s, origin, X, Y, Nrm, width, mat, obj, depth=0.03):
        V, F = text_mesh(s, 1.0, depth)
        span = V[:, 0].max() - V[:, 0].min()
        V *= np.array([width / span, width / span, 1.0])
        Vw = origin[None, :] + V[:, :1] * X[None, :] + V[:, 1:2] * Y[None, :] + V[:, 2:3] * Nrm[None, :]
        self.M.add(mat, Vw, [(F, np.zeros((len(F), 3, 2)))], orient="ref",
                   ref=np.tile(Nrm, (len(Vw), 1)), obj=obj)

    def logo(self, fp=None):
        self.text_on_surface("sympl studios", "front", self.X(0.3), self.Z(8.3), self.X(4.7), "logo", "logo")

    # ---- sides
    def side_zips(self):
        H = self.H
        # clamshell (front block / main body) over the top, down to ~0.1 H both sides
        P, N, T = self.loop_zip(self.y_cz, 0.1 * H, 0.1 * H, 1.2)
        n = len(P)
        # pulls parked near the upper corners: -X side end is near the end of the array
        self.tab_pull(P, N, T, 0.70, pref=(0, 0, -1))
        self.tab_pull(P, N, T, 0.30, pref=(0, 0, -1))
        # laptop compartment zip (top access), down to ~0.55 H
        P, N, T = self.loop_zip(self.y_lap, 0.55 * H, 0.55 * H, 1.1)
        self.tab_pull(P, N, T, 0.62, pref=(0, 0, -1), L=3.8)
        # back-panel border piping and front-block piping
        tcb = self.rc * (1 - math.sin(math.pi / 4))
        self.piping_loop(self.B0 - tcb, r=0.17)

    def side_pocket_zip(self):
        H = self.H
        ys = self.y_spz
        q = np.stack([np.full(60, ys), np.linspace(0.22 * H, 0.80 * H, 60)], 1)
        P, N = project(self.body_surf, "left", q)
        P, N, T = self.aqua_zip(P, N, 1.1)
        self.tab_pull(P, N, T, 0.96, pref=(0, 0, -1), L=4.2)

    def side_grab(self):
        """vertical padded grab strip on the -X side, mid depth"""
        H = self.H
        ys = 0.2 * self.B0
        k = 60
        zz = np.linspace(0.25 * H, 0.68 * H, k)
        base, nn = project(self.body_surf, "left", np.stack([np.full(k, ys), zz], 1))
        nn = nrm(smooth_rows(nn, 3))
        s = np.linspace(0, 1, k)
        lift = 0.55 * np.sin(np.pi * s) ** 0.35
        C = base + nn * (0.25 + lift[:, None])
        sweep(self.M, "strap", C, nn, prof_pillow(2.6, 0.9, 20, 3.0), tile=2.0)
        for e in (0, -1):
            rbox(self.M, "trim", base[e] + nn[e] * 0.25, np.array([0, 0, 1.0]), nn[e], 1.3, 2.9, 0.4, 0.15)

    def side_label(self):
        H = self.H
        yl = -0.47 * self.B0
        zc = 0.135 * H
        o = rr_outline(yl, zc, 1.4, 6.0, 0.25)
        self.patch("left", o, 0.12, 0.08, "label", obj="label")
        self.text_on_surface("sympl studios", "left", yl, zc, 4.6, "logo_mark", "label_text",
                             udir=(0, 1), depth=0.02, lift=0.0)

    def bottle_pocket(self):
        """stretch mesh sleeve on +X: front ~75 % of the side depth, up to ~0.46 H"""
        H, B0 = self.H, self.B0
        nu, nv = 34, 22
        u = np.linspace(0, 1, nu)
        v = np.linspace(0, 1, nv)
        y0, y1 = self.y_cz + 1.0, B0 - 2.2
        y = y0 + u * (y1 - y0)
        z0, z1 = 0.6, 0.5 * H
        rows, refs = [], []
        for jv in range(nv):
            zt = z1 - 1.6 * np.sin(np.pi * u) ** 1.5  # elastic top dips toward the middle (photo 15)
            zz = z0 + v[jv] * (zt - z0)
            q = np.stack([y, zz], 1)
            loc, nn = project(self.body_surf, "right", q)
            h = 0.12 + 0.95 * np.sin(np.pi * u) ** 0.5 * (0.45 + 0.55 * sstep(0, 0.55, v[jv]))
            h = np.where((u == 0) | (u == 1), 0.06, h)
            if jv == 0:
                h = np.full(nu, 0.05)
            rows.append(loc + nn * h[:, None]); refs.append(nn)
        G = np.array(rows).transpose(1, 0, 2)
        Rf = np.array(refs).transpose(1, 0, 2)
        V = G.reshape(-1, 3)
        F, corners = grid_faces(nu, nv, False, False)
        L_ = arclen(G[:, nv // 2])
        Zg = G[0, :, 2]
        UVg = np.stack(np.meshgrid(L_ / 2.0, (Zg - Zg[0]) / 2.0, indexing="ij"), -1)
        L = np.stack([UVg[a, b] for a, b in corners], 1)
        self.M.add("sleeve", V, [(F, L)], orient="ref", ref=Rf.reshape(-1, 3), obj="sleeve")
        self.surf.add(V, [F])
        # bound elastic top hem (flat binding) and front/back binding
        top = G[:, -1]; nt = nrm(Rf[:, -1])
        sweep(self.M, "trim", top + nt * 0.05, nt, prof_rrect(0.9, 0.25, 0.1, 2), tile=1.0)
        for kk in (0, nu - 1):
            sweep(self.M, "trim", G[kk] + Rf[kk] * 0.05, Rf[kk], prof_rrect(0.9, 0.22, 0.08, 2), tile=1.0)

    def side_loop(self):
        """small webbing keeper loop high on the +X side (photo 10)"""
        H = self.H
        loc, n = self.body_surf.ray(*view_ray("right", (0.05 * self.B0, 0.76 * H)))
        down = np.array([0, 0, -1.0]); down = nrm(down - np.dot(down, n) * n)
        rbox(self.M, "webbing", loc + n * 0.12 + down * 0.4, down, n, 2.4, 2.0, 0.22, 0.08)
        b = nrm(np.cross(n, down))
        base = loc + n * 0.4 + down * 1.5
        pts = [base + b * x for x in np.linspace(-1.0, 1.0, 6)]
        for t_ in np.linspace(0, np.pi, 14)[1:-1]:
            pts.append(base + b * 1.0 * math.cos(t_) + down * 1.4 * math.sin(t_))
        tube(self.M, "plastic", np.array(pts), n, 0.2, 6, closed=True)

    # ---- top handle: round, padded, glossy tube
    def top_handle(self):
        spec = self.S["top_handle"] or {}
        span = float(spec.get("width", 12.8)) / 2
        rise = float(spec.get("rise", 2.6))
        dia = float(spec.get("diameter", 2.2))
        yh = self.B0 - 0.40 * (2 * self.B0 + self.bulge)
        k = 80
        t = np.linspace(-1, 1, k)
        xs = span * np.sign(t) * np.abs(t) ** 0.9
        base, nn = project(self.body_surf, "top", np.stack([xs, np.full(k, yh)], 1))
        lift = (rise - dia * 0.5) * (1 - np.abs(t) ** 2.6) ** 0.55
        C = base.copy()
        C[:, 2] = base[:, 2] + lift
        # sink the ends into the body
        C[:3, 2] -= dia * 0.45 * np.array([1, 0.6, 0.25]); C[-3:, 2] -= dia * 0.45 * np.array([0.25, 0.6, 1])
        C = smooth_rows(C, 2, iters=2)
        C[:, 0] = xs
        ups = np.tile([0, 1.0, 0], (k, 1))
        sc = np.stack([np.ones(k), np.ones(k)], 1)
        sweep(self.M, "handle", C, ups, prof_pillow(dia * 1.05, dia, 22, 2.4), scale=sc, tile=2.0, obj="handle")

    # ---- back: moulded pads, luggage pass-through, laptop zip
    def back_panel(self):
        H = self.H
        th = float((self.S["back_panel"] or {}).get("thickness", 1.4))
        self.z0 = 0.89 * H
        xf = self.X(11.8)
        g = 0.5
        blocks = [
            (g, xf, 0.53 * H, 0.70 * H, 1.6), (g, xf, 0.715 * H, 0.885 * H, 2.6),
            (g, xf, 0.09 * H, 0.255 * H, 1.6), (g, xf, 0.27 * H, 0.42 * H, 1.6),
        ]
        for x0, x1, z0, z1, rr in blocks:
            for sgn in (-1, 1):
                o = rect_outline(sgn * x0, sgn * x1, z0, z1, rr)
                self.patch("back", o, th, 0.55, "pad", crown=0.04, obj="pads", ds=0.6, tile=2.0)
        # embossed "studios" on the lower pad (seen from the back, reads left->right = -X)
        # luggage pass-through band
        xl = self.X(11.6)
        lz0, lz1 = 0.43 * H, 0.515 * H
        o = rect_outline(-xl, xl, lz0, lz1, 0.5)
        self.patch("back", o, 0.35 + th * 0.35, 0.3, "webbing", obj="luggage", tile=1.0)
        for zf in (0.3, 0.5, 0.7):
            zz = lz0 + (lz1 - lz0) * zf
            xs = np.linspace(-xl + 0.8, xl - 0.8, 60)
            loc, nn = project(self.surf, "back", np.stack([xs, np.full(60, zz)], 1))
            sweep(self.M, "stripe", loc + nn * 0.03, nn, prof_rrect(0.32, 0.06, 0.02, 1), tile=1.0, obj="stripes")
        for sgn in (-1, 1):
            loc, n = self.surf.ray(*view_ray("back", (sgn * (xl - 0.5), 0.5 * (lz0 + lz1))))
            rbox(self.M, "trim", loc + n * 0.05, np.array([0, 0, 1.0]), n, lz1 - lz0 + 0.3, 1.0, 0.3, 0.1)
        self.text_on_surface("studios", "back", self.X(-5.6), 0.175 * H, 4.0, "logo_mark", "pad_text", udir=(-1, 0),
                             depth=0.05, lift=-0.02)

    # ---- shoulder harness
    def straps(self, front_min=None):
        st = self.S["straps"]
        H, A0 = self.H, self.A0
        w, T = float(st.get("width", 7.0)), float(st.get("thickness", 1.8))
        n = 100
        t = np.linspace(0, 1, n)
        z_top, z_end = 0.905 * H, 0.14 * H
        xa, xe = w / 2 + 0.8, self.X(10.6)
        sx = 1.0 - 0.14 * sstep(0.15, 0.85, t)           # ~7.4 -> 6.4 cm
        sx = np.where(t > 0.86, sx * (1 - 0.55 * sstep(0.86, 1.0, t)), sx)   # tapers into the webbing
        sy = 0.75 + 0.25 * sstep(0, 0.12, t)
        sy = np.where(t > 0.85, sy * (1 - 0.45 * sstep(0.85, 1.0, t)), sy)
        # side-view centre-line of the padded strap (photos 15 / 20): it leaves the rounded top-back
        # corner, arcs out ~6 cm from the back pads around mid-height and comes back in at the bottom
        lp, _ = self.surf.ray(*view_ray("back", (0.0, 0.3 * H)))
        yp = float(lp[1])                                  # back-pad plane
        ctrl = np.array([(yp - 2.6, 0.985 * H), (yp + 0.8, 0.955 * H), (yp + 4.3, 0.87 * H),
                         (yp + 6.4, 0.72 * H), (yp + 6.9, 0.55 * H), (yp + 6.0, 0.38 * H),
                         (yp + 4.5, 0.24 * H), (yp + 3.4, z_end)])
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
            x = sgn * (xa + (xe - xa) * (0.55 * t + 0.45 * sstep(0, 1, t)))
            z = yz[:, 1]
            y = yz[:, 0] + T * sy * 0.4
            P = np.stack([x, y, z], 1)
            ups = np.tile([0, 1.0, 0], (n, 1))
            G, Tn, Nn, Bn = sweep(M, "strap", P, ups, prof_strap(w, T, 28), scale=np.stack([sx, sy], 1),
                                  tile=2.0, obj="straps")
            frames[sgn] = (P, Tn, Nn, Bn)
            for e in (1, -1):
                Ep = P + Bn * (e * w / 2 * sx * 0.985)[:, None] + Nn * (0.02 * T * sy)[:, None]
                tube(M, "trim", Ep, Nn, 0.2, 6, obj="strap_trim")
            # black webbing overlay on the lower strap (photos 14, 24)
            r0, r1 = int(0.6 * (n - 1)), int(0.97 * (n - 1))
            ov = P[r0:r1] + Nn[r0:r1] * (0.62 * T * sy[r0:r1] * 0.97 + 0.02)[:, None]
            sweep(M, "webbing", ov, Nn[r0:r1], prof_rrect(1.0, 0.14, 0.05, 1),
                  scale=np.stack([0.62 * w * sx[r0:r1], np.ones(r1 - r0)], 1), tile=1.0, obj="webbing")
            # print across the strap just above the webbing (reads across the strap width)
            i = int(0.53 * (n - 1))
            Nd = nrm(Nn[i])
            Xd = nrm(Bn[i] if Bn[i][0] < 0 else -Bn[i])           # reads left->right seen from behind
            Yd = nrm(np.cross(Nd, Xd))
            org = P[i] + Nn[i] * (0.62 * T * sy[i]) * 0.98
            self.text_frame("sympl studios", org, Xd, Yd, Nd, 4.6 * sx[i], "print", "strap_print", depth=0.03)
            # webbing: padded end -> ladder lock -> bottom-corner anchor, plus tail
            Pend = P[-1]
            anc, an = self.surf.ray(*view_ray("back", (sgn * (A0 - 2.4), 1.8)))
            dv0 = nrm(P[-1] - P[-4])
            bk = Pend + dv0 * 2.2
            seg1 = Pend + (bk - Pend) * np.linspace(0, 1, 14)[:, None]
            seg2 = bk + (anc + an * 0.2 - bk) * np.linspace(0, 1, 10)[:, None]
            web = np.vstack([seg1, seg2[1:]])
            web = smooth_rows(web, 1, iters=1); web[0] = Pend; web[-1] = anc + an * 0.2
            sweep(M, "webbing", web, np.tile([0, 1.0, 0], (len(web), 1)), prof_rrect(2.5, 0.16, 0.05, 1),
                  tile=1.0, obj="webbing")
            dv = nrm(bk - Pend)
            # ladder lock: frame + bars
            Yv = np.array([0, 1.0, 0])
            rbox(M, "plastic", bk + Yv * 0.25, dv, Yv, 2.8, 3.6, 0.55, 0.2)
            bx = nrm(np.cross(Yv, dv))
            for off_ in (-0.75, 0.0, 0.75):
                rbox(M, "plastic", bk + Yv * 0.62 + dv * off_, bx, Yv, 3.4, 0.4, 0.32, 0.12)
            # tail hanging from the ladder lock
            tl = float(np.clip(bk[2] - 1.0, 1.5, 4.5))
            tail = bk + Yv * 0.7 + np.array([sgn * 0.35, 0.25, -1.0]) * np.linspace(0.3, tl, 14)[:, None]
            tail[:, 2] = np.maximum(tail[:, 2], 0.6)
            sweep(M, "webbing", tail, np.tile([0, 1.0, 0], (14, 1)), prof_rrect(2.5, 0.16, 0.05, 1),
                  tile=1.0, obj="webbing")
            rbox(M, "band", anc + an * 0.2, np.array([0, 0, 1.0]), an, 3.0, 3.4, 0.45, 0.15)
        self.strap_info = {"padded_end_z": round(z_end, 1), "max_standoff": round(float(off.max()), 2)}
        # sternum rails + strap
        ts = 0.3
        i = int(ts * (n - 1))
        outs = {}
        for sgn in (-1, 1):
            P, Tn, Nn, Bn = frames[sgn]
            r0, r1 = int(0.15 * (n - 1)), int(0.46 * (n - 1))
            rail = P[r0:r1] + Nn[r0:r1] * (0.62 * T * sy[r0:r1] * 0.97)[:, None]
            sweep(M, "webbing", rail, Nn[r0:r1], prof_rrect(2.0, 0.12, 0.04, 1), tile=1.0, obj="webbing")
            for zf in (-0.6, 0.6):
                sweep(M, "stripe", rail + Nn[r0:r1] * 0.07 + Bn[r0:r1] * zf, Nn[r0:r1],
                      prof_rrect(0.18, 0.04, 0.015, 1), tile=1.0, obj="stripes")
            outs[sgn] = (P[i] + Nn[i] * (0.62 * T * sy[i] + 0.3), Nn[i], Tn[i], Bn[i])
        pa, pb = outs[-1][0], outs[1][0]
        k = 30
        pts = pa + (pb - pa) * np.linspace(0, 1, k)[:, None]
        pts[:, 1] += 0.4 * np.sin(np.pi * np.linspace(0, 1, k))
        sweep(M, "webbing", pts, np.tile([0, 1.0, 0], (k, 1)), prof_rrect(1.9, 0.15, 0.05, 1), tile=1.0, obj="webbing")
        mid = pts[k // 2] + np.array([0, 0.35, 0])
        Xa = np.array([1.0, 0, 0]); Ya = np.array([0, 1.0, 0])
        rbox(M, "plastic", mid - Xa * 1.3, Xa, Ya, 2.7, 2.5, 0.9, 0.3)
        rbox(M, "plastic", mid + Xa * 1.2, Xa, Ya, 2.4, 2.2, 0.7, 0.25)
        for sgn in (-1, 1):
            p, nn_, tt, bb = outs[sgn]
            rbox(M, "plastic", p - nn_ * 0.08, tt, nn_, 1.6, 2.6, 0.5, 0.15)

    def laptop_zip(self):
        pass  # handled in side_zips (loop over the top)

    # ---- assemble
    def build(self):
        self.build_body()
        self.band_seam()
        self.front_panel()
        self.logo()
        self.side_zips()
        self.side_pocket_zip()
        self.side_grab()
        self.side_label()
        self.bottle_pocket()
        self.side_loop()
        self.top_handle()
        self.back_panel()
        self.front_min = self.M.all_vertices()[:, 1].min()
        self.straps()
