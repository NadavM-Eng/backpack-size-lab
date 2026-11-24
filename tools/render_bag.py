#!/usr/bin/env python3
"""
Render any bag GLB from the angles of your reference photos, on a light-grey studio background.
Needs Blender's Python module (bpy), numpy and Pillow (see requirements.txt). A view takes a few seconds.

    python tools/render_bag.py <id>                                      # front, side, back, threequarter
    python tools/render_bag.py <id> --views front,side_left,top
    python tools/render_bag.py <id> --view photo07:-35:12:60             # name:azimuth:elevation:lens
    python tools/render_bag.py <id> --glb path/to/other.glb --out work/renders/other

Writes <out>/<id>_<view>.png (default out: work/renders/<id>/).

Angles: azimuth 0 = straight at the front, 90 = the bag's right side (+x, on your left when you face
the bag's back), -90 = its left side, 180 = the back. Elevation is degrees above the bag's middle.
Lens is in mm (full-frame): 50-85 for studio photos, 24-35 for phone photos taken close.
Named views: front, back, side (= right, +x), side_left (-x), threequarter (front, from the bag's
left), threequarter_right, threequarter_back, top.

The frame is the GLB's (glTF: metres, Y up, front +Z); Blender imports it Z up with the front at -Y.
Generalised from builders/sympl_render.py.
"""
import argparse
import math
import os
import sys

import numpy as np
import bpy
from mathutils import Vector

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPOSURE = -3.2
NAMED = {   # azimuth, elevation, lens
    "front": (0, 3, 70), "back": (180, 7, 70), "side": (90, 6, 70), "side_left": (-90, 6, 70),
    "threequarter": (-36, 15, 60), "threequarter_right": (36, 15, 60), "threequarter_back": (-145, 18, 60),
    "top": (0, 85, 60),
}


def direction(az, el):
    a, e = math.radians(az), math.radians(el)
    return np.array([math.sin(a) * math.cos(e), -math.cos(a) * math.cos(e), math.sin(e)])


def composite(fp):
    """put the transparent render (bag + shadow-catcher shadow) on a light-grey studio sweep"""
    from PIL import Image
    im = np.asarray(Image.open(fp).convert("RGBA")).astype(float) / 255.0
    h, w = im.shape[:2]
    yy = np.linspace(0, 1, h)[:, None, None]
    bg = np.ones((h, w, 3)) * (np.array([0.905, 0.91, 0.918]) - 0.06 * yy)
    a = im[..., 3:4]
    out = im[..., :3] * a + bg * (1 - a)
    Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8)).save(fp)


def main():
    argv = sys.argv[1:]
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("id")
    ap.add_argument("--glb", help="default docs/models/<id>.glb")
    ap.add_argument("--views", default="front,side,back,threequarter", help="comma-separated named views")
    ap.add_argument("--view", action="append", default=[], help="name:azimuth:elevation:lens (repeatable)")
    ap.add_argument("--out", help="default work/renders/<id>")
    ap.add_argument("--size", type=int, default=900)
    ap.add_argument("--samples", type=int, default=48)
    a = ap.parse_args(argv)
    views = []
    for v in [x for x in a.views.split(",") if x]:
        if v not in NAMED:
            raise SystemExit("unknown view %r; named views: %s (or use --view name:az:el:lens)" % (v, ", ".join(NAMED)))
        views.append((v,) + NAMED[v])
    for v in a.view:
        parts = v.split(":")
        if len(parts) != 4:
            raise SystemExit("--view must be name:azimuth:elevation:lens, e.g. photo07:-35:12:60")
        views.append((parts[0], float(parts[1]), float(parts[2]), float(parts[3])))
    if a.view and a.views == ap.get_default("views"):
        views = views[len(a.views.split(",")):]   # only the custom views when --views was not given
    path = a.glb or os.path.join(ROOT, "docs", "models", a.id + ".glb")
    if not os.path.exists(path):
        raise SystemExit("missing %s (build it first: python build_models.py --only %s)" % (path, a.id))
    out_dir = a.out or os.path.join(ROOT, "work", "renders", a.id)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=os.path.abspath(path))
    sc = bpy.context.scene
    mn, mx = np.full(3, 1e9), np.full(3, -1e9)
    for ob in sc.objects:
        if ob.type == "MESH":
            co = np.array([ob.matrix_world @ v.co for v in ob.data.vertices])
            mn = np.minimum(mn, co.min(0)); mx = np.maximum(mx, co.max(0))
    c = (mn + mx) / 2
    ext = (mx - mn).max()

    fm = bpy.data.materials.new("studio"); fm.use_nodes = True
    fb = next(n for n in fm.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    fb.inputs["Base Color"].default_value = (0.62, 0.63, 0.65, 1); fb.inputs["Roughness"].default_value = 0.9
    bpy.ops.mesh.primitive_plane_add(size=20, location=(0, 0, mn[2]))
    fl = bpy.context.active_object
    fl.data.materials.append(fm)
    fl.is_shadow_catcher = True
    world = bpy.data.worlds.new("w"); sc.world = world; world.use_nodes = True
    nt = world.node_tree
    bg = nt.nodes.get("Background")
    bg.inputs["Color"].default_value = (0.75, 0.76, 0.78, 1); bg.inputs["Strength"].default_value = 0.35
    lp = nt.nodes.new("ShaderNodeLightPath")
    bg2 = nt.nodes.new("ShaderNodeBackground"); bg2.inputs["Color"].default_value = (0.70, 0.71, 0.73, 1)
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(lp.outputs["Is Camera Ray"], mix.inputs[0])
    nt.links.new(bg.outputs[0], mix.inputs[1])
    nt.links.new(bg2.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], nt.nodes.get("World Output").inputs["Surface"])
    tgt = bpy.data.objects.new("tgt", None); sc.collection.objects.link(tgt); tgt.location = Vector(c)

    def light(name, energy, size_):
        ld = bpy.data.lights.new(name, "AREA"); ld.energy = energy; ld.size = size_
        ob = bpy.data.objects.new(name, ld); sc.collection.objects.link(ob)
        con = ob.constraints.new("TRACK_TO"); con.target = tgt
        con.track_axis = "TRACK_NEGATIVE_Z"; con.up_axis = "UP_Y"
        return ob
    key, fill, rim, top = light("key", 420, 1.6), light("fill", 140, 2.0), light("rim", 260, 1.2), light("top", 160, 1.5)
    cd = bpy.data.cameras.new("cam")
    cam = bpy.data.objects.new("cam", cd); sc.collection.objects.link(cam); sc.camera = cam
    con = cam.constraints.new("TRACK_TO"); con.target = tgt
    con.track_axis = "TRACK_NEGATIVE_Z"; con.up_axis = "UP_Y"
    r = sc.render
    r.resolution_x = r.resolution_y = a.size
    r.engine = "CYCLES"
    sc.cycles.device = "CPU"; sc.cycles.samples = a.samples
    try:
        sc.cycles.use_denoising = True
    except Exception:
        pass
    r.film_transparent = True
    sc.view_settings.view_transform = "Standard"
    sc.view_settings.look = "None"
    sc.view_settings.exposure = EXPOSURE
    os.makedirs(out_dir, exist_ok=True)
    up = np.array([0, 0, 1.0])
    for name, az, el, lens in views:
        d = direction(az, el)
        cd.lens = lens
        fov = 2 * math.atan(18.0 / lens)
        dist = (ext * 0.62) / math.tan(fov / 2) + ext * 0.3
        cam.location = Vector(c + d * dist)
        if abs(d[2]) > 0.98:   # straight down: keep the front at the bottom of the image
            cam.location = Vector(c + np.array([0, -0.02, 1.0]) * dist)
        right = np.cross(d, up); right = right / (np.linalg.norm(right) or 1)
        key.location = Vector(c + (d * 1.0 - right * 0.9 + up * 1.0) * 1.6)
        fill.location = Vector(c + (d * 1.0 + right * 1.1 + up * 0.2) * 1.8)
        rim.location = Vector(c + (-d * 1.0 + right * 0.6 + up * 0.9) * 1.6)
        top.location = Vector(c + (up * 1.5 + d * 0.2))
        fp = os.path.join(out_dir, "%s_%s.png" % (a.id, name))
        r.filepath = fp
        bpy.ops.render.render(write_still=True)
        composite(fp)
        print("WROTE", fp)


if __name__ == "__main__":
    main()
