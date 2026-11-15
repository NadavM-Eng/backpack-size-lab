#!/usr/bin/env python3
"""
Photo-matched preview renders for the LTT Original GLB (adapted from builders/sympl_render.py).

    python3 builders/ltt_render.py [bag-id] [--size 900] [--samples 64] [--views front,side,back,threequarter]

Writes previews/<id>_{front,side,back,threequarter}.png (plus side_left if asked) on a light-grey
studio background.  Views match the reference photos in references/ltt/:
  front        straight front (front.jpg, camera slightly above)
  side         straight side, +X / side-pocket side, front on the left (side_with_front_side_pocket.jpg)
  side_left    straight side, -X / carabiner side, front on the right (side_other_with_carabiner.jpg)
  back         straight back (back.jpg)
  threequarter 3/4 front from the viewer's right, +X side pocket visible
"""
import math
import os
import sys

import numpy as np
import bpy
from mathutils import Vector

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EXPOSURE = -3.2

VIEWS = {
    # direction from the bag centre to the camera (Blender frame: front = -Y), lens mm
    "front": ((0.0, -1.0, 0.10), 70),
    "side": ((1.0, 0.0, 0.05), 70),
    "side_left": ((-1.0, 0.0, 0.05), 70),
    "back": ((0.0, 1.0, 0.08), 70),
    "threequarter": ((0.75, -1.0, 0.28), 60),
}


def main():
    argv = sys.argv[1:]
    bid = "ltt-original"
    size, samples = 900, 64
    views = ["front", "side", "back", "threequarter"]
    out_dir = os.path.join(HERE, "previews")
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--size":
            size = int(argv[i + 1]); i += 2; continue
        if a == "--samples":
            samples = int(argv[i + 1]); i += 2; continue
        if a == "--views":
            views = argv[i + 1].split(","); i += 2; continue
        if a == "--out":
            out_dir = argv[i + 1]; i += 2; continue
        bid = a; i += 1
    path = os.path.join(HERE, "docs", "models", f"{bid}.glb")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=path)
    sc = bpy.context.scene
    mn, mx = np.full(3, 1e9), np.full(3, -1e9)
    for ob in sc.objects:
        if ob.type == "MESH":
            co = np.array([ob.matrix_world @ v.co for v in ob.data.vertices])
            mn = np.minimum(mn, co.min(0)); mx = np.maximum(mx, co.max(0))
    c = (mn + mx) / 2
    ext = (mx - mn).max()

    # light-grey studio: seamless floor + backdrop, soft area lights
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
    # camera rays see a light-grey backdrop; lighting uses the dimmer environment
    lp = nt.nodes.new("ShaderNodeLightPath")
    bg2 = nt.nodes.new("ShaderNodeBackground"); bg2.inputs["Color"].default_value = (0.70, 0.71, 0.73, 1)
    bg2.inputs["Strength"].default_value = 1.0
    mix = nt.nodes.new("ShaderNodeMixShader")
    out = nt.nodes.get("World Output")
    nt.links.new(lp.outputs["Is Camera Ray"], mix.inputs[0])
    nt.links.new(bg.outputs[0], mix.inputs[1])
    nt.links.new(bg2.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])

    tgt = bpy.data.objects.new("tgt", None); sc.collection.objects.link(tgt); tgt.location = c

    def light(name, energy, size_):
        ld = bpy.data.lights.new(name, "AREA"); ld.energy = energy; ld.size = size_
        ob = bpy.data.objects.new(name, ld); sc.collection.objects.link(ob)
        con = ob.constraints.new("TRACK_TO"); con.target = tgt
        con.track_axis = "TRACK_NEGATIVE_Z"; con.up_axis = "UP_Y"
        return ob

    key = light("key", 420, 1.6)
    fill = light("fill", 140, 2.0)
    rim = light("rim", 260, 1.2)
    top = light("top", 160, 1.5)

    cd = bpy.data.cameras.new("cam")
    cam = bpy.data.objects.new("cam", cd); sc.collection.objects.link(cam); sc.camera = cam
    con = cam.constraints.new("TRACK_TO"); con.target = tgt
    con.track_axis = "TRACK_NEGATIVE_Z"; con.up_axis = "UP_Y"

    r = sc.render
    r.resolution_x = size; r.resolution_y = int(size * 1.0)
    r.engine = "CYCLES"
    sc.cycles.device = "CPU"; sc.cycles.samples = samples
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
    for tag in views:
        dvec, lens = VIEWS[tag]
        d = np.array(dvec, float); d /= np.linalg.norm(d)
        cd.lens = lens
        fov = 2 * math.atan(18.0 / lens)
        dist = (ext * 0.57) / math.tan(fov / 2) + ext * 0.3
        cam.location = Vector(c + d * dist)
        # light rig relative to the camera
        right = np.cross(d, up); right /= np.linalg.norm(right)
        key.location = Vector(c + (d * 1.0 - right * 0.9 + up * 1.0) * 1.6)
        fill.location = Vector(c + (d * 1.0 + right * 1.1 + up * 0.2) * 1.8)
        rim.location = Vector(c + (-d * 1.0 + right * 0.6 + up * 0.9) * 1.6)
        top.location = Vector(c + (up * 1.5 + d * 0.2))
        fp = os.path.join(out_dir, f"{bid}_{tag}.png")
        r.filepath = fp
        bpy.ops.render.render(write_still=True)
        composite(fp)
        print("WROTE", fp)


def composite(fp):
    """put the transparent render (bag + shadow-catcher shadow) on a light-grey studio sweep"""
    from PIL import Image
    im = np.asarray(Image.open(fp).convert("RGBA")).astype(float) / 255.0
    h, w = im.shape[:2]
    yy = np.linspace(0, 1, h)[:, None, None]
    bg = np.ones((h, w, 3)) * (np.array([0.905, 0.91, 0.918]) - 0.06 * yy)   # soft vertical falloff
    a = im[..., 3:4]
    if os.environ.get("LTT_MASK"):
        Image.fromarray(((im[..., 3] > 0.98) * 255).astype(np.uint8)).save(fp.replace(".png", "_mask.png"))
    out = im[..., :3] * a + bg * (1 - a)
    Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8)).save(fp)


if __name__ == "__main__":
    main()
