#!/usr/bin/env python3
"""
Pack one bag into a single bag file for the "Add your bag" panel of Backpack Size Lab
(format backpack-size-lab/bag@1, see docs/ai-instructions.md). Plain Python 3, no Blender needed.

    python tools/export_bag_file.py <id>
    python tools/export_bag_file.py <id> --research references/<id>/research.json --out <id>.bag.json

It reads
    docs/models/<id>.glb                 the mesh (written by build_models.py)
    the research, the first that exists of:
      --research <file>                  bag-file fields (everything but "format" and "model")
      references/<id>/research.json      the same, kept next to the bag's MEASUREMENTS.md
      the bag's entry in docs/bags.json  this site's own data (converted here)
and writes <id>.bag.json (default: work/export/<id>.bag.json) with model.glb_base64 and model.geom.

model.geom.back lists the nodes the page lines up at the back panel: <id>_body plus the back padding
(<id>_back_pad, _pads, _back_mesh, _band, _mesh, _spine), the same names docs/index.html uses for the
built-in bags.
"""
import argparse
import base64
import json
import os
import re
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DOCS = os.path.join(ROOT, "docs")
FORMAT = "backpack-size-lab/bag@1"
MAX_GLB = 12 * 1024 * 1024        # the page refuses larger meshes
MAX_TRIS = 400000
BACK_PARTS = re.compile(r"^(body|band|back_pad|back_mesh|pads|mesh|spine)$")
FIELDS = ("id", "name", "short", "brand", "buyer_country", "dims_cm", "volume_l", "volume_external_l", "weight_kg",
          "laptop", "main_compartment_cm", "laptop_compartment_cm", "pockets", "material", "opening", "warranty",
          "price", "more_specs", "notes", "sources", "made_with")
CURRENCY = {"$": "USD", "€": "EUR", "£": "GBP", "¥": "JPY", "₪": "ILS"}
KNOWN_SPEC_KEYS = {"volume_internal_l", "volume_external_l", "dimensions_cm", "weight_kg", "laptop", "storage_compartment_cm",
                   "tech_compartment_cm", "pockets", "material", "price", "warranty", "opening", "sources"}


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def nums(s):
    return [float(x) for x in re.findall(r"\d+(?:\.\d+)?", str(s or ""))]


# --------------------------------------------------------------------------------------------
# this site's bags.json entry -> bag-file fields
# --------------------------------------------------------------------------------------------
def from_bags_json(bag):
    sp = bag.get("specs") or {}
    srcs = [s for s in sp.get("sources") or [] if isinstance(s, dict) and str(s.get("url", "")).startswith("http")]
    first = srcs[0]["url"] if srcs else None

    def fact(v, **extra):
        o = {"value": v}
        if first:
            o["source"] = first
        o.update({k: x for k, x in extra.items() if x})
        return o

    out = {k: bag[k] for k in ("name", "short", "brand") if bag.get(k)}
    d = nums(sp.get("dimensions_cm"))
    if len(d) >= 3:
        rest = re.sub(r"^[^a-zA-Z(]*", "", str(sp["dimensions_cm"])).strip(" ()")
        out["dims_cm"] = {"h": d[0], "w": d[1], "d": d[2]}
        if rest:
            out["dims_cm"]["includes"] = rest
        if first:
            out["dims_cm"]["source"] = first
    else:
        s = bag.get("shape") or {}
        out["dims_cm"] = {"h": s.get("height"), "w": s.get("width"), "d": s.get("depth_external"),
                          "includes": "the generator's shape (no listed size)"}
    if isinstance(sp.get("volume_internal_l"), (int, float)):
        out["volume_l"] = fact(sp["volume_internal_l"])
    if isinstance(sp.get("volume_external_l"), (int, float)):
        out["volume_external_l"] = fact(sp["volume_external_l"])
    if isinstance(sp.get("weight_kg"), (int, float)):
        out["weight_kg"] = fact(sp["weight_kg"])
    if sp.get("laptop"):
        out["laptop"] = fact(sp["laptop"])
    for k, key in (("storage_compartment_cm", "main_compartment_cm"), ("tech_compartment_cm", "laptop_compartment_cm")):
        c = nums(sp.get(k))
        if len(c) >= 3:
            out[key] = {"h": c[0], "w": c[1], "d": c[2], **({"source": first} if first else {})}
    for k in ("pockets", "material", "opening", "warranty"):
        if sp.get(k):
            out[k] = fact(sp[k])
    m = re.search(r"([$€£¥₪])\s?([\d.,]+)", str(sp.get("price") or ""))
    if m:
        out["price"] = {"value": float(m.group(2).replace(",", "")), "currency": CURRENCY[m.group(1)]}
        if first:
            out["price"]["url"] = first
        if not re.fullmatch(r"\s*[$€£¥₪]\s?[\d.,]+\s*", str(sp["price"])):
            out["price"]["note"] = "listed as " + str(sp["price"])
    more = []
    for k, v in sp.items():
        if k not in KNOWN_SPEC_KEYS and v not in (None, ""):
            more.append({"label": k.replace("_", " ").capitalize(), "value": str(v), **({"source": first} if first else {})})
    if more:
        out["more_specs"] = more
    out["sources"] = [{"title": s.get("title") or s["url"], "url": s["url"]} for s in srcs]
    out["notes"] = ("One of the bags built into Backpack Size Lab, packed as a bag file. Its facts come from this site's data; "
                    "each links the first of the sources listed for the bag.")
    return out


def default_made_with(bid):
    """For this site's own bags. Photo counts come from references/<folder>/NOTES.md when it says them."""
    out = {"route": "A", "tools": ["Blender 5.2 (bpy)", "Backpack Size Lab generator (build_models.py)", "web search", "vision"],
           "verified_from_photos": ["front", "side", "back", "three-quarter"]}
    folder = {"ltt-original": "ltt", "sympl-commuter-20": "sympl", "sympl-commuter-24": "sympl", "aer-cpp2-20": "aer",
              "aer-cpp2-24": "aer", "aer-tp4-28": "aer-tp4", "jansport-superbreak": "jansport"}.get(bid, bid)
    for name in ("NOTES.md", "MEASUREMENTS.md"):
        p = os.path.join(ROOT, "references", folder, name)
        if os.path.exists(p):
            m = re.search(r"(\d+)\s+(?:real\s+)?(?:reference\s+)?photos", open(p, encoding="utf-8").read())
            if m:
                out["photos_used"] = int(m.group(1))
                break
    out["notes"] = ("Built-in bag of Backpack Size Lab: modelled from real photos of the exact bag, measured to scale and compared "
                    "with the photos side by side over several rounds (the count was not recorded). The measurements are in "
                    "the project's references/%s/MEASUREMENTS.md." % folder)
    return out


# --------------------------------------------------------------------------------------------
# the mesh
# --------------------------------------------------------------------------------------------
def glb_json(data):
    """Return the JSON chunk of a GLB, after the checks the page also makes."""
    if len(data) < 20 or data[:4] != b"glTF":
        raise SystemExit("not a binary glTF (GLB) file")
    version, length = struct.unpack("<II", data[4:12])
    if version != 2 or length != len(data):
        raise SystemExit("GLB header is wrong (version %d, length %d of %d bytes)" % (version, length, len(data)))
    clen, ctype = struct.unpack("<II", data[12:20])
    if ctype != 0x4E4F534A:
        raise SystemExit("the first GLB chunk is not JSON")
    return json.loads(data[20:20 + clen].decode("utf-8"))


def check_glb(data):
    j = glb_json(data)

    def walk(v, path):
        if isinstance(v, dict):
            for k, x in v.items():
                if k == "uri":
                    raise SystemExit("%s points to an outside file or URI (%r); export a self-contained GLB" % (path or "the GLB", str(x)[:60]))
                walk(x, path + "." + k if path else k)
        elif isinstance(v, list):
            for i, x in enumerate(v):
                walk(x, "%s[%d]" % (path, i))
    walk(j, "")
    bad = [x for x in j.get("extensionsRequired") or [] if not re.match(r"^(KHR_materials_\w+|KHR_texture_transform|KHR_mesh_quantization)$", x)]
    if bad:
        raise SystemExit("the GLB requires %s, which the page does not load (no Draco, meshopt or KTX2)" % ", ".join(bad))
    acc = j.get("accessors") or []
    tris = 0
    for n in j.get("nodes") or []:
        if "mesh" not in n:
            continue
        for prim in j["meshes"][n["mesh"]].get("primitives") or []:
            if prim.get("mode", 4) != 4:
                continue
            a = prim.get("indices")
            cnt = acc[a]["count"] if a is not None else acc[prim["attributes"]["POSITION"]]["count"]
            tris += cnt // 3
    return j, tris


def back_nodes(j, bid):
    out = []
    for n in j.get("nodes") or []:
        name = n.get("name") or ""
        if "mesh" in n and name.startswith(bid + "_") and BACK_PARTS.match(re.sub(r"\.\d{3}$", "", name[len(bid) + 1:])):
            out.append(name)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("id")
    ap.add_argument("--research", help="bag-file fields (JSON) to use instead of references/<id>/research.json or bags.json")
    ap.add_argument("--made-with", help="JSON file with the made_with block (overrides the research's own)")
    ap.add_argument("--glb", help="default docs/models/<id>.glb")
    ap.add_argument("--out", help="default work/export/<id>.bag.json")
    a = ap.parse_args()
    bid = a.id
    glb = a.glb or os.path.join(DOCS, "models", bid + ".glb")
    if not os.path.exists(glb):
        raise SystemExit("missing %s (run build_models.py --only %s first)" % (glb, bid))

    rpath = a.research or os.path.join(ROOT, "references", bid, "research.json")
    if os.path.exists(rpath):
        research = load(rpath)
        fields = {k: research[k] for k in FIELDS if k in research}
        unknown = sorted(set(research) - set(FIELDS) - {"format", "model"})
        if unknown:
            print("  ! left out unknown fields: %s" % ", ".join(unknown))
        made_default = None
        print("research: %s" % os.path.relpath(rpath, ROOT))
    elif a.research:
        raise SystemExit("missing research %s" % a.research)
    else:
        bags = load(os.path.join(DOCS, "bags.json"))["bags"]
        entry = next((b for b in bags if b.get("id") == bid), None)
        if not entry:
            raise SystemExit("no references/%s/research.json and no bag %r in docs/bags.json" % (bid, bid))
        fields = from_bags_json(entry)
        made_default = default_made_with(bid)
        print("research: docs/bags.json (this site's own data)")
    if a.made_with:
        fields["made_with"] = load(a.made_with)
    elif "made_with" not in fields and made_default:
        fields["made_with"] = made_default
    if "made_with" not in fields:
        print("  ! no made_with block: add one to the research file (see the instructions), or pass --made-with")
    for k in ("name", "dims_cm"):
        if k not in fields:
            print("  ! the research has no %r; the page will refuse the file" % k)

    data = open(glb, "rb").read()
    if len(data) > MAX_GLB:
        raise SystemExit("the GLB is %.1f MB; the page accepts at most 12 MB. Use smaller textures or fewer segments." % (len(data) / 1e6))
    j, tris = check_glb(data)
    if tris > MAX_TRIS:
        raise SystemExit("the GLB has %d triangles; the page accepts at most %d." % (tris, MAX_TRIS))
    back = back_nodes(j, bid)
    if not back:
        raise SystemExit("no node named %s_body (or %s_back_pad, ...) in the GLB: the page could not find the back panel" % (bid, bid))

    out = {"format": FORMAT, "id": bid}
    out.update({k: v for k, v in fields.items() if k != "id"})
    out["model"] = {"glb_base64": base64.b64encode(data).decode("ascii"),
                    "geom": {"frame": "glTF: metres, Y up, front +Z, back -Z, origin at the bottom centre", "back": back}}
    dst = a.out or os.path.join(ROOT, "work", "export", bid + ".bag.json")
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("wrote %s (%.2f MB; GLB %.2f MB, %d triangles, %d materials, %d images; back panel: %s)" % (
        dst, os.path.getsize(dst) / 1e6, len(data) / 1e6, tris, len(j.get("materials") or []), len(j.get("images") or []), ", ".join(back)))


if __name__ == "__main__":
    main()
