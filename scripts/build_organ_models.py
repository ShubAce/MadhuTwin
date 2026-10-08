"""Build anatomical 3D organs for the dashboard's virtual patient from BodyParts3D.

BodyParts3D (Database Center for Life Science, Japan) segments organs from a real human body
into meshes that share one coordinate frame, so the organs keep their true sizes, shapes and
positions relative to each other. Licence: CC BY-SA 2.1 Japan; credit: "BodyParts3D,
Copyright (c) 2008 Life Science Integrated Database Center, licensed under CC BY-SA 2.1 Japan".
The derived file dashboard/src/assets/organs.glb carries the same licence (see NOTICE.md).

The organ set is placed inside the MakeHuman body (scripts/build_body_model.py) by a scale and
translation fitted on four skeletal landmarks present in both: the femoral-head centres and the
sternal ends of the clavicles, separately for the male and female body shapes.

    pip install trimesh fast-simplification     # build-time only
    python scripts/build_organ_models.py
"""

from __future__ import annotations

import json
import urllib.request

import numpy as np
import trimesh

from twin.paths import RAW, ROOT

COMMIT = "f0eeb6e843380cfe6b83797cf8c3e1af74de5e61"  # github.com/Kevin-Mattheus-Moerman/BodyParts3D (mirror of BodyParts3D 3.0)
STL_URL = f"https://raw.githubusercontent.com/Kevin-Mattheus-Moerman/BodyParts3D/{COMMIT}/assets/BodyParts3D_data/stl"
CACHE = RAW / "bodyparts3d"
ASSETS = ROOT / "dashboard" / "src" / "assets"

# node name -> (FMA id, triangle budget). Prefix = the twin's organ key; "ctx_" = anatomical context only.
PARTS = {
    "heart": ("FMA7274", 9000),
    "liver": ("FMA7197", 5000),
    "liver_gallbladder": ("FMA7202", 600),
    "pancreas": ("FMA7198nsn", 2500),
    "gut_stomach": ("FMA7148", 3000),
    "gut_duodenum": ("FMA7206", 1500),
    "gut_jejunum": ("FMA7207", 4000),
    "gut_ileum": ("FMA7208", 4000),
    "gut_colon": ("FMA14543nsn", 5000),
    "kidney_right": ("FMA7204", 2000),
    "kidney_left": ("FMA7205", 2000),
    "muscle_right_rectus_femoris": ("FMA38928", 1800),
    "muscle_left_rectus_femoris": ("FMA38929", 1800),
    "muscle_right_vastus_lateralis": ("FMA38930", 1800),
    "muscle_left_vastus_lateralis": ("FMA38931", 1800),
    "muscle_right_vastus_medialis": ("FMA38932", 1500),
    "muscle_left_vastus_medialis": ("FMA38933", 1500),
    "ctx_lung_right_upper": ("FMA7333", 1500),
    "ctx_lung_right_middle": ("FMA7383", 1000),
    "ctx_lung_right_lower": ("FMA7337", 1800),
    "ctx_lung_left_upper": ("FMA7370", 1800),
    "ctx_lung_left_lower": ("FMA7371", 1800),
    "ctx_aorta_ascending": ("FMA3736", 400),
    "ctx_aorta_arch": ("FMA3768", 600),
    "ctx_aorta_descending": ("FMA3784", 1200),
}
BONES = {"femur_right": "FMA24474", "femur_left": "FMA24475", "clavicle_right": "FMA13322", "clavicle_left": "FMA13323"}


def stl(fma: str) -> trimesh.Trimesh:
    path = CACHE / f"{fma}.stl"
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(f"{STL_URL}/{fma}.stl", timeout=180) as r:
            path.write_bytes(r.read())
    return trimesh.load(path, force="mesh")


def axes(liver: trimesh.Trimesh, heart: trimesh.Trimesh, femur: trimesh.Trimesh, clavicle: trimesh.Trimesh, spine_back: np.ndarray):
    """Map BodyParts3D axes onto the body frame (+y up, +z front, +x patient's left)."""
    up = clavicle.centroid - femur.centroid
    iu = int(np.argmax(np.abs(up)))
    left = heart.centroid - liver.centroid  # the liver lies on the patient's right
    left[iu] = 0
    il = int(np.argmax(np.abs(left)))
    ifr = 3 - iu - il
    R = np.zeros((3, 3))
    R[0, il] = np.sign(left[il])
    R[1, iu] = np.sign(up[iu])
    front = (heart.centroid - spine_back)[ifr]  # the heart is in front of the descending aorta
    R[2, ifr] = np.sign(front)
    return R


def main() -> None:
    meshes = {k: stl(f) for k, (f, _) in PARTS.items()}
    bones = {k: stl(f) for k, f in BONES.items()}
    R = axes(meshes["liver"], meshes["heart"], bones["femur_right"], bones["clavicle_right"], meshes["ctx_aorta_descending"].centroid)
    to_body = lambda v: (np.asarray(v) @ R.T) / 1000.0  # noqa: E731  millimetres -> metres

    # skeletal landmarks in BodyParts3D: femoral-head centres and sternal clavicle ends
    def femoral_head(m: trimesh.Trimesh) -> np.ndarray:
        v = to_body(m.vertices)
        return v[v[:, 1] >= np.quantile(v[:, 1], 0.96)].mean(0)

    def sternal_end(m: trimesh.Trimesh) -> np.ndarray:
        v = to_body(m.vertices)
        return v[np.abs(v[:, 0]) <= np.quantile(np.abs(v[:, 0]), 0.05)].mean(0)

    src = np.array([femoral_head(bones["femur_left"]), femoral_head(bones["femur_right"]),
                    sternal_end(bones["clavicle_left"]), sternal_end(bones["clavicle_right"])])
    body = json.loads((ASSETS / "human.json").read_text(encoding="utf-8"))
    fits = {}
    for sex in ("male", "female"):
        L = body["landmarks"][sex]
        dst = np.array([L["l-upper-leg"], L["r-upper-leg"], L["l-clavicle"], L["r-clavicle"]])
        ps, qs = src - src.mean(0), dst - dst.mean(0)
        s = float((ps * qs).sum() / (ps * ps).sum())
        t = dst.mean(0) - s * src.mean(0)
        err = float(np.sqrt(((s * src + t - dst) ** 2).sum(1)).mean())
        fits[sex] = {"scale": round(s, 5), "offset": [round(float(x), 5) for x in t], "landmark_error_m": round(err, 4)}
        print(f"{sex}: scale {s:.3f}, mean landmark error {err * 100:.1f} cm")

    scene = trimesh.Scene()
    tris = 0
    for name, (_, budget) in PARTS.items():
        m = meshes[name]
        if len(m.faces) > budget:
            m = m.simplify_quadric_decimation(face_count=budget)
        m = trimesh.Trimesh(to_body(m.vertices), m.faces, process=True)
        m.fix_normals()
        tris += len(m.faces)
        scene.add_geometry(m, node_name=name, geom_name=name)
    glb = trimesh.exchange.gltf.export_glb(scene, include_normals=True)
    (ASSETS / "organs.glb").write_bytes(glb)
    meta = {"source": f"BodyParts3D 3.0 (CC BY-SA 2.1 JP), mirror commit {COMMIT}", "units": "metres",
            "frame": "body: +y up, +z front, +x patient's left; apply scale then offset per body shape",
            "fit": fits, "parts": {k: v[0] for k, v in PARTS.items()}}
    (ASSETS / "organs.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"wrote organs.glb ({len(glb) / 1e6:.1f} MB, {tris} triangles, {len(PARTS)} parts)")


if __name__ == "__main__":
    main()
