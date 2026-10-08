"""Build the 3D virtual-patient body for the dashboard from the MakeHuman base mesh (CC0).

Output
    dashboard/src/assets/human.glb           skin surface + 6 morph targets (male, female, and
                                             heavier / lighter variants of each, for BMI)
    dashboard/src/assets/human.json          anatomical landmarks (joint centres) per sex, in metres

The MakeHuman base mesh, gender targets and weight targets were released as CC0 1.0 by the
MakeHuman project (https://github.com/makehumancommunity/makehuman, LICENSE.md, section C).
Downloads are pinned to one commit and cached under data/raw/makehuman.

    python scripts/build_body_model.py
"""

from __future__ import annotations

import json
import struct
import urllib.request

import numpy as np

from twin.paths import RAW, ROOT

COMMIT = "a8bc2d54ff0ac92e78ff71431b1023eda42bf482"
BASE_URL = f"https://raw.githubusercontent.com/makehumancommunity/makehuman/{COMMIT}/makehuman/data"
CACHE = RAW / "makehuman"
OUT = ROOT / "dashboard" / "src" / "assets"  # bundled into the lazy 3D chunk
ETHNIC = ("african", "asian", "caucasian")  # averaged: MakeHuman's own neutral default, no single ethnicity
LANDMARKS = ("neck", "head", "spine-1", "spine-2", "spine-3", "spine-4", "pelvis", "l-upper-leg", "r-upper-leg",
             "l-knee", "r-knee", "l-shoulder", "r-shoulder", "l-eye", "r-eye", "l-clavicle", "r-clavicle")


def fetch(rel: str) -> str:
    path = CACHE / rel.replace("/", "__")
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(f"{BASE_URL}/{rel}", timeout=120) as r:
            path.write_bytes(r.read())
    return path.read_text(encoding="utf-8", errors="ignore")


def read_obj(text: str) -> tuple[np.ndarray, dict[str, list[list[int]]]]:
    verts, groups, cur = [], {}, None
    for line in text.splitlines():
        if line.startswith("v "):
            verts.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("g "):
            cur = line.split(maxsplit=1)[1].strip()
            groups.setdefault(cur, [])
        elif line.startswith("f "):
            groups[cur].append([int(t.split("/")[0]) - 1 for t in line.split()[1:]])
    return np.asarray(verts, dtype=np.float64), groups


def read_target(text: str, n: int) -> np.ndarray:
    d = np.zeros((n, 3))
    for line in text.splitlines():
        if line and not line.startswith("#"):
            i, x, y, z = line.split()
            d[int(i)] = (float(x), float(y), float(z))
    return d


def vertex_normals(v: np.ndarray, tri: np.ndarray) -> np.ndarray:
    fn = np.cross(v[tri[:, 1]] - v[tri[:, 0]], v[tri[:, 2]] - v[tri[:, 0]])  # area-weighted face normals
    n = np.zeros_like(v)
    for k in range(3):
        np.add.at(n, tri[:, k], fn)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def write_glb(path, pos: np.ndarray, nrm: np.ndarray, tri: np.ndarray, targets: list[tuple[str, np.ndarray, np.ndarray]]) -> None:
    """Minimal glTF 2.0 binary: one mesh, one primitive, morph targets (position deltas)."""
    blobs, views, accessors = [], [], []

    def add(arr: np.ndarray, comp: int, kind: str, target: int | None, minmax: bool = False) -> int:
        data = arr.tobytes()
        offset = sum(len(b) for b in blobs)
        blobs.append(data + b"\0" * (-len(data) % 4))
        view = {"buffer": 0, "byteOffset": offset, "byteLength": len(data)}
        if target is not None:
            view["target"] = target
        views.append(view)
        acc = {"bufferView": len(views) - 1, "componentType": comp, "count": int(arr.shape[0]) if kind != "SCALAR" else int(arr.size), "type": kind}
        if minmax:
            acc["min"], acc["max"] = arr.min(0).astype(float).tolist(), arr.max(0).astype(float).tolist()
        accessors.append(acc)
        return len(accessors) - 1

    F32, U16, U32 = 5126, 5123, 5125
    idx = tri.astype(np.uint16 if len(pos) < 65536 else np.uint32).ravel()
    a_pos = add(pos.astype(np.float32), F32, "VEC3", 34962, True)
    a_nrm = add(nrm.astype(np.float32), F32, "VEC3", 34962)
    a_idx = add(idx, U16 if idx.dtype == np.uint16 else U32, "SCALAR", 34963)
    # position deltas only: the dashboard recomputes normals after blending (halves the file)
    prim_targets = [{"POSITION": add(dp.astype(np.float32), F32, "VEC3", 34962, True)} for _, dp, _ in targets]
    binary = b"".join(blobs)
    gltf = {
        "asset": {"version": "2.0", "generator": "MadhuTwin build_body_model.py",
                  "copyright": "Body mesh and targets: MakeHuman project, CC0 1.0 (makehumancommunity/makehuman)"},
        "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0, "name": "body"}],
        "meshes": [{"name": "body", "primitives": [{"attributes": {"POSITION": a_pos, "NORMAL": a_nrm}, "indices": a_idx, "mode": 4,
                                                     "targets": prim_targets}],
                    "weights": [0.0] * len(targets), "extras": {"targetNames": [t[0] for t in targets]}}],
        "buffers": [{"byteLength": len(binary)}], "bufferViews": views, "accessors": accessors,
    }
    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(binary)))
        f.write(struct.pack("<II", len(js), 0x4E4F534A) + js)
        f.write(struct.pack("<II", len(binary), 0x004E4942) + binary)


def main() -> None:
    V, groups = read_obj(fetch("3dobjs/base.obj"))
    n = len(V)
    tgt = lambda name: read_target(fetch(f"targets/macrodetails/{name}.target"), n)  # noqa: E731
    sex = {s: np.mean([tgt(f"{e}-{s}-young") for e in ETHNIC], axis=0) for s in ("male", "female")}
    weight = {(s, w): tgt(f"universal-{s}-young-averagemuscle-{w}weight") for s in ("male", "female") for w in ("max", "min")}

    # the skin surface only: no helper geometry (tights, hair, teeth, eyelashes, joints)
    quads = np.asarray(groups["body"])
    keep = np.unique(quads)
    remap = np.full(n, -1)
    remap[keep] = np.arange(len(keep))
    q = remap[quads]
    tri = np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]])

    scale = 0.1  # MakeHuman units are decimetres
    floor = (V + sex["male"])[keep][:, 1].min()  # feet on the ground (y = 0) for the default shape
    to_m = lambda x: (x - np.array([0.0, floor, 0.0])) * scale  # noqa: E731

    base = to_m(V)[keep]
    n_base = vertex_normals(base, tri)
    targets = []
    for name, delta in (("male", sex["male"]), ("female", sex["female"]),
                        ("male_heavy", weight[("male", "max")]), ("male_light", weight[("male", "min")]),
                        ("female_heavy", weight[("female", "max")]), ("female_light", weight[("female", "min")])):
        if name in ("male", "female"):
            shape = base + delta[keep] * scale
            n_shape = vertex_normals(shape, tri)
        else:  # weight targets apply on top of the matching sex shape
            s = sex[name.split("_")[0]][keep]
            shape_sex = base + s * scale
            shape = shape_sex + delta[keep] * scale
            n_shape = vertex_normals(shape, tri) - vertex_normals(shape_sex, tri) + n_base
        targets.append((name, shape - base, n_shape - n_base))

    OUT.mkdir(parents=True, exist_ok=True)
    write_glb(OUT / "human.glb", base, n_base, tri, targets)

    def joint(name: str, shape: np.ndarray) -> list[float]:
        idx = np.unique(np.asarray(groups[f"joint-{name}"]))
        return [round(float(x), 4) for x in to_m(shape[idx]).mean(0)]

    marks = {s: {j: joint(j, V + sex[s]) for j in LANDMARKS} for s in ("male", "female")}
    heights = {s: round(float(to_m(V + sex[s])[keep][:, 1].max()), 3) for s in ("male", "female")}
    meta = {"units": "metres", "up": "+y", "front": "+z", "patient_left": "+x", "height": heights, "landmarks": marks,
            "targets": [t[0] for t in targets], "vertices": int(len(base)), "triangles": int(len(tri)),
            "source": f"MakeHuman base mesh and macro targets, CC0 1.0, commit {COMMIT}"}
    (OUT / "human.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"wrote {OUT / 'human.glb'} ({(OUT / 'human.glb').stat().st_size / 1e6:.1f} MB, {len(base)} vertices, {len(tri)} triangles); heights {heights}")


if __name__ == "__main__":
    main()
