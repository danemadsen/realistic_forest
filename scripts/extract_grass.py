#!/usr/bin/env python3
"""Extract grass pack 2 without an editor or third-party packages.

Run from any directory: python3 scripts/extract_grass.py
Validate checked-in outputs: python3 scripts/extract_grass.py --validate

Each mesh is a separate variant, never an LOD. Bake the full source hierarchy's
rotation/scale into mesh attributes, discard its display translation, and move
the lowest vertex to Y=0. Preserve the author's X/Z root pivot and metre scale.
Textures are shared external PNGs; GLBs contain only their own geometry. Source
BLEND materials become double-sided alpha masks for stable instanced vegetation.
"""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import struct


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "assets/models"
TEXTURE_DIR = ROOT / "assets/textures/grass"
MANIFEST_PATH = MODEL_DIR / "grass-manifest.json"
IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
COMPONENTS = {5120: "b", 5121: "B", 5122: "h", 5123: "H", 5125: "I", 5126: "f"}
WIDTHS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}
PACKS = [(2, {0: "grass-pack-2"}, set())]


def read_glb(path):
    data = path.read_bytes()
    magic, version, length = struct.unpack_from("<4sII", data)
    assert magic == b"glTF" and version == 2 and length == len(data), path
    chunks = {}
    offset = 12
    while offset < length:
        size, kind = struct.unpack_from("<I4s", data, offset)
        assert size % 4 == 0 and offset + 8 + size <= length
        chunks[kind] = data[offset + 8 : offset + 8 + size]
        offset += 8 + size
    return json.loads(chunks[b"JSON"]), chunks[b"BIN\0"]


def write_glb(path, document, binary):
    document["buffers"] = [{"byteLength": len(binary)}]
    encoded = json.dumps(document, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 4)
    binary += b"\0" * (-len(binary) % 4)
    length = 12 + 8 + len(encoded) + 8 + len(binary)
    path.write_bytes(
        struct.pack("<4sII", b"glTF", 2, length)
        + struct.pack("<I4s", len(encoded), b"JSON")
        + encoded
        + struct.pack("<I4s", len(binary), b"BIN\0")
        + binary
    )


def multiply(a, b):
    return [sum(a[k * 4 + row] * b[col * 4 + k] for k in range(4))
            for col in range(4) for row in range(4)]


def direction(matrix, vector):
    return tuple(sum(matrix[col * 4 + row] * vector[col] for col in range(3))
                 for row in range(3))


def unit(vector):
    length = math.sqrt(sum(value * value for value in vector))
    assert length > 1e-8
    return tuple(value / length for value in vector)


def world_matrices(document):
    matrices = {}

    def visit(index, parent):
        node = document["nodes"][index]
        # Both supplied libraries use matrices; fail clearly on future formats.
        assert not any(key in node for key in ("translation", "rotation", "scale"))
        matrix = multiply(parent, node.get("matrix", IDENTITY))
        matrices[index] = matrix
        for child in node.get("children", []):
            visit(child, matrix)

    for index in document["scenes"][document.get("scene", 0)]["nodes"]:
        visit(index, IDENTITY)
    return matrices


def read_accessor(document, binary, index):
    accessor = document["accessors"][index]
    assert "sparse" not in accessor
    view = document["bufferViews"][accessor["bufferView"]]
    pattern = "<" + COMPONENTS[accessor["componentType"]] * WIDTHS[accessor["type"]]
    size = struct.calcsize(pattern)
    stride = view.get("byteStride", size)
    start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    assert start + (accessor["count"] - 1) * stride + size <= len(binary)
    return [struct.unpack_from(pattern, binary, start + item * stride)
            for item in range(accessor["count"])]


def add_accessor(document, binary, values, kind, component=5126, bounds=False):
    binary.extend(b"\0" * (-len(binary) % 4))
    offset = len(binary)
    pattern = "<" + COMPONENTS[component] * WIDTHS[kind]
    for value in values:
        binary.extend(struct.pack(pattern, *value))
    view_index = len(document["bufferViews"])
    document["bufferViews"].append({
        "buffer": 0, "byteOffset": offset, "byteLength": len(binary) - offset,
        "target": 34963 if kind == "SCALAR" else 34962,
    })
    accessor = {"bufferView": view_index, "componentType": component,
                "count": len(values), "type": kind}
    if bounds:
        accessor["min"] = [min(row[axis] for row in values) for axis in range(WIDTHS[kind])]
        accessor["max"] = [max(row[axis] for row in values) for axis in range(WIDTHS[kind])]
    index = len(document["accessors"])
    document["accessors"].append(accessor)
    return index


def extract_material(document, binary, material_index, family):
    material = copy.deepcopy(document["materials"][material_index])
    material["alphaMode"] = "MASK"
    material["alphaCutoff"] = 0.5
    material["doubleSided"] = True
    slots = [
        (material["pbrMetallicRoughness"]["baseColorTexture"], "base-color"),
        (material["pbrMetallicRoughness"]["metallicRoughnessTexture"], "orm"),
        (material["normalTexture"], "normal"),
    ]
    images, textures, files = [], [], {}
    for index, (slot, label) in enumerate(slots):
        source_texture = document["textures"][slot["index"]]
        source_image = document["images"][source_texture["source"]]
        assert source_image["mimeType"] == "image/png"
        view = document["bufferViews"][source_image["bufferView"]]
        offset = view.get("byteOffset", 0)
        image = binary[offset : offset + view["byteLength"]]
        assert image[:8] == b"\x89PNG\r\n\x1a\n"
        name = f"{family}-{label}.png"
        (TEXTURE_DIR / name).write_bytes(image)
        files[label] = f"textures/grass/{name}"
        images.append({"uri": f"../textures/grass/{name}", "mimeType": "image/png"})
        textures.append({"sampler": 0, "source": index})
        slot["index"] = index
    material["occlusionTexture"]["index"] = 1
    return material, images, textures, files


def slug(name):
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def extract():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    TEXTURE_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": 1,
        "coordinate_system": "right-handed, Y up; metres; root pivot at base Y=0",
        "modifications": "Separate variants; bake orientation/scale; remove display translations; "
                         "move mesh base to Y=0; share external textures; BLEND to MASK at 0.5.",
        "materials": [], "models": [], "sources": [],
    }
    for pack, families, excluded in PACKS:
        source = ROOT / f"tmp/grass-pack-{pack}.glb"
        document, binary = read_glb(source)
        matrices = world_matrices(document)
        source_info = copy.deepcopy(document["asset"].get("extras", {}))
        manifest["sources"].append({
            "file": source.name, "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            **source_info,
            "excluded_meshes": [document["meshes"][i]["name"] for i in sorted(excluded)],
        })
        materials = {}
        for index, family in families.items():
            material = extract_material(document, binary, index, family)
            materials[index] = material
            manifest["materials"].append({"name": family, "textures": material[3],
                                           "normal_scale": material[0]["normalTexture"].get("scale", 1),
                                           "alpha_cutoff": 0.5})
        mesh_nodes = {node["mesh"]: index for index, node in enumerate(document["nodes"])
                      if "mesh" in node}
        for mesh_index, mesh in enumerate(document["meshes"]):
            if mesh_index in excluded:
                continue
            assert len(mesh["primitives"]) == 1
            primitive = mesh["primitives"][0]
            assert primitive.get("mode", 4) == 4
            material_index = primitive["material"]
            assert material_index in families, mesh["name"]
            material, images, textures, files = materials[material_index]
            # The source mesh name ends in its material name and variant index.
            suffix = "_" + document["materials"][material_index]["name"] + "_0"
            assert mesh["name"].endswith(suffix)
            source_name = mesh["name"][:-len(suffix)]
            name = f"grass-pack-{pack}-{slug(source_name.removeprefix('Grass '))}"
            path = MODEL_DIR / f"{name}.glb"
            transform = matrices[mesh_nodes[mesh_index]]
            # These assets contain only uniform positive scale + rotation. For
            # those transforms, normalized linear vectors also give normals.
            columns = [tuple(transform[col * 4 + row] for row in range(3)) for col in range(3)]
            scales = [math.sqrt(sum(x * x for x in column)) for column in columns]
            assert max(scales) - min(scales) < 1e-5 and min(scales) > 0
            assert all(abs(sum(columns[a][i] * columns[b][i] for i in range(3))) < 1e-5
                       for a, b in [(0, 1), (0, 2), (1, 2)])
            determinant = sum(columns[0][i] * (columns[1][(i + 1) % 3] * columns[2][(i + 2) % 3]
                                             - columns[1][(i + 2) % 3] * columns[2][(i + 1) % 3])
                              for i in range(3))
            assert determinant > 0
            output = {
                "asset": {"version": "2.0", "generator": "realistic_forest/scripts/extract_grass.py",
                          "extras": {**source_info, "modifications": manifest["modifications"]}},
                "scene": 0, "scenes": [{"name": name, "nodes": [0]}],
                "nodes": [{"name": name, "mesh": 0}], "meshes": [],
                "accessors": [], "bufferViews": [],
                "materials": [copy.deepcopy(material)], "images": copy.deepcopy(images),
                "textures": copy.deepcopy(textures), "samplers": copy.deepcopy(document["samplers"]),
                "extensionsUsed": copy.deepcopy(document.get("extensionsUsed", [])),
            }
            output_binary = bytearray()
            output_primitive = {"attributes": {}, "mode": 4, "material": 0}
            positions = [direction(transform, row) for row in read_accessor(
                document, binary, primitive["attributes"]["POSITION"])]
            base = min(row[1] for row in positions)
            positions = [(x, y - base, z) for x, y, z in positions]
            for semantic, accessor_index in primitive["attributes"].items():
                values = read_accessor(document, binary, accessor_index)
                if semantic == "POSITION":
                    values = positions
                elif semantic == "NORMAL":
                    values = [unit(direction(transform, row)) for row in values]
                elif semantic == "TANGENT":
                    values = [(*unit(direction(transform, row[:3])), row[3]) for row in values]
                accessor = document["accessors"][accessor_index]
                assert accessor["componentType"] == 5126
                output_primitive["attributes"][semantic] = add_accessor(
                    output, output_binary, values, accessor["type"], bounds=semantic == "POSITION")
            indices = read_accessor(document, binary, primitive["indices"])
            output_primitive["indices"] = add_accessor(output, output_binary, indices, "SCALAR", 5125)
            output["meshes"].append({"name": name, "primitives": [output_primitive]})
            write_glb(path, output, output_binary)
            bounds = {key: [round(function(row[axis] for row in positions), 7) for axis in range(3)]
                      for key, function in [("min", min), ("max", max)]}
            manifest["models"].append({
                "file": f"models/{path.name}", "name": name, "source_pack": pack,
                "source_mesh": mesh_index, "source_name": source_name,
                "material": families[material_index], "vertices": len(positions),
                "triangles": len(indices) // 3, "bounds": bounds,
                "height": round(bounds["max"][1], 7),
                "radius": round(max(math.hypot(x, z) for x, _, z in positions), 7),
            })
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def validate():
    manifest = json.loads(MANIFEST_PATH.read_text())
    assert len(manifest["models"]) == 9
    assert all(model["source_pack"] == 2 for model in manifest["models"])
    assert len({model["file"] for model in manifest["models"]}) == 9
    assert {path.name for path in MODEL_DIR.glob("grass-pack-*.glb")} == {
        Path(model["file"]).name for model in manifest["models"]
    }
    for model in manifest["models"]:
        path = ROOT / "assets" / model["file"]
        document, binary = read_glb(path)
        assert len(document["meshes"]) == len(document["nodes"]) == 1
        assert "matrix" not in document["nodes"][0]
        assert document["asset"]["extras"]["license"].startswith("CC-BY-4.0")
        primitive = document["meshes"][0]["primitives"][0]
        positions = read_accessor(document, binary, primitive["attributes"]["POSITION"])
        assert len(positions) == model["vertices"]
        assert abs(min(row[1] for row in positions)) < 1e-6
        assert 0.1 < max(row[1] for row in positions) < 1.1
        assert all(math.isfinite(x) for row in positions for x in row)
        indices = read_accessor(document, binary, primitive["indices"])
        assert len(indices) == model["triangles"] * 3
        assert all(0 <= index[0] < len(positions) for index in indices)
        for semantic in ["NORMAL", "TANGENT"]:
            rows = read_accessor(document, binary, primitive["attributes"][semantic])
            assert len(rows) == len(positions)
            assert all(abs(sum(value * value for value in row[:3]) - 1) < 1e-5 for row in rows)
        for image in document["images"]:
            texture = (path.parent / image["uri"]).resolve()
            assert texture.is_relative_to(TEXTURE_DIR.resolve())
            assert texture.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        assert document["materials"][0]["alphaMode"] == "MASK"
    textures = {path for material in manifest["materials"] for path in material["textures"].values()}
    assert len(textures) == 3
    assert {path.name for path in TEXTURE_DIR.glob("grass-pack-*.png")} == {
        Path(path).name for path in textures
    }
    print(f"Validated {len(manifest['models'])} GLBs, {len(textures)} shared PNGs, "
          f"{sum(model['triangles'] for model in manifest['models'])} total triangles.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", action="store_true", help="validate outputs without source packs")
    args = parser.parse_args()
    if not args.validate:
        extract()
    validate()
