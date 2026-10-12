"""Export one building of the ~/buildings kit to a GLB the forest can place.

Run headless, once per building file; the original .blend is only read, never saved:

    /Applications/Blender.app/Contents/MacOS/Blender --background ~/buildings/house-a.blend \\
        --python-exit-code 1 --python scripts/export-buildings.py -- --assets assets

scripts/export-buildings.sh runs it over every building in the kit.

What it writes:

    assets/models/buildings/<file name>.glb
        part.<object>       every mesh in "Collection" (the house), in building space
        door.<object>       every mesh in "Collection 2", its node origin on the hinge axis
        collision.<kind>    every COL_<kind> mesh in "Collision": closed convex pieces, no material; each face's
                        UV holds the number of the loose part (piece) it belongs to, which the glTF
                        loses otherwise: touching pieces share corners
    assets/textures/buildings/<image>.<png|jpg>
        the kit's packed images, written once and shared by every building: the GLBs reference
        them by relative URI instead of each carrying its own copy

Each material's Mapping node (UV scale and rotation) is baked into the exported UVs, so a texture
is sampled with the UVs as they come. Faces longer than LARGE_EDGE are subdivided, and every
corner of the parts and doors carries COLOR_0: red is how much of the sky the surface sees past
the building itself (its ambient light), green how much rain reaches it from above. Glass lets
both through.
"""

import argparse
import json
import math
import os
import struct
import sys

import bmesh
import bpy
from mathutils import Euler
from mathutils import Matrix
from mathutils import Vector
from mathutils.bvhtree import BVHTree


HOUSE_COLLECTION_NAME = "Collection"
DOOR_COLLECTION_NAME = "Collection 2"
COLLISION_COLLECTION_NAME = "Collision"
COLLISION_PREFIX = "COL_"
EXPORT_COLLECTION_NAME = "forest-export"

# Faces are cut until no edge is longer than this, so the sky visibility has corners to vary over.
LARGE_EDGE = 1.25
SUBDIVIDE_PASSES = 6
# Rays per corner: a cosine-weighted hemisphere for the sky, a cone about straight up for rain.
SKY_RAYS = 48
RAIN_RAYS = 12
RAIN_CONE_DEGREES = 22.0
RAY_REACH = 60.0
RAY_LIFT = 0.012
SKY_COLOUR_NAME = "sky"

TEXTURE_DIRECTORY = "textures/buildings"
MODEL_DIRECTORY = "models/buildings"
# Images keep their packed bytes; only the name is tidied.
NAME_TOKENS_DROPPED = ("_4K", "_2K-PNG")


def parseArguments():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description="Export a building of the kit to a GLB.")
    parser.add_argument("--assets", required=True, help="the forest's assets directory")
    parser.add_argument("--name", default=None, help="building name (default: the .blend's file name)")
    return parser.parse_args(argv)


def kebabName(image_name):
    name = image_name
    for token in NAME_TOKENS_DROPPED:
        name = name.replace(token, "")
    characters = []
    for index, character in enumerate(name):
        previous = name[index - 1] if index > 0 else ""
        starts_word = character.isupper() and previous.islower()
        if starts_word:
            characters.append("-")
        characters.append(character.lower())
    return "".join(characters).replace("_", "-")


# ---------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------


def linkedNode(socket):
    if not socket.is_linked:
        return None
    return socket.links[0].from_node


def imageFeeding(socket):
    """The image texture node feeding a socket directly or through a Separate Color or Normal Map."""
    node = linkedNode(socket)
    if node is None:
        return None
    if node.type == "TEX_IMAGE":
        return node
    if node.type in ("SEPARATE_COLOR", "NORMAL_MAP"):
        return imageFeeding(node.inputs["Color"])
    return None


def mappingOf(image_node):
    """The UV transform a texture node samples through, as a function of (u, v)."""
    mapping = linkedNode(image_node.inputs["Vector"])
    if mapping is None or mapping.type != "MAPPING":
        return None
    location = Vector(mapping.inputs["Location"].default_value)
    rotation = Euler(mapping.inputs["Rotation"].default_value).to_matrix()
    scale = Vector(mapping.inputs["Scale"].default_value)
    return (mapping.vector_type, location, rotation, scale)


def sameMapping(first, second):
    if first is None or second is None:
        return first is None and second is None
    first_type, first_location, first_rotation, first_scale = first
    second_type, second_location, second_rotation, second_scale = second
    if first_type != second_type:
        return False
    location_gap = (first_location - second_location).length
    scale_gap = (first_scale - second_scale).length
    rotation_gap = max(abs(a - b) for row_a, row_b in zip(first_rotation, second_rotation) for a, b in zip(row_a, row_b))
    return location_gap < 1e-4 and scale_gap < 1e-4 and rotation_gap < 1e-4


def applyMapping(mapping, uv):
    if mapping is None:
        return uv
    vector_type, location, rotation, scale = mapping
    point = Vector((uv[0], uv[1], 0.0))
    if vector_type == "POINT":
        scaled = Vector((point.x * scale.x, point.y * scale.y, point.z * scale.z))
        mapped = rotation @ scaled + location
    elif vector_type == "TEXTURE":
        moved = rotation.inverted() @ (point - location)
        mapped = Vector((moved.x / scale.x, moved.y / scale.y, moved.z / scale.z))
    else:
        raise RuntimeError(f"unsupported mapping type {vector_type}")
    return (mapped.x, mapped.y)


class KitMaterial:
    """What the forest needs of a material: its images and the one UV mapping they share."""

    def __init__(self, material):
        self.name = material.name
        self.base_color = None
        self.roughness = None
        self.metallic = None
        self.normal = None
        self.mapping = None
        self.colour_factor = [1.0, 1.0, 1.0, 1.0]
        self.roughness_factor = 1.0
        self.glass = False
        nodes = material.node_tree.nodes
        principled = next((node for node in nodes if node.type == "BSDF_PRINCIPLED"), None)
        if principled is None:
            # The kit's Glass is a Glass, Translucent or Principled BSDF depending on the file.
            self.glass = True
            return
        if self.name == "Glass":
            self.glass = True
            return
        self.base_color = imageFeeding(principled.inputs["Base Color"])
        self.roughness = imageFeeding(principled.inputs["Roughness"])
        self.metallic = imageFeeding(principled.inputs["Metallic"])
        self.normal = imageFeeding(principled.inputs["Normal"])
        if self.base_color is None:
            colour = principled.inputs["Base Color"].default_value
            self.colour_factor = [colour[0], colour[1], colour[2], 1.0]
            self.roughness_factor = principled.inputs["Roughness"].default_value
        textured = [node for node in (self.base_color, self.roughness, self.metallic, self.normal) if node is not None]
        mappings = [mappingOf(node) for node in textured]
        for mapping in mappings[1:]:
            if not sameMapping(mapping, mappings[0]):
                raise RuntimeError(f"{self.name}: its textures are sampled through different mappings")
        self.mapping = mappings[0] if mappings else None


def kitMaterials():
    materials = {}
    for material in bpy.data.materials:
        if material.node_tree is None:
            continue
        materials[material.name] = KitMaterial(material)
    return materials


# ---------------------------------------------------------------------------
# Export objects
# ---------------------------------------------------------------------------


def exportCollection(scene):
    collection = bpy.data.collections.new(EXPORT_COLLECTION_NAME)
    scene.collection.children.link(collection)
    return collection


def copyObject(source, name, collection, keep_transform):
    """A copy of a mesh object with its own mesh, in building space unless it keeps its transform."""
    mesh = source.data.copy()
    mesh.name = name
    copy = bpy.data.objects.new(name, mesh)
    collection.objects.link(copy)
    if keep_transform:
        copy.matrix_world = source.matrix_world.copy()
    else:
        mesh.transform(source.matrix_world)
        copy.matrix_world = Matrix.Identity(4)
    return copy


def bakeMappings(mesh, materials):
    """Put each face's material mapping into its UVs, so the exported UVs sample the textures directly."""
    uv_layer = mesh.uv_layers.active
    if uv_layer is None:
        return
    for polygon in mesh.polygons:
        slot_material = mesh.materials[polygon.material_index] if mesh.materials else None
        if slot_material is None:
            continue
        mapping = materials[slot_material.name].mapping
        if mapping is None:
            continue
        for loop_index in polygon.loop_indices:
            uv = uv_layer.data[loop_index].uv
            mapped = applyMapping(mapping, (uv[0], uv[1]))
            uv_layer.data[loop_index].uv = mapped


def tagLooseParts(mesh):
    """Number the mesh's loose parts and write each face's part into its UVs (u = part + 0.5)."""
    while mesh.uv_layers:
        mesh.uv_layers.remove(mesh.uv_layers[0])
    bm = bmesh.new()
    bm.from_mesh(mesh)
    uv_layer = bm.loops.layers.uv.new("piece")
    bm.faces.ensure_lookup_table()
    part_of = {}
    part_count = 0
    for start in bm.faces:
        if start.index in part_of:
            continue
        stack = [start]
        part_of[start.index] = part_count
        while stack:
            face = stack.pop()
            for vertex in face.verts:
                for neighbour in vertex.link_faces:
                    if neighbour.index not in part_of:
                        part_of[neighbour.index] = part_count
                        stack.append(neighbour)
        part_count += 1
    for face in bm.faces:
        part = part_of[face.index]
        for loop in face.loops:
            loop[uv_layer].uv = (part + 0.5, 0.5)
    bm.to_mesh(mesh)
    bm.free()
    return part_count


def subdivideLargeFaces(mesh):
    bm = bmesh.new()
    bm.from_mesh(mesh)
    for _ in range(SUBDIVIDE_PASSES):
        long_edges = [edge for edge in bm.edges if edge.calc_length() > LARGE_EDGE]
        if not long_edges:
            break
        bmesh.ops.subdivide_edges(bm, edges=long_edges, cuts=1, use_grid_fill=True)
    bm.to_mesh(mesh)
    bm.free()


# ---------------------------------------------------------------------------
# Sky visibility and rain
# ---------------------------------------------------------------------------


def occluderTree(objects, materials):
    """A BVH of everything that stops light, in world space: the building and its closed doors, not glass."""
    vertices = []
    polygons = []
    for obj in objects:
        mesh = obj.data
        matrix = obj.matrix_world
        offset = len(vertices)
        for vertex in mesh.vertices:
            vertices.append(matrix @ vertex.co)
        for polygon in mesh.polygons:
            slot_material = mesh.materials[polygon.material_index] if mesh.materials else None
            if slot_material is not None and materials[slot_material.name].glass:
                continue
            polygons.append([offset + index for index in polygon.vertices])
    return BVHTree.FromPolygons(vertices, polygons)


def hemisphereDirections(count):
    """Cosine-weighted directions about +Z on a Hammersley set, so every corner samples alike."""
    directions = []
    for index in range(count):
        first = (index + 0.5) / count
        bits = index
        second = 0.0
        scale = 0.5
        while bits:
            if bits & 1:
                second += scale
            bits >>= 1
            scale *= 0.5
        radius = math.sqrt(first)
        angle = 2.0 * math.pi * second
        height = math.sqrt(max(1.0 - first, 0.0))
        directions.append(Vector((radius * math.cos(angle), radius * math.sin(angle), height)))
    return directions


def rainDirections(count):
    """Rays up into the sky within the cone wind-driven rain falls through."""
    cone = math.radians(RAIN_CONE_DEGREES)
    directions = [Vector((0.0, 0.0, 1.0))]
    for index in range(count - 1):
        angle = 2.0 * math.pi * index / (count - 1)
        tilt = cone * (0.55 if index % 2 == 0 else 1.0)
        directions.append(Vector((math.sin(tilt) * math.cos(angle), math.sin(tilt) * math.sin(angle), math.cos(tilt))))
    return directions


def frameAbout(normal):
    """A rotation taking +Z to the normal."""
    up = Vector((0.0, 0.0, 1.0))
    if abs(normal.z) > 0.999:
        return Matrix.Identity(3) if normal.z > 0.0 else Matrix.Rotation(math.pi, 3, "X")
    return up.rotation_difference(normal).to_matrix()


def escapedShare(tree, origin, directions):
    escaped = 0
    for direction in directions:
        hit = tree.ray_cast(origin, direction, RAY_REACH)
        if hit[0] is None:
            escaped += 1
    return escaped / len(directions)


def bakeSky(obj, tree, sky_directions, rain_directions):
    """Write the sky and rain shares of every face corner into the object's corner colour attribute."""
    mesh = obj.data
    matrix = obj.matrix_world
    normal_matrix = matrix.to_3x3().inverted().transposed()
    attribute = mesh.color_attributes.new(SKY_COLOUR_NAME, "FLOAT_COLOR", "CORNER")
    cache = {}
    for polygon in mesh.polygons:
        world_normal = (normal_matrix @ polygon.normal).normalized()
        centre = matrix @ polygon.center
        frame = frameAbout(world_normal)
        for loop_index in polygon.loop_indices:
            vertex_index = mesh.loops[loop_index].vertex_index
            corner = matrix @ mesh.vertices[vertex_index].co
            # A hair in from the corner toward the face's centre, so rays leave the face itself and
            # not the edge it shares with the next.
            inward = (centre - corner) * 0.02
            key = (round(corner.x, 3), round(corner.y, 3), round(corner.z, 3),
                   round(world_normal.x, 2), round(world_normal.y, 2), round(world_normal.z, 2))
            shares = cache.get(key)
            if shares is None:
                origin = corner + inward + world_normal * RAY_LIFT
                rotated = [frame @ direction for direction in sky_directions]
                sky = escapedShare(tree, origin, rotated)
                rain = 0.0
                if world_normal.z > -0.2:
                    rain = escapedShare(tree, origin, rain_directions)
                shares = (sky, rain)
                cache[key] = shares
            attribute.data[loop_index].color = (shares[0], shares[1], 0.0, 1.0)
    mesh.color_attributes.active_color = attribute
    mesh.color_attributes.render_color_index = mesh.color_attributes.active_color_index


# ---------------------------------------------------------------------------
# GLB post-processing
# ---------------------------------------------------------------------------


def readGlb(path):
    with open(path, "rb") as handle:
        data = handle.read()
    magic, version, _ = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF" or version != 2:
        raise RuntimeError(f"{path} is not a glTF 2 binary")
    json_length, json_type = struct.unpack_from("<I4s", data, 12)
    if json_type != b"JSON":
        raise RuntimeError(f"{path}: the first chunk is not JSON")
    document = json.loads(data[20:20 + json_length])
    binary_offset = 20 + json_length
    binary_length, binary_type = struct.unpack_from("<I4s", data, binary_offset)
    if binary_type != b"BIN\x00":
        raise RuntimeError(f"{path}: the second chunk is not BIN")
    binary = data[binary_offset + 8:binary_offset + 8 + binary_length]
    return document, binary


def writeGlb(path, document, binary):
    json_bytes = json.dumps(document, separators=(",", ":")).encode("utf-8")
    json_bytes += b" " * ((4 - len(json_bytes) % 4) % 4)
    padded_binary = binary + b"\x00" * ((4 - len(binary) % 4) % 4)
    total = 12 + 8 + len(json_bytes) + 8 + len(padded_binary)
    header = struct.pack("<4sII", b"glTF", 2, total)
    json_header = struct.pack("<I4s", len(json_bytes), b"JSON")
    binary_header = struct.pack("<I4s", len(padded_binary), b"BIN\x00")
    with open(path, "wb") as handle:
        handle.write(header)
        handle.write(json_header)
        handle.write(json_bytes)
        handle.write(binary_header)
        handle.write(padded_binary)


class TextureTable:
    """The glTF images, samplers and textures the materials point at, one per shared image file."""

    def __init__(self, document, uri_prefix):
        self.document = document
        self.uri_prefix = uri_prefix
        self.indices = {}
        document["images"] = []
        document["textures"] = []
        sampler = {"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}
        document["samplers"] = [sampler]

    def textureFor(self, file_name):
        if file_name in self.indices:
            return self.indices[file_name]
        image = {"uri": self.uri_prefix + file_name, "name": file_name}
        self.document["images"].append(image)
        texture = {"sampler": 0, "source": len(self.document["images"]) - 1}
        self.document["textures"].append(texture)
        index = len(self.document["textures"]) - 1
        self.indices[file_name] = index
        return index


def describeMaterials(document, materials, files, uri_prefix):
    """Point every exported material at the shared image files and give it the kit's factors."""
    table = TextureTable(document, uri_prefix)
    # The exporter's own material extensions (specular, IOR) describe Blender's look, not the kit's.
    for key in ("extensionsUsed", "extensionsRequired"):
        kept = [name for name in document.get(key, []) if not name.startswith("KHR_materials_")]
        if kept:
            document[key] = kept
        else:
            document.pop(key, None)
    for entry in document.get("materials", []):
        kit = materials[entry["name"]]
        for key in ("pbrMetallicRoughness", "normalTexture", "occlusionTexture", "emissiveTexture", "extensions"):
            entry.pop(key, None)
        if kit.glass:
            entry["alphaMode"] = "BLEND"
            entry["doubleSided"] = True
            entry["pbrMetallicRoughness"] = {
                "baseColorFactor": [0.86, 0.9, 0.92, 0.18],
                "metallicFactor": 0.0,
                "roughnessFactor": 0.05,
            }
            continue
        entry.pop("alphaMode", None)
        pbr = {
            "baseColorFactor": kit.colour_factor,
            "metallicFactor": 1.0 if kit.metallic is not None else 0.0,
            "roughnessFactor": kit.roughness_factor,
        }
        if kit.base_color is not None:
            pbr["baseColorTexture"] = {"index": table.textureFor(files[kit.base_color.image.name])}
        # A plain roughness map is grey, so its green channel is the roughness glTF reads; the
        # metallic factor of 0 keeps its blue channel from making wood metal.
        metallic_roughness = files.get(("orm", kit.name))
        if metallic_roughness is None and kit.roughness is not None:
            metallic_roughness = files[kit.roughness.image.name]
        if metallic_roughness is not None:
            pbr["metallicRoughnessTexture"] = {"index": table.textureFor(metallic_roughness)}
        entry["pbrMetallicRoughness"] = pbr
        if kit.normal is not None:
            entry["normalTexture"] = {"index": table.textureFor(files[kit.normal.image.name]), "scale": 1.0}


# ---------------------------------------------------------------------------
# Shared textures
# ---------------------------------------------------------------------------


def writeIfChanged(path, data):
    if os.path.isfile(path):
        with open(path, "rb") as handle:
            if handle.read() == data:
                return False
    with open(path, "wb") as handle:
        handle.write(data)
    return True


def writePackedImage(image, directory):
    """The image's packed bytes as they are, under a tidied name; returns the file name."""
    if image.packed_file is None:
        raise RuntimeError(f"{image.name} is not packed")
    data = bytes(image.packed_file.data)
    extension = "jpg" if data[:3] == b"\xff\xd8\xff" else "png"
    if extension == "png" and data[:8] != b"\x89PNG\r\n\x1a\n":
        raise RuntimeError(f"{image.name}: packed bytes are neither JPEG nor PNG")
    file_name = f"{kebabName(image.name)}.{extension}"
    if writeIfChanged(os.path.join(directory, file_name), data):
        print(f"TEXTURE wrote {file_name} ({len(data) / 1048576:.1f} MiB)")
    return file_name


def writeMetalOrm(kit, directory):
    """glTF reads roughness from green and metalness from blue of one image: pack the two maps into one."""
    import numpy

    roughness = kit.roughness.image
    metalness = kit.metallic.image
    width, height = roughness.size
    if tuple(metalness.size) != (width, height):
        raise RuntimeError(f"{kit.name}: roughness and metalness differ in size")
    file_name = f"{kebabName(roughness.name).replace('-roughness', '')}-orm.png"
    path = os.path.join(directory, file_name)
    if os.path.isfile(path):
        return file_name
    texel_count = width * height
    roughness_pixels = numpy.empty(texel_count * 4, dtype=numpy.float32)
    metalness_pixels = numpy.empty(texel_count * 4, dtype=numpy.float32)
    roughness.pixels.foreach_get(roughness_pixels)
    metalness.pixels.foreach_get(metalness_pixels)
    packed = numpy.ones(texel_count * 4, dtype=numpy.float32)
    packed[1::4] = roughness_pixels[0::4]
    packed[2::4] = metalness_pixels[0::4]
    image = bpy.data.images.new(file_name, width, height, alpha=False, float_buffer=False, is_data=True)
    image.pixels.foreach_set(packed)
    image.filepath_raw = path
    image.file_format = "PNG"
    image.save()
    print(f"TEXTURE wrote {file_name}")
    return file_name


def writeTextures(materials, directory):
    os.makedirs(directory, exist_ok=True)
    files = {}
    for kit in materials.values():
        packs_metalness = kit.metallic is not None and kit.roughness is not None
        images = [kit.base_color, kit.normal]
        if not packs_metalness:
            images.append(kit.roughness)
        for node in images:
            if node is None or node.image.name in files:
                continue
            files[node.image.name] = writePackedImage(node.image, directory)
        if packs_metalness:
            files[("orm", kit.name)] = writeMetalOrm(kit, directory)
    return files


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def collectionMeshes(name):
    collection = bpy.data.collections.get(name)
    if collection is None:
        raise RuntimeError(f"the file has no '{name}' collection")
    return [obj for obj in collection.objects if obj.type == "MESH"]


def exportGlb(path, objects):
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    export_options = {
        "filepath": path,
        "export_format": "GLB",
        "use_selection": True,
        "export_image_format": "NONE",
        "export_materials": "EXPORT",
        "export_yup": True,
        "export_apply": False,
        "export_texcoords": True,
        "export_normals": True,
        "export_tangents": False,
        "export_vertex_color": "NAME",
        "export_vertex_color_name": SKY_COLOUR_NAME,
        "export_all_vertex_colors": False,
        "export_active_vertex_color_when_no_material": False,
        "export_attributes": False,
        "export_extras": False,
        "export_cameras": False,
        "export_lights": False,
        "export_animations": False,
        "export_skins": False,
        "export_morph": False,
    }
    bpy.ops.export_scene.gltf(**export_options)


def main():
    arguments = parseArguments()
    building_name = arguments.name or os.path.splitext(os.path.basename(bpy.data.filepath))[0]
    assets = os.path.abspath(arguments.assets)
    model_directory = os.path.join(assets, MODEL_DIRECTORY)
    texture_directory = os.path.join(assets, TEXTURE_DIRECTORY)
    os.makedirs(model_directory, exist_ok=True)

    materials = kitMaterials()
    files = writeTextures(materials, texture_directory)

    parts = collectionMeshes(HOUSE_COLLECTION_NAME)
    doors = collectionMeshes(DOOR_COLLECTION_NAME)
    collisions = collectionMeshes(COLLISION_COLLECTION_NAME)
    tree = occluderTree(parts + doors, materials)

    scene = bpy.context.scene
    collection = exportCollection(scene)
    exported = []
    sky_directions = hemisphereDirections(SKY_RAYS)
    rain_directions = rainDirections(RAIN_RAYS)
    for source in parts:
        part = copyObject(source, f"part.{source.name}", collection, keep_transform=False)
        bakeMappings(part.data, materials)
        subdivideLargeFaces(part.data)
        bakeSky(part, tree, sky_directions, rain_directions)
        exported.append(part)
        print(f"PART {source.name}: {len(part.data.polygons)} faces")
    for source in doors:
        door = copyObject(source, f"door.{source.name}", collection, keep_transform=True)
        bakeMappings(door.data, materials)
        bakeSky(door, tree, sky_directions, rain_directions)
        exported.append(door)
    for source in collisions:
        if not source.name.startswith(COLLISION_PREFIX):
            continue
        kind = source.name[len(COLLISION_PREFIX):]
        collision = copyObject(source, f"collision.{kind}", collection, keep_transform=False)
        collision.data.materials.clear()
        piece_count = tagLooseParts(collision.data)
        exported.append(collision)
        print(f"COLLISION {kind}: {piece_count} pieces")

    glb_path = os.path.join(model_directory, f"{building_name}.glb")
    exportGlb(glb_path, exported)
    document, binary = readGlb(glb_path)
    uri_prefix = os.path.relpath(texture_directory, model_directory).replace(os.sep, "/") + "/"
    describeMaterials(document, materials, files, uri_prefix)
    writeGlb(glb_path, document, binary)

    node_names = [node.get("name", "") for node in document.get("nodes", [])]
    part_count = sum(1 for name in node_names if name.startswith("part."))
    door_count = sum(1 for name in node_names if name.startswith("door."))
    collision_count = sum(1 for name in node_names if name.startswith("collision."))
    size = os.path.getsize(glb_path) / 1048576
    print(f"EXPORTED {glb_path}: {part_count} parts, {door_count} doors, {collision_count} collision meshes, "
          f"{len(document.get('materials', []))} materials, {size:.1f} MiB")


main()
