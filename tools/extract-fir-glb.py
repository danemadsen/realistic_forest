#!/usr/bin/env python3
"""Split the fir tree pack GLB into one GLB per tree per LOD.

The source pack (tmp/fir.glb, Sketchfab "Realistic fir trees pack (LODS,
gameready)", CC-BY-4.0) holds nine trees laid out in a row, each with four LOD
nodes stacked along world -Z for display. Every tree's LOD0-LOD2 are two
primitives -- bark and branches -- that share one bark material and one branch
material across the whole pack. Every LOD3 is a six-faced impostor atlas with
its own unique material and textures.

This script writes:

  assets/models/fir/fir-{small,medium,large}-{n}-lod-{k}.glb
  assets/models/fir/fir-bark-{basecolor,normal,metallicroughness}.png
  assets/models/fir/fir-branches-{basecolor,normal,metallicroughness}.png

The six shared pack textures become real files so the 36 GLBs do not each
carry their own 40 MB copy; the per-tree LOD3 billboard textures stay embedded
because they are unique per tree.

Geometry is transformed so each output GLB is a standalone tree with its trunk
foot at the origin, +Y up, standing on the XZ plane. The pack stores its meshes
upside down (mesh +Y is world -Y) with the LODs offset along Z by their stack
position; both are removed here, so an instance transform is a pure
translate/rotate/scale with no correction term.

  tools/extract-fir-glb.py [--source tmp/fir.glb] [--out assets/models/fir]
"""

import argparse
import io
import json
import os
import struct
import sys

import numpy as np
from PIL import Image

# Node index of each tree's placeholder in the source scene, in pack order.
# The name after " tree_" is the pack's own 1-based tree number.
TREES = [
    ('large', 1, 3), ('large', 2, 15), ('large', 3, 27),
    ('medium', 1, 39), ('medium', 2, 51), ('medium', 3, 63),
    ('small', 1, 75), ('small', 2, 87), ('small', 3, 99),
]

# Component type table for glTF accessors, as (struct code, numpy dtype, size).
COMPONENT_TYPES = {
    5120: ('b', np.int8, 1),
    5121: ('B', np.uint8, 1),
    5122: ('h', np.int16, 2),
    5123: ('H', np.uint16, 2),
    5125: ('I', np.uint32, 4),
    5126: ('f', np.float32, 4),
}
TYPE_COUNTS = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}

# The six pack textures the bark and branch materials point at, and the file
# each one becomes. Base colour and normal carry the look; the
# metallic-roughness maps are all-white or near-black in this pack -- every
# material is a fully rough, non-metallic dielectric -- but they are extracted
# anyway so the written materials stay a faithful, valid description.
SHARED_TEXTURES = {
    0: 'fir-bark-basecolor.png',
    1: 'fir-bark-metallicroughness.png',
    2: 'fir-bark-normal.png',
    3: 'fir-branches-basecolor.png',
    4: 'fir-branches-metallicroughness.png',
    5: 'fir-branches-normal.png',
}


class Glb:
    """A parsed binary glTF: its JSON tree and its single binary chunk."""

    def __init__(self, path):
        with open(path, 'rb') as handle:
            raw = handle.read()
        magic, version, length = struct.unpack('<III', raw[:12])
        if magic != 0x46546C67 or version != 2:
            raise ValueError(f'{path}: not a glTF 2.0 binary')
        if length != len(raw):
            raise ValueError(f'{path}: header length {length} != file size {len(raw)}')
        self.json = None
        self.bin = b''
        offset = 12
        while offset < length:
            chunk_len, chunk_type = struct.unpack('<II', raw[offset:offset + 8])
            body = raw[offset + 8:offset + 8 + chunk_len]
            if chunk_type == 0x4E4F534A:
                self.json = json.loads(body.decode('utf-8'))
            elif chunk_type == 0x004E4942:
                self.bin = body
            offset += 8 + chunk_len + (-chunk_len % 4)
        if self.json is None:
            raise ValueError(f'{path}: no JSON chunk')

    def accessor(self, index):
        """Read one accessor as an (count, components) numpy array."""
        acc = self.json['accessors'][index]
        code, dtype, size = COMPONENT_TYPES[acc['componentType']]
        components = TYPE_COUNTS[acc['type']]
        view = self.json['bufferViews'][acc['bufferView']]
        base = view.get('byteOffset', 0) + acc.get('byteOffset', 0)
        stride = view.get('byteStride') or size * components
        if stride == size * components:
            count = acc['count'] * components
            return np.frombuffer(self.bin, dtype=dtype, count=count, offset=base) \
                .reshape(acc['count'], components)
        rows = np.empty((acc['count'], components), dtype=dtype)
        for row in range(acc['count']):
            start = base + row * stride
            rows[row] = np.frombuffer(self.bin, dtype=dtype, count=components, offset=start)
        return rows

    def image_bytes(self, index):
        image = self.json['images'][index]
        if 'bufferView' not in image:
            raise ValueError('image is not embedded')
        view = self.json['bufferViews'][image['bufferView']]
        start = view.get('byteOffset', 0)
        return self.bin[start:start + view['byteLength']]


def local_matrix(node):
    """The node's own transform, as a row-major 4x4."""
    if 'matrix' in node:
        # glTF matrices are column-major.
        return np.array(node['matrix'], dtype=np.float64).reshape(4, 4).T
    matrix = np.eye(4)
    if 'scale' in node:
        matrix = np.diag([*node['scale'], 1.0]) @ matrix
    if 'rotation' in node:
        x, y, z, w = node['rotation']
        rotation = np.eye(4)
        rotation[:3, :3] = [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
        matrix = rotation @ matrix
    if 'translation' in node:
        matrix[:3, 3] = node['translation']
    return matrix


def world_matrix(gltf, index, parents):
    """The node's transform in scene space."""
    matrix = local_matrix(gltf.json['nodes'][index])
    parent = parents.get(index)
    while parent is not None:
        matrix = local_matrix(gltf.json['nodes'][parent]) @ matrix
        parent = parents.get(parent)
    return matrix


def build_parents(gltf):
    parents = {}
    for index, node in enumerate(gltf.json['nodes']):
        for child in node.get('children', []):
            parents[child] = index
    return parents


def png_bytes(image):
    """Re-encode a decoded image as a plain PNG.

    The pack stores several maps as 1-bit greyscale or palette PNGs. Decoding
    and re-encoding normalises them to 8-bit RGB/RGBA, which every consumer
    reads, without touching a single sample value.
    """
    buffer = io.BytesIO()
    if image.mode == 'P':
        # Palette entries carry the actual channel values; keep them verbatim.
        image = image.convert('RGBA' if 'transparency' in image.info else 'RGB')
    image.save(buffer, format='PNG', optimize=True)
    return buffer.getvalue()


class Builder:
    """Accumulates accessors, buffer views and buffer bytes for one GLB."""

    def __init__(self):
        self.json = {
            'asset': {
                'version': '2.0',
                'generator': 'forest tools/extract-fir-glb.py',
            },
            'scene': 0,
            'scenes': [{'nodes': [0]}],
            'nodes': [],
            'meshes': [],
            'materials': [],
            'textures': [],
            'images': [],
            'samplers': [],
            'accessors': [],
            'bufferViews': [],
            'buffers': [],
        }
        self.bytes = bytearray()

    def _view(self, data, target=None):
        # glTF requires accessor byte offsets to be multiples of the component
        # size, so pad every view to 4 bytes.
        while len(self.bytes) % 4:
            self.bytes.append(0)
        view = {'buffer': 0, 'byteOffset': len(self.bytes), 'byteLength': len(data)}
        if target is not None:
            view['target'] = target
        self.json['bufferViews'].append(view)
        self.bytes.extend(data)
        return len(self.json['bufferViews']) - 1

    def attribute(self, array, component_type, gltf_type, minimum=None, maximum=None):
        view = self._view(array.tobytes(), target=34962)
        accessor = {
            'bufferView': view,
            'componentType': component_type,
            'count': int(array.shape[0]),
            'type': gltf_type,
        }
        if minimum is not None:
            accessor['min'] = [float(v) for v in minimum]
            accessor['max'] = [float(v) for v in maximum]
        self.json['accessors'].append(accessor)
        return len(self.json['accessors']) - 1

    def indices(self, array):
        view = self._view(array.tobytes(), target=34963)
        self.json['accessors'].append({
            'bufferView': view,
            'componentType': 5125,
            'count': int(array.shape[0]),
            'type': 'SCALAR',
        })
        return len(self.json['accessors']) - 1

    def embedded_image(self, data):
        view = self._view(data)
        self.json['images'].append({'bufferView': view, 'mimeType': 'image/png'})
        return len(self.json['images']) - 1

    def external_image(self, uri):
        self.json['images'].append({'uri': uri})
        return len(self.json['images']) - 1

    def finish(self):
        self.json['buffers'] = [{'byteLength': len(self.bytes)}]
        binary = bytes(self.bytes)
        binary += b'\0' * (-len(binary) % 4)
        document = json.dumps(self.json, separators=(',', ':')).encode('utf-8')
        document += b' ' * (-len(document) % 4)
        total = 12 + 8 + len(document) + 8 + len(binary)
        out = bytearray()
        out += struct.pack('<III', 0x46546C67, 2, total)
        out += struct.pack('<II', len(document), 0x4E4F534A) + document
        out += struct.pack('<II', len(binary), 0x004E4942) + binary
        return bytes(out)


def write_shared_textures(gltf, out_dir, log):
    for index, name in sorted(SHARED_TEXTURES.items()):
        image = Image.open(io.BytesIO(gltf.image_bytes(index)))
        image.load()
        data = png_bytes(image)
        path = os.path.join(out_dir, name)
        with open(path, 'wb') as handle:
            handle.write(data)
        log(f'  texture {name:38s} {image.size[0]}x{image.size[1]} {image.mode:5s}'
            f' {len(data) / 1024:8.0f} KB')
    return {index: name for index, name in SHARED_TEXTURES.items()}


def copy_material(gltf, index, builder, shared, keep_extensions):
    """Copy one pack material, re-pointing its shared textures at files.

    Materials whose textures are all shared (bark, branches) get external
    URIs; materials that own any unique texture (the LOD3 billboards) keep
    every one of their textures embedded, so the whole material stays
    self-contained in its GLB.
    """
    material = json.loads(json.dumps(gltf.json['materials'][index]))
    pbr = material.get('pbrMetallicRoughness', {})
    referenced = set()

    def texture_slot(holder, key):
        if key in holder:
            referenced.add(holder[key]['index'])

    texture_slot(pbr, 'baseColorTexture')
    texture_slot(pbr, 'metallicRoughnessTexture')
    texture_slot(material, 'normalTexture')
    texture_slot(material, 'occlusionTexture')
    texture_slot(material, 'emissiveTexture')
    specular = material.get('extensions', {}).get('KHR_materials_specular', {})
    texture_slot(specular, 'specularTexture')
    texture_slot(specular, 'specularColorTexture')

    # A material is shared only if every texture it names is in the shared set.
    shared_material = referenced and referenced <= set(shared)

    images = {}
    for texture_index in sorted(referenced):
        texture = gltf.json['textures'][texture_index]
        source = texture.get('source')
        if source is None:
            raise ValueError(f'texture {texture_index} has no source image')
        if source in images:
            continue
        if shared_material:
            images[source] = builder.external_image(shared[source])
        else:
            images[source] = builder.embedded_image(
                png_bytes(_decode(gltf, source)))

    # The pack ships one sampler; carry it across when anything needs it.
    sampler_index = None
    for texture_index in sorted(referenced):
        if 'sampler' in gltf.json['textures'][texture_index]:
            if sampler_index is None:
                sampler_index = len(builder.json['samplers'])
                builder.json['samplers'].append(
                    gltf.json['samplers'][gltf.json['textures'][texture_index]['sampler']])
            break

    remapped = {}
    for texture_index in sorted(referenced):
        texture = gltf.json['textures'][texture_index]
        entry = {'source': images[texture['source']]}
        if sampler_index is not None:
            entry['sampler'] = sampler_index
        builder.json['textures'].append(entry)
        remapped[texture_index] = len(builder.json['textures']) - 1

    def repoint(holder, key):
        if key in holder:
            holder[key] = dict(holder[key])
            holder[key]['index'] = remapped[holder[key]['index']]

    repoint(pbr, 'baseColorTexture')
    repoint(pbr, 'metallicRoughnessTexture')
    repoint(material, 'normalTexture')
    repoint(material, 'occlusionTexture')
    repoint(material, 'emissiveTexture')
    repoint(specular, 'specularTexture')
    repoint(specular, 'specularColorTexture')

    if not keep_extensions:
        material.pop('extensions', None)

    builder.json['materials'].append(material)
    return len(builder.json['materials']) - 1


def _decode(gltf, image_index):
    image = Image.open(io.BytesIO(gltf.image_bytes(image_index)))
    image.load()
    return image


def extract_tree(gltf, parents, size, number, root, out_dir, log):
    """Write the four LOD GLBs for one tree; returns per-LOD summaries."""
    node = gltf.json['nodes'][root]
    lod_nodes = node['children']
    summaries = []

    for lod, lod_node in enumerate(lod_nodes):
        builder = Builder()
        # The LOD nodes sit at their stack offsets in the source and each tree
        # sits at its own place in the pack's display row; both are dropped, so
        # the written tree stands with its trunk foot on the origin. The world
        # matrix's rotation is kept -- it is the pack's own 180 degree turn
        # about X that stands the geometry the right way up.
        transform = world_matrix(gltf, lod_node, parents).copy()
        transform[:3, 3] = 0.0
        rotation = transform[:3, :3]

        primitives = []
        totals = {'vertices': 0, 'triangles': 0}
        for child in gltf.json['nodes'][lod_node].get('children', []):
            node_child = gltf.json['nodes'][child]
            mesh = gltf.json['meshes'][node_child['mesh']]
            # A mesh node's own transform must be folded in before the bake.
            mesh_matrix = rotation @ local_matrix(node_child)[:3, :3]
            for primitive in mesh['primitives']:
                attributes = {}
                position = gltf.accessor(primitive['attributes']['POSITION']).astype(np.float64)
                position = position @ mesh_matrix.T + transform[:3, 3]
                normal = gltf.accessor(primitive['attributes']['NORMAL']).astype(np.float64)
                normal = normal @ mesh_matrix.T
                # Re-normalise: the bake is a rigid rotation, so only the pack's
                # own node scales could have stretched these.
                normal /= np.linalg.norm(normal, axis=1, keepdims=True)
                uv = gltf.accessor(primitive['attributes']['TEXCOORD_0']).astype(np.float64)
                tangent = gltf.accessor(primitive['attributes']['TANGENT']).astype(np.float64)
                xyz = tangent[:, :3] @ mesh_matrix.T
                xyz /= np.linalg.norm(xyz, axis=1, keepdims=True)
                tangent = np.concatenate([xyz, tangent[:, 3:4]], axis=1)
                index = gltf.accessor(primitive['indices']).astype(np.uint32).ravel()

                attributes['POSITION'] = builder.attribute(
                    position.astype(np.float32), 5126, 'VEC3',
                    position.min(axis=0), position.max(axis=0))
                attributes['NORMAL'] = builder.attribute(normal.astype(np.float32), 5126, 'VEC3')
                attributes['TEXCOORD_0'] = builder.attribute(uv.astype(np.float32), 5126, 'VEC2')
                attributes['TANGENT'] = builder.attribute(tangent.astype(np.float32), 5126, 'VEC4')

                material = copy_material(
                    gltf, primitive['material'], builder, shared_textures,
                    keep_extensions=True)
                primitives.append({
                    'attributes': attributes,
                    'indices': builder.indices(index),
                    'material': material,
                })
                totals['vertices'] += int(position.shape[0])
                totals['triangles'] += int(index.shape[0] // 3)

        name = f'fir-{size}-{number}-lod-{lod}'
        builder.json['nodes'].append({'name': name, 'mesh': 0})
        builder.json['meshes'].append({'name': name, 'primitives': primitives})
        if 'extensionsUsed' in gltf.json:
            used = sorted({extension
                           for material in builder.json['materials']
                           for extension in material.get('extensions', {})})
            if used:
                builder.json['extensionsUsed'] = used

        path = os.path.join(out_dir, f'{name}.glb')
        data = builder.finish()
        with open(path, 'wb') as handle:
            handle.write(data)

        box_min = np.array([a['min'] for a in builder.json['accessors'] if 'min' in a]).min(axis=0)
        box_max = np.array([a['max'] for a in builder.json['accessors'] if 'max' in a]).max(axis=0)
        summaries.append({
            'name': name, 'lod': lod, 'bytes': len(data),
            'vertices': totals['vertices'], 'triangles': totals['triangles'],
            'min': box_min, 'max': box_max,
            'primitives': len(primitives),
        })
        log(f'  {name:26s} {totals["vertices"]:6d} verts {totals["triangles"]:6d} tris '
            f'{len(data) / 1024:7.0f} KB  '
            f'foot y={box_min[1]:+.3f} top y={box_max[1]:+.3f} '
            f'span x=[{box_min[0]:+.2f},{box_max[0]:+.2f}] z=[{box_min[2]:+.2f},{box_max[2]:+.2f}]')
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', default='tmp/fir.glb')
    parser.add_argument('--out', default='assets/models/fir')
    arguments = parser.parse_args()

    def log(message):
        print(message, flush=True)

    gltf = Glb(arguments.source)
    parents = build_parents(gltf)
    os.makedirs(arguments.out, exist_ok=True)

    log(f'source {arguments.source}: {len(gltf.json["nodes"])} nodes, '
        f'{len(gltf.json["meshes"])} meshes, {len(gltf.json["images"])} images')

    global shared_textures
    log('shared textures:')
    shared_textures = write_shared_textures(gltf, arguments.out, log)

    log('trees:')
    manifest = {}
    for size, number, root in TREES:
        manifest[f'{size}-{number}'] = extract_tree(
            gltf, parents, size, number, root, arguments.out, log)

    total = sum(os.path.getsize(os.path.join(arguments.out, name))
                for name in os.listdir(arguments.out))
    log(f'{len(TREES) * 4} GLBs written to {arguments.out}, '
        f'{total / 1024 / 1024:.1f} MB total')

    with open(os.path.join(arguments.out, 'fir-manifest.json'), 'w') as handle:
        json.dump(manifest, handle, indent=2, default=lambda v: v.tolist())
    log('wrote fir-manifest.json')
    return 0


if __name__ == '__main__':
    sys.exit(main())
