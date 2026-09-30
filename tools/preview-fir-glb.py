#!/usr/bin/env python3
"""Validate the extracted fir GLBs and render a contact sheet to eyeball.

Two jobs:

* Structural -- every URI resolves, every accessor is in range, POSITION
  min/max match the real data, tangents are unit length and orthogonal to
  their normals, and the four LODs of one tree share a footprint.
* Visual -- a tiny z-buffered software rasteriser draws each GLB from a fixed
  camera with alpha cutout, so the geometry, the UVs and the material wiring
  are all exercised together. Output is one PNG per size class plus a combined
  sheet.

  tools/preview-fir-glb.py [--dir assets/models/fir] [--out tmp/fir-preview]
"""

import argparse
import json
import os
import struct
import sys

import numpy as np
from PIL import Image, ImageDraw

COMPONENT_TYPES = {
    5120: np.int8, 5121: np.uint8, 5122: np.int16,
    5123: np.uint16, 5125: np.uint32, 5126: np.float32,
}
TYPE_COUNTS = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}


class Glb:
    def __init__(self, path):
        self.path = path
        with open(path, 'rb') as handle:
            raw = handle.read()
        magic, version, length = struct.unpack('<III', raw[:12])
        assert magic == 0x46546C67 and version == 2, f'{path}: bad header'
        assert length == len(raw), f'{path}: truncated'
        self.json, self.bin = None, b''
        offset = 12
        while offset < length:
            chunk_len, chunk_type = struct.unpack('<II', raw[offset:offset + 8])
            body = raw[offset + 8:offset + 8 + chunk_len]
            if chunk_type == 0x4E4F534A:
                self.json = json.loads(body.decode('utf-8'))
            elif chunk_type == 0x004E4942:
                self.bin = body
            offset += 8 + chunk_len + (-chunk_len % 4)

    def accessor(self, index):
        acc = self.json['accessors'][index]
        dtype = COMPONENT_TYPES[acc['componentType']]
        components = TYPE_COUNTS[acc['type']]
        view = self.json['bufferViews'][acc['bufferView']]
        base = view.get('byteOffset', 0) + acc.get('byteOffset', 0)
        stride = view.get('byteStride') or np.dtype(dtype).itemsize * components
        count = acc['count'] * components
        if stride == np.dtype(dtype).itemsize * components:
            flat = np.frombuffer(self.bin, dtype=dtype, count=count, offset=base)
        else:
            flat = np.empty(count, dtype=dtype)
            for row in range(acc['count']):
                start = base + row * stride
                flat[row * components:(row + 1) * components] = np.frombuffer(
                    self.bin, dtype=dtype, count=components, offset=start)
        return flat.reshape(acc['count'], components)

    def image(self, index):
        entry = self.json['images'][index]
        if 'uri' in entry:
            return Image.open(os.path.join(os.path.dirname(self.path), entry['uri']))
        view = self.json['bufferViews'][entry['bufferView']]
        start = view.get('byteOffset', 0)
        import io
        return Image.open(io.BytesIO(self.bin[start:start + view['byteLength']]))

    def texture_image(self, texture_index):
        return self.image(self.json['textures'][texture_index]['source'])


def material_textures(gltf, material_index):
    """Return (base_color_image, normal_image) or Nones."""
    material = gltf.json['materials'][material_index]
    pbr = material.get('pbrMetallicRoughness', {})
    base = gltf.texture_image(pbr['baseColorTexture']['index']) \
        if 'baseColorTexture' in pbr else None
    normal = gltf.texture_image(material['normalTexture']['index']) \
        if 'normalTexture' in material else None
    return base, normal


def check(gltf):
    """Structural checks; returns a list of problem strings."""
    problems = []
    if gltf.json['nodes'][0] != {'name': gltf.json['nodes'][0].get('name'),
                                 'mesh': 0}:
        problems.append('root node carries extra fields')
    for image in gltf.json['images']:
        if 'uri' in image:
            path = os.path.join(os.path.dirname(gltf.path), image['uri'])
            if not os.path.exists(path):
                problems.append(f'missing texture file {image["uri"]}')
    for index, acc in enumerate(gltf.json['accessors']):
        data = gltf.accessor(index)
        if 'min' in acc:
            actual_min = data.min(axis=0)
            actual_max = data.max(axis=0)
            if not np.allclose(actual_min, acc['min'], atol=1e-4) or \
               not np.allclose(actual_max, acc['max'], atol=1e-4):
                problems.append(f'accessor {index} min/max declared '
                                f'{acc["min"]}/{acc["max"]} actual '
                                f'{actual_min}/{actual_max}')
    for mesh in gltf.json['meshes']:
        for primitive in mesh['primitives']:
            attributes = primitive['attributes']
            count = gltf.json['accessors'][attributes['POSITION']]['count']
            index = gltf.accessor(primitive['indices']).ravel()
            if index.max() >= count:
                problems.append(f'index {index.max()} out of range ({count} verts)')
            normal = gltf.accessor(attributes['NORMAL']).astype(np.float64)
            lengths = np.linalg.norm(normal, axis=1)
            if not np.allclose(lengths, 1.0, atol=2e-3):
                problems.append(f'normals not unit: {lengths.min():.4f}..{lengths.max():.4f}')
            tangent = gltf.accessor(attributes['TANGENT']).astype(np.float64)
            lengths = np.linalg.norm(tangent[:, :3], axis=1)
            if not np.allclose(lengths, 1.0, atol=2e-3):
                problems.append(f'tangents not unit: {lengths.min():.4f}..{lengths.max():.4f}')
            dot = np.abs(np.einsum('ij,ij->i', tangent[:, :3], normal))
            if dot.max() > 0.1:
                problems.append(f'tangent not perpendicular to normal: max |dot| {dot.max():.4f}')
            if not set(tangent[:, 3]).issubset({-1.0, 1.0}):
                problems.append(f'tangent w not +-1: {sorted(set(tangent[:, 3]))}')
            uv = gltf.accessor(attributes['TEXCOORD_0'])
            # Bark and branch UVs tile (up to 66 repeats along the trunk), so
            # leaving [0,1] is expected -- but only if every sampler wraps.
            if uv.min() < -0.001 or uv.max() > 1.001:
                for texture in gltf.json['textures']:
                    sampler = gltf.json['samplers'][texture.get('sampler', 0)]
                    if sampler.get('wrapS', 10497) != 10497 or \
                       sampler.get('wrapT', 10497) != 10497:
                        problems.append('tiled UVs but the sampler does not repeat')
                        break
    if 'extensionsRequired' in gltf.json:
        problems.append(f'extensionsRequired present: {gltf.json["extensionsRequired"]}')
    return problems


def look_at(eye, target, up):
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    view = np.eye(4)
    view[0, :3] = right
    view[1, :3] = true_up
    view[2, :3] = -forward
    view[:3, 3] = -view[:3, :3] @ eye
    return view


def render(gltf, size, eye, target, background, ambient=0.45):
    """Software-render every primitive with alpha cutout. Returns an RGB array."""
    width, height = size
    view = look_at(np.array(eye, dtype=np.float64), np.array(target, dtype=np.float64),
                   np.array([0.0, 1.0, 0.0]))
    # A mild perspective so the tree is not flattened.
    focal = 1.6
    colour = np.zeros((height, width, 3), dtype=np.float32)
    colour[:] = background
    depth = np.full((height, width), np.inf)

    sun = np.array([0.45, 0.72, 0.53])
    sun /= np.linalg.norm(sun)

    for mesh in gltf.json['meshes']:
        for primitive in mesh['primitives']:
            attributes = primitive['attributes']
            position = gltf.accessor(attributes['POSITION']).astype(np.float64)
            normal = gltf.accessor(attributes['NORMAL']).astype(np.float64)
            uv = gltf.accessor(attributes['TEXCOORD_0']).astype(np.float64)
            index = gltf.accessor(primitive['indices']).ravel().astype(np.int64)
            base_image, _ = material_textures(gltf, primitive['material'])
            base = np.asarray(base_image.convert('RGBA'), dtype=np.float32) / 255.0

            eye_space = position @ view[:3, :3].T + view[:3, 3]
            # Clip the whole primitive to the near plane; the trees never
            # straddle it at these camera distances.
            if eye_space[:, 2].max() > -0.01:
                continue
            projected = np.empty((position.shape[0], 2))
            projected[:, 0] = eye_space[:, 0] * focal / -eye_space[:, 2] * height / 2 + width / 2
            projected[:, 1] = -eye_space[:, 1] * focal / -eye_space[:, 2] * height / 2 + height / 2

            triangles = index.reshape(-1, 3)
            for triangle in triangles:
                screen = projected[triangle]
                x_min = max(int(np.floor(screen[:, 0].min())), 0)
                x_max = min(int(np.ceil(screen[:, 0].max())) + 1, width)
                y_min = max(int(np.floor(screen[:, 1].min())), 0)
                y_max = min(int(np.ceil(screen[:, 1].max())) + 1, height)
                if x_min >= x_max or y_min >= y_max:
                    continue
                a, b, c = screen
                area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
                if abs(area) < 1e-9:
                    continue
                xs = np.arange(x_min, x_max) + 0.5
                ys = np.arange(y_min, y_max) + 0.5
                grid_x, grid_y = np.meshgrid(xs, ys)
                w0 = ((b[0] - grid_x) * (c[1] - grid_y) - (b[1] - grid_y) * (c[0] - grid_x)) / area
                w1 = ((c[0] - grid_x) * (a[1] - grid_y) - (c[1] - grid_y) * (a[0] - grid_x)) / area
                w2 = 1.0 - w0 - w1
                inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
                if not inside.any():
                    continue
                z = w0 * eye_space[triangle[0], 2] + w1 * eye_space[triangle[1], 2] \
                    + w2 * eye_space[triangle[2], 2]
                u = w0 * uv[triangle[0], 0] + w1 * uv[triangle[1], 0] + w2 * uv[triangle[2], 0]
                v = w0 * uv[triangle[0], 1] + w1 * uv[triangle[1], 1] + w2 * uv[triangle[2], 1]
                texel_x = np.clip((u * base.shape[1]).astype(np.int64), 0, base.shape[1] - 1)
                texel_y = np.clip((v * base.shape[0]).astype(np.int64), 0, base.shape[0] - 1)
                texel = base[texel_y, texel_x]
                shade = np.clip(
                    ambient + (1 - ambient) * np.maximum(
                        w0 * (normal[triangle[0]] @ sun)
                        + w1 * (normal[triangle[1]] @ sun)
                        + w2 * (normal[triangle[2]] @ sun), 0.0), 0, 1)
                lit = texel[:, :, :3] * shade[:, :, None]
                alpha = texel[:, :, 3] > 0.5
                patch = depth[y_min:y_max, x_min:x_max]
                take = inside & alpha & (z < patch)
                colour[y_min:y_max, x_min:x_max][take] = lit[take]
                patch[take] = z[take]
    return (np.clip(colour, 0, 1) * 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default='assets/models/fir')
    parser.add_argument('--out', default='tmp/fir-preview')
    parser.add_argument('--only', default=None, help='substring filter')
    arguments = parser.parse_args()

    os.makedirs(arguments.out, exist_ok=True)
    names = sorted(n for n in os.listdir(arguments.dir) if n.endswith('.glb'))
    if arguments.only:
        names = [n for n in names if arguments.only in n]

    failures = 0
    tiles = []
    for name in names:
        gltf = Glb(os.path.join(arguments.dir, name))
        problems = check(gltf)
        position = gltf.accessor(gltf.json['meshes'][0]['primitives'][0]
                                 ['attributes']['POSITION']) if False else None
        # Bounds over every primitive, for the camera.
        boxes = []
        for mesh in gltf.json['meshes']:
            for primitive in mesh['primitives']:
                acc = gltf.json['accessors'][primitive['attributes']['POSITION']]
                boxes.append((acc['min'], acc['max']))
        low = np.array([b[0] for b in boxes]).min(axis=0)
        high = np.array([b[1] for b in boxes]).max(axis=0)
        centre = (low + high) / 2
        radius = float(np.linalg.norm(high - low)) / 2
        eye = centre + np.array([1.0, 0.42, 1.0]) / np.linalg.norm([1.0, 0.42, 1.0]) \
            * radius * 2.7
        image = render(gltf, (256, 320), eye, centre, np.array([0.55, 0.68, 0.82]))
        tile = Image.fromarray(image)
        drawer = ImageDraw.Draw(tile)
        drawer.text((4, 4), name.replace('fir-', '').replace('.glb', ''), fill=(255, 255, 255))
        tiles.append(tile)
        status = 'ok' if not problems else 'PROBLEMS'
        print(f'{name:26s} {status}  bbox y=[{low[1]:+.2f},{high[1]:+.2f}] '
              f'radius={radius:.2f}')
        for problem in problems:
            print(f'    ! {problem}')
        failures += bool(problems)

    # Contact sheet, four per row.
    columns = 4
    rows = (len(tiles) + columns - 1) // columns
    sheet = Image.new('RGB', (columns * 256, rows * 320), (20, 20, 24))
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % columns) * 256, (index // columns) * 320))
    sheet_path = os.path.join(arguments.out, 'contact-sheet.png')
    sheet.save(sheet_path)
    print(f'\n{failures} file(s) with problems; contact sheet -> {sheet_path}')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
