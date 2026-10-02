# Grass asset attribution

The active grass pack was created by [LOLIPOP](https://sketchfab.com/lolipop_1707)
and is licensed under [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
The author, license and source link below are copied from the supplied GLB's
`asset.extras` metadata and are retained in every extracted GLB.

| Source | Extracted models | Textures |
| --- | --- | --- |
| [Grass (pack of 9 vars, lowpoly, game ready)](https://sketchfab.com/3d-models/grass-pack-of-9-vars-lowpoly-game-ready-0561204a1fa14c17939300ee1108948b) | `grass-*.glb`: 9 distinct grass variants | `../textures/grass/grass-*.png` |

Modifications: split each grass mesh into a separate GLB; bake the source orientation and scale;
remove display translations; place the lowest vertex at Y=0 while retaining the
authored X/Z root pivot; change alpha blending to double-sided alpha masking
with a cutoff of 0.5. Texture image bytes and all mesh triangles are preserved.
All files use glTF's Y-up convention and metre units. These are separate grass
variants, not levels of detail.

The GLBs reference their shared PNG textures using relative paths. Keep the
`models` and `textures` directories together when distributing these assets.
`grass-manifest.json` records every source mesh, output file, bounds, texture
family and source-file checksum.

The files were extracted from the source library `grass-pack-2.glb` (its
checksum is in the manifest) and later renamed from `grass-pack-2-*` to
`grass-*`; the extraction script is no longer part of the repository.
