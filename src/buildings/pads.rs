//! Building pads: the ground levelled under and around each building.
//!
//! A pad is a rectangle turned with its building, flat at one height, and a
//! skirt around it over which the ground eases back into the land: the pad's
//! weight falls from one to nothing as a smootherstep of the distance from
//! the rectangle, so the slope is continuous at both edges of the skirt and
//! its corners are round. The skirt is as wide as the cut or fill at the
//! pad's edge needs to stay gentle.
//!
//! It is the last thing applied to the ground everywhere the ground is
//! measured: the clipmap, the lighting and shore heightfields, the grass
//! capture and the plants' rooting on the GPU (`building-pads.wgslinc`), the
//! player's footing and the snow on the CPU ([`PadIndex::level`]). Erosion
//! simulates the land as it was; the pad overrides whatever it made of the
//! ground under the building, and the skirt blends the eroded ground into it.

use bevy::math::Vec2;
use std::collections::HashMap;

/// Side of a lookup cell, metres, on the CPU and in the uploaded grid alike.
pub const CELL_SIZE: f32 = 64.0;
/// Cells across the uploaded grid: it covers 4 km about the player.
pub const GPU_GRID_CELLS: usize = 64;
/// The uploaded grid is recentred in steps of this.
pub const GPU_GRID_SNAP: f32 = 512.0;
/// Pads one cell lists at most, on both sides.
pub const MAX_CANDIDATES: usize = 12;
/// Words of the header, and of one pad.
pub const HEADER_WORDS: usize = 8;
pub const PAD_WORDS: usize = 12;
/// Words the uploaded buffer holds: header, grid, index lists and pads.
pub const GPU_WORDS: usize = HEADER_WORDS + GPU_GRID_CELLS * GPU_GRID_CELLS * (2 + MAX_CANDIDATES) + MAX_GPU_PADS * PAD_WORDS;
/// Pads the GPU holds; the nearest are kept.
pub const MAX_GPU_PADS: usize = 2048;

/// What a pad does to the ground.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PadKind {
    /// A building's: levelled, its footprint bare, its yard trodden.
    #[default]
    Building,
    /// A town's main street: trodden into a dirt road, its ground left as
    /// it lies.
    Street,
}

/// One levelled rectangle and its skirt, world space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pad {
    pub centre: Vec2,
    /// Cosine and sine of the building's turn about +Y.
    pub axis: Vec2,
    /// Half the flat rectangle's extent along the building's X and Z.
    pub half: Vec2,
    pub height: f32,
    pub skirt: f32,
    /// How far inside the flat rectangle the building's footprint ends.
    pub inset: f32,
    pub kind: PadKind,
}

fn smootherstep(t: f32) -> f32 {
    let x = t.clamp(0.0, 1.0);
    x * x * x * (x * (x * 6.0 - 15.0) + 10.0)
}

impl Pad {
    /// A world point in the pad's own axes, from its centre.
    pub fn local(&self, point: Vec2) -> Vec2 {
        let offset = point - self.centre;
        Vec2::new(offset.x * self.axis.x - offset.y * self.axis.y, offset.x * self.axis.y + offset.y * self.axis.x)
    }

    /// Distance from the flat rectangle, zero inside it.
    pub fn outside(&self, point: Vec2) -> f32 {
        let local = self.local(point);
        let beyond = (local.abs() - self.half).max(Vec2::ZERO);
        beyond.length()
    }

    /// How much of the pad's level holds at a point: none of a street's.
    pub fn weight(&self, point: Vec2) -> f32 {
        if self.kind == PadKind::Street {
            return 0.0;
        }
        1.0 - smootherstep(self.outside(point) / self.skirt)
    }

    /// World XZ bounds of everything the pad changes.
    pub fn reach(&self) -> (Vec2, Vec2) {
        let extent = self.half + Vec2::splat(self.skirt);
        let across = Vec2::new(
            extent.x * self.axis.x.abs() + extent.y * self.axis.y.abs(),
            extent.x * self.axis.y.abs() + extent.y * self.axis.x.abs(),
        );
        (self.centre - across, self.centre + across)
    }
}

fn cell_of(x: f32, z: f32) -> [i32; 2] {
    [(x / CELL_SIZE).floor() as i32, (z / CELL_SIZE).floor() as i32]
}

/// Every pad in force, looked up by 64 m cell.
#[derive(Clone, Debug, Default)]
pub struct PadIndex {
    pads: Vec<Pad>,
    cells: HashMap<[i32; 2], Vec<u32>>,
}

impl PadIndex {
    /// The pads, in an order both sides apply them in where they overlap.
    pub fn new(pads: Vec<Pad>) -> Self {
        let mut cells: HashMap<[i32; 2], Vec<u32>> = HashMap::new();
        for (index, pad) in pads.iter().enumerate() {
            let (minimum, maximum) = pad.reach();
            let first = cell_of(minimum.x, minimum.y);
            let last = cell_of(maximum.x, maximum.y);
            for cz in first[1]..=last[1] {
                for cx in first[0]..=last[0] {
                    let list = cells.entry([cx, cz]).or_default();
                    if list.len() < MAX_CANDIDATES {
                        list.push(index as u32);
                    }
                }
            }
        }
        Self { pads, cells }
    }

    #[cfg(test)]
    pub fn pads(&self) -> &[Pad] {
        &self.pads
    }

    fn candidates(&self, x: f32, z: f32) -> &[u32] {
        self.cells.get(&cell_of(x, z)).map_or(&[], Vec::as_slice)
    }

    /// The ground at `(x, z)` once every pad reaching it has levelled it.
    pub fn level(&self, x: f32, z: f32, height: f32) -> f32 {
        let point = Vec2::new(x, z);
        let mut levelled = height;
        for &index in self.candidates(x, z) {
            let pad = &self.pads[index as usize];
            let weight = pad.weight(point);
            levelled += (pad.height - levelled) * weight;
        }
        levelled
    }

    /// The buffer `building-pads.wgslinc` reads, for a grid of
    /// [`GPU_GRID_CELLS`] cells about `centre` snapped to [`GPU_GRID_SNAP`]:
    ///
    /// - header: grid origin X and Z and cell size (f32 bits), cells across,
    ///   word offset of the pads, pad count, two spare words;
    /// - per cell, row by row: the word offset of its pad list and its count;
    /// - the lists of pad numbers;
    /// - the pads, [`PAD_WORDS`] each: centre XZ, axis, half extents, level,
    ///   skirt, inset (f32 bits), the kind (0 a building, 1 a street, as a
    ///   u32), two spare words.
    pub fn gpu_words(&self, centre: [f32; 2]) -> Vec<u32> {
        let span = GPU_GRID_CELLS as f32 * CELL_SIZE;
        let snapped = [
            (centre[0] / GPU_GRID_SNAP).round() * GPU_GRID_SNAP,
            (centre[1] / GPU_GRID_SNAP).round() * GPU_GRID_SNAP,
        ];
        let origin = [snapped[0] - span * 0.5, snapped[1] - span * 0.5];
        let first_cell = cell_of(origin[0], origin[1]);
        // The pads the grid sees, renumbered in their own order, nearest the
        // centre kept if there are too many.
        let mut visible: Vec<u32> = (0..self.pads.len() as u32)
            .filter(|&index| {
                let (minimum, maximum) = self.pads[index as usize].reach();
                maximum.x >= origin[0] && maximum.y >= origin[1] && minimum.x <= origin[0] + span && minimum.y <= origin[1] + span
            })
            .collect();
        if visible.len() > MAX_GPU_PADS {
            let distance = |index: &u32| self.pads[*index as usize].centre.distance(Vec2::from(snapped));
            visible.sort_by(|a, b| distance(a).total_cmp(&distance(b)));
            visible.truncate(MAX_GPU_PADS);
            visible.sort();
        }
        let renumbered: HashMap<u32, u32> = visible.iter().enumerate().map(|(new, &old)| (old, new as u32)).collect();

        let cells = GPU_GRID_CELLS * GPU_GRID_CELLS;
        let mut words = vec![0u32; HEADER_WORDS + cells * 2];
        let mut lists: Vec<u32> = Vec::new();
        for row in 0..GPU_GRID_CELLS {
            for column in 0..GPU_GRID_CELLS {
                let key = [first_cell[0] + column as i32, first_cell[1] + row as i32];
                let Some(list) = self.cells.get(&key) else {
                    continue;
                };
                let entry = HEADER_WORDS + (row * GPU_GRID_CELLS + column) * 2;
                let start = HEADER_WORDS + cells * 2 + lists.len();
                let kept: Vec<u32> = list.iter().filter_map(|index| renumbered.get(index).copied()).collect();
                words[entry] = start as u32;
                words[entry + 1] = kept.len() as u32;
                lists.extend(kept);
            }
        }
        words.extend_from_slice(&lists);
        let pad_offset = words.len();
        for &index in &visible {
            let pad = &self.pads[index as usize];
            let values = [
                pad.centre.x,
                pad.centre.y,
                pad.axis.x,
                pad.axis.y,
                pad.half.x,
                pad.half.y,
                pad.height,
                pad.skirt,
                pad.inset,
                0.0,
                0.0,
                0.0,
            ];
            words.extend(values.iter().map(|value| value.to_bits()));
            // The kind replaces the first spare float: 0 a building, 1 a street.
            let kind_word = words.len() - 3;
            words[kind_word] = match pad.kind {
                PadKind::Building => 0,
                PadKind::Street => 1,
            };
        }
        words[0] = origin[0].to_bits();
        words[1] = origin[1].to_bits();
        words[2] = CELL_SIZE.to_bits();
        words[3] = if visible.is_empty() { 0 } else { GPU_GRID_CELLS as u32 };
        words[4] = pad_offset as u32;
        words[5] = visible.len() as u32;
        debug_assert!(words.len() <= GPU_WORDS);
        words
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pad(centre: Vec2, yaw: f32, height: f32) -> Pad {
        Pad {
            centre,
            axis: Vec2::new(yaw.cos(), yaw.sin()),
            half: Vec2::new(7.0, 5.0),
            height,
            skirt: 6.0,
            inset: 1.5,
            kind: PadKind::Building,
        }
    }

    /// What `buildingPadHeight` in building-pads.wgslinc computes from the
    /// uploaded words, word for word.
    fn shader_level(words: &[u32], point: Vec2, height: f32) -> f32 {
        let float = |index: usize| f32::from_bits(words[index]);
        let resolution = words[3] as usize;
        if resolution == 0 {
            return height;
        }
        let origin = Vec2::new(float(0), float(1));
        let cell = ((point - origin) / float(2)).floor();
        if cell.x < 0.0 || cell.y < 0.0 || cell.x >= resolution as f32 || cell.y >= resolution as f32 {
            return height;
        }
        let entry = HEADER_WORDS + (cell.y as usize * resolution + cell.x as usize) * 2;
        let (first, count) = (words[entry] as usize, (words[entry + 1] as usize).min(MAX_CANDIDATES));
        let mut levelled = height;
        for i in 0..count {
            let base = words[4] as usize + words[first + i] as usize * PAD_WORDS;
            let pad = Pad {
                centre: Vec2::new(float(base), float(base + 1)),
                axis: Vec2::new(float(base + 2), float(base + 3)),
                half: Vec2::new(float(base + 4), float(base + 5)),
                height: float(base + 6),
                skirt: float(base + 7),
                inset: float(base + 8),
                kind: if words[base + 9] == 1 { PadKind::Street } else { PadKind::Building },
            };
            levelled += (pad.height - levelled) * pad.weight(point);
        }
        levelled
    }

    #[test]
    fn a_pad_is_flat_over_its_rectangle_and_blends_continuously_into_the_land() {
        let index = PadIndex::new(vec![pad(Vec2::new(100.0, -40.0), 0.6, 12.0)]);
        // Gentle land a little above and below the pad's level, as a site is.
        let land = |x: f32, z: f32| 12.0 + 0.03 * (x - 100.0) - 0.02 * (z + 40.0);
        // Flat inside.
        for (dx, dz) in [(0.0, 0.0), (4.0, 3.0), (-5.0, -2.5)] {
            let local = Vec2::new(dx, dz);
            let axis = index.pads()[0].axis;
            let world = Vec2::new(local.x * axis.x + local.y * axis.y, -local.x * axis.y + local.y * axis.x) + Vec2::new(100.0, -40.0);
            assert!((index.level(world.x, world.y, land(world.x, world.y)) - 12.0).abs() < 1e-4);
        }
        // The land itself beyond the skirt.
        let far = Vec2::new(130.0, -40.0);
        assert_eq!(index.level(far.x, far.y, land(far.x, far.y)), land(far.x, far.y));
        // No step anywhere along a line through it: neighbouring samples a
        // centimetre apart never differ by more than the land's own slope and
        // the skirt's steepest rise.
        let mut previous = None;
        for step in 0..4000 {
            let x = 70.0 + step as f32 * 0.015;
            let z = -40.0 + (x - 100.0) * 0.3;
            let height = index.level(x, z, land(x, z));
            if let Some(last) = previous {
                let rise: f32 = height - last;
                assert!(rise.abs() < 0.015 * 0.6, "a step of {rise} at {x}");
            }
            previous = Some(height);
        }
    }

    #[test]
    fn the_uploaded_grid_levels_the_ground_as_the_cpu_does() {
        let mut pads = Vec::new();
        for i in 0..40 {
            let angle = i as f32 * 0.37;
            let centre = Vec2::new((i % 7) as f32 * 31.0 - 90.0, (i / 7) as f32 * 29.0 - 70.0);
            pads.push(pad(centre, angle, 5.0 + i as f32 * 0.1));
        }
        let index = PadIndex::new(pads);
        let words = index.gpu_words([10.0, -20.0]);
        assert!(words.len() <= GPU_WORDS);
        for i in 0..5000 {
            let point = Vec2::new(((i * 7919) % 400) as f32 * 0.61 - 120.0, ((i * 104729) % 300) as f32 * 0.57 - 100.0);
            let land = 3.0 + (point.x * 0.05).sin();
            let cpu = index.level(point.x, point.y, land);
            let gpu = shader_level(&words, point, land);
            assert!((cpu - gpu).abs() < 1e-5, "{point}: {cpu} against {gpu}");
        }
    }

    #[test]
    fn every_ground_shader_pastes_the_same_pad_block() {
        let common = include_str!("../../assets/shaders/building-pads.wgslinc").trim();
        for source in [
            include_str!("../../assets/shaders/terrain-vs.wgsl"),
            include_str!("../../assets/shaders/terrain-fs.wgsl"),
            include_str!("../../assets/shaders/vegetation-cull.wgsl"),
        ] {
            assert!(source.contains(common), "the building pads diverged");
        }
    }
}
