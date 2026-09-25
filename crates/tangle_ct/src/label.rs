//! Connected components and the enclosed holes of a mask.

use crate::{strides, voxel_count, Shape};

/// Which of a voxel's 26 neighbors (and itself, at the center) connect:
/// `structure[dz + 1][dy + 1][dx + 1]`, as SciPy's `label(structure=...)`.
pub type Structure = [[[bool; 3]; 3]; 3];

/// Face connectivity (SciPy's default structure).
pub const FACES: Structure = {
    let mut s = [[[false; 3]; 3]; 3];
    s[1][1][1] = true;
    s[0][1][1] = true;
    s[2][1][1] = true;
    s[1][0][1] = true;
    s[1][2][1] = true;
    s[1][1][0] = true;
    s[1][1][2] = true;
    s
};

fn find(parent: &mut [u32], mut x: u32) -> u32 {
    while parent[x as usize] != x {
        let up = parent[parent[x as usize] as usize];
        parent[x as usize] = up;
        x = up;
    }
    x
}

/// Components of the `true` voxels under `structure`, numbered 1, 2, … in
/// the order a C-order scan first meets them (as SciPy numbers them); 0 is
/// background. Returns the labels and the component count.
pub fn label(mask: &[bool], shape: Shape, structure: &Structure) -> (Vec<i32>, usize) {
    assert_eq!(mask.len(), voxel_count(shape));
    let s = strides(shape);
    // Neighbors met earlier in a C-order scan.
    let mut earlier = Vec::new();
    for dz in -1i64..=1 {
        for dy in -1i64..=1 {
            for dx in -1i64..=1 {
                let before = (dz, dy, dx) < (0, 0, 0);
                if before && structure[(dz + 1) as usize][(dy + 1) as usize][(dx + 1) as usize] {
                    earlier.push((dz, dy, dx));
                }
            }
        }
    }
    let n = mask.len();
    let mut parent: Vec<u32> = (0..n as u32).collect();
    for k in 0..shape[0] {
        for j in 0..shape[1] {
            for i in 0..shape[2] {
                let index = k * s[0] + j * s[1] + i;
                if !mask[index] {
                    continue;
                }
                for &(dz, dy, dx) in &earlier {
                    let (kk, jj, ii) = (k as i64 + dz, j as i64 + dy, i as i64 + dx);
                    if kk < 0 || jj < 0 || ii < 0 || jj >= shape[1] as i64 || ii >= shape[2] as i64
                    {
                        continue;
                    }
                    let other = kk as usize * s[0] + jj as usize * s[1] + ii as usize;
                    if mask[other] {
                        let (a, b) = (
                            find(&mut parent, index as u32),
                            find(&mut parent, other as u32),
                        );
                        if a != b {
                            let (low, high) = if a < b { (a, b) } else { (b, a) };
                            parent[high as usize] = low;
                        }
                    }
                }
            }
        }
    }
    let mut labels = vec![0i32; n];
    let mut number = vec![0i32; n];
    let mut count = 0usize;
    for index in 0..n {
        if !mask[index] {
            continue;
        }
        let root = find(&mut parent, index as u32) as usize;
        if number[root] == 0 {
            count += 1;
            number[root] = count as i32;
        }
        labels[index] = number[root];
    }
    (labels, count)
}

/// Enclosed holes no larger than `max_area` voxels, slice by slice along
/// each axis: void components (face-connected within the slice) that do not
/// reach the slice's edge. A hollow fiber is a closed ring in the slices
/// across it, which a 3D fill misses. Slices are filled independently, spread
/// over the available cores.
pub fn core_holes(mask: &[bool], shape: Shape, max_area: f64) -> Vec<bool> {
    assert_eq!(mask.len(), voxel_count(shape));
    let mut holes = vec![false; mask.len()];
    let s = strides(shape);
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    for axis in 0..3 {
        let (p, q) = match axis {
            0 => (1, 2),
            1 => (0, 2),
            _ => (0, 1),
        };
        let slices = shape[axis];
        let per_thread = slices.div_ceil(threads.min(slices).max(1)).max(1);
        let found: Vec<Vec<usize>> = std::thread::scope(|scope| {
            let workers: Vec<_> = (0..slices)
                .step_by(per_thread)
                .map(|first| {
                    scope.spawn(move || {
                        let mut out = Vec::new();
                        let (rows, cols) = (shape[p], shape[q]);
                        let mut void = vec![false; rows * cols];
                        for t in first..(first + per_thread).min(slices) {
                            let base = t * s[axis];
                            for r in 0..rows {
                                for c in 0..cols {
                                    void[r * cols + c] = !mask[base + r * s[p] + c * s[q]];
                                }
                            }
                            for at in slice_holes(&void, rows, cols, max_area) {
                                out.push(base + (at / cols) * s[p] + (at % cols) * s[q]);
                            }
                        }
                        out
                    })
                })
                .collect();
            workers.into_iter().map(|w| w.join().expect("hole worker")).collect()
        });
        for index in found.into_iter().flatten() {
            holes[index] = true;
        }
    }
    holes
}

/// The positions of the `void` components (4-connected) of a `rows × cols`
/// slice that do not touch its edge and have at most `max_area` pixels.
fn slice_holes(void: &[bool], rows: usize, cols: usize, max_area: f64) -> Vec<usize> {
    let mut seen = vec![false; void.len()];
    let mut holes = Vec::new();
    let mut stack = Vec::new();
    let mut component = Vec::new();
    for start in 0..void.len() {
        if !void[start] || seen[start] {
            continue;
        }
        seen[start] = true;
        stack.push(start);
        component.clear();
        let mut edge = false;
        while let Some(at) = stack.pop() {
            component.push(at);
            let (r, c) = (at / cols, at % cols);
            edge |= r == 0 || c == 0 || r + 1 == rows || c + 1 == cols;
            let mut visit = |next: usize| {
                if void[next] && !seen[next] {
                    seen[next] = true;
                    stack.push(next);
                }
            };
            if r > 0 {
                visit(at - cols);
            }
            if r + 1 < rows {
                visit(at + cols);
            }
            if c > 0 {
                visit(at - 1);
            }
            if c + 1 < cols {
                visit(at + 1);
            }
        }
        if !edge && component.len() as f64 <= max_area {
            holes.extend_from_slice(&component);
        }
    }
    holes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_blobs_numbered_in_scan_order() {
        let shape = [1, 3, 5];
        let mask = [
            true, true, false, false, true, //
            false, false, false, false, true, //
            true, false, false, false, false,
        ];
        let (labels, count) = label(&mask, shape, &FACES);
        assert_eq!(count, 3);
        assert_eq!(labels, vec![1, 1, 0, 0, 2, 0, 0, 0, 0, 2, 3, 0, 0, 0, 0]);
    }

    #[test]
    fn a_ring_encloses_one_hole_in_its_slices() {
        let shape = [3, 5, 5];
        let mut mask = vec![false; 75];
        for k in 0..3 {
            for j in 1..4 {
                for i in 1..4 {
                    if (j, i) != (2, 2) {
                        mask[(k * 5 + j) * 5 + i] = true;
                    }
                }
            }
        }
        let holes = core_holes(&mask, shape, 4.0);
        let found: Vec<usize> = (0..75).filter(|&x| holes[x]).collect();
        assert_eq!(found, vec![12, 37, 62]);
    }

    /// The labeling route the slice fill replaced.
    fn holes_by_labeling(mask: &[bool], shape: Shape, max_area: f64) -> Vec<bool> {
        let void: Vec<bool> = mask.iter().map(|&m| !m).collect();
        let mut holes = vec![false; mask.len()];
        let s = strides(shape);
        for axis in 0..3 {
            let others: Vec<usize> = (0..3).filter(|&x| x != axis).collect();
            let mut structure = [[[false; 3]; 3]; 3];
            for (a, b) in [(1, 1), (0, 1), (2, 1), (1, 0), (1, 2)] {
                let mut offset = [1usize; 3];
                offset[others[0]] = a;
                offset[others[1]] = b;
                structure[offset[0]][offset[1]][offset[2]] = true;
            }
            let (ids, count) = label(&void, shape, &structure);
            let mut sizes = vec![0usize; count + 1];
            for &id in &ids {
                sizes[id as usize] += 1;
            }
            let mut small: Vec<bool> = sizes.iter().map(|&n| n as f64 <= max_area).collect();
            small[0] = false;
            for &other in &others {
                for edge in [0, shape[other] - 1] {
                    for (index, &id) in ids.iter().enumerate() {
                        if (index / s[other]) % shape[other] == edge {
                            small[id as usize] = false;
                        }
                    }
                }
            }
            for (hole, &id) in holes.iter_mut().zip(&ids) {
                *hole |= small[id as usize];
            }
        }
        holes
    }

    #[test]
    fn slice_fill_matches_labeling() {
        let shape = [23, 31, 29];
        // A speckled mask: about 60% foreground, with small enclosed voids.
        let mask: Vec<bool> = (0..voxel_count(shape))
            .map(|i| (i.wrapping_mul(2654435761) >> 7) % 10 < 6)
            .collect();
        for max_area in [1.0, 4.0, 40.0] {
            assert_eq!(core_holes(&mask, shape, max_area), holes_by_labeling(&mask, shape, max_area));
        }
    }
}
