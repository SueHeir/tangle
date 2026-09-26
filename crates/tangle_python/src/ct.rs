//! `tangle.ct` volume operations (the `tangle_ct` crate) on NumPy arrays.
//!
//! Arrays come in through the buffer protocol without copies, and results
//! are written into arrays the caller allocates, so the Python side stays a
//! thin wrapper (`tangle/ct/_native.py`). Every array must be C-contiguous
//! with the element type named in the signature.

use pyo3::buffer::{Element, PyBuffer};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use tangle_ct::confidence::{node_confidence, ConfidenceSettings};
use tangle_ct::grey::{overlap, render_grey};
use tangle_ct::hessian::HessianField;
use tangle_ct::line::resample;
use tangle_ct::moves::{
    merge_fragments, remove_unsupported, resolve_side_by_side, split_kinks, trim_duplicates,
    MergeSettings, SideBySide, SplitSettings,
};
use tangle_ct::refine::{curvature_ratio, cut_void, end_step, support, OwnerLookup, VoidRules};
use tangle_ct::render::{box_size, local_residual, render_occupancy, Corner};
use tangle_ct::trace::{trace_fibers, FiberSearch, TraceSettings, Tracer};
use tangle_ct::Shape;

fn read<'a, T: Element>(buffer: &'a PyBuffer<T>, name: &str) -> PyResult<&'a [T]> {
    if !buffer.is_c_contiguous() {
        return Err(PyValueError::new_err(format!(
            "{name} must be C-contiguous"
        )));
    }
    // SAFETY: the buffer is C-contiguous with item_count elements of T, and
    // the exporting array outlives this call.
    Ok(unsafe { std::slice::from_raw_parts(buffer.buf_ptr() as *const T, buffer.item_count()) })
}

#[allow(clippy::mut_from_ref)]
fn write<'a, T: Element>(buffer: &'a PyBuffer<T>, name: &str) -> PyResult<&'a mut [T]> {
    if buffer.readonly() {
        return Err(PyValueError::new_err(format!("{name} must be writable")));
    }
    if !buffer.is_c_contiguous() {
        return Err(PyValueError::new_err(format!(
            "{name} must be C-contiguous"
        )));
    }
    // SAFETY: as in `read`; the caller passes a distinct output array.
    Ok(unsafe { std::slice::from_raw_parts_mut(buffer.buf_ptr() as *mut T, buffer.item_count()) })
}

fn volume_shape<T: Element>(buffer: &PyBuffer<T>, name: &str) -> PyResult<Shape> {
    match buffer.shape() {
        [nz, ny, nx] => Ok([*nz, *ny, *nx]),
        _ => Err(PyValueError::new_err(format!(
            "{name} must be a 3D (z, y, x) array"
        ))),
    }
}

fn same_shape<A: Element, B: Element>(
    a: &PyBuffer<A>,
    b: &PyBuffer<B>,
    name: &str,
) -> PyResult<()> {
    if a.shape() != b.shape() {
        return Err(PyValueError::new_err(format!(
            "{name} must have the input's shape {:?}",
            a.shape()
        )));
    }
    Ok(())
}

fn points(buffer: &PyBuffer<f64>, name: &str) -> PyResult<Vec<[f64; 3]>> {
    let values = read(buffer, name)?;
    if values.len() % 3 != 0 {
        return Err(PyValueError::new_err(format!(
            "{name} must be (n, 3) points"
        )));
    }
    Ok(values.chunks_exact(3).map(|p| [p[0], p[1], p[2]]).collect())
}

/// `out[...] = gaussian_filter(image, sigma, order=orders)` (orders along z, y, x).
#[pyfunction]
pub(crate) fn ct_gaussian_filter(
    image: PyBuffer<f32>,
    out: PyBuffer<f32>,
    sigma: f64,
    orders: (usize, usize, usize),
) -> PyResult<()> {
    let shape = volume_shape(&image, "image")?;
    same_shape(&image, &out, "out")?;
    let result = tangle_ct::filter::gaussian_filter(
        read(&image, "image")?,
        shape,
        sigma,
        [orders.0, orders.1, orders.2],
    );
    write(&out, "out")?.copy_from_slice(&result);
    Ok(())
}

/// Trilinear samples of a float32 volume at (x, y, z) points into `out`.
#[pyfunction]
pub(crate) fn ct_sample(
    image: PyBuffer<f32>,
    points_xyz: PyBuffer<f64>,
    out: PyBuffer<f64>,
    fill: f64,
) -> PyResult<()> {
    let shape = volume_shape(&image, "image")?;
    let values = read(&image, "image")?;
    let pts = points(&points_xyz, "points")?;
    let out = write(&out, "out")?;
    if out.len() != pts.len() {
        return Err(PyValueError::new_err("out must hold one value per point"));
    }
    for (o, p) in out.iter_mut().zip(&pts) {
        *o = tangle_ct::sample::trilinear(values, shape, *p, fill);
    }
    Ok(())
}

/// The foreground's distance transform and its 3×3×3 maximum.
#[pyfunction]
pub(crate) fn ct_foreground_depth(
    foreground: PyBuffer<u8>,
    edt: PyBuffer<f32>,
    peak: PyBuffer<f32>,
) -> PyResult<()> {
    let shape = volume_shape(&foreground, "foreground")?;
    same_shape(&foreground, &edt, "edt")?;
    same_shape(&foreground, &peak, "peak")?;
    let mask: Vec<bool> = read(&foreground, "foreground")?
        .iter()
        .map(|&v| v != 0)
        .collect();
    let distance = tangle_ct::edt::distance_transform(&mask, shape);
    let maximum = tangle_ct::filter::maximum_filter3(&distance, shape);
    write(&edt, "edt")?.copy_from_slice(&distance);
    write(&peak, "peak")?.copy_from_slice(&maximum);
    Ok(())
}

/// Enclosed holes of `mask` up to `max_area` voxels, slice by slice along each axis.
#[pyfunction]
pub(crate) fn ct_core_holes(mask: PyBuffer<u8>, out: PyBuffer<u8>, max_area: f64) -> PyResult<()> {
    let shape = volume_shape(&mask, "mask")?;
    same_shape(&mask, &out, "out")?;
    let values: Vec<bool> = read(&mask, "mask")?.iter().map(|&v| v != 0).collect();
    let holes = tangle_ct::label::core_holes(&values, shape, max_area);
    for (o, h) in write(&out, "out")?.iter_mut().zip(holes) {
        *o = u8::from(h);
    }
    Ok(())
}

fn split_lines(nodes: &[[f64; 3]], counts: &[usize]) -> PyResult<Vec<Vec<[f64; 3]>>> {
    if counts.iter().sum::<usize>() != nodes.len() {
        return Err(PyValueError::new_err(
            "counts must add up to the number of nodes",
        ));
    }
    let mut start = 0;
    Ok(counts
        .iter()
        .map(|&n| {
            let line = nodes[start..start + n].to_vec();
            start += n;
            line
        })
        .collect())
}

/// Nearest-fiber ownership: fills `labels`, `distance` and `segment` (see `_geometry.rasterize`).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn ct_rasterize(
    nodes: PyBuffer<f64>,
    counts: Vec<usize>,
    radii: Vec<f64>,
    reach: Vec<f64>,
    signed: bool,
    labels: PyBuffer<i32>,
    distance: PyBuffer<f32>,
    segment: PyBuffer<i32>,
) -> PyResult<()> {
    let shape = volume_shape(&labels, "labels")?;
    same_shape(&labels, &distance, "distance")?;
    same_shape(&labels, &segment, "segment")?;
    if radii.len() != counts.len() || reach.len() != counts.len() {
        return Err(PyValueError::new_err(
            "give one radius and one reach per line",
        ));
    }
    let lines = split_lines(&points(&nodes, "nodes")?, &counts)?;
    let raster = tangle_ct::raster::rasterize(shape, &lines, &radii, &reach, signed);
    write(&labels, "labels")?.copy_from_slice(&raster.labels);
    write(&distance, "distance")?.copy_from_slice(&raster.distance);
    write(&segment, "segment")?.copy_from_slice(&raster.segment);
    Ok(())
}

/// Sets `target`'s voxels within `reach` of polyline `line` to `value`, in place.
#[pyfunction]
pub(crate) fn ct_paint(
    target: PyBuffer<i32>,
    line: PyBuffer<f64>,
    reach: f64,
    value: i32,
    only_empty: bool,
) -> PyResult<()> {
    let shape = volume_shape(&target, "target")?;
    let line = points(&line, "line")?;
    if line.is_empty() {
        return Ok(());
    }
    tangle_ct::raster::paint(
        write(&target, "target")?,
        shape,
        &line,
        reach,
        value,
        only_empty,
    );
    Ok(())
}

/// The Gaussian-scale Hessian of a float32 volume, read at points.
#[pyclass(name = "CtHessian", module = "tangle._tangle")]
pub(crate) struct PyCtHessian {
    field: HessianField,
}

#[pymethods]
impl PyCtHessian {
    #[new]
    fn new(image: PyBuffer<f32>, sigma: f64) -> PyResult<Self> {
        let shape = volume_shape(&image, "image")?;
        if !(sigma.is_finite() && sigma > 0.0) {
            return Err(PyValueError::new_err("sigma must be positive"));
        }
        Ok(Self {
            field: HessianField::new(read(&image, "image")?, shape, sigma),
        })
    }

    #[getter]
    fn sigma(&self) -> f64 {
        self.field.sigma
    }

    /// Fills `out` (n, 3, 3) with the Hessian at each (x, y, z) point.
    fn at(&self, points_xyz: PyBuffer<f64>, out: PyBuffer<f64>) -> PyResult<()> {
        let pts = points(&points_xyz, "points")?;
        let out = write(&out, "out")?;
        if out.len() != 9 * pts.len() {
            return Err(PyValueError::new_err("out must be (n, 3, 3)"));
        }
        for (block, p) in out.chunks_exact_mut(9).zip(&pts) {
            let h = self.field.at(*p);
            for i in 0..3 {
                block[3 * i..3 * i + 3].copy_from_slice(&h[i]);
            }
        }
        Ok(())
    }

    /// Fills `axis` (n, 3) with the tube direction and `tubularity` (n) at each point.
    fn directions(
        &self,
        points_xyz: PyBuffer<f64>,
        axis: PyBuffer<f64>,
        tubularity: PyBuffer<f64>,
    ) -> PyResult<()> {
        let pts = points(&points_xyz, "points")?;
        let axis = write(&axis, "axis")?;
        let tubularity = write(&tubularity, "tubularity")?;
        if axis.len() != 3 * pts.len() || tubularity.len() != pts.len() {
            return Err(PyValueError::new_err(
                "axis must be (n, 3) and tubularity (n,)",
            ));
        }
        for (k, p) in pts.iter().enumerate() {
            let (direction, score) = self.field.direction(*p);
            axis[3 * k..3 * k + 3].copy_from_slice(&direction);
            tubularity[k] = score;
        }
        Ok(())
    }
}

fn trace_settings(
    radius: f64,
    min_bend_radius: f64,
    step: f64,
    peak_floor: f64,
) -> PyResult<TraceSettings> {
    if !(radius > 0.0 && step > 0.0) {
        return Err(PyValueError::new_err("radius and step must be positive"));
    }
    Ok(TraceSettings {
        radius,
        min_bend_radius,
        step,
        peak_floor,
    })
}

/// One-way trace from `start` along `direction` (see `_trace.Tracer.trace_one_way`).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn ct_trace_one_way(
    image: PyBuffer<f32>,
    hessian: PyRef<'_, PyCtHessian>,
    claimed: PyBuffer<i32>,
    radius: f64,
    min_bend_radius: f64,
    step: f64,
    start: [f64; 3],
    direction: [f64; 3],
    max_steps: usize,
    own_label: i32,
) -> PyResult<Vec<[f64; 3]>> {
    let shape = volume_shape(&image, "image")?;
    same_shape(&image, &claimed, "claimed")?;
    let settings = trace_settings(radius, min_bend_radius, step, 0.0)?;
    let tracer = Tracer::new(read(&image, "image")?, shape, &hessian.field, settings);
    Ok(tracer.trace_one_way(
        start,
        direction,
        max_steps,
        read(&claimed, "claimed")?,
        own_label,
    ))
}

/// Traces fibers from ridge seeds, painting `claimed` in place (see `_trace.trace_fibers`).
#[pyfunction]
#[pyo3(signature = (image, hessian, claimed, edt, peak, radius, min_bend_radius, step, min_length, node_spacing, label_offset, max_fibers, seed_depth_radii, bright_seed_strength=None, peak_floor=0.0, claim_radii=1.1))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn ct_trace_fibers(
    image: PyBuffer<f32>,
    hessian: PyRef<'_, PyCtHessian>,
    claimed: PyBuffer<i32>,
    edt: PyBuffer<f32>,
    peak: PyBuffer<f32>,
    radius: f64,
    min_bend_radius: f64,
    step: f64,
    min_length: f64,
    node_spacing: f64,
    label_offset: i32,
    max_fibers: Option<usize>,
    seed_depth_radii: f64,
    bright_seed_strength: Option<f32>,
    peak_floor: f64,
    claim_radii: f64,
) -> PyResult<Vec<Vec<[f64; 3]>>> {
    let shape = volume_shape(&image, "image")?;
    same_shape(&image, &claimed, "claimed")?;
    same_shape(&image, &edt, "edt")?;
    same_shape(&image, &peak, "peak")?;
    let search = FiberSearch {
        trace: trace_settings(radius, min_bend_radius, step, peak_floor)?,
        min_length,
        node_spacing,
        label_offset,
        max_fibers,
        seed_depth_radii,
        bright_seed_strength,
        claim_radii,
    };
    Ok(trace_fibers(
        read(&image, "image")?,
        shape,
        &hessian.field,
        write(&claimed, "claimed")?,
        read(&edt, "edt")?,
        read(&peak, "peak")?,
        search,
    ))
}

/// Lines as flat `(x, y, z)` values and node counts, for returning to Python.
type Packed = (Vec<f64>, Vec<usize>);

fn pack(lines: &[Vec<[f64; 3]>]) -> Packed {
    let counts = lines.iter().map(Vec::len).collect();
    let flat = lines.iter().flatten().flatten().copied().collect();
    (flat, counts)
}

fn lines_of(nodes: &PyBuffer<f64>, counts: &[usize]) -> PyResult<Vec<Vec<[f64; 3]>>> {
    split_lines(&points(nodes, "nodes")?, counts)
}

fn per_line(values: &[f64], counts: &[usize], name: &str) -> PyResult<()> {
    if values.len() != counts.len() {
        return Err(PyValueError::new_err(format!("give one {name} per line")));
    }
    Ok(())
}

/// Owners of voxels `(k, j, i)` (see `_geometry.OwnerLookup`) into `out`.
#[pyfunction]
pub(crate) fn ct_owners(
    nodes: PyBuffer<f64>,
    counts: Vec<usize>,
    radii: Vec<f64>,
    voxels: PyBuffer<i64>,
    out: PyBuffer<i32>,
) -> PyResult<()> {
    per_line(&radii, &counts, "radius")?;
    let owners = OwnerLookup::new(&lines_of(&nodes, &counts)?, &radii);
    let voxels = read(&voxels, "voxels")?;
    let out = write(&out, "out")?;
    if voxels.len() != 3 * out.len() {
        return Err(PyValueError::new_err("voxels must be (n, 3) and out (n,)"));
    }
    for (o, v) in out.iter_mut().zip(voxels.chunks_exact(3)) {
        if v.iter().any(|&c| c < 0) {
            return Err(PyValueError::new_err("voxel indices must not be negative"));
        }
        *o = owners.owner([v[0] as usize, v[1] as usize, v[2] as usize]);
    }
    Ok(())
}

/// Fiber ends grown or trimmed (see `_refine.end_step`); returns packed lines.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn ct_end_step(
    image: PyBuffer<f32>,
    nodes: PyBuffer<f64>,
    counts: Vec<usize>,
    radii: Vec<f64>,
    reach: Vec<f64>,
    step: f64,
    max_moves: usize,
) -> PyResult<Packed> {
    let shape = volume_shape(&image, "image")?;
    per_line(&radii, &counts, "radius")?;
    per_line(&reach, &counts, "reach")?;
    let lines = lines_of(&nodes, &counts)?;
    let owners = OwnerLookup::new(&lines, &radii);
    let out = end_step(
        read(&image, "image")?,
        shape,
        &lines,
        &reach,
        step,
        &owners,
        max_moves,
    );
    Ok(pack(&out))
}

/// Fits cut where they sit in void (see `_refine.cut_void`). `directions`,
/// when given, is called as `directions(fit, points)` with a (2, 3) array
/// and returns the scan's fiber axis at both points. Returns the packed
/// pieces, each piece's source fit, and the trimmed, split and bridged counts.
#[pyfunction]
#[pyo3(signature = (image, nodes, counts, radii, level, min_gap_radii, bridge_level, bridge_offset_radii, aligned_level, aligned_angle_degrees, directions=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn ct_cut_void(
    py: Python<'_>,
    image: PyBuffer<f32>,
    nodes: PyBuffer<f64>,
    counts: Vec<usize>,
    radii: Vec<f64>,
    level: f64,
    min_gap_radii: f64,
    bridge_level: f64,
    bridge_offset_radii: f64,
    aligned_level: f64,
    aligned_angle_degrees: f64,
    directions: Option<Py<PyAny>>,
) -> PyResult<(Packed, Vec<usize>, (usize, usize, usize))> {
    let shape = volume_shape(&image, "image")?;
    per_line(&radii, &counts, "radius")?;
    let lines = lines_of(&nodes, &counts)?;
    let rules = VoidRules {
        level,
        min_gap_radii,
        bridge_level,
        bridge_offset_radii,
        aligned_level,
        aligned_angle_degrees,
    };
    let array = py.import("numpy")?.getattr("array")?;
    let mut failure: Option<PyErr> = None;
    let mut call = |index: usize, ends: [[f64; 3]; 2]| -> [[f64; 3]; 2] {
        let result = (|| -> PyResult<[[f64; 3]; 2]> {
            let callback = directions.as_ref().expect("only called with directions");
            let points = array.call1((ends.to_vec(),))?;
            let axes: Vec<[f64; 3]> = callback.call1(py, (index, points))?.extract(py)?;
            match axes.as_slice() {
                [a, b] => Ok([*a, *b]),
                _ => Err(PyValueError::new_err("directions must return two axes")),
            }
        })();
        result.unwrap_or_else(|error| {
            failure.get_or_insert(error);
            [[0.0; 3]; 2]
        })
    };
    let (pieces, source, cut) = cut_void(
        read(&image, "image")?,
        shape,
        &lines,
        &radii,
        rules,
        if directions.is_some() {
            Some(&mut call as &mut dyn FnMut(usize, [[f64; 3]; 2]) -> [[f64; 3]; 2])
        } else {
            None
        },
    );
    if let Some(error) = failure {
        return Err(error);
    }
    Ok((
        pack(&pieces),
        source,
        (cut.trimmed, cut.splits, cut.bridged),
    ))
}

/// Each line resampled to `spacing` (see `_geometry.resample`); returns packed lines.
#[pyfunction]
pub(crate) fn ct_resample(
    nodes: PyBuffer<f64>,
    counts: Vec<usize>,
    spacing: f64,
) -> PyResult<Packed> {
    if !(spacing > 0.0) {
        return Err(PyValueError::new_err("spacing must be positive"));
    }
    let lines = lines_of(&nodes, &counts)?;
    let out: Vec<Vec<[f64; 3]>> = lines.iter().map(|line| resample(line, spacing)).collect();
    Ok(pack(&out))
}

/// Mean image value along each line (see `_refine.support`).
#[pyfunction]
pub(crate) fn ct_support(
    image: PyBuffer<f32>,
    nodes: PyBuffer<f64>,
    counts: Vec<usize>,
) -> PyResult<Vec<f64>> {
    let shape = volume_shape(&image, "image")?;
    Ok(support(
        read(&image, "image")?,
        shape,
        &lines_of(&nodes, &counts)?,
    ))
}

/// Largest curvature times `min_bend_radius`, per line (see `_refine.curvature_ratio`).
#[pyfunction]
pub(crate) fn ct_curvature_ratio(
    nodes: PyBuffer<f64>,
    counts: Vec<usize>,
    min_bend_radius: f64,
) -> PyResult<Vec<f64>> {
    Ok(curvature_ratio(
        &lines_of(&nodes, &counts)?,
        min_bend_radius,
    ))
}

fn line_refs(lines: &[Vec<[f64; 3]>]) -> Vec<&[[f64; 3]]> {
    lines.iter().map(Vec::as_slice).collect()
}

/// Soft union occupancy of capsules over box `[low, high)` (x, y, z) into
/// `out`, a `(z, y, x)` array of the box (see `_moves.render_occupancy`).
#[pyfunction]
pub(crate) fn ct_render_occupancy(
    low: Corner,
    high: Corner,
    nodes: PyBuffer<f64>,
    counts: Vec<usize>,
    radii: Vec<f64>,
    edge: f64,
    out: PyBuffer<f64>,
) -> PyResult<()> {
    per_line(&radii, &counts, "radius")?;
    let lines = lines_of(&nodes, &counts)?;
    let out = write(&out, "out")?;
    if out.len() != box_size(low, high) {
        return Err(PyValueError::new_err("out must have the box's shape"));
    }
    out.copy_from_slice(&render_occupancy(
        low,
        high,
        &line_refs(&lines),
        &radii,
        edge,
    ));
    Ok(())
}

/// Squared residual of rendering the lines (max-union with `base`) against
/// the scan over box `[low, high)` (see `_ends.local_residual`).
#[pyfunction]
#[pyo3(signature = (image, low, high, nodes, counts, radii, edge, base=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn ct_local_residual(
    image: PyBuffer<f32>,
    low: Corner,
    high: Corner,
    nodes: PyBuffer<f64>,
    counts: Vec<usize>,
    radii: Vec<f64>,
    edge: f64,
    base: Option<PyBuffer<f64>>,
) -> PyResult<f64> {
    let shape = volume_shape(&image, "image")?;
    per_line(&radii, &counts, "radius")?;
    let lines = lines_of(&nodes, &counts)?;
    let base = match &base {
        Some(buffer) => {
            let values = read(buffer, "base")?;
            if values.len() != box_size(low, high) {
                return Err(PyValueError::new_err("base must have the box's shape"));
            }
            Some(values)
        }
        None => None,
    };
    Ok(local_residual(
        read(&image, "image")?,
        shape,
        low,
        high,
        &line_refs(&lines),
        &radii,
        base,
        edge,
    ))
}

/// Duplicates removed and trimmed (see `_moves.trim_duplicates`).
#[pyfunction]
pub(crate) fn ct_trim_duplicates(
    nodes: PyBuffer<f64>,
    counts: Vec<usize>,
    radii: Vec<f64>,
    min_length: f64,
    closeness: f64,
) -> PyResult<(Packed, Vec<f64>)> {
    per_line(&radii, &counts, "radius")?;
    let (lines, radii) =
        trim_duplicates(&lines_of(&nodes, &counts)?, &radii, min_length, closeness);
    Ok((pack(&lines), radii))
}

/// Short or unsupported fibers removed (see `_moves.remove_unsupported`).
#[pyfunction]
pub(crate) fn ct_remove_unsupported(
    image: PyBuffer<f32>,
    nodes: PyBuffer<f64>,
    counts: Vec<usize>,
    radii: Vec<f64>,
    min_length: f64,
    min_support: f64,
) -> PyResult<(Packed, Vec<f64>)> {
    let shape = volume_shape(&image, "image")?;
    per_line(&radii, &counts, "radius")?;
    let (lines, radii) = remove_unsupported(
        read(&image, "image")?,
        shape,
        &lines_of(&nodes, &counts)?,
        &radii,
        min_length,
        min_support,
    );
    Ok((pack(&lines), radii))
}

/// Fragments joined (see `_moves.merge_fragments`); returns the lines,
/// radii and number of joins.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn ct_merge_fragments(
    image: PyBuffer<f32>,
    nodes: PyBuffer<f64>,
    counts: Vec<usize>,
    radii: Vec<f64>,
    max_gap: f64,
    max_angle_degrees: f64,
    min_bridge_support: f64,
    min_bend_radius: Option<f64>,
    kink_threshold: f64,
    end_cost: f64,
    scale: f64,
    max_prior_gap: Option<f64>,
    max_prior_angle_degrees: f64,
) -> PyResult<(Packed, Vec<f64>, usize)> {
    let shape = volume_shape(&image, "image")?;
    per_line(&radii, &counts, "radius")?;
    let settings = MergeSettings {
        max_gap,
        max_angle_degrees,
        min_bridge_support,
        min_bend_radius,
        kink_threshold,
        end_cost,
        scale,
        max_prior_gap,
        max_prior_angle_degrees,
    };
    let (lines, radii, merges) = merge_fragments(
        read(&image, "image")?,
        shape,
        &lines_of(&nodes, &counts)?,
        &radii,
        settings,
    );
    Ok((pack(&lines), radii, merges))
}

/// Kinks split or smoothed, overlong fibers cut (see `_moves.split_kinks`);
/// returns the lines, radii and number of splits.
#[pyfunction]
#[pyo3(signature = (nodes, counts, radii, min_bend_radius, min_length, max_length, threshold, min_angle_degrees, end_cost, scale, image=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn ct_split_kinks(
    nodes: PyBuffer<f64>,
    counts: Vec<usize>,
    radii: Vec<f64>,
    min_bend_radius: f64,
    min_length: f64,
    max_length: Option<f64>,
    threshold: f64,
    min_angle_degrees: f64,
    end_cost: f64,
    scale: f64,
    image: Option<PyBuffer<f32>>,
) -> PyResult<(Packed, Vec<f64>, usize)> {
    per_line(&radii, &counts, "radius")?;
    let image = match &image {
        Some(buffer) => Some((read(buffer, "image")?, volume_shape(buffer, "image")?)),
        None => None,
    };
    let settings = SplitSettings {
        min_bend_radius,
        min_length,
        max_length,
        threshold,
        min_angle_degrees,
        end_cost,
        scale,
    };
    let (lines, radii, splits) = split_kinks(image, &lines_of(&nodes, &counts)?, &radii, settings);
    Ok((pack(&lines), radii, splits))
}

/// Side-by-side fits of one fiber merged (see `_moves.resolve_side_by_side`);
/// returns the lines, radii and number of pairs merged.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn ct_resolve_side_by_side(
    image: PyBuffer<f32>,
    nodes: PyBuffer<f64>,
    counts: Vec<usize>,
    radii: Vec<f64>,
    min_length: f64,
    reach: f64,
    end_cost: f64,
    scale: f64,
    max_angle_degrees: f64,
) -> PyResult<(Packed, Vec<f64>, usize)> {
    let shape = volume_shape(&image, "image")?;
    per_line(&radii, &counts, "radius")?;
    let settings = SideBySide {
        min_length,
        reach,
        end_cost,
        scale,
        max_angle_degrees,
    };
    let (lines, radii, changed) = resolve_side_by_side(
        read(&image, "image")?,
        shape,
        &lines_of(&nodes, &counts)?,
        &radii,
        settings,
    );
    Ok((pack(&lines), radii, changed))
}

/// Line integrals of a `(z, y, x)` sample at `angles` (radians) into `out`,
/// an `(angles, z, width)` array (see `tangle_ct::scan::project`).
#[pyfunction]
pub(crate) fn ct_project(
    sample: PyBuffer<f32>,
    angles: Vec<f64>,
    out: PyBuffer<f32>,
) -> PyResult<()> {
    let shape = volume_shape(&sample, "sample")?;
    let width = match out.shape() {
        [a, z, w] if *a == angles.len() && *z == shape[0] => *w,
        _ => {
            return Err(PyValueError::new_err(
                "out must be (angles, sample z, width)",
            ))
        }
    };
    let result = tangle_ct::scan::project(read(&sample, "sample")?, shape, &angles, width);
    write(&out, "out")?.copy_from_slice(&result);
    Ok(())
}

/// The back-projection of `(angles, z, width)` rows over the `(z, y, x)`
/// volume `out` (see `tangle_ct::scan::back_project`).
#[pyfunction]
pub(crate) fn ct_back_project(
    projections: PyBuffer<f32>,
    angles: Vec<f64>,
    out: PyBuffer<f32>,
) -> PyResult<()> {
    let shape = volume_shape(&out, "out")?;
    let width = match projections.shape() {
        [a, z, w] if *a == angles.len() && *z == shape[0] => *w,
        _ => {
            return Err(PyValueError::new_err(
                "projections must be (angles, out z, width)",
            ))
        }
    };
    let result =
        tangle_ct::scan::back_project(read(&projections, "projections")?, &angles, width, shape);
    write(&out, "out")?.copy_from_slice(&result);
    Ok(())
}

/// The grey the lines should show over box `[low, high)` into `out`, a
/// `(z, y, x)` array of the box (see `_grey.render_grey`).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn ct_render_grey(
    low: Corner,
    high: Corner,
    nodes: PyBuffer<f64>,
    counts: Vec<usize>,
    radii: Vec<f64>,
    profiles: Vec<Vec<f64>>,
    void: f64,
    edge: f64,
    out: PyBuffer<f64>,
) -> PyResult<()> {
    per_line(&radii, &counts, "radius")?;
    if profiles.len() != counts.len() {
        return Err(PyValueError::new_err("give one profile per line"));
    }
    let lines = lines_of(&nodes, &counts)?;
    let out = write(&out, "out")?;
    if out.len() != box_size(low, high) {
        return Err(PyValueError::new_err("out must have the box's shape"));
    }
    let profiles: Vec<&[f64]> = profiles.iter().map(Vec::as_slice).collect();
    out.copy_from_slice(&render_grey(
        low,
        high,
        &line_refs(&lines),
        &radii,
        &profiles,
        void,
        edge,
    ));
    Ok(())
}

/// Per-line node confidence and the same without stability, per-line
/// per-sample confidence, and per component per line per sample (see
/// `_confidence.node_confidence`).
type ConfidenceOut = (
    Vec<Vec<f64>>,
    Vec<Vec<f64>>,
    Vec<Vec<f64>>,
    Vec<Vec<Vec<f64>>>,
);

#[pyfunction]
#[pyo3(signature = (image, depth, nodes, counts, radii, spacing, margin, thickness_margin, ring, thickness_tolerance, previous_nodes=None, previous_counts=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn ct_node_confidence(
    image: PyBuffer<f32>,
    depth: PyBuffer<f32>,
    nodes: PyBuffer<f64>,
    counts: Vec<usize>,
    radii: Vec<f64>,
    spacing: f64,
    margin: f64,
    thickness_margin: f64,
    ring: usize,
    thickness_tolerance: f64,
    previous_nodes: Option<PyBuffer<f64>>,
    previous_counts: Option<Vec<usize>>,
) -> PyResult<ConfidenceOut> {
    let shape = volume_shape(&image, "image")?;
    same_shape(&image, &depth, "depth")?;
    per_line(&radii, &counts, "radius")?;
    if spacing.is_nan() || spacing <= 0.0 || ring == 0 {
        return Err(PyValueError::new_err("spacing and ring must be positive"));
    }
    let lines = lines_of(&nodes, &counts)?;
    let previous = match (&previous_nodes, &previous_counts) {
        (Some(nodes), Some(counts)) => {
            if counts.len() != lines.len() {
                return Err(PyValueError::new_err("give one previous line per line"));
            }
            Some(lines_of(nodes, counts)?)
        }
        _ => None,
    };
    let settings = ConfidenceSettings {
        spacing,
        margin,
        thickness_margin,
        ring,
        thickness_tolerance,
    };
    let c = node_confidence(
        read(&image, "image")?,
        read(&depth, "depth")?,
        shape,
        &lines,
        &radii,
        previous.as_deref(),
        settings,
    );
    Ok((
        c.per_node,
        c.settled,
        c.per_sample,
        c.parts.into_iter().collect(),
    ))
}

/// Per voxel of box `[low, high)`, how many fits beyond the first contain it,
/// into `out` (`uint16`, the box's `(z, y, x)` shape).
#[pyfunction]
pub(crate) fn ct_overlap(
    low: Corner,
    high: Corner,
    nodes: PyBuffer<f64>,
    counts: Vec<usize>,
    radii: Vec<f64>,
    out: PyBuffer<u16>,
) -> PyResult<()> {
    per_line(&radii, &counts, "radius")?;
    let lines = lines_of(&nodes, &counts)?;
    let out = write(&out, "out")?;
    if out.len() != box_size(low, high) {
        return Err(PyValueError::new_err("out must have the box's shape"));
    }
    out.copy_from_slice(&overlap(low, high, &line_refs(&lines), &radii));
    Ok(())
}
