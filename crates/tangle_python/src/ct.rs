//! `tangle.ct` volume operations (the `tangle_ct` crate) on NumPy arrays.
//!
//! Arrays come in through the buffer protocol without copies, and results
//! are written into arrays the caller allocates, so the Python side stays a
//! thin wrapper (`tangle/ct/_native.py`). Every array must be C-contiguous
//! with the element type named in the signature.

use pyo3::buffer::{Element, PyBuffer};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use tangle_ct::hessian::HessianField;
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

fn trace_settings(radius: f64, min_bend_radius: f64, step: f64) -> PyResult<TraceSettings> {
    if !(radius > 0.0 && step > 0.0) {
        return Err(PyValueError::new_err("radius and step must be positive"));
    }
    Ok(TraceSettings {
        radius,
        min_bend_radius,
        step,
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
    let settings = trace_settings(radius, min_bend_radius, step)?;
    let tracer = Tracer::new(read(&image, "image")?, shape, &hessian.field, settings);
    Ok(tracer.trace_one_way(start, direction, max_steps, read(&claimed, "claimed")?, own_label))
}

/// Traces fibers from ridge seeds, painting `claimed` in place (see `_trace.trace_fibers`).
#[pyfunction]
#[pyo3(signature = (image, hessian, claimed, edt, peak, radius, min_bend_radius, step, min_length, node_spacing, label_offset, max_fibers, seed_depth_radii))]
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
) -> PyResult<Vec<Vec<[f64; 3]>>> {
    let shape = volume_shape(&image, "image")?;
    same_shape(&image, &claimed, "claimed")?;
    same_shape(&image, &edt, "edt")?;
    same_shape(&image, &peak, "peak")?;
    let search = FiberSearch {
        trace: trace_settings(radius, min_bend_radius, step)?,
        min_length,
        node_spacing,
        label_offset,
        max_fibers,
        seed_depth_radii,
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
