"""Independent PuMA measurements and a reproducible report for the saved felt crop.

Run needled_section first, then run this script in an environment with pumapy.
TANGLE measurements come only from the Rust-generated JSON files.
"""
from pathlib import Path
import gc
import hashlib
import json
import os
import platform
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pumapy as puma

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output" / "needled_section"

def tensor(tangents):
    t = np.asarray(tangents, dtype=np.float64)
    return np.einsum("ni,nj->ij", t, t) / len(t)

def run():
    os.chdir(OUTPUT)  # Keep PuMA's logs beside the analysis outputs.
    native = json.loads((OUTPUT / "section_centerline_analysis.json").read_text())
    meta = json.loads((OUTPUT / "section_metadata.json").read_text())
    full = json.loads((OUTPUT / "full_specimen_analysis.json").read_text())
    ref = np.asarray(native["volume_weighted_orientation_tensor"])
    volume = meta["crop_width_m"]**3
    results = []
    slice_data = None
    for h in (3, 2, 1):
        start = time.perf_counter()
        bundle = OUTPUT / f"h_{h}um"
        ws = puma.import_vti(str(bundle / "domain.vti"), import_ws=True)
        manifest = json.loads((bundle / "manifest.json").read_text())
        assert ws.matrix.shape == tuple(manifest["grid"]["voxel_counts"])
        vf = float(puma.compute_volume_fraction(ws, (1, 2)))
        assert abs(vf - manifest["grid"]["voxel_volume_fraction"]) < 1e-12
        occupied = ws.matrix != 0
        exported = tensor(ws.orientation[occupied])
        phases = []
        for material in native["materials"]:
            phase = material["material_id"] + 1
            mask = ws.matrix == phase
            phases.append({
                "name": material["material_name"], "phase_id": phase,
                "voxel_volume_fraction": float(puma.compute_volume_fraction(ws, (phase, phase))),
                "exported_orientation_tensor": tensor(ws.orientation[mask]).tolist(),
            })
        mil = puma.compute_mean_intercept_length(ws, (0, 0))
        interface = puma.import_vti(str(bundle / "interface.vti"), import_ws=True)
        area, puma_specific = puma.compute_surface_area(interface, (128, 255), flag_gaussian=False)
        del interface
        result = {
            "voxel_size_um": h, "voxel_counts": list(ws.matrix.shape),
            "voxel_volume_fraction": vf, "phases": phases,
            "exported_orientation_tensor": exported.tolist(),
            "exported_orientation_error_frobenius": float(np.linalg.norm(exported-ref)),
            "mean_intercept_length_um": (np.asarray(mil)*1e6).tolist(),
            "surface_area_m2": float(area),
            "puma_specific_surface_area_per_m": float(puma_specific),
            "surface_area_per_crop_volume_per_m": float(area/volume),
        }
        # Estimate directions from image morphology independently of exported
        # tangents. Keep the smoothing lengths fixed in physical units.
        if h in (2, 1):
            detected = ws.copy()
            sigma_um, rho_um = 1.4, 2.8
            puma.compute_orientation_st(detected, (1,2), sigma=sigma_um/h, rho=rho_um/h, edt=True)
            angles, mean, std = puma.compute_angular_differences(ws.matrix, ws.orientation, detected.orientation, (1,2))
            st_tensor = tensor(detected.orientation[occupied])
            margin = int(np.ceil(3*rho_um/h))
            inner = occupied.copy()
            for axis in range(3):
                sl = [slice(None)]*3
                sl[axis] = slice(0,margin); inner[tuple(sl)] = False
                sl[axis] = slice(-margin,None); inner[tuple(sl)] = False
            result["structure_tensor"] = {
                "sigma_um": sigma_um, "rho_um": rho_um,
                "orientation_tensor": st_tensor.tolist(),
                "tensor_error_frobenius": float(np.linalg.norm(st_tensor-ref)),
                "angular_mean_deg": float(mean), "angular_std_deg": float(std),
                "angular_median_deg": float(np.median(angles[occupied])),
                "angular_p90_deg": float(np.percentile(angles[occupied],90)),
                "angular_interior_mean_deg": float(angles[inner].mean()),
                "interior_margin_um": margin*h,
                "by_material": [
                    {"phase_id": phase,
                     "angular_mean_deg": float(angles[ws.matrix==phase].mean()),
                     "orientation_tensor": tensor(detected.orientation[ws.matrix==phase]).tolist()}
                    for phase in (1,2)
                ],
            }
            del detected, angles, inner
            gc.collect()
        if h == 1:
            # Six-neighbor connectivity, bounded crop. Measure actual component
            # sizes rather than assuming labels are sorted by size.
            pores = puma.identify_porespace(ws, (1,2), connectivity=1)
            ids, counts = np.unique(pores[pores>0], return_counts=True)
            largest_id = ids[counts.argmax()]
            spanning = []
            for axis in range(3):
                spanning.append(bool(np.any(np.take(pores,0,axis=axis)==largest_id)
                                     and np.any(np.take(pores,-1,axis=axis)==largest_id)))
            result["pores"] = {"components": len(ids),
                "largest_fraction_of_void": float(counts.max()/counts.sum()),
                "largest_spans_xyz": spanning, "neighbor_connectivity": 6}
            del pores
            slice_data = [np.take(ws.matrix,ws.matrix.shape[axis]//2,axis=axis).copy() for axis in (2,1,0)]
        result["wall_seconds"] = time.perf_counter()-start
        results.append(result)
        (OUTPUT / f"analysis_{h}um.json").write_text(json.dumps(result,indent=2)+"\n")
        print(f"FINISHED h={h} um: Vf={vf:.8f}, seconds={result['wall_seconds']:.1f}",flush=True)
        del ws, occupied
        gc.collect()
    summary = {"source_sha256":hashlib.sha256(Path(meta["source_path"]).read_bytes()).hexdigest(),
        "pumapy_version":puma.__version__, "numpy_version":np.__version__,
        "python_version":platform.python_version(), "native":native,
        "metadata":meta,"full_specimen":full,"resolutions":results}
    (OUTPUT / "comparison.json").write_text(json.dumps(summary,indent=2)+"\n")
    make_figures(summary,slice_data)
    write_report(summary)

def make_figures(s,slices):
    plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False})
    native=s["native"]; rows=s["resolutions"]; fine=rows[-1]
    fig,ax=plt.subplots(1,3,figsize=(13,3.8),constrained_layout=True)
    hs=[r["voxel_size_um"] for r in rows]
    ax[0].plot(hs,[100*r["voxel_volume_fraction"] for r in rows],"o-",label="PuMA occupied voxels")
    ax[0].axhline(100*native["nominal_swept_volume_fraction"],color="black",ls="--",label="TANGLE nominal")
    ax[0].set(xlabel="Voxel size (µm)",ylabel="Solid volume fraction (%)",title="Volume fraction")
    ax[0].legend(fontsize=8)
    x=np.arange(3); w=.24
    for dx,key,label,color in [(-w,None,"TANGLE centerlines","#222222"),(0,"exported_orientation_tensor","Voxelized exact tangents","#2676a4"),(w,"st","PuMA image estimate","#df8c25")]:
        a=native["volume_weighted_orientation_tensor"] if key is None else fine["structure_tensor"]["orientation_tensor"] if key=="st" else fine[key]
        ax[1].bar(x+dx,np.diag(a),w,label=label,color=color)
    ax[1].set(xticks=x,xticklabels=["Axx","Ayy","Azz"],ylabel="Volume-weighted orientation",title="Direction distribution · 1 µm")
    ax[1].set_ylim(0, .67)
    ax[1].legend(fontsize=7, loc="upper right")
    ax[2].bar(["x","y","z"],fine["mean_intercept_length_um"],color=["#2676a4","#2676a4","#df8c25"])
    ax[2].set(ylabel="Void mean intercept length (µm)",title="PuMA pore-space measurement")
    fig.savefig(OUTPUT/"comparison.png",dpi=180);plt.close(fig)
    fig,ax=plt.subplots(1,3,figsize=(11.5,4),constrained_layout=True)
    cmap=ListedColormap(["white","#2879af","#e68c32"])
    for a,data,title,labels in zip(ax,slices,["XY · mid-thickness","XZ · mid-y","YZ · mid-x"],[("x","y"),("x","z"),("y","z")]):
        a.imshow(data.T,origin="lower",extent=[0,240,0,240],cmap=cmap,vmin=0,vmax=2,interpolation="nearest")
        a.set(title=title,xlabel=f"{labels[0]} (µm)",ylabel=f"{labels[1]} (µm)")
    fig.suptitle("Central 240 µm cube · blue: 7 µm fibers · orange: 19 µm fibers")
    fig.savefig(OUTPUT/"section_slices.png",dpi=180);plt.close(fig)

def write_report(s):
    n=s["native"]; m=s["metadata"]; f=s["resolutions"][-1]; whole=s["full_specimen"]
    st=f["structure_tensor"]; native_vf=n["nominal_swept_volume_fraction"]
    delta=(f["voxel_volume_fraction"]-native_vf)
    diag=lambda a: ", ".join(f"{x:.5f}" for x in np.diag(a))
    loc=lambda a: ", ".join(f"{x*1e6:.3f}" for x in a)
    lines=["# Needled felt: TANGLE–PuMA section analysis", "",
        "Analysis date: 2026-09-22. Source: the final saved TANGLE `needled_capsules.data` from the 20-ply, approximately 1 mm × 1 mm specimen (export dated 2026-09-20). This is the generated structure before DIRT loading.","",
        f"The central section has **{100*native_vf:.3f}% nominal solid volume in TANGLE** and **{100*f['voxel_volume_fraction']:.3f}% occupied volume in PuMA** at 1 µm resolution. The difference is **{100*delta:.3f} percentage points ({100*delta/native_vf:.2f}% relative)**. The exact-tangent orientation agrees closely after voxelization; PuMA's independently estimated image orientation has a mean local error of {st['angular_mean_deg']:.2f}°.","",
        "## Specimen and section", "",
        f"- Full specimen: {whole['fibers']} physical fibers, {whole['segments']:,} exported capsule segments; nominal Vf {100*whole['nominal_swept_volume_fraction']:.3f}%.",
        f"- Full cell dimensions: {loc(whole['cell']['lengths'])} µm; x/y periodic, z bounded.",
        f"- Section: 240 × 240 × 240 µm, geometrically centered in the saved cell. Lower corner ({loc(m['crop_low_m'])}) µm; upper corner ({loc(m['crop_high_m'])}) µm.",
        f"- {m['physical_fibers_with_centerline_in_crop']} distinct source fibers intersect the section centerline window; {m['clipped_segment_count']:,} clipped centerline segments.",
        f"- Reconstructed all intra-fiber bonds with maximum periodic endpoint mismatch {m['maximum_periodic_endpoint_join_error_m']:.3e} m.",
        "- The section is a bounded observation window. Parent periodic images are included, and full segments extending outside the crop are retained during voxelization to avoid artificial caps at crop faces.",
        "- A central crop is an illustrative sample, not an established representative volume element. It cannot quantify a needling effect without matched control and multiple spatial samples.","",
        "![Central section slices](section_slices.png)","",
        f"This section is less dense than the full specimen ({100*native_vf:.2f}% versus {100*whole['nominal_swept_volume_fraction']:.2f}% nominal Vf). Its local solid-volume split is approximately {100*n['materials'][0]['nominal_swept_volume_fraction']/native_vf:.1f}% small fibers and {100*n['materials'][1]['nominal_swept_volume_fraction']/native_vf:.1f}% large fibers, despite the approximately 50/50 whole-specimen design. This is one reason to analyze several windows before treating this crop as representative.","",
        "## Matched quantities", "",
        "| Quantity | TANGLE centerlines | PuMA, 1 µm voxels |", "|---|---:|---:|",
        f"| Solid volume fraction | {100*native_vf:.4f}% | {100*f['voxel_volume_fraction']:.4f}% |"]
    for mat,pm in zip(n["materials"],f["phases"]):
        lines.append(f"| {mat['material_name']} volume / cell volume | {100*mat['nominal_swept_volume_fraction']:.4f}% | {100*pm['voxel_volume_fraction']:.4f}% |")
    lines += [f"| Orientation diagonal Axx, Ayy, Azz | {diag(n['volume_weighted_orientation_tensor'])} | {diag(f['exported_orientation_tensor'])} |",
        f"| Nominal lateral / measured union surface area per crop volume | {m['nominal_specific_lateral_area_per_m']:.0f} m⁻¹ | {f['surface_area_per_crop_volume_per_m']:.0f} m⁻¹ |","",
        "TANGLE volume is the sum of cross-section area × centerline length inside the crop. PuMA volume counts occupied voxels of the union of swept capsules. Their difference includes sampling, bends/end caps, cross-section clipping at crop faces, and any residual intersections; it is not by itself an overlap measurement.","",
        "Orientation uses area × segment-length weighting in TANGLE and equal occupied-voxel weighting in the image. The orientation field exported into PuMA carries TANGLE tangents, so that comparison checks voxelization/weighting rather than independent fiber detection.","",
        "Surface area is an approximate comparison: TANGLE sums 2πrL without subtracting hidden interfaces; PuMA runs marching cubes on the smooth capsule interface with cutoff (128, 255), Gaussian filtering disabled, and open crop faces. PuMA's raw specific-area result uses (Nx−1)(Ny−1)(Nz−1)h³; the table instead divides its measured area by the physical crop volume (Nx Ny Nz h³) for consistent normalization.","",
        "## Resolution check", "",
        "| Voxel edge | Voxels across 7 µm fiber | Grid | PuMA solid Vf | ΔVf vs TANGLE (percentage points) | Exact-tangent tensor error |",
        "|---:|---:|---:|---:|---:|---:|"]
    for r in s["resolutions"]:
        lines.append(f"| {r['voxel_size_um']} µm | {7/r['voxel_size_um']:.2f} | {r['voxel_counts'][0]}³ | {100*r['voxel_volume_fraction']:.4f}% | {100*(r['voxel_volume_fraction']-native_vf):+.4f} | {r['exported_orientation_error_frobenius']:.5f} |")
    lines += ["",f"The 2 → 1 µm change in occupied volume is {100*(f['voxel_volume_fraction']-s['resolutions'][1]['voxel_volume_fraction']):+.4f} percentage points. This supports numerical stability of volume fraction for this crop; it does not establish convergence of permeability, transport, or mechanical response.","",
        "![Comparison charts](comparison.png)","",
        "## Independent PuMA orientation estimation", "",
        "PuMA estimates fiber directions from the segmented solid using a Euclidean-distance-transform structure tensor. Smoothing lengths are held fixed at sigma = 1.4 µm and rho = 2.8 µm for the 2 and 1 µm grids. The material-specific numbers below are subsets of this joint solid-image estimate.","",
        "| Grid | Mean angle error | Median | 90th percentile | Interior mean | Estimated Azz |",
        "|---|---:|---:|---:|---:|---:|"]
    for r in s["resolutions"]:
        if "structure_tensor" in r:
            q=r["structure_tensor"]
            lines.append(f"| {r['voxel_size_um']} µm | {q['angular_mean_deg']:.2f}° | {q['angular_median_deg']:.2f}° | {q['angular_p90_deg']:.2f}° | {q['angular_interior_mean_deg']:.2f}° | {q['orientation_tensor'][2][2]:.5f} |")
    lines += ["",f"At 1 µm, mean errors are {st['by_material'][0]['angular_mean_deg']:.2f}° for the 7 µm phase and {st['by_material'][1]['angular_mean_deg']:.2f}° for the 19 µm phase. Angles are sign-invariant (0–90°). The interior comparison excludes a {st['interior_margin_um']} µm band from every crop face. Image-derived directions can be ambiguous at contacts, bends, and near boundaries.","",
        "## Additional PuMA measurements", "",
        f"- Void mean intercept lengths (x, y, z): {', '.join(f'{v:.2f}' for v in f['mean_intercept_length_um'])} µm. These are PuMA's directional voxel-transition statistics for this bounded crop, not permeability or complete pore diameters.",
        f"- Six-neighbor pore connectivity: {f['pores']['components']} connected void components; {100*f['pores']['largest_fraction_of_void']:.6f}% of void belongs to the largest. Largest component spans x/y/z: {f['pores']['largest_spans_xyz']}.",
        "- No native TANGLE pore-space counterpart exists in this implementation, so these are complementary PuMA measurements rather than cross-validation.","",
        "## Limits and reproduction", "",
        "The legacy restart could not be decoded by the current checkpoint structs. This report therefore uses the final capsule export and verified geometric continuity. Rest centerlines, stored bending limits, solver residuals, and manufacturing-state diagnostics are unavailable. Any placeholder rest/curvature-limit fields in the reconstructed raw assembly reports must not be interpreted as measured strain or admissibility.","",
        "The native crop JSON represents clipped segment pieces; its `fibers` count is a piece count. Use `physical_fibers_with_centerline_in_crop` in `section_metadata.json` for the number of source fibers. The bundle-local `tangle_analysis.json` describes the full halo segments used for rasterization; the crop reference is `section_centerline_analysis.json`. Segment-owner IDs are mapped back to physical fibers in `raster_source_map.json`.","",
        f"Environment: PuMA {s['pumapy_version']}, NumPy {s['numpy_version']}, Python {s['python_version']}. PuMA is imported directly. Raw values, both surface-area normalization conventions, and timing are in [comparison.json](comparison.json).", "",
        f"Source SHA-256: `{s['source_sha256']}`", "",
        "From the TANGLE repository root:","", "```bash",
        "cargo run -p puma_cross_validation --bin needled_section",
        "python examples/puma_cross_validation/analyze_needled_section.py",
        "```", "",
        "The second command requires an environment with PuMA. This analysis used an isolated Python 3.12 environment with NumPy 1.26 for the PuMA source build; it did not change the notebook environment."]
    (OUTPUT/"report.md").write_text("\n".join(lines)+"\n")
    # A portable HTML report embeds both figures so it can be shared alone.
    import base64
    import markdown
    html=markdown.markdown("\n".join(lines),extensions=["tables","fenced_code"])
    for name in ("comparison.png","section_slices.png"):
        html=html.replace(f'src="{name}"',f'src="data:image/png;base64,{base64.b64encode((OUTPUT/name).read_bytes()).decode()}"')
    (OUTPUT/"report.html").write_text('<!doctype html><html><head><meta charset="utf-8"><title>Needled felt — TANGLE and PuMA</title><style>body{max-width:1100px;margin:40px auto;padding:0 24px;font:16px/1.55 Arial;color:#202020;background:white}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border-bottom:1px solid #ccc;padding:8px;text-align:left}img{max-width:100%}pre{white-space:pre-wrap;background:#f5f5f5;padding:14px}h1,h2{line-height:1.2}</style></head><body>'+html+'</body></html>')

if __name__ == "__main__":
    run()
