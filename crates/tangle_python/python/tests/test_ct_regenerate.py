"""regenerate_kept.py: the audit cut applied again to remade scans."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "ct_unet"
sys.path.insert(0, str(EXAMPLES))
spec = importlib.util.spec_from_file_location("regenerate_kept", EXAMPLES / "regenerate_kept.py")
regen = importlib.util.module_from_spec(spec)
sys.modules["regenerate_kept"] = regen  # so --jobs worker processes can find recut_row
spec.loader.exec_module(regen)


def scan(folder: Path, name: str, blur: float, diameter: float, fiber_grey: float, fibers: bool = True) -> Path:
    """A tiny make_data-like scan: one straight fiber along x in noisy void, and its settings json."""
    rng = np.random.default_rng(0)
    n = 24
    volume = rng.normal(0.0, 1.0, (n, n, n)).astype(np.float32)
    labels = np.zeros((n, n, n), np.int32)
    xs = np.arange(n, dtype=float)
    if fibers:
        labels[10:14, 10:14, :] = 1
        volume[labels > 0] = fiber_grey
        pos = np.stack([xs, np.full(n, 12.0), np.full(n, 12.0)], 1)
        tan, fid, typ = np.tile([1.0, 0.0, 0.0], (n, 1)), np.ones(n, int), np.zeros(n, int)
    else:
        pos, tan, fid, typ = np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0, int), np.zeros(0, int)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.npz"
    np.savez(path, volume=volume, labels=labels, p_pos=pos, p_tan=tan, p_fid=fid, p_typ=typ)
    settings = {"family": "mixed", "scanner": {"resolution": blur}, "types": [{"diameter": diameter}]}
    path.with_suffix(".json").write_text(json.dumps({"settings": settings}))
    return path


def test_recut_applies_the_audit_rule(tmp_path):
    folder = tmp_path / "data_mixed"
    scan(folder, "mixed_1", blur=2.0, diameter=8.0, fiber_grey=10.0)  # clear: kept
    scan(folder, "mixed_2", blur=7.0, diameter=8.0, fiber_grey=10.0)  # blur 0.875 x diameter: cut
    scan(folder, "mixed_3", blur=2.0, diameter=8.0, fiber_grey=0.0)  # fiber as dark as the void: cut
    scan(folder, "mixed_4", blur=9.0, diameter=8.0, fiber_grey=0.0, fibers=False)  # no fibers: never cut
    cut = regen.recut(tmp_path, ["data_mixed"], jobs=1)
    assert cut == ["data_mixed/mixed_2", "data_mixed/mixed_3"]
    assert (tmp_path / "exclude_recut.txt").read_text().split() == cut
    assert len((tmp_path / "recut.csv").read_text().splitlines()) == 5


def test_recut_in_worker_processes_matches(tmp_path):
    folder = tmp_path / "data_mixed"
    for i, blur in enumerate((2.0, 7.0, 2.0, 7.0)):
        scan(folder, f"mixed_{i}", blur=blur, diameter=8.0, fiber_grey=10.0)
    assert regen.recut(tmp_path, ["data_mixed"], jobs=2) == ["data_mixed/mixed_1", "data_mixed/mixed_3"]
