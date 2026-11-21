"""``tangle.ct.find_fibers``: the CT map network tutorial's pipeline in one call that returns a Tangle fit.

The tests that run the network need PyTorch. No trained network is published yet, so they use a tiny untrained
one (the pipeline runs end to end and finds nothing); with ``TANGLE_CT_NETWORK`` set to a trained network's .pt
file, one more test checks that ``find_fibers`` finds the same fibers as the tutorial's ``find_fibers.py``.
"""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import tangle.ct as ct
from tangle.ct._find import _command_line

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "ct_unet"


def test_package_copy_is_up_to_date():
    spec = importlib.util.spec_from_file_location("sync_package", EXAMPLES / "sync_package.py")
    sync = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sync)
    assert sync.stale() == [], "tangle/ct/_network is out of date: run examples/ct_unet/sync_package.py"


def test_settings_become_the_tutorial_options():
    settings = ct.NetworkSettings(network="net.pt", voxel_size=1.3e-6, diameters=[12e-6, 30e-6], bonded=True,
                                  crop={"x": (100, 612), "z": (200, 456)}, levels=(-1000, 85), min_length=50e-6,
                                  stacks=False, binder=True, device="cpu")
    argv = _command_line("scan.tif", settings, "out")
    assert argv[:3] == ["scan.tif", "--weights", "net.pt"]
    pairs = dict(zip(argv[3::1], argv[4::1]))
    assert pairs["--voxel-size"] == "1.3um"
    assert pairs["--diameters"] == "12um,30um"
    assert pairs["--bonded"] == "yes"
    assert pairs["--crop"] == "x=100:612,z=200:456"
    assert pairs["--min-length"] == "50um"
    assert pairs["--stacks"] == "no"
    assert pairs["--out"] == "out"
    assert "--levels=-1000,85" in argv and "--binder" in argv
    assert _command_line("scan.tif", ct.NetworkSettings(network="net.pt"), None)[3:] == [
        "--bonded", "unknown", "--stacks", "auto", "--device", "auto"]


def _scan(folder: Path) -> Path:
    """A 64-voxel cube: three bright straight fibers, 6 voxels across, on a noisy dark background."""
    z, y, x = np.mgrid[:64, :64, :64]
    volume = 10.0 + np.random.default_rng(0).normal(0.0, 1.0, (64, 64, 64))
    for y0, z0 in ((20, 20), (40, 30), (30, 45)):
        volume[(y - y0) ** 2 + (z - z0) ** 2 < 9] = 100.0
    path = folder / "scan.npy"
    np.save(path, volume.astype(np.float32))
    return path


def test_bonds_csv_gives_where_each_bond_holds_its_fibers(tmp_path):
    from tangle.ct._network.fiber_outputs import Fibers, write_tables

    lines = [np.array([[0.0, 0, 5], [10, 0, 5], [20, 0, 5]]), np.array([[12.0, -5, 7], [12, 5, 7]])]
    fibers = Fibers(lines=lines, radii=np.array([1.0, 1.0]), types=np.array([0, 0]), support=np.ones(2),
                    bonds=[{"fibers": (0, 1), "center": (12.0, 0.5, 6.0), "strength": 0.9}], voxel_um=2.0,
                    shape=(10, 10, 30), type_names=["4 um"])
    write_tables(fibers, tmp_path)
    import csv
    with (tmp_path / "bonds.csv").open(newline="") as handle:
        (row,) = csv.DictReader(handle)
    a = [float(row[f"a_{k}_um"]) for k in "xyzs"]
    b = [float(row[f"b_{k}_um"]) for k in "xyzs"]
    assert a == pytest.approx([24.0, 0.0, 10.0, 24.0])  # micrometers: 2 um voxels
    assert b == pytest.approx([24.0, 1.0, 14.0, 11.0])


@pytest.fixture(scope="module")
def tiny_network(tmp_path_factory) -> Path:
    torch = pytest.importorskip("torch")
    from tangle.ct._network import maps

    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("network") / "tiny.pt"
    torch.save({"model": maps.UNet3D(base=8, condition=True).state_dict(), "base": 8, "layout": "classes"}, path)
    return path


def test_find_fibers_returns_a_tangle_fit(tiny_network, tmp_path):
    settings = ct.NetworkSettings(network=tiny_network, voxel_size=2e-6, diameters=[12e-6], bonded=False,
                                  device="cpu")
    fit = ct.find_fibers(_scan(tmp_path), settings, out=tmp_path / "found")
    assert isinstance(fit, ct.FitResult)
    assert fit.folder == tmp_path / "found"
    assert fit.voxel_size == pytest.approx(2e-6)
    assert fit.shape == (64, 64, 64)
    for name in ("summary.txt", "run.json", "fit.json", "fibers.csv", "overlay.png", "overlay.tif", "labels.tif"):
        assert (fit.folder / name).exists(), name
    rows = (fit.folder / "fibers.csv").read_text().splitlines()
    assert len(fit.centerlines) == len(rows) - 1
    assert fit.bonds == []


@pytest.mark.skipif(not os.environ.get("TANGLE_CT_NETWORK"), reason="set TANGLE_CT_NETWORK to a trained network")
def test_find_fibers_matches_the_tutorial_script(tmp_path):
    pytest.importorskip("torch")
    network = os.environ["TANGLE_CT_NETWORK"]
    scan = _scan(tmp_path)
    fit = ct.find_fibers(scan, ct.NetworkSettings(network=network, voxel_size=2e-6, device="cpu"), out=tmp_path / "api")
    subprocess.run([sys.executable, str(EXAMPLES / "find_fibers.py"), str(scan), "--weights", network,
                    "--voxel-size", "2um", "--device", "cpu", "--out", str(tmp_path / "script")],
                   check=True, cwd=EXAMPLES, capture_output=True)
    ours, theirs = (json.loads((tmp_path / name / "fit.json").read_text())["fibers"] for name in ("api", "script"))
    assert len(fit.centerlines) == len(ours) >= 3  # the three fibers of the test scan, at least
    assert [f["centerline"] for f in ours] == [f["centerline"] for f in theirs]
    assert [f["diameter"] for f in ours] == [f["diameter"] for f in theirs]


def _binder_between(a, b, x_range, half=1):
    """Binder voxel centers (x, y, z) filling the gap between fibers along x at y = a and y = b (z 10), for x in
    ``x_range``, ``half`` voxels either side in z."""
    return np.array([(x + 0.5, y + 0.5, z + 0.5) for x in x_range for y in range(a, b)
                     for z in range(10 - half, 10 + half + 1)], float)


def test_bonds_come_from_binder_filling_the_gap_between_fibers():
    pytest.importorskip("torch")  # bonds.py reads the map layout from maps.py
    from tangle.ct._network.bonds import bonds_from_binder

    # three parallel fibers of radius 2 along x, surfaces 2 voxels apart: y = 10, 16, 22, all at z = 10.5
    lines = [np.array([[0.0, y + 0.5, 10.5], [40.0, y + 0.5, 10.5]]) for y in (10, 16, 22)]
    radii = [2.0, 2.0, 2.0]
    # a neck of binder between fibers 0 and 1 around x = 10: one bond, near there
    bonds = bonds_from_binder(_binder_between(13, 14, range(8, 13)), lines, radii)
    assert [b["fibers"] for b in bonds] == [(0, 1)]
    assert bonds[0]["center"][0] == pytest.approx(10.5, abs=0.6)
    # one web around x = 30 filling both gaps and wrapping over the middle fiber: neighbours bonded, but not
    # the outer two to each other
    over = np.array([(x + 0.5, y + 0.5, z + 0.5) for x in range(28, 33) for y in range(13, 20) for z in (12, 13)])
    web = np.concatenate([_binder_between(13, 14, range(28, 33)), _binder_between(19, 20, range(28, 33)), over])
    assert sorted(b["fibers"] for b in bonds_from_binder(web, lines, radii)) == [(0, 1), (1, 2)]
    # a coating along one side of fiber 0, away from the others: no bond
    coating = np.array([(x + 0.5, 7.5, z + 0.5) for x in range(5, 35) for z in range(9, 12)], float)
    assert bonds_from_binder(coating, lines, radii) == []
    # loose binder far from every fiber: no bond
    assert bonds_from_binder(np.array([[20.5, 40.5, 30.5]]), lines, radii) == []


def test_three_way_call_starts_from_a_bond_point_network():
    torch = pytest.importorskip("torch")
    from tangle.ct._network import maps

    torch.manual_seed(0)
    old = maps.UNet3D(base=8)
    out = torch.randn(1, maps.HEAD_CHANNELS, 2, 2, 2) * 3
    # a background logit of 0 next to the old fiber and binder logits: the softmax keeps the old calls' order
    out[:, maps.VOID] = 0.0
    p = torch.softmax(maps.class_logits(out), dim=1)
    assert torch.allclose(p.sum(1), torch.ones(1, 2, 2, 2))
    fiber, binder = torch.zeros(1, 1, 2, 2, 2), torch.zeros(1, 1, 2, 2, 2)
    fiber[..., 0, 0, 0] = 1.0
    binder[..., 0, 0, :] = 1.0  # one voxel marked both: fiber wins
    target = maps.class_target({"fiber": fiber, "binder": binder})
    assert target[0, 0, 0, 0] == 1 and target[0, 0, 0, 1] == 2 and target[0, 1, 1, 1] == 0
    assert old.head.out_channels == maps.HEAD_CHANNELS
