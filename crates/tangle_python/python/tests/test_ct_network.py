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


@pytest.fixture(scope="module")
def tiny_network(tmp_path_factory) -> Path:
    torch = pytest.importorskip("torch")
    from tangle.ct._network import maps

    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("network") / "tiny.pt"
    torch.save({"model": maps.UNet3D(base=8, condition=True).state_dict(), "base": 8}, path)
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
