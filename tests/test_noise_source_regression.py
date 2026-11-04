"""Regressions of ``eigsep-build-noise-source`` on the bench data.

They need the bench files, so they run only where these are given:

- ``EIGSEP_NS_REGRESSION_ROOT``: a campaign tree holding
  ``derived/noise_source/v0001/`` (its ``config.json`` and the published
  ``solutions.npz``) and the bench run it names. The builder's defaults,
  with v0001's config, must rebuild v0001's arrays.
- ``EIGSEP_NS_V0002_ROOT``: a campaign tree holding the bench runs that
  ``derived/noise_source/v0002/config.json`` (under the same tree) names.
  v0002's per-channel excess must agree, band by band, with the
  independent reduction it ports (EIGSEP D5 analysis notebook 033: band
  medians of the excess at P over its kept channels) within that
  reduction's bench-terms 1 sigma.
"""

import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from eigsep_data.calibration import build_noise_source

V0001_ROOT = os.environ.get("EIGSEP_NS_REGRESSION_ROOT")
V0002_ROOT = os.environ.get("EIGSEP_NS_V0002_ROOT")
BANDS = ((50, 88), (108, 150), (150, 200), (200, 245))
#: Notebook 033 (canonical at eigsep_analysis_code eb5c274): band medians
#: of the excess at P (K) and the bench terms' 1 sigma (fraction).
NB033_EXCESS_K = (509.42, 453.59, 424.64, 420.26)
NB033_BENCH_SIGMA = (0.0382, 0.0285, 0.0336, 0.0335)


def rebuild(root, version, tmp_path):
    src = root / "derived" / "noise_source" / version
    out = tmp_path / "products"
    d = out / "derived" / "noise_source" / version
    d.mkdir(parents=True)
    shutil.copy(src / "config.json", d / "config.json")
    build_noise_source.main(
        [version, "--data", str(root), "--products", str(out)]
    )
    return d


@pytest.mark.skipif(not V0001_ROOT, reason="no bench data")
def test_defaults_rebuild_v0001(tmp_path):
    root = Path(V0001_ROOT)
    d = rebuild(root, "v0001", tmp_path)
    old = np.load(
        root / "derived" / "noise_source" / "v0001" / "solutions.npz"
    )
    new = np.load(d / "solutions.npz")
    assert set(new.files) == set(old.files)
    for k in old.files:
        np.testing.assert_array_equal(new[k], old[k], err_msg=k)


@pytest.mark.skipif(not V0002_ROOT, reason="no bench data")
def test_v0002_matches_the_reduction_it_ports(tmp_path):
    d = rebuild(Path(V0002_ROOT), "v0002", tmp_path)
    with np.load(d / "solutions.npz") as z:
        f, ex, good = z["freqs"], z["excess_k"], z["good"]
        assert str(z["smooth_model"]) == "dpss"
    for (lo, hi), ref, sig in zip(BANDS, NB033_EXCESS_K, NB033_BENCH_SIGMA):
        k = good & (f >= lo) & (f < hi)
        assert abs(np.median(ex[k]) / ref - 1) < sig, (lo, hi)
