"""Build ``derived/noise_source/<version>/`` from a bench run.

The measurement is :func:`eigsep_data.noise_source_bench.measure` and the
product format :func:`eigsep_data.products.noise_source.write`. The run's
choices (where its files are, how the bench was wired, the thresholds) are
the version's ``config.json``; the manifest records it with the SHA-256 of
every bench file and of the lab reflection of the noise-source path. Run::

    EIGSEP_CAMPAIGN_ROOT=/path/to/marjum-2026-07 eigsep-build-noise-source v0001

The bench files and the lab file are read from the campaign tree given by
``--data`` (default the campaign root); the lab file is looked for under
``--products`` first, where it may be tracked on a branch.
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..index import MetadataIndex
from ..noise_source_bench import BenchSetup, measure
from ..paths import get_campaign_root
from ..products import noise_source
from ..provenance import sha256
from ._common import code_provenance, config_record, load_config


def main(argv=None):
    from eigsep_cal import S11

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("version", help="product version to write, e.g. v0001")
    ap.add_argument(
        "--data",
        type=Path,
        default=None,
        help="campaign tree holding the inputs (default: the campaign root)",
    )
    ap.add_argument(
        "--products",
        type=Path,
        default=None,
        help="campaign tree holding the config and receiving "
        "derived/noise_source/ (default: --data)",
    )
    ap.add_argument(
        "--config", help="build config (default: <version dir>/config.json)"
    )
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)

    data_root = a.data or Path(get_campaign_root(required=True))
    out_root = a.products or data_root
    cfg, cfg_path = load_config(out_root, "noise_source", a.version, a.config)
    run_dir = data_root / cfg["bench_dir"]
    gamma_path = next(
        p
        for p in (
            out_root / cfg["gamma_ns_file"],
            data_root / cfg["gamma_ns_file"],
        )
        if p.is_file()
    )
    setup = BenchSetup(
        **{
            k: tuple(v) if isinstance(v, list) else v
            for k, v in cfg["setup"].items()
        }
    )
    ch = cfg["choices"]

    # every stream: the thermistor resistances are not among the index's curated fields
    index = MetadataIndex(str(run_dir), cache=False, streams="all")
    s11 = S11(str(run_dir), year=cfg["s11_year"], pattern=cfg["s11_pattern"])
    s11.calibrate()
    with np.load(gamma_path) as g:
        gamma = {
            "freqs_mhz": g["freqs"] / 1e6,
            "on": g[setup.non],
            "off": g[setup.noff],
        }

    res = measure(
        index,
        s11,
        gamma,
        setup,
        steady_per_min=ch["steady_per_min"],
        sweep_guard=tuple(ch["sweep_guard_s"]),
        band_mhz=tuple(ch["band_mhz"]),
        chi2_factor=ch["chi2_factor"],
        max_rel_err=ch["max_rel_err"],
    )

    files = sorted(run_dir.glob("*.h5"))
    digest = hashlib.sha256()
    file_sums = {}
    for f in files:
        file_sums[f.name] = sha256(f)
        digest.update(f"{f.name} {file_sums[f.name]}\n".encode())
    provenance = {
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "builder": "eigsep_data.calibration.build_noise_source (eigsep-build-noise-source)",
        "argv": sys.argv[1:],
        "config": config_record(cfg_path, out_root),
        "bench_run": cfg["bench_run"],
        "run_record": cfg["run_record"],
        "setup": cfg["setup"],
        "choices": ch,
        "inputs": {
            "bench_files": {
                "dir": cfg["bench_dir"],
                "n_files": len(files),
                "digest_of_sha256s": digest.hexdigest(),
                "sha256": file_sums,
            },
            "gamma_ns": {
                "path": cfg["gamma_ns_file"],
                "sha256": sha256(gamma_path),
            },
        },
        "code": code_provenance(),
        "model": "eigsep_cal minimal tier: no receiver noise waves, no switch-path loss; "
        "T_NON receiver-independent with the lab Gamma_NS at P",
    }
    d = noise_source.write(
        out_root,
        a.version,
        freqs=res["freqs"],
        t_non_k=res["t_non_k"],
        t_non_sigma_k=res["t_non_sigma_k"],
        t_off_k=res["t_off_k"],
        good=res["good"],
        gamma_ns=res["gamma_ns"],
        provenance=provenance,
        validation=res["validation"],
        smooth_deg=ch["smooth_deg"],
        overwrite=a.overwrite,
    )
    print(f"wrote {d}")
    print(json.dumps(res["validation"], indent=1, default=float))


if __name__ == "__main__":
    main()
