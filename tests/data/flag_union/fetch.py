"""Read two bounded released masks; write fixtures only in the caller's tree.

Run with the task's installed Python. Remote reads never build an index or
write into the shared campaign. The original raw hash is the released
dataset's source_sha256 attestation, not a newly computed raw-file hash.
"""

import json
from pathlib import Path
import subprocess

REMOTE = r"""
import base64
import hashlib
import json
from pathlib import Path
import zlib
import h5py
import numpy as np

root = Path('/mnt/data02/eigsep/marjum-2026-07')
curation = root / 'curation/boxair_emi_presence.jsonl'
entries = [json.loads(x) for x in curation.read_text().splitlines()]
out = []
for name in ['corr_20260716_011800Z.h5', 'corr_20260717_085537Z.h5']:
    path = root / 'flags/v3' / ('flags_' + name[5:13] + '.h5')
    with h5py.File(path, 'r') as h:
        d = h['mask/' + name + '/4']
        flags = np.asarray(d[:], dtype='<u2')
        freqs = np.asarray(h['freqs_mhz'][:], dtype='<f8')
        records = [x for x in entries if x.get('file') == name]
        assert len(records) == 1
        out.append(dict(
            file=name, input_key='4', phase='C', shape=list(flags.shape),
            flags_zlib_b85=base64.b85encode(zlib.compress(flags.tobytes())).decode(),
            flags_sha256=hashlib.sha256(flags.tobytes()).hexdigest(),
            frequencies_mhz=freqs.tolist(),
            frequencies_sha256=hashlib.sha256(freqs.tobytes()).hexdigest(),
            released_attributes={str(k):str(v) for k,v in d.attrs.items()},
            released_path=str(path), curation_entry=records[0],
            curation_path=str(curation),
            curation_sha256=hashlib.sha256(curation.read_bytes()).hexdigest(),
            flag_bits=json.loads((root/'flags/v3/flag_bits.json').read_text()),
            provenance_scope=('released dataset source_sha256 is '
                              'the raw-file attestation'),
        ))
print(json.dumps(out))
"""


def main():
    command = [
        "ssh",
        "-x",
        "-oBatchMode=yes",
        "-oConnectTimeout=15",
        "-oServerAliveInterval=5",
        "-oServerAliveCountMax=2",
        "limbo",
        "/mnt/data02/cbye/research/papers/eigsep_analysis/"
        "code-061-native-calibration-order/.venv/bin/python -B -",
    ]
    result = subprocess.run(
        command,
        input=REMOTE,
        capture_output=True,
        text=True,
        check=False,
        timeout=45,
    )
    if result.returncode:
        raise RuntimeError(result.stderr)
    for record in json.loads(result.stdout):
        target = Path(__file__).parent / (record["file"] + ".json")
        if target.exists():
            raise FileExistsError(target)
        target.write_text(json.dumps(record, indent=2) + "\n")
        print(target.name, target.stat().st_size)


if __name__ == "__main__":
    main()
