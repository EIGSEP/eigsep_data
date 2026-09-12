"""The browser must work headless, and fail clearly without widgets."""

import importlib
import sys

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from eigsep_data.browse import StateBrowser  # noqa: E402
from eigsep_data.index import MetadataIndex  # noqa: E402

from conftest import RFSWITCH_LADDER, write_corr_file  # noqa: E402

#: Rows the ``corr_dir`` fixture puts in each file, read off the fixture
#: rather than off the index, so a test comparing against them is not
#: comparing the code under test with itself.
N_RFANT = RFSWITCH_LADDER.count("RFANT")
N_RFAMB = 30
FILE_RFANT = "corr_20260717_150041Z.h5"
FILE_CAL = "corr_20260717_152041Z.h5"


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


class TestStateBrowser:
    def test_import_and_build_without_ipywidgets(self, corr_dir, monkeypatch):
        # Importing the module must not require ipywidgets; only the
        # interactive controls do. ipywidgets is installed here, so the
        # only way to test that is to take it away: None in sys.modules
        # makes `import ipywidgets` raise, and the module is reimported
        # under that block.
        monkeypatch.setitem(sys.modules, "ipywidgets", None)
        monkeypatch.delitem(sys.modules, "eigsep_data.browse", raising=False)
        browse = importlib.import_module("eigsep_data.browse")

        sel = MetadataIndex(corr_dir, cache=False).select(rfswitch="RFANT")
        b = browse.StateBrowser(sel, keys=["0"], controls=False)
        assert b.nfiles == 1
        assert np.isfinite(b.ax_w.images[0].get_array()).all()

        with pytest.raises(ImportError, match=r"vis"):
            browse.StateBrowser(sel, keys=["0"], controls=True)

    def test_goto_reads_one_file(self, corr_dir):
        sel = MetadataIndex(corr_dir, cache=False).select(
            rfswitch=["RFANT", "RFAMB"]
        )
        # Two files, and neither one holds the whole selection: a read
        # scoped to one file is distinguishable from a read of the lot.
        assert sel.nrows == N_RFANT + N_RFAMB
        b = StateBrowser(sel, keys=["0"], controls=False)
        assert b.nfiles == 2

        b.goto(0)
        assert b.pos == 0
        assert list(b.loaded.meta.file.unique()) == [FILE_RFANT]
        assert b.loaded.data["0"].shape[0] == N_RFANT

        b.goto(1)
        assert b.pos == 1
        assert list(b.loaded.meta.file.unique()) == [FILE_CAL]
        assert b.loaded.data["0"].shape[0] == N_RFAMB

        # Past the end clamps to the last file rather than raising.
        b.goto(99)
        assert b.pos == 1

    def test_empty_selection_raises(self, corr_dir):
        sel = MetadataIndex(corr_dir, cache=False).select(rfswitch="VNAO")
        with pytest.raises(ValueError, match="no integrations"):
            StateBrowser(sel, keys=["0"], controls=False)

    def test_key_absent_from_the_file_is_not_drawn(self, tmp_path):
        # A wiring phase that never recorded key "0": the waterfall must
        # fall through to the key the file does carry, not draw a panel
        # of NaN, and the absent key must not reach the spectrum panel.
        write_corr_file(
            tmp_path / "corr_20260714_120041Z.h5",
            ntimes=20,
            keys=("4",),
            rfswitch=["RFANT"] * 20,
        )
        sel = MetadataIndex(tmp_path, cache=False).select(rfswitch="RFANT")
        b = StateBrowser(sel, keys=["0", "4"], controls=False)
        img = b.ax_w.images[0].get_array()
        assert img.shape[0] == 20
        assert np.isfinite(img).all()
        assert len(b.ax_s.lines) == 1
