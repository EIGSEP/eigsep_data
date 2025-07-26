from datetime import datetime
import numpy as np
from scipy import signal

from cmt_vna import calkit
from eigsep_observing import io


class S11:

    def __init__(self, fpath):
        """
        Parameters
        ----------
        fpath : pathlib.Path
            File path of S11 measurement.

        """
        self.data, self.cal_data, self.hdr, self.meta = io.read_s11_file(fpath)
        self.time = datetime.fromisoformat(fpath.name[-18:-3])
        self.timestamp = self.time.timestamp()
        self.freqs = np.array(self.hdr["freqs"]) / 1e6  # in MHz
        self.dlys = (
            np.fft.fftfreq(self.freqs.size, d=self.freqs[1] - self.freqs[0])
            * 1e3
        )  # in ns

        osl_model = np.array([1, -1, 0])  # open, short, load
        osl_model.shape = (3, 1)  # second axis is freq
        self.osl_model = np.repeat(osl_model, self.freqs.size, axis=1)

        self._s11_cal = {}
        self._s11_dly = {}

    def calibrate_s11(self, key):
        """
        First stage calibration at internal reference plane.
        S-parameters of internal network must also be de-embedded.

        Parameters
        ----------
        key : str
            Which measurment to calibrate; `ant`, `noise`, or `load`
            for `ants11` files or `rec` for `recs11` files.
        
        Returns
        -------
        np.ndarray
            Calibrated S11 data for the given key.

        """
        o = self.cal_data["VNAO"]
        s = self.cal_data["VNAS"]
        l = self.cal_data["VNAL"]
        osl = np.array([o, s, l])
        network_sparams = calkit.network_sparams(self.osl_model, osl)
        return calkit.de_embed_sparams(network_sparams, self.data[key])

    @property
    def s11_cal(self):
        if not self._s11_cal:
            self._s11_cal = {k: self.calibrate_s11(k) for k in self.data}
        return self._s11_cal

    @property
    def s11_dly(self):
        if not self._s11_dly:
            bh = signal.windows.blackmanharris(self.freqs.size, sym=False)
            self._s11_dly = {
                k: np.abs(np.fft.fft(self.s11_cal[k] * bh)) for k in self.data
            }
        return self._s11_dly
