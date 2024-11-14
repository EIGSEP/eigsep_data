from eigsep_corr import io
import glob
import numpy as np

class EigsepData:
    
    autos = [str(i) for i in range(6)]
    cross = [f"{i}{i+2}" for i in range(4)] + ["04", "15"]

    def __init__(self, data_dir, load_file=None):
        """
        Initialize the data object.

        Parameters
        ----------
        data_dir : str
            The directory containing the .eig files.
        load_file : str
            An npz file containing measurements of the 50 Ohm load for
            calibration. If None, no calibration will be performed.

        """
        self.files = np.array(sorted(glob.glob(f"{data_dir}/*.eig")))
        self.freqs = io.read_header(self.files[0])["freqs"]
        if load_file:
            self.load = np.load(load_file)

    def read_data(
        self,
        pairs=autos+cross,
        indices=None,
        gain_normalize=True,
        pacific_to_mountain=True,
    ):
        """
        Read the data from the .eig files.

        Parameters
        ----------
        pairs : list of str
            The pairs of antennas to read. The default is reading all pairs.
        indices : list of int
            The indices of files to read. The default is reading all files.
        gain_normalize : bool
            If True, normalize the data by the gain of the 50 Ohm load.
            This requires a calibration file to be loaded.
        pacific_to_mountain : bool
            If True, convert the times from Pacific to Mountain time. Usually
            necessary since Raspberry Pi is synced to Pacific time.

        """
        if indices:
            files = self.files[indices]
        else:
            files = self.files

        if gain_normalize:
            try:
                load = self.load
            except AttributeError:
                warnings.warn("No load file loaded. Data will not be normalized.")
                gain_normalize = False
                load = {k: 1 for k in pairs}

        data = {}
        acc_cnt = []
        times = []
        for f in files:
            hdr, dat = io.read_file(f)
            times.append(hdr["times"])
            acc_cnt.append(hdr["acc_cnt"])
            for k, d in dat.items():
                if k not in pairs:
                    continue
                d.dtype = io.build_dtype(*hdr["dtype"])
                if len(k) == 1:
                    d = d[..., 0]  # only real part for autos
                else:
                    d = d[..., 0] + 1j * d[..., 1]  # complex for cross
                data[k] = data.get(k, []) + [d]
        for k, v in data.items():
            data[k] = np.concatenate(v, axis=0) / load[k]
        self.data = data
        self.acc_cnt = np.concatenate(acc_cnt, axis=0)
        self.times = np.concatenate(times, axis=0) + 3600 * int(pacific_to_mountain)




