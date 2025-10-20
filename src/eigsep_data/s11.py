from datetime import datetime
import numpy as np
from scipy import signal

from cmt_vna import calkit as cal
import os
from eigsep_observing import io


class S11:

    def __init__(self, fpath, standards_dir='./cal_data/'):
        """
        INPUT:
            filename : EIGSEP hdf5 S11 file.
            standards_dir : should contain 'fieldOSL_characterizations.npz' and 'system_sparameters.npz'.

        PROPERTIES:
            gamma_primes : uncalibrated S11s.
            cal_data : measured internal calibration standards, in OSL order.
            hdr : header from s11 file.
            meta : metadata from s11 file.
            mode : should be 'ant' if it's an ants11 file, and 'rec' if it's a recs11 file.
            time : time data was taken, as a datetime object. 
            timestamp : timestamp data was taken.
            freqs : frequency data in MHz.
            dlys : time delay axis for fft of S11 in nanoseconds. 
            sparams : a dictionary of all sparameters that were de embedded from the S11. For mode 'feed', also includes the receiver to DUT path sparams that were embedded into the DUT S11.
            standards_dir: directory with system sparameters and the characterized OSL.
            calibrated_gammas : what it sounds like, in dictionary form. will match keys of S11 hdf5 fed in.
            isolated_gammas : Only in 'feed' mode. This is the S11 with the paths to VNA de-embedded, but without the receiver paths embedded.
        

        """
        self.gamma_primes, self.cal_data, self.hdr, self.meta = io.read_s11_file(fpath)
        self.mode=list(self.gamma_primes.keys())[0]
        self.time = datetime.fromisoformat(fpath.name[-18:-3])
        self.timestamp = self.time.timestamp()
        self.freqs = np.array(self.hdr["freqs"]) / 1e6  # in MHz
        self.cal_data = np.array([self.cal_data['VNAO'], self.cal_data['VNAS'], self.cal_data['VNAL']])
        self.sparams = dict()
        self.standards_dir = standards_dir
        self._s11_dly = {}
        
        self.dlys = (
            np.fft.fftfreq(self.freqs.size, d=self.freqs[1] - self.freqs[0])* 1e3
        )  # in ns
        #read osl model, and retrieve vna sparameters from them
        standards_filename = os.path.join(self.standards_dir, 'fieldOSL_characterization.npz')
        vna_sprms = self._get_vna_sparams()
        self.sparams['vna'] = vna_sprms
        
        modes = {
                'ant': ['feed_cables', 'ns_cables'],
                'rec': ['rf_cables']
                }

        sys_sprms = {i: np.load(os.path.join(self.standards_dir, 'system_sparameters.npz'))[i] for i in modes[self.mode]}
        self.sparams = self.sparams | sys_sprms
        self._calibrate()
        self._ref_plane_to_rec()

    def _get_vna_sparams(self):
        "loads characterized standards and returns the vna sparameters."
        standards_filename = os.path.join(self.standards_dir, 'fieldOSL_characterizations.npz')
        self.osl_model = np.load(standards_filename)['fieldOSL']
        return cal.network_sparams(gamma_true=self.osl_model, gamma_meas=self.cal_data)

    def _calibrate(self):
        "calibrates the gamma_primes hased on vna sparameters and the relevant cable sparameters, with a hard-coded mapping."
        mapping = {'rec': 'rf_cables', 'ant':'feed_cables', 'load':'ns_cables', 'noise':'ns_cables'}
        self.calibrated_gammas = self.gamma_primes.copy()
        
        for key in self.gamma_primes.keys():
            self.calibrated_gammas[key] = cal.de_embed_sparams(gamma_prime=self.calibrated_gammas[key], sparams=self.sparams['vna'])
            try:
                assert mapping[key] in self.sparams.keys()
            except AssertionError:
                self.calibrated_gammas = dict()
                print('check your mode and file.')
                return
            self.calibrated_gammas[key] = cal.de_embed_sparams(gamma_prime=self.calibrated_gammas[key], sparams=self.sparams[mapping[key]]) 

    def _ref_plane_to_rec(self):
        "embeds the s parameters of the cable between the feed/noise source into the S11 of each. Can only be used in 'feed' mode."
        try:
            assert self.mode == 'ant'
        except AssertionError:
            return
        sys_sprms = np.load(os.path.join(self.standards_dir, 'system_sparameters.npz'))
        self.isolated_gammas = self.calibrated_gammas.copy()
        self.calibrated_gammas = dict()

        mapping = {'ant':'rffeed_cables', 'load':'rfns_cables', 'noise':'rfns_cables'}
        for key, data in self.isolated_gammas.items():
            self.calibrated_gammas[key] = cal.embed_sparams(sparams=sys_sprms[mapping[key]], gamma=self.isolated_gammas[key])
            self.sparams[mapping[key]] =  sys_sprms[mapping[key]]

    @property
    def s11_dly(self):
        if not self._s11_dly:
            bh = signal.windows.blackmanharris(self.freqs.size, sym=False)
            self._s11_dly = {
                k: np.abs(np.fft.ifft(self.calibrated_gammas[k] * bh)) for k in self.calibrated_gammas.keys()
                }
        return self._s11_dly
