import healpy as hp
import numpy as np

import eigsep_data.beam_mapping as bm

PSI = 142.164


def test_stripe_split_holds_out_whole_azimuth_stripes():
    az = np.arange(-180.0, 180.0, 1.0)
    split = bm.stripe_split(az)
    k = np.floor((az + 180) / 12).astype(int)
    np.testing.assert_array_equal(split[k % 5 == 1], 1)
    np.testing.assert_array_equal(split[k % 5 == 3], 2)
    np.testing.assert_array_equal(split[(k % 5 != 1) & (k % 5 != 3)], 0)
    assert np.all(split[:12] == split[0])                       # a stripe is never split


def dipole_fields(nfreq=2, nside=16):
    v = np.array(hp.pix2vec(nside, np.arange(hp.nside2npix(nside))))
    e = np.array([1.0, 0, 0])[:, None] - v[0][None, :] * v
    return np.repeat(e[None].astype(complex), nfreq, axis=0)


def small_data(n=50, seed=0):
    rng = np.random.default_rng(seed)
    az, el = rng.uniform(-180, 180, n), rng.uniform(-180, 180, n)
    channels = np.array([640, 648])
    return bm.ToothData(t=np.arange(n), az_deg=az, el_deg=el, data=rng.normal(size=(2, n)),
                        good=np.ones((2, n), bool), channels=channels, freqs_mhz=channels * 250 / 1024,
                        split=bm.stripe_split(az))


def test_hfss_power_matches_the_transmitter_coupling_path():
    data = small_data()
    model = bm.TxGeometryModel(PSI, [0, 0, 91.0], [7, 3, 0], alpha0_deg=40.0, az_offset_deg=-16.0)
    params = np.array([1.0, 0.5, -0.4, 0.3, -0.2])
    fields = dipole_fields()
    a = bm.hfss_power(fields, data, model, params)
    R = model.rotations(data.az_deg, data.el_deg, params)
    for f, arm in enumerate(data.arms):
        b = bm.transmitter_power(fields[f:f + 1], R, model.transmitter(params), arm)
        np.testing.assert_allclose(a[f], b[0], atol=1e-12)


def test_score_and_subset():
    data = small_data()
    records = bm.score(data, data.data, 'perfect')
    assert {r['split'] for r in records} == {'train', 'validation', 'test', 'all'}
    assert all(r['fractional_rms'] == 0 for r in records if r['n'])
    sub = data.subset(np.arange(10))
    assert sub.data.shape == (2, 10) and sub.az_deg.shape == (10,)
    np.testing.assert_array_equal(sub.arms, data.arms)


def test_nan_data_is_masked():
    data = small_data()
    data.data[0, 3] = np.nan
    data = bm.ToothData(**{**data.__dict__})
    assert not data.good[0, 3]
