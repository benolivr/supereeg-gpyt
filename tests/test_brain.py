import pytest
import os
import supereeg as se
import numpy as np
import pandas as pd
import nibabel as nib

bo = se.simulate_bo(n_samples=10, sample_rate=100)

nii = se.load('example_nifti')
bo_n = se.Brain(nii)

mo = se.load('example_model')
bo_m = se.Brain(mo)

def test_create_bo():
    assert isinstance(bo, se.Brain)

def test_bo_data_nifti():
    assert isinstance(bo_n, se.Brain)

def test_bo_data_model():
    assert isinstance(bo_m, se.Brain)

def test_bo_data_df():
    assert isinstance(bo.data, pd.DataFrame)

def test_bo_locs_df():
    assert isinstance(bo.locs, pd.DataFrame)

def test_bo_sessions_series():
    assert isinstance(bo.sessions, pd.Series)

def test_bo_nelecs_int():
    assert isinstance(bo.n_elecs, int)

def test_bo_nsecs_list():
    assert (bo.dur is None) or (type(bo.dur) is np.ndarray) or (type(bo.dur) is int) or (type(bo.dur) is float)

def test_bo_nsessions_int():
    assert isinstance(bo.n_sessions, int)

def test_bo_kurtosis_list():
    assert isinstance(bo.kurtosis, np.ndarray)

def test_samplerate_array():
    assert (bo.sample_rate is None) or (type(bo.sample_rate) is list)

def test_bo_getdata_nparray():
    assert isinstance(bo.get_data().values, np.ndarray)

def test_bo_zscoredata_nparray():
    assert isinstance(bo.get_zscore_data(), np.ndarray)

def test_bo_get_locs_nparray():
    assert isinstance(bo.get_locs().values, np.ndarray)

def test_bo_get_slice():
    bo_d = bo.get_slice(sample_inds=[1, 2], loc_inds=[1])
    assert isinstance(bo_d, se.Brain)
    assert bo_d.data.shape==(2,1)

def test_bo_resample():
    bo.resample(resample_rate=60)
    assert isinstance(bo, se.Brain)
    assert bo.sample_rate == [60]

def test_bo_save(tmpdir):
    p = tmpdir.mkdir("sub").join("example")
    print(p)
    print(type(p))
    bo.save(fname=p.strpath)
    test_bo = se.load(os.path.join(p.strpath + '.bo'))
    assert isinstance(test_bo, se.Brain)

def test_nii_nifti():
    assert isinstance(bo.to_nii(), se.Nifti)

def test_brain_load_str():
    bo = se.Brain('std')
    assert isinstance(bo, se.Brain)

def test_brain_brain():
    bo = se.simulate_bo(n_samples=10, sample_rate=100)
    bo = se.Brain(bo)
    assert isinstance(bo, se.Brain)

def test_brain_getitem():
    bo = se.simulate_bo(n_samples=10, sample_rate=100)
    bo = bo[:2]
    assert bo.data.shape[0]==2

def test_brain_getitem_2():
    bo = se.simulate_bo(n_samples=10, sample_rate=100)
    bos = [b for b in bo[:2]]
    assert all(isinstance(b, se.Brain) for b in bos)

def test_brain_getrowcols():
    bo = se.simulate_bo(n_samples=10, sample_rate=100)
    bo = bo[:5, 3]
    assert bo.data.shape==(5, 1)

def test_brain_filter():
    data = np.random.rand(10, 2)
    locs = np.random.rand(2, 3)
    bo = se.Brain(data=data, locs=locs, filter=None, sample_rate=1000)
    assert bo.get_data().shape==(10,2)
    assert bo.get_locs().shape==(2,3)

import matplotlib
import matplotlib.pyplot as plt

def test_bo_plot_data():
    # Smoke test: plot_data should not raise.  The Agg backend is set in
    # conftest.py so this works in headless CI environments.
    # plt.show() emits a UserWarning in non-interactive mode; that is expected.
    import warnings
    bo_tmp = se.simulate_bo(n_samples=10, sample_rate=100)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        bo_tmp.plot_data()      # exercises the normal (multi-sample) path
    plt.close('all')

def test_bo_plot_locs(tmpdir):
    # Smoke test: plot_locs should not raise.
    # nilearn 0.13+ added colorbar=True as default to plot_connectome, which
    # requires a .cmap attribute that GlassBrainAxes doesn't expose when only
    # nodes (no edge colormap) are drawn.  Fixed by passing colorbar=False.
    import warnings
    bo_tmp = se.simulate_bo(n_samples=10, sample_rate=100)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        bo_tmp.plot_locs()      # no file: exercises the ni_plt.show() path
    plt.close('all')

    # Also verify the save-to-file path works
    p = tmpdir.join("locs.pdf")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        bo_tmp.plot_locs(pdfpath=str(p))
    assert p.exists()
