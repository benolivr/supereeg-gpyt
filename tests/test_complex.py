"""
Integration tests that exercise multi-step reconstruct → carve → compare
pipelines.  These are slower than unit tests and depend on Model.predict.
"""
import numpy as np
import supereeg as se
from supereeg.helpers import _count_overlapping

# ── shared fixtures ──────────────────────────────────────────────────────────

locs = np.array([[-61., -77.,  -3.],
                 [-41., -77., -23.],
                 [-21., -97.,  17.],
                 [-21., -37.,  77.],
                 [-21.,  63.,  -3.],
                 [ -1., -37.,  37.],
                 [ -1.,  23.,  17.],
                 [ 19., -57., -23.],
                 [ 19.,  23.,  -3.],
                 [ 39., -57.,  17.],
                 [ 39.,   3.,  37.],
                 [ 59., -17.,  17.]])

n_subs  = 3
n_elecs = 5

np.random.seed(42)
data = [se.simulate_model_bos(n_samples=10, locs=locs, sample_locs=n_elecs)
        for _ in range(n_subs)]
test_model = se.Model(data=data, locs=locs, rbf_width=100)

bo_full  = se.simulate_bo(n_samples=10, sessions=2, sample_rate=10, locs=locs)
sub_locs = bo_full.locs.iloc[6:]
sub_data = bo_full.data.iloc[:, sub_locs.index]
bo = se.Brain(data=sub_data.values, sessions=bo_full.sessions,
              locs=sub_locs, sample_rate=10)

# ── tests ────────────────────────────────────────────────────────────────────

def test_recon_carved():
    """
    Reconstruct a held-out electrode two ways and verify predictions agree:

      1. Full model  → predict on (N-1) electrodes → slice to target electrode.
      2. Carved model (model subset that overlaps with bo) → predict on (N-1)
         electrodes → slice to target electrode.

    The two paths should agree because the carved model is a consistent
    sub-matrix of the full correlation model; the GP posterior at the target
    location only depends on the rows/columns corresponding to bo's electrodes.

    Note on _count_overlapping(X, Y):
        Returns a boolean array of length |Y|, marking which rows of Y are
        present in X.  To find which rows of A are in B, call
        _count_overlapping(B, A).
    """
    elec_ind   = 1
    other_inds = [i for i in range(sub_locs.shape[0]) if i != elec_ind]

    # held-out electrode Brain object
    bo_l = bo.get_slice(loc_inds=elec_ind,   inplace=False)
    # input to the model (everything except the held-out electrode)
    bo_t = bo.get_slice(loc_inds=other_inds, inplace=False)

    # ── path 1: full model ───────────────────────────────────────────────────
    bo_r    = test_model.predict(bo_t, nearest_neighbor=False)
    # which rows of bo_r.locs are present in bo_l.locs?
    bo_inds = np.where(_count_overlapping(bo_l.get_locs(), bo_r.get_locs()))[0]
    bo_p    = bo_r.get_slice(loc_inds=bo_inds, inplace=False)

    # ── path 2: carved model ─────────────────────────────────────────────────
    # which rows of test_model.locs are present in bo.locs?
    mo_carve_mask = _count_overlapping(bo.get_locs(), test_model.get_locs())
    carved_mat    = test_model.get_slice(loc_inds=np.where(mo_carve_mask)[0])

    bo_c          = carved_mat.predict(bo_t, nearest_neighbor=False)
    # which rows of bo_c.locs are present in bo_l.locs?
    bo_carve_inds = np.where(_count_overlapping(bo_l.get_locs(), bo_c.get_locs()))[0]
    bo_p_2        = bo_c.get_slice(loc_inds=bo_carve_inds, inplace=False)

    # Both paths should produce the same reconstruction at the target electrode.
    assert bo_p.data.shape == bo_p_2.data.shape, (
        f"shape mismatch: full={bo_p.data.shape}, carved={bo_p_2.data.shape}"
    )
    assert np.allclose(bo_p.get_data(), bo_p_2.get_data()), \
        "full and carved reconstructions do not agree at the target electrode"
