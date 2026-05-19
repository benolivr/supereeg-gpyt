.. _methodology:

Algorithm and implementation notes
====================================

This page maps the equations in `Owen et al. (2020)`_ to the functions that
implement them, and documents deliberate implementation choices and corrections
made relative to the original codebase.

.. _Owen et al. (2020): https://doi.org/10.1093/cercor/bhaa115

Overview
--------

SuperEEG uses Gaussian process regression to infer full-brain activity from a
sparse set of implanted electrodes.  The algorithm has two stages:

1. **Model building** — learn a full-brain correlation structure from many
   subjects' recordings.
2. **Reconstruction** — given a new subject's sparse recordings and the
   population model, infer activity at all unobserved locations.

Preprocessing
-------------

**Paper (Methods, p. 4):** fourth-order Butterworth notch filter at 60 Hz,
then downsample all recordings to 250 Hz.  Electrodes with maximum kurtosis
≥ 10 across sessions are excluded.

**Code:**

- Filtering: :func:`~supereeg.Brain.apply_filter` on the Brain object.
- Kurtosis exclusion: :func:`~supereeg.filter_elecs` and
  :func:`~supereeg.filter_subj`, both with ``threshold=10`` and using
  ``kurtosis >= threshold`` (i.e. the boundary value 10 is excluded, matching
  the paper's "10 or greater").

Radial basis function — Equation 1
-----------------------------------

.. math::

   \text{rbf}(x \mid \eta, \lambda) = \exp\!\left(-\frac{\|x - \eta\|^2}{\lambda}\right)

with :math:`\lambda = 20` (mm²).

**Code:** ``helpers._log_rbf(to_coords, from_coords, width=20)`` returns
:math:`\log \text{rbf}` directly (i.e. :math:`-\|x-\eta\|^2 / \lambda`) for
numerical stability.  The ``width`` parameter defaults to 20 and is exposed as
``rbf_width`` on the :class:`~supereeg.Model` constructor.

Per-patient correlation matrix — Equations 2–4
------------------------------------------------

.. math::

   \bar{C}_s = r\!\left(\frac{1}{n}\sum_{k=1}^{n} z(C_{s,k})\right)

where :math:`z` is the Fisher z-transform and :math:`r = z^{-1}` is its
inverse.  Sessions are averaged with **equal weight** (simple :math:`1/n`
mean).

**Code:** ``helpers._get_corrmat(bo)``.  Each session's correlation matrix is
z-transformed and summed; the sum is divided by the number of sessions and
then back-transformed.

Full-brain correlation estimate — Equations 6–8
------------------------------------------------

.. math::

   \hat{N}_s(x, y) &= \sum_{i}\sum_{j < i} W(x,i)\cdot W(y,j)\cdot z\!\left(\bar{C}_s(i,j)\right) \\
   \hat{D}_s(x, y) &= \sum_{i}\sum_{j < i} W(x,i)\cdot W(y,j) \\
   \hat{C}_s &= r\!\left(\frac{\hat{N}_s}{\hat{D}_s}\right)

where :math:`W(x, i) = \text{rbf}(i \mid x, \lambda)`.

**Code:** ``helpers._blur_corrmat(Z, Zp, weights, gpu)`` implements Equations
6–8 entirely in log-space to avoid overflow with the exponentially large weight
products.  ``_zero_pad_corrmat`` expands the subject's electrode-level matrix
:math:`Z` to the full location set before blurring.

Merged correlation model — Equation 9
--------------------------------------

.. math::

   \hat{K} = r\!\left(\frac{\sum_s \hat{N}_s}{\sum_s \hat{D}_s}\right)

**Code:** :meth:`~supereeg.Model.update` accumulates numerators and
denominators across subjects in log-complex space
(``_to_log_complex`` / ``_to_exp_real``); :meth:`~supereeg.Model.get_model`
recovers :math:`\hat{K}` on demand.

Reconstruction — Equation 12
------------------------------

.. math::

   \hat{Y}_{s,k,\beta_s} = \left(\hat{K}_{\beta_s,\alpha_s}\cdot\hat{K}_{\alpha_s,\alpha_s}^{-1}\cdot Y_{s,k,\alpha_s}^{\mathsf{T}}\right)^{\mathsf{T}}

where :math:`\alpha_s` indexes observed electrode locations and :math:`\beta_s`
indexes all other locations.

**Code:** ``helpers._reconstruct_activity(Y, Kba, Kaa_inv)`` computes
``(Kba @ Kaa_inv @ Y.T).T``, matching Equation 12 exactly.

Key implementation notes
------------------------

:math:`K_{\alpha\alpha}` inversion
    The paper uses the standard matrix inverse.  The code uses
    ``numpy.linalg.inv`` (not the pseudoinverse), matching the paper's
    assumption that :math:`K_{\alpha\alpha}` is invertible.  If your electrode
    set produces a near-singular :math:`K_{\alpha\alpha}` (e.g. many
    co-located electrodes), ``predict`` may raise a ``LinAlgError``.

Reconstruction amplitude attenuation
    Observed electrode columns in the output of :meth:`~supereeg.Model.predict`
    have z-scored amplitudes (std ≈ 1).  Reconstructed columns will have
    **lower amplitude** (std < 1).  GP regression is a smoother: it integrates
    information from spatially distributed electrodes via the learned
    correlation structure, which attenuates variance.  This attenuation is real
    and reflects reconstruction uncertainty.  It is intentionally not corrected
    post-hoc so that downstream analyses can account for it.  The paper's
    reported performance metrics (correlation between reconstructed and
    observed) are scale-invariant and are unaffected by this choice.

Session averaging
    Equation 2 uses a simple :math:`1/n` average across sessions.  The code
    matches this exactly.  An earlier version of the codebase used
    duration-weighted averaging; this has been corrected.

Kurtosis threshold boundary
    The paper excludes electrodes with kurtosis "of 10 or greater."
    :func:`~supereeg.filter_elecs` now uses ``kurtosis >= threshold``; an
    earlier version used strict ``>`` and would have retained electrodes with
    kurtosis exactly equal to the threshold.
