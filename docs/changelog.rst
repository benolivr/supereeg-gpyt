.. _changelog:

Changelog
=========

0.3.0 (2025)
------------

This release modernises the codebase for Python 3.12+, NumPy 2.x, and current
versions of nibabel, nilearn, and h5py.  All 134 tests pass.

Breaking changes
~~~~~~~~~~~~~~~~

- **File format**: ``.bo`` and ``.mo`` files are now stored in HDF5 via
  ``h5py`` instead of deepdish.  Files saved with supereeg ≤ 0.2 must be
  converted once — see :ref:`migration`.

Algorithm corrections (paper alignment)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following bugs were identified by cross-referencing the codebase against
Owen et al. (2020) and corrected:

- **Kurtosis threshold**: ``filter_elecs`` and ``filter_subj`` now use
  ``kurtosis >= threshold`` (was ``>``) to match the paper's "10 or greater"
  criterion.
- **Session averaging**: ``_get_corrmat`` now uses a simple ``1/n`` average
  across sessions (Equation 2) instead of the previous duration-weighted
  average.
- **K\ :sub:`αα` inversion**: ``_timeseries_recon`` now uses
  ``numpy.linalg.inv`` (was ``pinv``) matching the paper's Equation 12.
- **Post-reconstruction re-zscore removed**: the previous codebase applied an
  additional zscore to all output columns after reconstruction, inflating the
  amplitude of reconstructed signals to match observed ones.  This step is not
  in the paper and masked genuine GP-induced variance attenuation.  Reconstructed
  electrode columns will now have lower amplitude than observed ones — see
  :ref:`methodology`.
- **``_blur_corrmat`` argument order**: fixed a bug where ``_timeseries_recon``
  passed ``rbf_weights`` as the ``Zp`` positional argument, leaving the ``gpu``
  argument unbound and causing a ``TypeError`` on any call to ``predict``.
- **``Model.plot_locs``**: fixed ``_plot_locs_connectome`` being called with
  ``pdfpath`` in the ``label`` positional slot, causing a node-colour mismatch
  error with nilearn 0.13.

New features
~~~~~~~~~~~~

- :meth:`~supereeg.Nifti.to_bo` and :meth:`~supereeg.Nifti.to_mo` convenience
  methods on the ``Nifti`` class.
- :meth:`~supereeg.Model.plot_data_scaled`: correlation matrix heatmap with
  the colorbar scaled to the data's actual range (RdBu_r, centered at the
  mean), rows/columns sorted by k-means cluster membership (``n_clusters=7``,
  matching the paper's Figure 1D).
- ``supereeg-migrate`` command-line tool and ``supereeg.hdf5.migrate_deepdish_file``
  Python API for converting legacy deepdish files.
- :class:`~supereeg.Brain` ``kurtosis`` property computed on construction.

Bug fixes
~~~~~~~~~

- ``Brain.plot_data``: fixed ``plt.savefig(filename=filepath)`` →
  ``plt.savefig(filepath)`` (wrong keyword argument name).
- ``Nifti.plot_anat`` / ``Nifti.plot_glass_brain``: fixed nibabel
  ``ImageSlicer`` error when calling ``index_img`` on a ``Nifti1Image``
  subclass by casting to a plain ``Nifti1Image`` first.
- ``_plot_locs_connectome``: added ``colorbar=False`` to both
  ``plot_connectome`` calls to fix an ``AttributeError`` introduced by
  nilearn 0.13 changing the default to ``colorbar=True``.
- Tutorial RST stubs: replaced ``.. code:: ipython2`` with ``.. code:: python``
  throughout (ipython2 lexer removed in modern Pygments).

Documentation
~~~~~~~~~~~~~

- Sidebar navigation restored: removed ``:orphan:`` directive from all doc
  pages, populated ``index.rst`` toctree.
- Added :ref:`quickstart`, :ref:`methodology`, :ref:`migration`, and this
  changelog.
- API reference now uses ``.. autoclass:: :members:`` to render all method
  docstrings (was missing method-level pages).
- ``Model.predict`` docstring updated to explain reconstruction amplitude
  attenuation.
- Added ``CITATION.cff`` (CFF 1.2.0) with preferred citation for the
  Cerebral Cortex paper.

Dependencies
~~~~~~~~~~~~

- Minimum Python raised to 3.9 (tested on 3.12).
- Replaced ``deepdish`` with ``h5py`` for file I/O.
- ``scikit-learn`` added as dependency (used by ``plot_data_scaled``).
- ``cupy`` is optional (GPU support); mocked in docs build.

0.2.1
-----

Initial public release accompanying Owen et al. (2020).
