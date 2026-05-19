.. _quickstart:

Quickstart
==========

This page gets you from raw ECoG data to a full-brain reconstruction in a few
steps.  For a deeper walkthrough see the :ref:`tutorial` and :ref:`examples-index`.

Installation
------------

.. code-block:: bash

   pip install supereeg

Core workflow
-------------

.. code-block:: python

   import supereeg as se

   # 1. Load or create a Brain object from your recordings
   bo = se.Brain(data=my_data,          # samples × electrodes array
                 locs=my_locs,          # electrodes × 3 MNI coordinate array
                 sample_rate=250)

   # 2. Load the bundled population model (or supply your own)
   mo = se.load('example_model')

   # 3. Reconstruct full-brain activity via Gaussian process regression
   bo_recon = mo.predict(bo)

   # 4. Export to NIfTI for use with any neuroimaging tool
   nii = se.Nifti(bo_recon)
   nii.to_filename('full_brain.nii.gz')

``bo_recon`` is a :class:`~supereeg.Brain` object with a time series at every
model location.  Columns corresponding to your original electrodes retain their
z-scored amplitudes; reconstructed columns will have lower amplitude — this is
expected and reflects genuine reconstruction uncertainty (see :ref:`methodology`).

Loading data from NumPy archives
---------------------------------

If your data is stored as a ``.npz`` file (e.g. from the RAM project):

.. code-block:: python

   import numpy as np
   import supereeg as se

   d = np.load('subject.npz')           # Y, R, samplerate keys
   bo = se.Brain(data=d['Y'][:500, :],  # slice for a quick test
                 locs=d['R'],
                 sample_rate=float(d['samplerate'][0]))

   mo = se.load('example_model')
   bo_recon = mo.predict(bo)

Loading saved objects
---------------------

.. code-block:: python

   # Save
   bo.save('my_recording')      # writes my_recording.bo
   mo.save('my_model')          # writes my_model.mo

   # Reload
   bo2 = se.load('my_recording.bo')
   mo2 = se.load('my_model.mo')

If you have files saved with an older version of supereeg (deepdish format)
see :ref:`migration`.

Filtering electrodes
--------------------

The paper excludes electrodes with kurtosis ≥ 10 before building or using a
model.  The :func:`~supereeg.filter_elecs` function applies this automatically:

.. code-block:: python

   bo_clean = se.filter_elecs(bo)           # drops channels with kurtosis ≥ 10
   bo_recon = mo.predict(bo_clean)

Building a model from your own data
-------------------------------------

.. code-block:: python

   # Single subject
   mo = se.Model(bo_clean)

   # Multiple subjects — pass a list
   mo = se.Model([bo1, bo2, bo3])

   # Incrementally update an existing model
   mo.update(bo_new)

Visualisation
-------------

.. code-block:: python

   bo.plot_data()                          # EEG waterfall, first 10 s
   bo.plot_locs()                          # glass-brain electrode map

   mo.plot_data()                          # correlation matrix, 0–1 / turbo
   mo.plot_data_scaled()                   # mean-centred / RdBu_r (reveals structure)
   mo.plot_locs()

   nii = se.Nifti(bo_recon)
   nii.plot_anat()                         # anatomical overlay
   nii.plot_glass_brain()                  # glass-brain view
