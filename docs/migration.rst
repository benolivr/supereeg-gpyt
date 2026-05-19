.. _migration:

Migrating legacy files
======================

Files saved with older versions of supereeg used the
`deepdish <https://github.com/uchicago-cs/deepdish>`_ HDF5 format.
That library is no longer supported and is not compatible with NumPy 2.x or
Python 3.12+.

Files loaded through :func:`supereeg.load` (e.g. ``se.load('example')``) are
migrated to the new format automatically on first use.  Any ``.bo``, ``.mo``,
or ``.locs`` files you have saved to disk must be converted once using one of
the methods below.

Command-line tool
-----------------

A ``supereeg-migrate`` command is installed alongside the package:

.. code-block:: bash

   # Convert a single file in place
   supereeg-migrate path/to/file.bo --overwrite

   # Convert every legacy file in a directory (recursively)
   supereeg-migrate path/to/data/ --overwrite --recursive

Run ``supereeg-migrate --help`` for the full list of options.

Python API
----------

.. code-block:: python

   from supereeg.hdf5 import migrate_deepdish_file

   # Convert in place (replaces the original file)
   migrate_deepdish_file("recording.bo", overwrite=True)

   # Convert to a new path (original is preserved)
   migrate_deepdish_file("recording.bo", dst_path="recording_new.bo")

Both ``.bo`` (Brain) and ``.mo`` (Model) extensions are supported.

After migration the files are readable with :func:`supereeg.load` and the
standard ``se.Brain``/``se.Model`` constructors as normal. The deepdish
package does **not** need to be installed.
