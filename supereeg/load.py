import os
import warnings
import requests
import numpy as np
import pandas as pd
from . import hdf5 as hdf5_io
from datetime import datetime
from .brain import Brain
from .model import Model
from .nifti import Nifti
from .location import Location
from .helpers import _resample_nii

BASE_URL = 'https://docs.google.com/uc?export=download'
USERCONTENT_URL = 'https://drive.usercontent.google.com/download'
homedir = os.path.expanduser('~')
datadir = os.path.join(homedir, 'supereeg_data')


datadict = { #TODO: do the data types need to be specified or could they be inferred from the downloaded objects?
    'example_data': ['1kijSKt-QLEZ1O3J5Pk-8aByn33bPCAFl', 'bo'],
    'example_model': ['1l4s7mE0KbPMmIcIA9JQzSZHCA8LWFq1I', 'mo'],
    'example_nifti': ['17VeBTruexTERwBjq1UvU6BbBRt0jkBzi', 'nii'],
    'example_filter': ['1eHcYg1idIK8y2LMLK_tqSxB7jI_l7OsL', 'bo'],
    'std': ['1P-WcEBVYnoMQAYhvSCf1BBMIDMe9VZIM', 'nii'],
    'gray': ['1a8wptBaMIFEl4j8TFhlTQVUAbyC0sN4p', 'nii'],
    'pyFR_k10r20_20mm': ['1l4s7mE0KbPMmIcIA9JQzSZHCA8LWFq1I', 'mo'],
    'pyFR_k10r20_6mm': ['1yH47fldoeuK0AQtOhMM-P2P0Dv_zH5G6', 'mo']
}


def load(fname, vox_size=None, return_type=None, sample_inds=None,
         loc_inds=None, field=None):
    """
    Load nifti file, brain or model object, or example data.

    This function can load in example data, as well as nifti objects (.nii), brain objects (.bo)
    and model objects (.mo) by detecting the extension and calling the appropriate
    load function.  Thus, be sure to include the file extension in the fname
    parameter.

    Parameters
    ----------
    fname : str

        The name of the example data or a filepath.


        Examples include :

            example_data - example brain object (n = 64)

            example_filter - load example patient data with kurtosis thresholded channels (n = 40)

            example_model - example model object with locations from gray masked brain downsampled to 20mm (n = 210)

            example_nifti - example nifti file from gray masked brain downsampled to 20mm (n = 210)


        Nifti templates :

            gray - load gray matter masked MNI 152 brain

            std - load MNI 152 standard brain


        Models :

            pyfr - model used for analyses from Owen LLW and Manning JR (2017) Towards Human Super EEG. bioRxiv: 121020`

                vox_size options: 6mm and 20mm


    vox_size : int or float

        Voxel size for loading and resampling nifti image

    return_type : str

        Option for loading data

            'bo' - returns supereeg.Brain

            'mo' - returns supereeg.Model

            'nii' - returns supereeg.Nifti

    sample_inds : int, list or slice
        Indices of samples you'd like to load in. Only works for Brain object.

    loc_inds : int, list or slice
        Indices of slices you'd like to load in. Only works for Brain object.

    field : str
        The particular field of the data you want to load. This will work for
        Brain objects and Model objects.

    Returns
    ----------
    data : supereeg.Nifti, supereeg.Brain or supereeg.Model
        Data to be returned

    """
    if field is not None and (sample_inds is not None or loc_inds is not None):
        raise ValueError("Using both field and slicing currently not supported.")

    if fname in datadict.keys():
        data = _load_example(fname, datadict[fname], sample_inds, loc_inds, field)
    else:
        data = _load_from_path(fname, sample_inds, loc_inds, field)
    if field is None:
        return _convert(data, return_type, vox_size)
    else:
        return data


def _convert(data, return_type, vox_size):
    """ Converts between bo, mo and nifti """
    if return_type is None and vox_size is None:
        return data
    elif return_type is None and vox_size is not None:
        if type(data) is Nifti:
            return _resample_nii(data, target_res=vox_size)
        else:
            warnings.warn('Data is not a Nifti file, therefore vox_size was not computed '
                          ' Please specify nii as return_type if you would like a Nifti returned.')
            return data
    elif return_type == 'nii':
        if type(data) is not Nifti:
            data = Nifti(data)
        if vox_size:
            return _resample_nii(data, target_res=vox_size)
        else:
            return data
    elif return_type == 'bo':
        if type(data) is not Brain:
            data = Brain(data)
        return data
    elif return_type == 'mo':
        if type(data) is not Model:
            data = Model(data)
        return data

def _load_example(fname, fileid, sample_inds, loc_inds, field):
    """ Loads in dataset given a google file id """
    # ensure PyTables is initialized for pandas HDFStore
    try:
        pd.io.pytables._tables()
    except Exception:
        pass

    fullpath = os.path.join(homedir, 'supereeg_data', fname + '.' + fileid[1])
    if not os.path.exists(datadir):
        os.makedirs(datadir)
    if not os.path.exists(fullpath):
        try:
            _download(fname, _load_stream(fileid[0]), fileid[1])
            _migrate_if_legacy(fullpath)
            data = _load_from_cache(fname, fileid[1], sample_inds, loc_inds, field)
        except ValueError as e:
            print(e)
            raise ValueError('Download failed.')
    else:
        try:
            data = _load_from_cache(fname, fileid[1], sample_inds, loc_inds, field)
        except Exception:
            try:
                _migrate_if_legacy(fullpath)
                data = _load_from_cache(fname, fileid[1], sample_inds, loc_inds, field)
            except Exception:
                try:
                    _download(fname, _load_stream(fileid[0]), fileid[1])
                    _migrate_if_legacy(fullpath)
                    data = _load_from_cache(fname, fileid[1], sample_inds, loc_inds, field)
                except ValueError as e:
                    print(e)
                    raise ValueError('Download failed. Try deleting cache data in'
                                     ' {}.'.format(datadir))
    return data


def _migrate_if_legacy(fullpath):
    """ Migrate deepdish legacy file to new HDF5 format if needed """
    ext = fullpath.rsplit('.', 1)[-1]
    if ext not in ('bo', 'mo'):
        return
    import h5py
    needs_migration = False
    try:
        with h5py.File(fullpath, 'r') as h5file:
            needs_migration = 'format_version' not in h5file.attrs
    except Exception:
        return
    if needs_migration:
        hdf5_io.migrate_deepdish_file(fullpath, overwrite=True)


def _load_stream(fileid):
    """ Retrieve data from google drive """
    import re

    def _get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    def _is_html(response):
        content_type = response.headers.get('Content-Type', '')
        if 'text/html' in content_type:
            return True
        # peek at first bytes without consuming the stream
        return response.content[:15].lstrip().startswith(b'<')

    def _extract_uuid(html_text):
        match = re.search(r'name="uuid"\s+value="([^"]+)"', html_text)
        return match.group(1) if match else None

    session = requests.Session()
    response = session.get(BASE_URL, params={'id': fileid}, stream=True)

    # Legacy cookie-based confirmation (small files)
    token = _get_confirm_token(response)
    if token:
        response = session.get(BASE_URL, params={'id': fileid, 'confirm': token}, stream=True)

    # New virus-scan warning page (large files) — parse uuid from HTML form
    if _is_html(response):
        html_text = response.text
        uuid = _extract_uuid(html_text)
        params = {'id': fileid, 'export': 'download', 'confirm': 't'}
        if uuid:
            params['uuid'] = uuid
        response = session.get(USERCONTENT_URL, params=params, stream=True)

    return response


def _download(fname, data, ext):
    """ Download data to cache """
    content = data.content
    if content.lstrip()[:9] in (b'<!DOCTYPE', b'<!doctype') or content.lstrip()[:5] == b'<html':
        raise ValueError(
            'Download returned an HTML page instead of file data. '
            'The Google Drive link for "{}" may be broken or require manual download.'.format(fname)
        )
    fullpath = os.path.join(homedir, 'supereeg_data', fname)
    with open(fullpath + '.' + ext, 'wb') as f:
        f.write(content)


def _load_from_path(fpath, sample_inds=None, loc_inds=None, field=None):
    """ Load a file from a local path """
    try:
        ext = fpath.split('.')[-1]
    except (AttributeError, IndexError):
        raise ValueError("Must specify a file extension.")
    if field is not None:
        if ext in ['bo', 'mo']:
            return _load_field(fpath, field)
        else:
            raise ValueError("Can only load field from Brain or Model object.")
    elif ext=='bo':
        if sample_inds!=None or loc_inds!=None:
            return Brain(**_load_slice(fpath, sample_inds, loc_inds))
        else:
            return Brain(**hdf5_io.load_brain(fpath))
    elif ext=='mo':
        return Model(**hdf5_io.load_model(fpath))
    elif ext=='locs':
        loc_data = hdf5_io.load_location(fpath)
        return Location(loc_data["locs"], meta=loc_data["meta"], date_created=loc_data["date_created"])
    elif ext in ('nii', 'gz'):
        return Nifti(fpath)
    else:
        raise ValueError("Filetype not recognized. Must be .bo, .mo, .locs or .nii.")


def _load_from_cache(fname, ftype, sample_inds=None, loc_inds=None, field=None):
    """ Load a file from local data cache """
    fullpath = os.path.join(homedir, 'supereeg_data', fname + '.' + ftype)
    if field is not None:
        if ftype in ['bo', 'mo']:
            return _load_field(fullpath, field)
        else:
            raise ValueError("Can only load field from Brain or Model object.")
    elif ftype == 'bo':
        if sample_inds is not None or loc_inds is not None:
            return Brain(**_load_slice(fullpath, sample_inds, loc_inds))
        else:
            return Brain(**hdf5_io.load_brain(fullpath))
    elif ftype == 'mo':
        # if the model was created using supereeg<0.2.0, load using the "old" format
        # (i.e. supereeg>=0.2.0 computes model in log space)
        date_created = _load_field(fullpath, field='date_created')
        if datetime.strptime(date_created, "%c")< datetime(2018, 7, 27, 14, 40, 48, 359141):
            num = _load_field(fullpath, field='numerator')
            den = _load_field(fullpath, field='denominator')
            locs = _load_field(fullpath, field='locs')
            n_subs = _load_field(fullpath, field='n_subs')
            return Model(data=np.divide(num, den), locs=locs, n_subs=n_subs)
        else:
            return Model(**hdf5_io.load_model(fullpath))
    elif ftype == 'nii':
        return Nifti(fullpath)
    elif ftype == 'locs':
        loc_data = hdf5_io.load_location(fullpath)
        return Location(loc_data["locs"], meta=loc_data["meta"], date_created=loc_data["date_created"])


def _load_field(fname, field):
    """ Loads a particular field of a file """
    return hdf5_io.load_field(fname, field)


def _load_slice(fname, sample_inds=None, loc_inds=None):
    """
    Load a slice of a brain object

    Parameters
    ----------
    fname : str
        Path to brain object

    sample_inds : int, list or slice
        Indices of samples you'd like to load in

    loc_inds : int, list or slice
        Indices of slices you'd like to load in

    Returns
    ----------
    data : dict
        Dictionary of contents to pass to brain object

    """

    return hdf5_io.load_slice(fname, sample_inds=sample_inds, loc_inds=loc_inds)


def migrate_deepdish(path, overwrite=False, compression='blosc', recursive=False):
    """Convert legacy deepdish .bo/.mo/.locs files to the new HDF5 format."""
    if os.path.isdir(path):
        candidates = []
        if recursive:
            for root, _, files in os.walk(path):
                for name in files:
                    fullpath = os.path.join(root, name)
                    if name.lower().endswith(('.bo', '.mo', '.locs')):
                        candidates.append(fullpath)
        else:
            for name in os.listdir(path):
                fullpath = os.path.join(path, name)
                if os.path.isfile(fullpath) and name.lower().endswith(('.bo', '.mo', '.locs')):
                    candidates.append(fullpath)

        converted = []
        for fullpath in sorted(candidates):
            converted.append(
                hdf5_io.migrate_deepdish_file(
                    fullpath, overwrite=overwrite, compression=compression
                )
            )
        return converted

    return [
        hdf5_io.migrate_deepdish_file(
            path, overwrite=overwrite, compression=compression
        )
    ]
