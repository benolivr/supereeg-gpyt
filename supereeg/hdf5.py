import json
import numbers
import os

import h5py
import numpy as np
import pandas as pd

FORMAT_VERSION = "1"
CREATED_BY = "supereeg"

_JSON_FIELDS = {
    "meta",
    "date_created",
    "label",
    "filter",
    "sample_rate",
}

_PANDAS_FIELDS = {
    "locs",
    "sessions",
}


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (np.bool_)):
        return bool(value)
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _json_dumps(value):
    try:
        return json.dumps(value, default=_json_default)
    except TypeError as exc:
        raise ValueError(
            "HDF5 save failed: field contains non-JSON-serializable values. "
            "Convert to basic Python types before saving."
        ) from exc


def _write_json_dataset(h5file, name, value):
    dtype = h5py.string_dtype(encoding="utf-8")
    if name in h5file:
        del h5file[name]
    h5file.create_dataset(name, data=_json_dumps(value), dtype=dtype)


def _read_json_dataset(h5file, name):
    raw = h5file[name][()]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    elif isinstance(raw, np.bytes_):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def _normalize_compression(compression):
    if compression is None:
        return None
    if not isinstance(compression, str):
        raise ValueError("compression must be a string or None")
    comp = compression.lower()
    if comp == "blosc":
        return "gzip"
    if comp in ("gzip", "lzf", "szip"):
        return comp
    if comp in ("none", "false"):
        return None
    raise ValueError("Unsupported compression: {0}".format(compression))


def _write_array_dataset(h5file, name, value, compression=None, chunks=False):
    if value is None:
        return
    data = np.asarray(value)
    if name in h5file:
        del h5file[name]
    kwargs = {}
    if data.ndim > 0 and compression is not None:
        kwargs["compression"] = compression
    if data.ndim > 0 and chunks:
        kwargs["chunks"] = True
    h5file.create_dataset(name, data=data, **kwargs)


def _read_array_dataset(h5file, name):
    return h5file[name][()]


def _ensure_locs_df(locs):
    if isinstance(locs, pd.DataFrame):
        if all(col in locs.columns for col in ["x", "y", "z"]):
            return locs[["x", "y", "z"]].copy()
        raise ValueError("locs must contain x, y, z columns")
    locs_arr = np.asarray(locs)
    if locs_arr.ndim != 2 or locs_arr.shape[1] != 3:
        raise ValueError("locs must be an (n, 3) array")
    return pd.DataFrame(locs_arr, columns=["x", "y", "z"])


def _ensure_sessions_series(sessions):
    if isinstance(sessions, pd.Series):
        return sessions
    return pd.Series(sessions)


def _write_root_attrs(h5file, object_type):
    h5file.attrs["format_version"] = FORMAT_VERSION
    h5file.attrs["object_type"] = object_type
    h5file.attrs["created_by"] = CREATED_BY


def _read_attr(h5file, name):
    value = h5file.attrs.get(name)
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.decode("utf-8")
    return value


def _raise_legacy_error(path):
    raise ValueError(
        "Legacy deepdish file detected at {0}. Run supereeg.hdf5.migrate_deepdish_file(\"{0}\", "
        "overwrite=True) to convert to the new HDF5 format.".format(path)
    )


def _require_format(h5file, path, expected_type=None):
    if "format_version" not in h5file.attrs:
        _raise_legacy_error(path)
    if expected_type is None:
        return
    object_type = _read_attr(h5file, "object_type")
    if object_type != expected_type:
        raise ValueError(
            "Unexpected object type in {0}: {1} (expected {2})".format(
                path, object_type, expected_type
            )
        )


def save_brain(brain, fname, compression="blosc"):
    comp = _normalize_compression(compression)
    with h5py.File(fname, "w") as h5file:
        _write_root_attrs(h5file, "brain")
        _write_array_dataset(h5file, "data", brain.data.values, comp, chunks=True)
        _write_json_dataset(h5file, "sample_rate", brain.sample_rate)
        _write_array_dataset(h5file, "kurtosis", brain.kurtosis, comp)
        _write_array_dataset(h5file, "kurtosis_threshold", brain.kurtosis_threshold, comp)
        _write_json_dataset(h5file, "meta", brain.meta)
        _write_json_dataset(h5file, "date_created", brain.date_created)
        _write_array_dataset(h5file, "minimum_voxel_size", brain.minimum_voxel_size, comp)
        _write_array_dataset(h5file, "maximum_voxel_size", brain.maximum_voxel_size, comp)
        _write_json_dataset(h5file, "label", brain.label)
        _write_json_dataset(h5file, "filter", brain.filter)

    locs_df = _ensure_locs_df(brain.locs)
    sessions_series = _ensure_sessions_series(brain.sessions)
    with pd.HDFStore(fname, mode="a") as store:
        store.put("locs", locs_df)
        store.put("sessions", sessions_series)


def save_model(model, fname, compression="blosc"):
    comp = _normalize_compression(compression)
    with h5py.File(fname, "w") as h5file:
        _write_root_attrs(h5file, "model")
        _write_array_dataset(h5file, "numerator", model.numerator, comp, chunks=True)
        _write_array_dataset(h5file, "denominator", model.denominator, comp, chunks=True)
        _write_array_dataset(h5file, "n_subs", model.n_subs, comp)
        _write_json_dataset(h5file, "meta", model.meta)
        _write_json_dataset(h5file, "date_created", model.date_created)
        _write_array_dataset(h5file, "rbf_width", model.rbf_width, comp)

    locs_df = _ensure_locs_df(model.locs)
    with pd.HDFStore(fname, mode="a") as store:
        store.put("locs", locs_df)


def save_location(location, fname, compression="blosc"):
    comp = _normalize_compression(compression)
    with h5py.File(fname, "w") as h5file:
        _write_root_attrs(h5file, "location")
        _write_json_dataset(h5file, "meta", location.meta)
        _write_json_dataset(h5file, "date_created", location.date_created)

    locs_df = _ensure_locs_df(location.locs)
    with pd.HDFStore(fname, mode="a") as store:
        store.put("locs", locs_df)


def load_brain(fname):
    with h5py.File(fname, "r") as h5file:
        _require_format(h5file, fname, expected_type="brain")
        data = _read_array_dataset(h5file, "data")
        sample_rate = _read_json_dataset(h5file, "sample_rate")
        kurtosis = _read_array_dataset(h5file, "kurtosis")
        kurtosis_threshold = _read_array_dataset(h5file, "kurtosis_threshold")
        meta = _read_json_dataset(h5file, "meta")
        date_created = _read_json_dataset(h5file, "date_created")
        minimum_voxel_size = _read_array_dataset(h5file, "minimum_voxel_size")
        maximum_voxel_size = _read_array_dataset(h5file, "maximum_voxel_size")
        label = _read_json_dataset(h5file, "label")
        filt = _read_json_dataset(h5file, "filter")

    with pd.HDFStore(fname, mode="r") as store:
        locs = store.get("locs")
        sessions = store.get("sessions")

    return dict(
        data=data,
        locs=locs,
        sessions=sessions,
        sample_rate=sample_rate,
        kurtosis=kurtosis,
        kurtosis_threshold=kurtosis_threshold,
        meta=meta,
        date_created=date_created,
        minimum_voxel_size=minimum_voxel_size,
        maximum_voxel_size=maximum_voxel_size,
        label=label,
        filter=filt,
    )


def load_model(fname):
    with h5py.File(fname, "r") as h5file:
        _require_format(h5file, fname, expected_type="model")
        numerator = _read_array_dataset(h5file, "numerator")
        denominator = _read_array_dataset(h5file, "denominator")
        n_subs = _read_array_dataset(h5file, "n_subs")
        meta = _read_json_dataset(h5file, "meta")
        date_created = _read_json_dataset(h5file, "date_created")
        rbf_width = _read_array_dataset(h5file, "rbf_width")

    with pd.HDFStore(fname, mode="r") as store:
        locs = store.get("locs")

    return dict(
        numerator=numerator,
        denominator=denominator,
        locs=locs,
        n_subs=int(n_subs),
        meta=meta,
        date_created=date_created,
        rbf_width=rbf_width,
    )


def load_location(fname):
    with h5py.File(fname, "r") as h5file:
        _require_format(h5file, fname, expected_type="location")
        meta = _read_json_dataset(h5file, "meta")
        date_created = _read_json_dataset(h5file, "date_created")

    with pd.HDFStore(fname, mode="r") as store:
        locs = store.get("locs")

    return dict(locs=locs, meta=meta, date_created=date_created)


def load_field(fname, field):
    if field in _PANDAS_FIELDS:
        with pd.HDFStore(fname, mode="r") as store:
            try:
                return store.get(field)
            except KeyError:
                raise KeyError("Field not found: {0}".format(field))

    with h5py.File(fname, "r") as h5file:
        _require_format(h5file, fname, expected_type=None)
        if field in _JSON_FIELDS:
            if field not in h5file:
                raise KeyError("Field not found: {0}".format(field))
            return _read_json_dataset(h5file, field)
        if field in h5file:
            return _read_array_dataset(h5file, field)

    raise KeyError("Field not found: {0}".format(field))


def _normalize_index(index):
    if index is None:
        return None
    if isinstance(index, slice):
        return index
    if isinstance(index, range):
        if index.step == 1:
            return slice(index.start, index.stop, index.step)
        return np.asarray(list(index), dtype=int)
    if isinstance(index, numbers.Integral):
        return int(index)
    if isinstance(index, (list, np.ndarray)):
        return np.asarray(index, dtype=int)
    return index


def load_slice(fname, sample_inds=None, loc_inds=None):
    if sample_inds is not None and loc_inds is not None:
        if not isinstance(sample_inds, numbers.Integral) and not isinstance(loc_inds, numbers.Integral):
            raise IndexError("Slicing with 2 lists is currently not supported.")

    sample_sel = _normalize_index(sample_inds)
    loc_sel = _normalize_index(loc_inds)

    with h5py.File(fname, "r") as h5file:
        _require_format(h5file, fname, expected_type="brain")
        data = h5file["data"]
        if sample_sel is None and loc_sel is None:
            data_sel = data[()]
        elif sample_sel is None:
            data_sel = data[:, loc_sel]
        elif loc_sel is None:
            data_sel = data[sample_sel, :]
        else:
            data_sel = data[sample_sel, loc_sel]
        meta = _read_json_dataset(h5file, "meta")
        date_created = _read_json_dataset(h5file, "date_created")
        sample_rate = _read_json_dataset(h5file, "sample_rate")

    with pd.HDFStore(fname, mode="r") as store:
        locs = store.get("locs")
        sessions = store.get("sessions")

    if loc_sel is None:
        locs_sel = locs
    else:
        locs_sel = locs.iloc[loc_sel]

    if sample_sel is None:
        sessions_sel = sessions
    else:
        sessions_sel = sessions.iloc[sample_sel]

    sessions_list = sessions_sel.tolist()
    if sample_rate is None:
        sample_rate_out = None
    else:
        sr_values = sample_rate
        if isinstance(sr_values, np.ndarray):
            sr_values = sr_values.tolist()
        sample_rate_out = [sr_values[int(s - 1)] for s in np.unique(sessions_list)]

    if isinstance(locs_sel, pd.DataFrame):
        locs_array = locs_sel.values
    else:
        locs_array = np.asarray(locs_sel)

    data_array = np.atleast_2d(data_sel)
    locs_array = np.atleast_2d(locs_array)
    if locs_array.shape[0] == 1:
        if data_array.shape[1] > data_array.shape[0]:
            data_array = data_array.T

    return dict(
        data=data_array,
        locs=locs_array,
        sample_rate=sample_rate_out,
        meta=meta,
        date_created=date_created,
    )


def _read_legacy_deepdish(path, ext):
    """Read a legacy deepdish/PyTables HDF5 file, returning a dict suitable
    for constructing Brain, Model, or Location objects.
    Uses PyTables (tables) to handle the compressed CARRAY datasets."""
    import tables

    def _decode(v):
        if isinstance(v, (bytes, np.bytes_)):
            return v.decode()
        if hasattr(v, 'item'):
            v = v.item()
            return v.decode() if isinstance(v, bytes) else v
        return v

    def _read_meta(node):
        title = _decode(getattr(node._v_attrs, 'TITLE', ''))
        if title.startswith('nonetype'):
            return None
        if title.startswith('dict'):
            skip = {'CLASS', 'TITLE', 'VERSION'}
            return {k: _decode(node._v_attrs[k])
                    for k in node._v_attrs._v_attrnames if k not in skip}
        return None

    def _read_list(node):
        title = _decode(getattr(node._v_attrs, 'TITLE', ''))
        if title.startswith('list:'):
            n = int(title.split(':')[1])
            return [node._v_attrs[f'i{i}'].item() for i in range(n)]
        return []

    def _read_pandas_series(node):
        index = node.index[:]
        values = node.values[:]
        return pd.Series(data=values, index=index)

    def _read_pandas_dataframe(node):
        cols = [_decode(c) for c in node.axis0[:]]
        values = node.block0_values[:]
        return pd.DataFrame(values, columns=cols)

    with tables.open_file(path, 'r') as f:
        root = f.root
        date_created = _decode(getattr(root._v_attrs, 'date_created', ''))

        if ext == '.bo':
            data = root.data[:]
            locs = root.locs[:]
            meta = _read_meta(root.meta)
            sample_rate = _read_list(root.sample_rate)
            sessions = _read_pandas_series(root.sessions)
            return dict(data=data, locs=locs, meta=meta,
                        sample_rate=sample_rate, sessions=sessions,
                        date_created=date_created)

        if ext == '.mo':
            numerator = root.numerator[:]
            denominator = root.denominator[:]
            n_subs = int(getattr(root._v_attrs, 'n_subs', 0))
            meta = _read_meta(root.meta)
            locs = _read_pandas_dataframe(root.locs)
            return dict(numerator=numerator, denominator=denominator,
                        n_subs=n_subs, meta=meta, locs=locs,
                        date_created=date_created)

    raise ValueError("Unsupported legacy extension: {0}".format(ext))


def migrate_deepdish_file(src_path, dst_path=None, compression="blosc", overwrite=False):
    if dst_path is None:
        dst_path = src_path

    if os.path.exists(dst_path) and not overwrite:
        raise ValueError(
            "Destination exists: {0}. Pass overwrite=True to replace it.".format(dst_path)
        )

    _, ext = os.path.splitext(src_path)
    ext = ext.lower()

    data = _read_legacy_deepdish(src_path, ext)

    if ext == ".bo":
        from .brain import Brain
        obj = Brain(**data)
        save_brain(obj, dst_path, compression=compression)
    elif ext == ".mo":
        from .model import Model
        obj = Model(**data)
        save_model(obj, dst_path, compression=compression)
    elif ext == ".locs":
        from .location import Location
        obj = Location(data.get("locs"), meta=data.get("meta"),
                       date_created=data.get("date_created"))
        save_location(obj, dst_path, compression=compression)
    else:
        raise ValueError("Unsupported legacy extension: {0}".format(ext))

    return dst_path
