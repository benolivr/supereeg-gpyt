import argparse
import sys

from .load import migrate_deepdish


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="supereeg-migrate",
        description="Convert legacy deepdish .bo/.mo/.locs files to the new HDF5 format.",
    )
    parser.add_argument(
        "path",
        help="Path to a legacy file or a directory containing legacy files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in place.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search directories recursively for legacy files.",
    )
    parser.add_argument(
        "--compression",
        default="blosc",
        help="Compression to use (blosc maps to gzip).",
    )
    return parser


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        converted = migrate_deepdish(
            args.path,
            overwrite=args.overwrite,
            compression=args.compression,
            recursive=args.recursive,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    for path in converted:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
