#!/usr/bin/env python3

from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


BASE = Path.home() / "Projects/cosmic_ai"

FILES = [
    BASE / "data/atomic_lines/m51_1096_pagamma_hydrogen_consistency_map.fits",
    BASE / "data/atomic_lines/m51_1096_pagamma_consistency_map.fits",
    BASE / "data/atomic_lines/m51_1096_pagamma_consistency_snr.fits",
    BASE / "data/atomic_lines/m51_1094_pagamma_multidimensional_line_map.fits",
    BASE / "data/atomic_lines/m51_1094_pagamma_multidimensional_snr_map.fits",
]


def print_header_value(header, key):
    value = header.get(key, "<not present>")
    print(f"  {key:20s}: {value}")


def summarize_array(data):
    arr = np.asarray(data, dtype=float)

    finite = np.isfinite(arr)

    if not np.any(finite):
        print("  No finite values.")
        return

    values = arr[finite]

    positive = values[values > 0]
    negative = values[values < 0]
    zero = np.count_nonzero(values == 0)

    print(f"  Shape                 : {arr.shape}")
    print(f"  dtype                 : {arr.dtype}")
    print(f"  finite pixels         : {finite.sum()}")
    print(f"  NaN/inf pixels        : {(~finite).sum()}")
    print(f"  zero pixels           : {zero}")
    print(f"  positive pixels       : {positive.size}")
    print(f"  negative pixels       : {negative.size}")

    print(f"  min                   : {values.min():.8g}")
    print(f"  max                   : {values.max():.8g}")
    print(f"  median                : {np.median(values):.8g}")
    print(f"  mean                  : {np.mean(values):.8g}")

    if positive.size:
        print(f"  positive minimum      : {positive.min():.8g}")
        print(f"  positive median       : {np.median(positive):.8g}")
        print(f"  positive maximum      : {positive.max():.8g}")


def inspect_wcs(header):

    try:
        wcs = WCS(header)

        print("\nWCS:")
        print(f"  pixel dimensions      : {wcs.pixel_n_dim}")
        print(f"  world dimensions      : {wcs.world_n_dim}")

        for i, name in enumerate(wcs.world_axis_names):
            print(f"  world axis {i}        : {name}")

        for i, ctype in enumerate(wcs.wcs.ctype):
            print(f"  CTYPE{i + 1}             : {ctype}")

        for i, cunit in enumerate(wcs.wcs.cunit):
            print(f"  CUNIT{i + 1}             : {cunit}")

        print(f"  CRVAL                 : {wcs.wcs.crval}")
        print(f"  CRPIX                 : {wcs.wcs.crpix}")

    except Exception as exc:
        print(f"\nWCS could not be interpreted: {exc}")


def inspect_file(path):

    print("\n")
    print("=" * 78)
    print("FILE")
    print("=" * 78)
    print(path)

    if not path.exists():
        print("\nSTATUS:")
        print("  FILE DOES NOT EXIST")
        return

    with fits.open(path) as hdul:

        print("\nHDU LIST")
        print("-" * 78)

        hdul.info()

        for hdu_index, hdu in enumerate(hdul):

            print("\n")
            print("-" * 78)
            print(f"HDU {hdu_index}: {hdu.name}")
            print("-" * 78)

            header = hdu.header

            print("\nImportant metadata:")

            for key in [
                "EXTNAME",
                "EXTVER",
                "BUNIT",
                "BTYPE",
                "DATAMODL",
                "TELESCOP",
                "INSTRUME",
                "DETECTOR",
                "EXP_TYPE",
                "FILTER",
                "PUPIL",
                "GRATING",
                "SRCTYPE",
                "SRCTYAPT",
                "TARG_RA",
                "TARG_DEC",
                "RA_REF",
                "DEC_REF",
                "CRVAL1",
                "CRVAL2",
                "CRVAL3",
                "CRPIX1",
                "CRPIX2",
                "CRPIX3",
                "CTYPE1",
                "CTYPE2",
                "CTYPE3",
                "CUNIT1",
                "CUNIT2",
                "CUNIT3",
                "CDELT1",
                "CDELT2",
                "CDELT3",
            ]:
                if key in header:
                    print_header_value(header, key)

            if hdu.data is None:
                print("\nDATA:")
                print("  None")
                continue

            print("\nDATA SUMMARY")
            print("-" * 78)

            if isinstance(hdu, fits.ImageHDU) or isinstance(hdu, fits.PrimaryHDU):

                try:
                    summarize_array(hdu.data)
                except Exception as exc:
                    print(f"  Could not summarize image: {exc}")

                inspect_wcs(header)

            elif isinstance(hdu, fits.BinTableHDU):

                print("  Binary table")
                print(f"  Rows                  : {len(hdu.data)}")
                print(f"  Columns               : {len(hdu.columns)}")

                for col in hdu.columns:
                    print(
                        f"    {col.name:25s}"
                        f" format={col.format}"
                        f" unit={col.unit}"
                    )


def main():

    print("=" * 78)
    print("M51 JWST NIRSPEC — Pa-GAMMA FITS PRODUCT AUDIT")
    print("=" * 78)

    print(
        """
Purpose:
Determine which local FITS product contains the actual spatial
Pa-gamma line flux and whether it is suitable for the
69-pixel JWST aperture analysis.

This script DOES NOT:
  - calculate extinction
  - modify any FITS files
  - select a preferred Pa-gamma product automatically
  - alter the Storey-Hummer analysis

It only audits the available products.
"""
    )

    for path in FILES:
        inspect_file(path)

    print("\n")
    print("=" * 78)
    print("TARGETED COMPARISON")
    print("=" * 78)

    print(
        """
For the subsequent spatial extinction experiment we need a map
whose pixel values represent measured Pa-gamma line flux.

The preferred candidate should have:

  1. spatial shape = (97, 125)
  2. flux-like numerical values
  3. appropriate physical units
  4. celestial WCS matching the S3D cube
  5. provenance showing that it is a Pa-gamma line map
  6. values that can legitimately be summed over the 69-pixel aperture

A consistency, S/N, residual, model, or ratio map must NOT be
treated as a Pa-gamma flux map merely because it has the same
spatial dimensions.
"""
    )

    print("=" * 78)
    print("AUDIT COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()
