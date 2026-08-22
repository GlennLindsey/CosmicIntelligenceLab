#!/usr/bin/env python3

"""
M51 JWST NIRSpec/IFU — Pa-gamma LINE-MAP PROVENANCE AUDIT

Purpose
-------
Determine how the local Pa-gamma spatial line map was produced and
whether it is suitable for quantitative Pa-beta / Pa-gamma flux-ratio
analysis.

This audit examines:

1. Pa-gamma FITS header and numerical properties
2. Pa-beta FITS header and numerical properties
3. Project scripts containing the Pa-gamma filename
4. Project scripts containing PAGAMMA_LINE
5. Project scripts containing the Pa-beta map filename
6. Relevant source-code context
7. Whether the two line maps appear to be constructed comparably
8. Possible units/conversion operations
9. Spectral fitting/integration methodology, if recoverable

IMPORTANT
---------
This script does NOT:

- modify FITS files
- modify scripts
- calculate extinction
- calculate a Pa-beta / Pa-gamma ratio
- declare a product scientifically valid unless the evidence supports it

If provenance cannot be recovered locally, the script reports that
explicitly.
"""

from pathlib import Path
import re
import textwrap

import numpy as np
from astropy.io import fits


# ---------------------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------------------

BASE = Path.home() / "Projects/cosmic_ai"

PAGAMMA = (
    BASE
    / "data/atomic_lines/m51_1094_pagamma_multidimensional_line_map.fits"
)

PAGAMMA_SNR = (
    BASE
    / "data/atomic_lines/m51_1094_pagamma_multidimensional_snr_map.fits"
)

PABETA = (
    BASE
    / "data/atomic_lines/m51_1284_pabeta_spatial_line_map.fits"
)

PAGAMMA_CONSISTENCY = (
    BASE
    / "data/atomic_lines/m51_1096_pagamma_hydrogen_consistency_map.fits"
)


# Search locations
SEARCH_ROOTS = [
    BASE / "scripts",
    BASE,
]


# Filename-related search strings
SEARCH_TERMS = [
    "m51_1094_pagamma_multidimensional_line_map",
    "PAGAMMA_LINE",
    "m51_1094_pagamma",
    "m51_1284_pabeta_spatial_line_map",
    "pabeta_spatial_line_map",
    "pagamma",
    "Pa-gamma",
    "Paγ",
]


# ---------------------------------------------------------------------
# GENERAL UTILITIES
# ---------------------------------------------------------------------

def section(title):
    print("\n")
    print("=" * 78)
    print(title)
    print("=" * 78)


def subsection(title):
    print("\n")
    print("-" * 78)
    print(title)
    print("-" * 78)


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def summarize_array(data):
    """Print numerical summary of an image."""

    arr = np.asarray(data, dtype=float)

    finite = np.isfinite(arr)

    print(f"Shape                    : {arr.shape}")
    print(f"Finite values            : {finite.sum()}")
    print(f"NaN/Inf                  : {(~finite).sum()}")

    if not np.any(finite):
        print("No finite values.")
        return

    values = arr[finite]

    positive = values[values > 0]
    negative = values[values < 0]

    print(f"Zeros                    : {np.count_nonzero(values == 0)}")
    print(f"Positive                 : {positive.size}")
    print(f"Negative                 : {negative.size}")

    print(f"Minimum                  : {values.min():.10g}")
    print(f"Maximum                  : {values.max():.10g}")
    print(f"Mean                     : {values.mean():.10g}")
    print(f"Median                   : {np.median(values):.10g}")

    if positive.size:
        print(f"Positive median          : {np.median(positive):.10g}")

    if negative.size:
        print(f"Negative median          : {np.median(negative):.10g}")


def print_header_keywords(header):
    """Print potentially useful provenance and unit keywords."""

    interesting_groups = {
        "Identity": [
            "EXTNAME",
            "EXTVER",
            "BUNIT",
            "BTYPE",
            "DATAMODL",
            "AUTHOR",
            "ORIGIN",
            "DATE",
        ],

        "JWST": [
            "TELESCOP",
            "INSTRUME",
            "DETECTOR",
            "EXP_TYPE",
            "FILTER",
            "PUPIL",
            "GRATING",
            "SRCTYPE",
        ],

        "Target": [
            "TARGNAME",
            "TARG_RA",
            "TARG_DEC",
            "RA_REF",
            "DEC_REF",
        ],

        "Spectral": [
            "CRVAL3",
            "CRPIX3",
            "CDELT3",
            "CTYPE3",
            "CUNIT3",
            "WAVSTART",
            "WAVEND",
        ],

        "Spatial": [
            "CRVAL1",
            "CRVAL2",
            "CRPIX1",
            "CRPIX2",
            "CDELT1",
            "CDELT2",
            "CTYPE1",
            "CTYPE2",
            "CUNIT1",
            "CUNIT2",
        ],

        "Processing": [
            "PIPELINE",
            "VERSION",
            "SOFTWARE",
            "CREATOR",
            "HISTORY",
        ],
    }

    for group, keys in interesting_groups.items():

        found = []

        for key in keys:
            if key in header:
                found.append((key, header[key]))

        if found:
            print(f"\n{group}:")

            for key, value in found:
                print(f"  {key:20s}: {value}")


def audit_fits(path, label):
    """Audit a FITS image."""

    section(f"{label}")

    print(f"File:\n  {path}")

    if not path.exists():
        print("\nSTATUS:")
        print("  FILE DOES NOT EXIST")
        return None

    with fits.open(path) as hdul:

        print("\nHDU LIST")
        hdul.info()

        for index, hdu in enumerate(hdul):

            subsection(f"HDU {index}: {hdu.name}")

            print_header_keywords(hdu.header)

            if hdu.data is None:
                print("\nDATA:")
                print("  None")
                continue

            print("\nDATA:")
            summarize_array(hdu.data)

            print("\nArray dtype:")
            print(f"  {hdu.data.dtype}")

            print("\nArray dimensionality:")
            print(f"  ndim = {hdu.data.ndim}")

    return True


# ---------------------------------------------------------------------
# HISTORY EXTRACTION
# ---------------------------------------------------------------------

def print_history(path):
    """Print FITS HISTORY cards."""

    section(f"FITS HISTORY — {path.name}")

    if not path.exists():
        print("File does not exist.")
        return

    with fits.open(path) as hdul:

        found = False

        for hdu_index, hdu in enumerate(hdul):

            history = hdu.header.get("HISTORY")

            if history is None:
                continue

            found = True

            print(f"\nHDU {hdu_index} ({hdu.name}):")

            if isinstance(history, str):
                print(f"  {history}")
            else:
                for item in history:
                    print(f"  {item}")

        if not found:
            print("No HISTORY cards found.")


# ---------------------------------------------------------------------
# PROJECT SOURCE SEARCH
# ---------------------------------------------------------------------

def source_files():

    files = []

    for root in SEARCH_ROOTS:

        if not root.exists():
            continue

        for path in root.rglob("*"):

            if not path.is_file():
                continue

            # Ignore common non-source / huge data locations
            if any(
                part in {
                    ".git",
                    "__pycache__",
                    ".ipynb_checkpoints",
                    "node_modules",
                }
                for part in path.parts
            ):
                continue

            suffix = path.suffix.lower()

            if suffix in {
                ".py",
                ".ipynb",
                ".md",
                ".txt",
                ".sh",
            }:
                files.append(path)

    # Remove duplicates
    return sorted(set(files))


def search_project(term):
    """Search text files in the project for a term."""

    matches = []

    for path in source_files():

        try:
            text = path.read_text(
                encoding="utf-8",
                errors="ignore",
            )
        except Exception:
            continue

        if term.lower() not in text.lower():
            continue

        lines = text.splitlines()

        for lineno, line in enumerate(lines, start=1):

            if term.lower() in line.lower():

                matches.append(
                    (
                        path,
                        lineno,
                        line.strip(),
                    )
                )

    return matches


def print_search_results(term, matches):

    print(f"\nSEARCH TERM:")
    print(f"  {term}")

    print(f"Matches:")
    print(f"  {len(matches)}")

    for path, lineno, line in matches[:100]:

        try:
            relative = path.relative_to(BASE)
        except ValueError:
            relative = path

        print(
            f"  {relative}:{lineno}: {line}"
        )


# ---------------------------------------------------------------------
# SOURCE CONTEXT
# ---------------------------------------------------------------------

def print_source_context(path, lineno, radius=8):

    try:
        lines = path.read_text(
            encoding="utf-8",
            errors="ignore",
        ).splitlines()

    except Exception:
        return

    start = max(1, lineno - radius)
    end = min(len(lines), lineno + radius)

    print(
        f"\nSOURCE CONTEXT: "
        f"{path.relative_to(BASE)}:{lineno}"
    )

    for number in range(start, end + 1):

        marker = ">>>" if number == lineno else "   "

        print(
            f"{marker} {number:5d} | {lines[number - 1]}"
        )


# ---------------------------------------------------------------------
# CODE ANALYSIS
# ---------------------------------------------------------------------

def analyze_source_line(line):

    lower = line.lower()

    indicators = []

    tests = {
        "spectral extraction": [
            "spectral",
            "spectrum",
            "wave",
            "wavelength",
        ],

        "line fitting": [
            "fit",
            "curve_fit",
            "least_squares",
            "gaussian",
            "model",
        ],

        "integration": [
            "integrate",
            "trapz",
            "trapezoid",
            "sum(",
            "np.sum",
        ],

        "continuum": [
            "continuum",
            "baseline",
            "background",
        ],

        "unit conversion": [
            "mjy",
            "jy",
            "erg",
            "sr",
            "arcsec",
            "conversion",
            "unit",
        ],

        "Pa-gamma": [
            "pagamma",
            "pa-gamma",
            "paγ",
        ],

        "Pa-beta": [
            "pabeta",
            "pa-beta",
            "paβ",
        ],
    }

    for label, patterns in tests.items():

        if any(pattern in lower for pattern in patterns):
            indicators.append(label)

    return indicators


def inspect_relevant_source_context():

    section("SOURCE-CODE PROVENANCE ANALYSIS")

    interesting_hits = []

    for term in SEARCH_TERMS:

        matches = search_project(term)

        for path, lineno, line in matches:

            indicators = analyze_source_line(line)

            interesting_hits.append(
                (
                    path,
                    lineno,
                    line,
                    indicators,
                )
            )

    # Deduplicate
    unique = {}

    for item in interesting_hits:

        key = (
            str(item[0]),
            item[1],
        )

        unique[key] = item

    interesting_hits = list(unique.values())

    if not interesting_hits:

        print(
            """
No relevant source-code matches were found.

This means the local project does not currently expose
the code that generated the Pa-gamma map.
"""
        )

        return

    # Prefer Python source
    interesting_hits.sort(
        key=lambda x: (
            0 if x[0].suffix == ".py" else 1,
            str(x[0]),
            x[1],
        )
    )

    print(
        f"Potential provenance matches: "
        f"{len(interesting_hits)}"
    )

    for path, lineno, line, indicators in interesting_hits[:100]:

        try:
            relative = path.relative_to(BASE)
        except ValueError:
            relative = path

        print(
            f"\n{relative}:{lineno}"
        )

        if indicators:
            print(
                "  Indicators: "
                + ", ".join(indicators)
            )

        print(
            f"  {line}"
        )


# ---------------------------------------------------------------------
# PA-BETA / PA-GAMMA COMPARISON
# ---------------------------------------------------------------------

def compare_maps():

    section("PA-BETA / PA-GAMMA MAP COMPARISON")

    if not PABETA.exists():
        print("Pa-beta map does not exist.")
        return

    if not PAGAMMA.exists():
        print("Pa-gamma map does not exist.")
        return

    with fits.open(PABETA) as pb_hdul:
        pb = pb_hdul[0]
        pb_data = np.asarray(pb.data, dtype=float)
        pb_header = pb.header

    with fits.open(PAGAMMA) as pg_hdul:
        pg = pg_hdul[0]
        pg_data = np.asarray(pg.data, dtype=float)
        pg_header = pg.header

    print("\nPa-beta:")
    print(f"  shape = {pb_data.shape}")
    print(f"  BUNIT = {pb_header.get('BUNIT')}")
    print(f"  EXTNAME = {pb_header.get('EXTNAME')}")

    print("\nPa-gamma:")
    print(f"  shape = {pg_data.shape}")
    print(f"  BUNIT = {pg_header.get('BUNIT')}")
    print(f"  EXTNAME = {pg_header.get('EXTNAME')}")

    print("\nShape comparison:")
    print(
        f"  Same shape = "
        f"{pb_data.shape == pg_data.shape}"
    )

    print("\nWCS comparison:")

    for key in [
        "CRVAL1",
        "CRVAL2",
        "CRPIX1",
        "CRPIX2",
        "CDELT1",
        "CDELT2",
        "CTYPE1",
        "CTYPE2",
        "CUNIT1",
        "CUNIT2",
    ]:

        pb_value = pb_header.get(key, "<missing>")
        pg_value = pg_header.get(key, "<missing>")

        same = pb_value == pg_value

        print(
            f"  {key:10s}: "
            f"Pa-beta={pb_value} | "
            f"Pa-gamma={pg_value} | "
            f"same={same}"
        )

    print("\nUnit comparison:")

    pb_unit = pb_header.get("BUNIT")
    pg_unit = pg_header.get("BUNIT")

    print(f"  Pa-beta  BUNIT = {pb_unit}")
    print(f"  Pa-gamma BUNIT = {pg_unit}")

    if pb_unit == pg_unit:
        print("  Units match.")
    else:
        print(
            "  WARNING: Units do NOT match."
        )

    print("\nFinite-pixel comparison:")

    pb_finite = np.isfinite(pb_data)
    pg_finite = np.isfinite(pg_data)

    print(
        f"  Pa-beta finite  = {pb_finite.sum()}"
    )

    print(
        f"  Pa-gamma finite = {pg_finite.sum()}"
    )

    print(
        f"  Both finite     = "
        f"{np.count_nonzero(pb_finite & pg_finite)}"
    )


# ---------------------------------------------------------------------
# FITS HEADER COMMENT / HISTORY SEARCH
# ---------------------------------------------------------------------

def search_header_text(path, terms):

    section(
        f"HEADER TEXT SEARCH — {path.name}"
    )

    if not path.exists():
        print("File does not exist.")
        return

    with fits.open(path) as hdul:

        for hdu_index, hdu in enumerate(hdul):

            header = hdu.header

            print(f"\nHDU {hdu_index}: {hdu.name}")

            found = False

            for card in header.cards:

                text = str(card).upper()

                if any(
                    term.upper() in text
                    for term in terms
                ):

                    print(
                        f"  {card.keyword:12s} "
                        f"{card.value}"
                    )

                    found = True

            if not found:
                print("  No matching header text.")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():

    section(
        "M51 JWST NIRSPEC/IFU — Pa-GAMMA LINE-MAP PROVENANCE AUDIT"
    )

    print(
        """
Purpose:
Determine how the local Pa-gamma line map was created and
whether its values can legitimately be treated as measured
spatial Pa-gamma line flux.

Primary candidate:
  m51_1094_pagamma_multidimensional_line_map.fits

This is a provenance audit only.
No extinction calculation is performed.
"""
    )

    # ---------------------------------------------------------------
    # 1. Primary Pa-gamma product
    # ---------------------------------------------------------------

    audit_fits(
        PAGAMMA,
        "1. PRIMARY Pa-GAMMA LINE MAP",
    )

    # ---------------------------------------------------------------
    # 2. FITS history
    # ---------------------------------------------------------------

    print_history(PAGAMMA)

    # ---------------------------------------------------------------
    # 3. Header text search
    # ---------------------------------------------------------------

    search_header_text(
        PAGAMMA,
        [
            "PAGAMMA",
            "LINE",
            "FIT",
            "FLUX",
            "CONTINUUM",
            "BACKGROUND",
            "SPECTR",
            "WAVE",
            "M51",
        ],
    )

    # ---------------------------------------------------------------
    # 4. Pa-beta comparison
    # ---------------------------------------------------------------

    audit_fits(
        PABETA,
        "4. Pa-BETA LINE MAP",
    )

    compare_maps()

    # ---------------------------------------------------------------
    # 5. SNR product
    # ---------------------------------------------------------------

    audit_fits(
        PAGAMMA_SNR,
        "5. Pa-GAMMA SNR MAP",
    )

    # ---------------------------------------------------------------
    # 6. Consistency product
    # ---------------------------------------------------------------

    audit_fits(
        PAGAMMA_CONSISTENCY,
        "6. Pa-GAMMA HYDROGEN-CONSISTENCY MAP",
    )

    # ---------------------------------------------------------------
    # 7. Search source tree
    # ---------------------------------------------------------------

    section("7. PROJECT-WIDE PROVENANCE SEARCH")

    print(
        f"Search roots:"
    )

    for root in SEARCH_ROOTS:
        print(f"  {root}")

    print(
        f"\nSource files inspected:"
        f" {len(source_files())}"
    )

    for term in SEARCH_TERMS:

        matches = search_project(term)

        print_search_results(
            term,
            matches,
        )

    # ---------------------------------------------------------------
    # 8. Analyze source context
    # ---------------------------------------------------------------

    inspect_relevant_source_context()

    # ---------------------------------------------------------------
    # 9. Final scientific assessment
    # ---------------------------------------------------------------

    section("FINAL PROVENANCE ASSESSMENT")

    print(
        """
The Pa-gamma map should be considered suitable for quantitative
Pa-beta / Pa-gamma flux analysis ONLY if the local evidence shows
that its pixel values represent measured line emission in a
consistent quantitative sense.

Evidence required:

  [ ] Explicit Pa-gamma line-map construction
  [ ] Spectral extraction or line fitting identified
  [ ] Continuum/background treatment identified
  [ ] Flux/integrated-line calculation identified
  [ ] Units or normalization identified
  [ ] Pa-beta and Pa-gamma construction shown to be comparable
  [ ] Spatial WCS verified
  [ ] No indication that the values are merely ratios,
      residuals, model values, or consistency diagnostics

If one or more of these cannot be recovered, the result should
remain classified as:

  PROVENANCE INCOMPLETE

rather than assuming that the map is a calibrated line-flux map.

Do NOT rerun the Storey-Hummer extinction analysis until this
assessment is complete.
"""
    )

    section("AUDIT COMPLETE")


if __name__ == "__main__":
    main()
