#!/usr/bin/env python3

"""
M51 JWST NIRSPEC — Pa-GAMMA SPECTRAL-WINDOW ROBUSTNESS TEST

Purpose
-------
Determine how sensitive the reconstructed Pa-gamma aperture flux is
to the spectral extraction window used on the JWST S3D cube.

The experiment:

1. Reads the JWST S3D SCI and ERR cubes.
2. Locates the spectral WCS directly from the FITS HDU headers.
3. Predicts the observed Pa-gamma wavelength for M51.
4. Builds a continuum-subtracted S3D cube.
5. Extracts Pa-gamma using several spectral windows:
       +/- 0 planes
       +/- 1 planes
       +/- 2 planes
       +/- 3 planes
       +/- 4 planes
6. Uses exactly the 69-pixel nominal JWST aperture.
7. Calculates integrated Pa-gamma flux for every window.
8. Compares each extraction with the existing questionable
   Pa-gamma product.
9. Calculates Pa-beta / Pa-gamma for each window.
10. Calculates diagnostic Storey-Hummer A(V) values.
11. Saves tables and figures.

IMPORTANT
---------
This script does not decide which Pa-gamma extraction is correct.

It establishes the stability of the direct S3D reconstruction.

The existing Pa-gamma product is used only for comparison.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.io import fits


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = (
    Path.home()
    / "Projects"
    / "cosmic_ai"
)

S3D_PATH = (
    PROJECT_DIR
    / "data/m51_jwst_level3/"
      "jw03435-o006_t010_nirspec_g140m-f100lp_s3d.fits"
)

APERTURE_PATH = (
    PROJECT_DIR
    / "data/atomic_lines/"
      "m51_jwst_extraction_aperture.csv"
)

PABETA_MAP_PATH = (
    PROJECT_DIR
    / "data/atomic_lines/"
      "m51_1284_pabeta_multidimensional_line_map.fits"
)

EXISTING_PAGAMMA_PATH = (
    PROJECT_DIR
    / "data/atomic_lines/"
      "m51_1096_pagamma_hydrogen_consistency_map.fits"
)

STOREY_HUMMER_PATH = (
    PROJECT_DIR
    / "data/atomic_lines/"
      "m51_hydrogen_required_av.csv"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "data/atomic_lines"
)

RESULTS_CSV = (
    OUTPUT_DIR
    / "m51_pagamma_spectral_window_robustness.csv"
)

PROFILE_CSV = (
    OUTPUT_DIR
    / "m51_pagamma_aperture_spectral_profile.csv"
)

SUMMARY_CSV = (
    OUTPUT_DIR
    / "m51_pagamma_spectral_window_summary.csv"
)

FIGURE_PATH = (
    PROJECT_DIR
    / "m51_pagamma_spectral_window_robustness.png"
)

PROFILE_FIGURE_PATH = (
    PROJECT_DIR
    / "m51_pagamma_aperture_spectral_profile.png"
)


# ============================================================
# SCIENTIFIC CONSTANTS
# ============================================================

PA_BETA_REST_NM = 1281.807000
PA_GAMMA_REST_NM = 1093.800000

M51_VELOCITY_KMS = 463.0

RESOLVING_POWER = 2700.0

# Number of spectral planes on either side
# of the nearest Pa-gamma plane.
WINDOW_PLANES = [0, 1, 2, 3, 4]

BLUE_CONTINUUM = (
    1080.0,
    1088.0,
)

RED_CONTINUUM = (
    1100.0,
    1108.0,
)

CHAOS_AV = 2.237203


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def banner(text):

    print()
    print("=" * 70)
    print(text)
    print("=" * 70)


def predicted_wavelength(
    rest_nm,
    velocity_kms,
):

    c_kms = 299792.458

    return (
        rest_nm
        * (
            1.0
            + velocity_kms / c_kms
        )
    )


def velocity_from_wavelength(
    rest_nm,
    observed_nm,
):

    c_kms = 299792.458

    return (
        (
            observed_nm / rest_nm
        )
        - 1.0
    ) * c_kms


# ============================================================
# SPECTRAL WCS
# ============================================================

def find_spectral_header(hdul):

    """
    Search every HDU for the spectral WCS.

    We specifically require:
        CRVAL3
        CRPIX3
        CDELT3

    This avoids assuming that the SCI header contains the
    spectral axis.
    """

    for index, hdu in enumerate(hdul):

        header = hdu.header

        if all(
            key in header
            for key in (
                "CRVAL3",
                "CRPIX3",
                "CDELT3",
            )
        ):

            print(
                f"Spectral WCS found in HDU "
                f"{index}: "
                f"{hdu.name}"
            )

            return header.copy()

    return None

def wavelength_from_header(
    header,
    nplanes,
):
    """
    Construct the spectral wavelength array from the JWST S3D
    spectral WCS.

    For this JWST S3D cube, CRVAL3 and CDELT3 are expressed
    in microns.

    Return wavelengths in nanometres.
    """

    crval3 = float(header["CRVAL3"])
    crpix3 = float(header["CRPIX3"])
    cdelt3 = float(header["CDELT3"])

    pixel = np.arange(nplanes, dtype=float) + 1.0

    wavelength_um = (
        crval3
        + (pixel - crpix3) * cdelt3
    )

    # JWST spectral WCS is in microns.
    # Convert microns -> nanometres.
    wavelength_nm = wavelength_um * 1000.0

    return wavelength_nm

    """
    Construct the spectral wavelength array from the JWST S3D
    spectral WCS.

    JWST S3D CRVAL3/CDELT3 are stored in metres for this cube.
    Return wavelengths in nanometres.
    """

    crval3 = float(
        header["CRVAL3"]
    )

    crpix3 = float(
        header["CRPIX3"]
    )

    cdelt3 = float(
        header["CDELT3"]
    )

    pixel = (
        np.arange(nplanes)
        + 1.0
    )

    wavelength_m = (
        crval3
        + (
            pixel - crpix3
        )
        * cdelt3
    )

    # FITS/JWST spectral WCS is in metres.
    # Convert metres -> nanometres.
    wavelength_nm = (
        wavelength_m * 1.0e9
    )

    return wavelength_nm

    """
    Construct a linear wavelength array from the spectral WCS.

    JWST S3D spectral WCS values are normally stored in metres.
    The returned array is always in nanometres.
    """

    crval3 = float(
        header["CRVAL3"]
    )

    crpix3 = float(
        header["CRPIX3"]
    )

    cdelt3 = float(
        header["CDELT3"]
    )

    pixel = (
        np.arange(nplanes)
        + 1.0
    )

    wavelength = (
        crval3
        + (
            pixel - crpix3
        )
        * cdelt3
    )

    # Convert metres -> nm when necessary.
    if np.nanmedian(
        np.abs(wavelength)
    ) < 1.0:

        wavelength *= 1e9

    return wavelength


# ============================================================
# APERTURE
# ============================================================

def load_aperture():

    df = pd.read_csv(
        APERTURE_PATH
    )

    required = {
        "x_pixel",
        "y_pixel",
        "inside_nominal_aperture",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:

        raise RuntimeError(
            "Aperture file is missing "
            f"columns: {missing}"
        )

    aperture = df[
        df[
            "inside_nominal_aperture"
        ].astype(bool)
    ].copy()

    aperture["x_pixel"] = (
        aperture[
            "x_pixel"
        ].astype(int)
    )

    aperture["y_pixel"] = (
        aperture[
            "y_pixel"
        ].astype(int)
    )

    return aperture


def aperture_values(
    map_data,
    aperture,
):

    values = []

    for _, row in aperture.iterrows():

        x = int(
            row["x_pixel"]
        )

        y = int(
            row["y_pixel"]
        )

        values.append(
            map_data[y, x]
        )

    return np.asarray(
        values,
        dtype=float,
    )


def aperture_statistics(
    map_data,
    aperture,
):

    values = aperture_values(
        map_data,
        aperture,
    )

    finite = np.isfinite(
        values
    )

    positive = (
        finite
        & (values > 0)
    )

    return {
        "values": values,
        "finite": int(
            np.sum(finite)
        ),
        "positive": int(
            np.sum(positive)
        ),
        "sum_finite": float(
            np.nansum(values)
        ),
        "sum_positive": float(
            np.nansum(
                np.where(
                    positive,
                    values,
                    0.0,
                )
            )
        ),
    }


# ============================================================
# CONTINUUM
# ============================================================

def build_continuum_cube(
    wavelength_nm,
    cube,
):

    blue_idx = np.where(
        (
            wavelength_nm
            >= BLUE_CONTINUUM[0]
        )
        &
        (
            wavelength_nm
            <= BLUE_CONTINUUM[1]
        )
    )[0]

    red_idx = np.where(
        (
            wavelength_nm
            >= RED_CONTINUUM[0]
        )
        &
        (
            wavelength_nm
            <= RED_CONTINUUM[1]
        )
    )[0]

    if len(blue_idx) == 0:

        raise RuntimeError(
            "No blue continuum planes found."
        )

    if len(red_idx) == 0:

        raise RuntimeError(
            "No red continuum planes found."
        )

    blue_wave = float(
        np.nanmedian(
            wavelength_nm[
                blue_idx
            ]
        )
    )

    red_wave = float(
        np.nanmedian(
            wavelength_nm[
                red_idx
            ]
        )
    )

    blue_continuum = (
        np.nanmedian(
            cube[
                blue_idx
            ],
            axis=0,
        )
    )

    red_continuum = (
        np.nanmedian(
            cube[
                red_idx
            ],
            axis=0,
        )
    )

    continuum_cube = (
        np.empty_like(
            cube,
            dtype=float,
        )
    )

    for index, wave in enumerate(
        wavelength_nm
    ):

        fraction = (
            (
                wave
                - blue_wave
            )
            /
            (
                red_wave
                - blue_wave
            )
        )

        continuum_cube[index] = (
            blue_continuum
            + fraction
            * (
                red_continuum
                - blue_continuum
            )
        )

    return (
        continuum_cube,
        blue_idx,
        red_idx,
    )


# ============================================================
# LINE EXTRACTION
# ============================================================

def extract_line_map(
    residual_cube,
    wavelength_nm,
    center_index,
    half_width_planes,
):

    first = max(
        0,
        center_index
        - half_width_planes,
    )

    last = min(
        len(wavelength_nm) - 1,
        center_index
        + half_width_planes,
    )

    indices = np.arange(
        first,
        last + 1,
    )

    if len(wavelength_nm) > 1:

        spacing = float(
            np.nanmedian(
                np.diff(
                    wavelength_nm
                )
            )
        )

    else:

        spacing = 1.0

    line_map = (
        np.nansum(
            residual_cube[
                indices
            ],
            axis=0,
        )
        * spacing
    )

    return (
        line_map,
        indices,
        spacing,
    )


# ============================================================
# STOREY-HUMMER
# ============================================================

def calculate_required_av(
    observed_ratio,
    intrinsic_ratio,
):

    A_BETA_OVER_V = 0.270821
    A_GAMMA_OVER_V = 0.349594

    delta = (
        A_GAMMA_OVER_V
        - A_BETA_OVER_V
    )

    return (
        np.log10(
            observed_ratio
            / intrinsic_ratio
        )
        /
        (
            0.4
            * delta
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    banner(
        "M51 JWST NIRSPEC — "
        "Pa-GAMMA SPECTRAL-WINDOW ROBUSTNESS TEST"
    )

    print(
        "Purpose:"
    )

    print(
        "Determine whether the direct Pa-gamma "
        "S3D reconstruction is stable against "
        "reasonable spectral extraction windows."
    )

    print()
    print(
        "The existing Pa-gamma product is treated "
        "only as a comparison product."
    )


    # ========================================================
    # 1. READ S3D
    # ========================================================

    banner(
        "1. READING JWST S3D"
    )

    with fits.open(
        S3D_PATH
    ) as hdul:

        sci = (
            hdul["SCI"]
            .data
            .astype(float)
        )

        err = (
            hdul["ERR"]
            .data
            .astype(float)
        )

        spectral_header = (
            find_spectral_header(
                hdul
            )
        )

    if spectral_header is None:

        raise RuntimeError(
            "Could not locate a FITS HDU "
            "containing CRVAL3/CRPIX3/CDELT3."
        )

    print()
    print(
        f"File:"
    )
    print(
        f"  {S3D_PATH}"
    )

    print()
    print(
        f"SCI shape:"
    )
    print(
        f"  {sci.shape}"
    )

    print()
    print(
        f"ERR shape:"
    )
    print(
        f"  {err.shape}"
    )


    # ========================================================
    # 2. WAVELENGTH
    # ========================================================

    banner(
        "2. BUILDING WAVELENGTH ARRAY"
    )

    nplanes = sci.shape[0]

    wavelength_nm = (
        wavelength_from_header(
            spectral_header,
            nplanes,
        )
    )

    spacing = float(
        np.nanmedian(
            np.diff(
                wavelength_nm
            )
        )
    )

    print(
        f"Wavelength range = "
        f"{wavelength_nm[0]:.9f}"
        f" - "
        f"{wavelength_nm[-1]:.9f} nm"
    )

    print(
        f"Spectral planes = "
        f"{nplanes}"
    )

    print(
        f"Spectral spacing = "
        f"{spacing:.9f} nm"
    )

    print()
    print(
        "Spectral WCS:"
    )

    print(
        f"  CRVAL3 = "
        f"{spectral_header['CRVAL3']}"
    )

    print(
        f"  CRPIX3 = "
        f"{spectral_header['CRPIX3']}"
    )

    print(
        f"  CDELT3 = "
        f"{spectral_header['CDELT3']}"
    )


    # ========================================================
    # 3. PREDICT PA-GAMMA
    # ========================================================

    banner(
        "3. PREDICTING PA-GAMMA WAVELENGTH"
    )

    pagamma_observed = (
        predicted_wavelength(
            PA_GAMMA_REST_NM,
            M51_VELOCITY_KMS,
        )
    )

    pagamma_fwhm = (
        pagamma_observed
        / RESOLVING_POWER
    )

    nearest_index = int(
        np.argmin(
            np.abs(
                wavelength_nm
                - pagamma_observed
            )
        )
    )

    nearest_wave = (
        wavelength_nm[
            nearest_index
        ]
    )

    nearest_velocity = (
        velocity_from_wavelength(
            PA_GAMMA_REST_NM,
            nearest_wave,
        )
    )

    print(
        f"Pa-gamma rest wavelength:"
        f" {PA_GAMMA_REST_NM:.6f} nm"
    )

    print(
        f"M51 velocity:"
        f" {M51_VELOCITY_KMS:+.3f} km/s"
    )

    print(
        f"Predicted observed wavelength:"
        f" {pagamma_observed:.9f} nm"
    )

    print(
        f"Instrumental FWHM:"
        f" {pagamma_fwhm:.9f} nm"
    )

    print(
        f"Nearest cube plane:"
        f" {nearest_index}"
    )

    print(
        f"Nearest cube wavelength:"
        f" {nearest_wave:.9f} nm"
    )

    print(
        f"Velocity at nearest plane:"
        f" {nearest_velocity:+.3f} km/s"
    )

    print()
    print(
        f"Predicted wavelength offset from "
        f"nearest plane:"
    )

    print(
        f"  "
        f"{nearest_wave - pagamma_observed:+.9f} nm"
    )


    # ========================================================
    # 4. CONTINUUM
    # ========================================================

    banner(
        "4. BUILDING CONTINUUM MODEL"
    )

    print(
        f"Blue continuum:"
        f" {BLUE_CONTINUUM[0]:.3f}"
        f" - {BLUE_CONTINUUM[1]:.3f} nm"
    )

    print(
        f"Red continuum:"
        f" {RED_CONTINUUM[0]:.3f}"
        f" - {RED_CONTINUUM[1]:.3f} nm"
    )

    (
        continuum_cube,
        blue_idx,
        red_idx,
    ) = build_continuum_cube(
        wavelength_nm,
        sci,
    )

    residual_cube = (
        sci
        - continuum_cube
    )

    print(
        f"Blue continuum planes:"
        f" {len(blue_idx)}"
    )

    print(
        f"Red continuum planes:"
        f" {len(red_idx)}"
    )


    # ========================================================
    # 5. APERTURE
    # ========================================================

    banner(
        "5. LOADING NOMINAL JWST APERTURE"
    )

    aperture = load_aperture()

    if len(aperture) != 69:

        raise RuntimeError(
            "Expected exactly 69 nominal "
            "aperture pixels."
        )

    print(
        f"Aperture file:"
    )

    print(
        f"  {APERTURE_PATH}"
    )

    print()
    print(
        f"Nominal aperture pixels:"
        f" {len(aperture)}"
    )

    print(
        "Confirmed: exactly 69 pixels."
    )


    # ========================================================
    # 6. PA-BETA
    # ========================================================

    banner(
        "6. LOADING VALIDATED PA-BETA MAP"
    )

    with fits.open(
        PABETA_MAP_PATH
    ) as hdul:

        pabeta_map = (
            hdul[0]
            .data
            .astype(float)
        )

    pabeta_stats = (
        aperture_statistics(
            pabeta_map,
            aperture,
        )
    )

    pabeta_flux = (
        pabeta_stats[
            "sum_positive"
        ]
    )

    print(
        f"File:"
    )

    print(
        f"  {PABETA_MAP_PATH}"
    )

    print()
    print(
        f"Pa-beta finite pixels:"
        f" {pabeta_stats['finite']}"
    )

    print(
        f"Pa-beta positive pixels:"
        f" {pabeta_stats['positive']}"
    )

    print(
        f"Pa-beta aperture flux:"
        f" {pabeta_flux:.8f}"
    )


    # ========================================================
    # 7. EXISTING PA-GAMMA PRODUCT
    # ========================================================

    banner(
        "7. LOADING EXISTING PA-GAMMA PRODUCT"
    )

    with fits.open(
        EXISTING_PAGAMMA_PATH
    ) as hdul:

        existing_map = (
            hdul[0]
            .data
            .astype(float)
        )

    existing_stats = (
        aperture_statistics(
            existing_map,
            aperture,
        )
    )

    existing_flux = (
        existing_stats[
            "sum_positive"
        ]
    )

    print(
        f"File:"
    )

    print(
        f"  {EXISTING_PAGAMMA_PATH}"
    )

    print()
    print(
        f"Existing finite pixels:"
        f" {existing_stats['finite']}"
    )

    print(
        f"Existing positive pixels:"
        f" {existing_stats['positive']}"
    )

    print(
        f"Existing aperture flux:"
        f" {existing_flux:.8f}"
    )


    # ========================================================
    # 8. SPECTRAL WINDOWS
    # ========================================================

    banner(
        "8. TESTING SPECTRAL WINDOWS"
    )

    results = []

    profile_rows = []

    for half_width in WINDOW_PLANES:

        (
            line_map,
            indices,
            window_spacing,
        ) = extract_line_map(
            residual_cube,
            wavelength_nm,
            nearest_index,
            half_width,
        )

        stats = aperture_statistics(
            line_map,
            aperture,
        )

        flux = (
            stats["sum_positive"]
        )

        low_wave = (
            wavelength_nm[
                indices[0]
            ]
        )

        high_wave = (
            wavelength_nm[
                indices[-1]
            ]
        )

        mean_wave = float(
            np.nanmean(
                wavelength_nm[
                    indices
                ]
            )
        )

        mean_velocity = (
            velocity_from_wavelength(
                PA_GAMMA_REST_NM,
                mean_wave,
            )
        )

        ratio = (
            pabeta_flux / flux
            if flux > 0
            else np.nan
        )

        existing_fraction = (
            flux / existing_flux
            if existing_flux > 0
            else np.nan
        )

        results.append(
            {
                "half_width_planes":
                    half_width,

                "number_of_planes":
                    len(indices),

                "first_plane":
                    int(indices[0]),

                "last_plane":
                    int(indices[-1]),

                "low_wavelength_nm":
                    low_wave,

                "high_wavelength_nm":
                    high_wave,

                "mean_wavelength_nm":
                    mean_wave,

                "mean_velocity_kms":
                    mean_velocity,

                "pagamma_flux":
                    flux,

                "finite_pixels":
                    stats["finite"],

                "positive_pixels":
                    stats["positive"],

                "pabeta_flux":
                    pabeta_flux,

                "pabeta_pagamma_ratio":
                    ratio,

                "existing_pagamma_flux":
                    existing_flux,

                "fraction_of_existing":
                    existing_fraction,
            }
        )

        # ----------------------------------------------------
        # Per-plane aperture spectral profile
        # ----------------------------------------------------

        for index in indices:

            values = []

            for _, row in aperture.iterrows():

                x = int(
                    row["x_pixel"]
                )

                y = int(
                    row["y_pixel"]
                )

                values.append(
                    residual_cube[
                        index,
                        y,
                        x,
                    ]
                )

            values = np.asarray(
                values,
                dtype=float,
            )

            finite = np.isfinite(
                values
            )

            if np.any(finite):

                aperture_signal = float(
                    np.nansum(
                        values
                    )
                )

            else:

                aperture_signal = np.nan

            profile_rows.append(
                {
                    "plane_index":
                        int(index),

                    "wavelength_nm":
                        wavelength_nm[index],

                    "velocity_kms":
                        velocity_from_wavelength(
                            PA_GAMMA_REST_NM,
                            wavelength_nm[index],
                        ),

                    "aperture_sum":
                        aperture_signal,

                    "window_half_width":
                        half_width,
                }
            )

        print()
        print(
            f"Window ±{half_width} planes"
        )

        print(
            f"  Number of planes:"
            f" {len(indices)}"
        )

        print(
            f"  Plane indices:"
            f" {indices[0]} - {indices[-1]}"
        )

        print(
            f"  Wavelength:"
            f" {low_wave:.6f}"
            f" - {high_wave:.6f} nm"
        )

        print(
            f"  Mean velocity:"
            f" {mean_velocity:+.3f} km/s"
        )

        print(
            f"  Finite aperture pixels:"
            f" {stats['finite']}"
        )

        print(
            f"  Positive aperture pixels:"
            f" {stats['positive']}"
        )

        print(
            f"  Pa-gamma flux:"
            f" {flux:.8f}"
        )

        print(
            f"  Pa-beta / Pa-gamma:"
            f" {ratio:.8f}"
        )

        print(
            f"  Fraction of existing:"
            f" {existing_fraction:.6f}"
        )


    results_df = pd.DataFrame(
        results
    )

    profile_df = pd.DataFrame(
        profile_rows
    )


    # ========================================================
    # 9. STOREY-HUMMER DIAGNOSTIC
    # ========================================================

    banner(
        "9. STOREY-HUMMER DIAGNOSTIC"
    )

    if STOREY_HUMMER_PATH.exists():

        sh = pd.read_csv(
            STOREY_HUMMER_PATH
        )

        required = {
            "Te",
            "ne",
            "intrinsic_ratio",
        }

        if required.issubset(
            sh.columns
        ):

            print(
                f"Grid:"
            )

            print(
                f"  {STOREY_HUMMER_PATH}"
            )

            print(
                f"Grid points:"
                f" {len(sh)}"
            )

            min_av = []
            max_av = []
            median_av = []
            best_te = []
            best_ne = []
            best_intrinsic = []
            best_av = []

            for _, row in (
                results_df.iterrows()
            ):

                ratio = row[
                    "pabeta_pagamma_ratio"
                ]

                av = calculate_required_av(
                    ratio,
                    sh[
                        "intrinsic_ratio"
                    ].values,
                )

                finite = np.isfinite(
                    av
                )

                if not np.any(
                    finite
                ):

                    min_av.append(
                        np.nan
                    )

                    max_av.append(
                        np.nan
                    )

                    median_av.append(
                        np.nan
                    )

                    best_te.append(
                        np.nan
                    )

                    best_ne.append(
                        np.nan
                    )

                    best_intrinsic.append(
                        np.nan
                    )

                    best_av.append(
                        np.nan
                    )

                    continue

                av_finite = av[
                    finite
                ]

                distance = np.abs(
                    av_finite
                    - CHAOS_AV
                )

                finite_indices = (
                    np.where(finite)[0]
                )

                best_local = int(
                    np.argmin(
                        distance
                    )
                )

                best_index = int(
                    finite_indices[
                        best_local
                    ]
                )

                min_av.append(
                    float(
                        np.nanmin(
                            av
                        )
                    )
                )

                max_av.append(
                    float(
                        np.nanmax(
                            av
                        )
                    )
                )

                median_av.append(
                    float(
                        np.nanmedian(
                            av
                        )
                    )
                )

                best_te.append(
                    sh.iloc[
                        best_index
                    ]["Te"]
                )

                best_ne.append(
                    sh.iloc[
                        best_index
                    ]["ne"]
                )

                best_intrinsic.append(
                    sh.iloc[
                        best_index
                    ]["intrinsic_ratio"]
                )

                best_av.append(
                    float(
                        av[
                            best_index
                        ]
                    )
                )

            results_df[
                "minimum_Av"
            ] = min_av

            results_df[
                "maximum_Av"
            ] = max_av

            results_df[
                "median_Av"
            ] = median_av

            results_df[
                "best_Te"
            ] = best_te

            results_df[
                "best_ne"
            ] = best_ne

            results_df[
                "best_intrinsic_ratio"
            ] = best_intrinsic

            results_df[
                "best_Av"
            ] = best_av

            print()
            print(
                f"CHAOS A(V) = "
                f"{CHAOS_AV:.6f} mag"
            )

            for _, row in (
                results_df.iterrows()
            ):

                print()

                print(
                    f"±{int(row['half_width_planes'])} "
                    f"planes:"
                )

                print(
                    f"  Pa-beta/Pa-gamma = "
                    f"{row['pabeta_pagamma_ratio']:.8f}"
                )

                print(
                    f"  Best Te = "
                    f"{row['best_Te']}"
                )

                print(
                    f"  Best ne = "
                    f"{row['best_ne']:.3e}"
                )

                print(
                    f"  Best A(V) = "
                    f"{row['best_Av']:.6f}"
                )


    # ========================================================
    # 10. SAVE TABLES
    # ========================================================

    banner(
        "10. SAVING RESULTS"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        RESULTS_CSV,
        index=False,
    )

    profile_df.to_csv(
        PROFILE_CSV,
        index=False,
    )

    summary = {
        "predicted_pagamma_nm":
            pagamma_observed,

        "nearest_plane":
            nearest_index,

        "nearest_plane_wavelength_nm":
            nearest_wave,

        "nearest_plane_velocity_kms":
            nearest_velocity,

        "spectral_spacing_nm":
            spacing,

        "instrumental_fwhm_nm":
            pagamma_fwhm,

        "pabeta_aperture_flux":
            pabeta_flux,

        "existing_pagamma_aperture_flux":
            existing_flux,

        "nominal_aperture_pixels":
            len(aperture),
    }

    pd.DataFrame(
        [summary]
    ).to_csv(
        SUMMARY_CSV,
        index=False,
    )

    print(
        f"Results:"
    )

    print(
        f"  {RESULTS_CSV}"
    )

    print(
        f"Profile:"
    )

    print(
        f"  {PROFILE_CSV}"
    )

    print(
        f"Summary:"
    )

    print(
        f"  {SUMMARY_CSV}"
    )


    # ========================================================
    # 11. ROBUSTNESS FIGURE
    # ========================================================

    banner(
        "11. CREATING ROBUSTNESS FIGURE"
    )

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    ax.plot(
        results_df[
            "number_of_planes"
        ],
        results_df[
            "pagamma_flux"
        ],
        marker="o",
        linewidth=2,
        label="Direct S3D reconstruction",
    )

    ax.axhline(
        existing_flux,
        linestyle="--",
        label="Existing Pa-gamma product",
    )

    ax.set_xlabel(
        "Number of spectral planes"
    )

    ax.set_ylabel(
        "Pa-gamma aperture flux"
    )

    ax.set_title(
        "M51 Pa-gamma Spectral-Window Robustness"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend()

    plt.tight_layout()

    plt.savefig(
        FIGURE_PATH,
        dpi=180,
    )

    plt.close()

    print(
        f"Saved:"
    )

    print(
        f"  {FIGURE_PATH}"
    )


    # ========================================================
    # 12. SPECTRAL PROFILE
    # ========================================================

    banner(
        "12. CREATING APERTURE SPECTRAL PROFILE"
    )

    profile_unique = (
        profile_df
        .drop_duplicates(
            subset=[
                "plane_index"
            ]
        )
        .sort_values(
            "plane_index"
        )
    )

    plt.figure(
        figsize=(11, 7)
    )

    plt.plot(
        profile_unique[
            "wavelength_nm"
        ],
        profile_unique[
            "aperture_sum"
        ],
        marker="o",
        linewidth=1.5,
    )

    plt.axvline(
        pagamma_observed,
        linestyle="--",
        label="Predicted Pa-gamma",
    )

    plt.axvline(
        nearest_wave,
        linestyle=":",
        label="Nearest cube plane",
    )

    plt.xlabel(
        "Wavelength (nm)"
    )

    plt.ylabel(
        "Aperture-summed residual"
    )

    plt.title(
        "M51 JWST Pa-gamma Aperture Spectral Profile"
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        PROFILE_FIGURE_PATH,
        dpi=180,
    )

    plt.close()

    print(
        f"Saved:"
    )

    print(
        f"  {PROFILE_FIGURE_PATH}"
    )


    # ========================================================
    # 13. FINAL SUMMARY
    # ========================================================

    banner(
        "FINAL Pa-GAMMA WINDOW ROBUSTNESS RESULT"
    )

    print(
        f"Predicted Pa-gamma:"
        f" {pagamma_observed:.9f} nm"
    )

    print(
        f"Nearest plane:"
        f" {nearest_index}"
    )

    print(
        f"Nearest wavelength:"
        f" {nearest_wave:.9f} nm"
    )

    print(
        f"Spectral spacing:"
        f" {spacing:.9f} nm"
    )

    print(
        f"Instrumental FWHM:"
        f" {pagamma_fwhm:.9f} nm"
    )

    print()
    print(
        f"Nominal aperture:"
        f" {len(aperture)} pixels"
    )

    print()
    print(
        "Window results:"
    )

    for _, row in (
        results_df.iterrows()
    ):

        print(
            f"  ±{int(row['half_width_planes'])}"
            f" planes "
            f"({int(row['number_of_planes'])} total): "
            f"Pa-gamma = "
            f"{row['pagamma_flux']:.6f}, "
            f"Pa-beta/Pa-gamma = "
            f"{row['pabeta_pagamma_ratio']:.6f}"
        )

    print()
    print(
        f"Existing product aperture flux:"
        f" {existing_flux:.6f}"
    )

    print()
    print(
        "Interpretation:"
    )

    print(
        "The spectral-window dependence must be "
        "examined before selecting a Pa-gamma flux "
        "for the extinction analysis."
    )

    print()
    print(
        "Pa-gamma spectral-window robustness "
        "experiment complete."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
