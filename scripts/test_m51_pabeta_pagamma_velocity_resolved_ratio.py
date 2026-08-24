#!/usr/bin/env python3

"""
M51 JWST NIRSPEC — VELOCITY-RESOLVED Pa-beta / Pa-gamma RATIO

Purpose
-------
Measure the Pa-beta / Pa-gamma ratio directly as a function of
velocity using the summed 69-pixel JWST nominal aperture.

This is a non-parametric diagnostic.

The experiment does NOT:
    - fit Gaussian velocity components
    - infer extinction
    - use the existing Pa-gamma spatial product as a flux map
    - average pixel-by-pixel ratios

Instead it asks:

    Do Pa-beta and Pa-gamma trace the same velocity structure?

and:

    Does Pa-beta / Pa-gamma change systematically with velocity?

The spectra are extracted directly from the JWST S3D SCI/ERR cube.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.io import fits


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path.home() / "Projects" / "cosmic_ai"

S3D_PATH = (
    PROJECT_DIR
    / "data"
    / "m51_jwst_level3"
    / "jw03435-o006_t010_nirspec_g140m-f100lp_s3d.fits"
)

APERTURE_PATH = (
    PROJECT_DIR
    / "data"
    / "atomic_lines"
    / "m51_jwst_extraction_aperture.csv"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "data"
    / "atomic_lines"
)

PLOT_DIR = PROJECT_DIR


# Hydrogen rest wavelengths
PA_BETA_REST_NM = 1281.807000
PA_GAMMA_REST_NM = 1093.800000

# M51 systemic velocity used throughout the preceding experiments
M51_VELOCITY_KMS = 463.0

# Speed of light
C_KMS = 299792.458


# Spectral regions
PABETA_RANGE = (
    1270.0,
    1298.0,
)

PAGAMMA_RANGE = (
    1088.0,
    1102.0,
)

# Continuum windows
PABETA_BLUE = (
    1270.0,
    1278.0,
)

PABETA_RED = (
    1288.0,
    1296.0,
)

PAGAMMA_BLUE = (
    1080.0,
    1088.0,
)

PAGAMMA_RED = (
    1100.0,
    1108.0,
)

# Velocity binning.
#
# The native spectral spacing is about 0.636 nm.
# We retain the native spectral sampling and calculate the
# ratio plane-by-plane.
#
# For optional binned diagnostics:
VELOCITY_BIN_KMS = 50.0

# Minimum positive flux required before forming a ratio.
MIN_FLUX = 0.0


# ============================================================
# UTILITIES
# ============================================================

def banner(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def velocity_from_wavelength(rest_nm, observed_nm):
    """
    Non-relativistic Doppler velocity.
    """

    return (
        (observed_nm / rest_nm) - 1.0
    ) * C_KMS


def wavelength_from_velocity(rest_nm, velocity_kms):
    """
    Convert velocity to observed wavelength.
    """

    return (
        rest_nm
        * (1.0 + velocity_kms / C_KMS)
    )


def find_sci_err_hdus(hdul):
    """
    Locate SCI and ERR HDUs by EXTNAME.
    """

    sci_hdu = None
    err_hdu = None

    for hdu in hdul:
        name = str(
            hdu.header.get(
                "EXTNAME",
                "",
            )
        ).strip().upper()

        if name == "SCI":
            sci_hdu = hdu

        elif name == "ERR":
            err_hdu = hdu

    if sci_hdu is None:
        raise RuntimeError(
            "Could not find SCI HDU in S3D FITS file."
        )

    if err_hdu is None:
        raise RuntimeError(
            "Could not find ERR HDU in S3D FITS file."
        )

    return sci_hdu, err_hdu


def build_wavelength_array(header, n_spec):
    """
    Build wavelength array from the S3D spectral WCS.

    JWST S3D CRVAL3/CDELT3 are in microns.

    Convert:
        microns -> nm
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

    unit = str(
        header.get(
            "CUNIT3",
            "um",
        )
    ).strip().lower()

    pixel = (
        np.arange(
            n_spec,
            dtype=float,
        )
        + 1.0
    )

    wavelength = (
        crval3
        + (pixel - crpix3)
        * cdelt3
    )

    if unit in (
        "um",
        "micron",
        "microns",
    ):
        wavelength_nm = (
            wavelength * 1000.0
        )

    elif unit in (
        "nm",
        "nanometer",
        "nanometers",
    ):
        wavelength_nm = wavelength

    elif unit in (
        "m",
        "meter",
        "meters",
    ):
        wavelength_nm = (
            wavelength * 1.0e9
        )

    else:
        raise RuntimeError(
            f"Unsupported spectral WCS unit: {unit}"
        )

    return wavelength_nm


def load_nominal_aperture(path):
    """
    Load exactly the pixels marked inside_nominal_aperture=True.
    """

    df = pd.read_csv(path)

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
            "Aperture file is missing columns: "
            + ", ".join(sorted(missing))
        )

    mask = (
        df["inside_nominal_aperture"]
        .astype(bool)
    )

    aperture = (
        df.loc[mask]
        .copy()
        .reset_index(drop=True)
    )

    return aperture


def aperture_spectrum(
    cube,
    err_cube,
    aperture,
):
    """
    Sum the aperture spectra.

    Flux:
        direct sum over the 69 spatial pixels.

    Error:
        quadrature sum of ERR values.
    """

    x = (
        aperture["x_pixel"]
        .astype(int)
        .to_numpy()
    )

    y = (
        aperture["y_pixel"]
        .astype(int)
        .to_numpy()
    )

    sci = cube[
        :,
        y,
        x,
    ]

    err = err_cube[
        :,
        y,
        x,
    ]

    spectrum = np.nansum(
        sci,
        axis=1,
    )

    variance = np.nansum(
        np.square(err),
        axis=1,
    )

    spectrum_err = np.sqrt(
        variance
    )

    return (
        spectrum,
        spectrum_err,
    )


def continuum_indices(
    wavelength_nm,
    blue_window,
    red_window,
):
    blue = (
        (wavelength_nm >= blue_window[0])
        & (wavelength_nm <= blue_window[1])
    )

    red = (
        (wavelength_nm >= red_window[0])
        & (wavelength_nm <= red_window[1])
    )

    return blue, red


def fit_linear_continuum(
    wavelength_nm,
    spectrum,
    blue_window,
    red_window,
):
    """
    Fit a linear continuum using the blue and red windows.
    """

    blue_mask, red_mask = continuum_indices(
        wavelength_nm,
        blue_window,
        red_window,
    )

    mask = (
        blue_mask
        | red_mask
    )

    valid = (
        mask
        & np.isfinite(wavelength_nm)
        & np.isfinite(spectrum)
    )

    if np.sum(valid) < 2:
        raise RuntimeError(
            "Insufficient continuum points."
        )

    coeff = np.polyfit(
        wavelength_nm[valid],
        spectrum[valid],
        1,
    )

    continuum = np.polyval(
        coeff,
        wavelength_nm,
    )

    return (
        continuum,
        coeff,
        int(np.sum(blue_mask)),
        int(np.sum(red_mask)),
    )


def extract_line_region(
    wavelength_nm,
    spectrum,
    spectrum_err,
    line_range,
    blue_window,
    red_window,
):
    """
    Extract a continuum-subtracted line region.
    """

    region = (
        (wavelength_nm >= line_range[0])
        & (wavelength_nm <= line_range[1])
    )

    continuum, coeff, n_blue, n_red = (
        fit_linear_continuum(
            wavelength_nm,
            spectrum,
            blue_window,
            red_window,
        )
    )

    residual = (
        spectrum
        - continuum
    )

    return {
        "mask": region,
        "continuum": continuum,
        "residual": residual,
        "coeff": coeff,
        "n_blue": n_blue,
        "n_red": n_red,
    }


def native_velocity_profile(
    wavelength_nm,
    residual,
    error,
    rest_nm,
):
    """
    Convert a line profile to velocity space.

    Each native spectral plane retains its original sampling.
    """

    velocity = velocity_from_wavelength(
        rest_nm,
        wavelength_nm,
    )

    return (
        velocity,
        residual,
        error,
    )


def integrate_positive_flux(
    wavelength_nm,
    residual,
    error,
    mask,
):
    """
    Integrate continuum-subtracted line flux
    over a specified wavelength mask.

    Only finite values are used.
    """

    valid = (
        mask
        & np.isfinite(residual)
    )

    if not np.any(valid):
        return (
            np.nan,
            np.nan,
        )

    x = wavelength_nm[valid]
    y = residual[valid]
    e = error[valid]

    flux = np.trapezoid(
        y,
        x,
    )

    valid_err = np.isfinite(e)

    if np.any(valid_err):
        variance = np.sum(
            np.square(e[valid_err])
            * np.square(
                np.median(
                    np.diff(x)
                )
            )
        )

        sigma = np.sqrt(
            variance
        )

    else:
        sigma = np.nan

    return (
        flux,
        sigma,
    )

def ratio_uncertainty(
    beta,
    gamma,
    beta_err,
    gamma_err,
):
    """
    Propagate uncertainty for beta/gamma.

    Works with either scalar values or NumPy arrays.
    """

    beta = np.asarray(beta, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    beta_err = np.asarray(beta_err, dtype=float)
    gamma_err = np.asarray(gamma_err, dtype=float)

    ratio = np.full(
        np.broadcast(
            beta,
            gamma,
            beta_err,
            gamma_err,
        ).shape,
        np.nan,
        dtype=float,
    )

    valid = (
        np.isfinite(beta)
        & np.isfinite(gamma)
        & np.isfinite(beta_err)
        & np.isfinite(gamma_err)
        & (beta > 0)
        & (gamma > 0)
    )

    ratio[valid] = (
        beta[valid]
        / gamma[valid]
    )

    ratio[valid] *= np.sqrt(
        (
            beta_err[valid]
            / beta[valid]
        ) ** 2
        +
        (
            gamma_err[valid]
            / gamma[valid]
        ) ** 2
    )

    if ratio.ndim == 0:
        return float(ratio)

    return ratio


def save_csv(df, path):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        path,
        index=False,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    banner(
        "M51 JWST NIRSPEC — "
        "VELOCITY-RESOLVED Pa-beta / Pa-gamma RATIO"
    )

    print(
        "Purpose:"
    )

    print(
        "Measure the Pa-beta / Pa-gamma ratio directly "
        "as a function of velocity."
    )

    print()
    print(
        "No Gaussian component decomposition is used."
    )

    print(
        "No extinction inference is performed."
    )

    print(
        "The existing Pa-gamma product is not used."
    )

    # --------------------------------------------------------
    # 1. READ S3D
    # --------------------------------------------------------

    banner(
        "1. READING JWST S3D"
    )

    with fits.open(
        S3D_PATH,
        memmap=True,
    ) as hdul:

        sci_hdu, err_hdu = (
            find_sci_err_hdus(
                hdul
            )
        )

        cube = np.asarray(
            sci_hdu.data,
            dtype=float,
        )

        err_cube = np.asarray(
            err_hdu.data,
            dtype=float,
        )

        wavelength_nm = (
            build_wavelength_array(
                sci_hdu.header,
                cube.shape[0],
            )
        )

    print(
        f"SCI shape = {cube.shape}"
    )

    print(
        f"ERR shape = {err_cube.shape}"
    )

    print(
        f"Wavelength range = "
        f"{wavelength_nm[0]:.12f} - "
        f"{wavelength_nm[-1]:.12f} nm"
    )

    print(
        f"Spectral spacing = "
        f"{np.median(np.diff(wavelength_nm)):.12f} nm"
    )

    # --------------------------------------------------------
    # 2. APERTURE
    # --------------------------------------------------------

    banner(
        "2. LOADING NOMINAL APERTURE"
    )

    aperture = load_nominal_aperture(
        APERTURE_PATH
    )

    print(
        f"Aperture pixels = {len(aperture)}"
    )

    if len(aperture) != 69:
        raise RuntimeError(
            "Expected exactly 69 aperture pixels."
        )

    # --------------------------------------------------------
    # 3. APERTURE SPECTRUM
    # --------------------------------------------------------

    banner(
        "3. EXTRACTING 69-PIXEL APERTURE SPECTRUM"
    )

    spectrum, spectrum_err = (
        aperture_spectrum(
            cube,
            err_cube,
            aperture,
        )
    )

    print(
        "Summed SCI aperture spectrum created."
    )

    print(
        "Quadrature aperture error spectrum created."
    )

    # --------------------------------------------------------
    # 4. PREDICT LINE LOCATIONS
    # --------------------------------------------------------

    banner(
        "4. PREDICTING Pa-beta / Pa-gamma"
    )

    pabeta_predicted = (
        wavelength_from_velocity(
            PA_BETA_REST_NM,
            M51_VELOCITY_KMS,
        )
    )

    pagamma_predicted = (
        wavelength_from_velocity(
            PA_GAMMA_REST_NM,
            M51_VELOCITY_KMS,
        )
    )

    pabeta_index = int(
        np.argmin(
            np.abs(
                wavelength_nm
                - pabeta_predicted
            )
        )
    )

    pagamma_index = int(
        np.argmin(
            np.abs(
                wavelength_nm
                - pagamma_predicted
            )
        )
    )

    print(
        f"Pa-beta predicted = "
        f"{pabeta_predicted:.12f} nm"
    )

    print(
        f"Pa-gamma predicted = "
        f"{pagamma_predicted:.12f} nm"
    )

    print(
        f"Pa-beta nearest plane = "
        f"{pabeta_index}"
    )

    print(
        f"Pa-beta wavelength = "
        f"{wavelength_nm[pabeta_index]:.12f} nm"
    )

    print(
        f"Pa-gamma nearest plane = "
        f"{pagamma_index}"
    )

    print(
        f"Pa-gamma wavelength = "
        f"{wavelength_nm[pagamma_index]:.12f} nm"
    )

    beta_nearest_velocity = (
        velocity_from_wavelength(
            PA_BETA_REST_NM,
            wavelength_nm[pabeta_index],
        )
    )

    gamma_nearest_velocity = (
        velocity_from_wavelength(
            PA_GAMMA_REST_NM,
            wavelength_nm[pagamma_index],
        )
    )

    print(
        f"Pa-beta nearest-plane velocity = "
        f"{beta_nearest_velocity:+.3f} km/s"
    )

    print(
        f"Pa-gamma nearest-plane velocity = "
        f"{gamma_nearest_velocity:+.3f} km/s"
    )

    # --------------------------------------------------------
    # 5. CONTINUUM SUBTRACTION
    # --------------------------------------------------------

    banner(
        "5. BUILDING CONTINUUM-SUBTRACTED PROFILES"
    )

    beta_profile = extract_line_region(
        wavelength_nm,
        spectrum,
        spectrum_err,
        PABETA_RANGE,
        PABETA_BLUE,
        PABETA_RED,
    )

    gamma_profile = extract_line_region(
        wavelength_nm,
        spectrum,
        spectrum_err,
        PAGAMMA_RANGE,
        PAGAMMA_BLUE,
        PAGAMMA_RED,
    )

    print(
        f"Pa-beta continuum planes: "
        f"{beta_profile['n_blue'] + beta_profile['n_red']}"
    )

    print(
        f"Pa-gamma continuum planes: "
        f"{gamma_profile['n_blue'] + gamma_profile['n_red']}"
    )

    # --------------------------------------------------------
    # 6. VELOCITY PROFILES
    # --------------------------------------------------------

    banner(
        "6. CONVERTING TO VELOCITY SPACE"
    )

    beta_velocity = velocity_from_wavelength(
        PA_BETA_REST_NM,
        wavelength_nm,
    )

    gamma_velocity = velocity_from_wavelength(
        PA_GAMMA_REST_NM,
        wavelength_nm,
    )

    beta_mask = (
        beta_profile["mask"]
    )

    gamma_mask = (
        gamma_profile["mask"]
    )

    beta_v = beta_velocity[
        beta_mask
    ]

    beta_flux = beta_profile[
        "residual"
    ][beta_mask]

    beta_err = spectrum_err[
        beta_mask
    ]

    gamma_v = gamma_velocity[
        gamma_mask
    ]

    gamma_flux = gamma_profile[
        "residual"
    ][gamma_mask]

    gamma_err = spectrum_err[
        gamma_mask
    ]

    # --------------------------------------------------------
    # 7. MATCH VELOCITY SAMPLING
    # --------------------------------------------------------

    banner(
        "7. MATCHING Pa-beta / Pa-gamma VELOCITY SAMPLING"
    )

    # The wavelength grids are identical, but the two lines
    # correspond to different velocities at each wavelength.
    #
    # Interpolate both profiles onto a common velocity grid.

    velocity_min = max(
        np.min(beta_v),
        np.min(gamma_v),
    )

    velocity_max = min(
        np.max(beta_v),
        np.max(gamma_v),
    )

    velocity_grid = np.arange(
        np.ceil(
            velocity_min
            / VELOCITY_BIN_KMS
        ) * VELOCITY_BIN_KMS,
        np.floor(
            velocity_max
            / VELOCITY_BIN_KMS
        ) * VELOCITY_BIN_KMS
        + VELOCITY_BIN_KMS,
        VELOCITY_BIN_KMS,
    )

    print(
        f"Common velocity range = "
        f"{velocity_min:.1f} to "
        f"{velocity_max:.1f} km/s"
    )

    print(
        f"Velocity bin = "
        f"{VELOCITY_BIN_KMS:.1f} km/s"
    )

    print(
        f"Velocity bins = "
        f"{len(velocity_grid)}"
    )

    # --------------------------------------------------------
    # 8. INTERPOLATE
    # --------------------------------------------------------

    beta_valid = (
        np.isfinite(beta_v)
        & np.isfinite(beta_flux)
    )

    gamma_valid = (
        np.isfinite(gamma_v)
        & np.isfinite(gamma_flux)
    )

    beta_order = np.argsort(
        beta_v[beta_valid]
    )

    gamma_order = np.argsort(
        gamma_v[gamma_valid]
    )

    beta_v_sorted = beta_v[
        beta_valid
    ][beta_order]

    beta_flux_sorted = beta_flux[
        beta_valid
    ][beta_order]

    beta_err_sorted = beta_err[
        beta_valid
    ][beta_order]

    gamma_v_sorted = gamma_v[
        gamma_valid
    ][gamma_order]

    gamma_flux_sorted = gamma_flux[
        gamma_valid
    ][gamma_order]

    gamma_err_sorted = gamma_err[
        gamma_valid
    ][gamma_order]

    beta_interp = np.interp(
        velocity_grid,
        beta_v_sorted,
        beta_flux_sorted,
        left=np.nan,
        right=np.nan,
    )

    gamma_interp = np.interp(
        velocity_grid,
        gamma_v_sorted,
        gamma_flux_sorted,
        left=np.nan,
        right=np.nan,
    )

    beta_err_interp = np.interp(
        velocity_grid,
        beta_v_sorted,
        beta_err_sorted,
        left=np.nan,
        right=np.nan,
    )

    gamma_err_interp = np.interp(
        velocity_grid,
        gamma_v_sorted,
        gamma_err_sorted,
        left=np.nan,
        right=np.nan,
    )

    # --------------------------------------------------------
    # 9. VELOCITY-RESOLVED RATIO
    # --------------------------------------------------------

    banner(
        "9. VELOCITY-RESOLVED Pa-beta / Pa-gamma"
    )

    ratio = np.full(
        len(velocity_grid),
        np.nan,
    )

    ratio_error = np.full(
        len(velocity_grid),
        np.nan,
    )

    valid_ratio = (
        np.isfinite(beta_interp)
        & np.isfinite(gamma_interp)
        & (beta_interp > MIN_FLUX)
        & (gamma_interp > MIN_FLUX)
    )

    ratio[
        valid_ratio
    ] = (
        beta_interp[valid_ratio]
        / gamma_interp[valid_ratio]
    )

    ratio_error[
        valid_ratio
    ] = ratio_uncertainty(
        beta_interp[valid_ratio],
        gamma_interp[valid_ratio],
        beta_err_interp[valid_ratio],
        gamma_err_interp[valid_ratio],
    )

    print(
        f"Valid velocity bins = "
        f"{np.sum(valid_ratio)}"
    )

    if np.any(valid_ratio):

        valid_ratios = ratio[
            valid_ratio
        ]

        print(
            f"Ratio minimum = "
            f"{np.nanmin(valid_ratios):.6f}"
        )

        print(
            f"Ratio maximum = "
            f"{np.nanmax(valid_ratios):.6f}"
        )

        print(
            f"Ratio median = "
            f"{np.nanmedian(valid_ratios):.6f}"
        )

    # --------------------------------------------------------
    # 10. VELOCITY INTEGRATED FLUXES
    # --------------------------------------------------------

    banner(
        "10. VELOCITY-INTEGRATED LINE FLUXES"
    )

    beta_flux_integrated = np.trapezoid(
        beta_flux_sorted,
        beta_v_sorted,
    )

    gamma_flux_integrated = np.trapezoid(
        gamma_flux_sorted,
        gamma_v_sorted,
    )

    beta_error_integrated = np.sqrt(
        np.sum(
            np.square(
                beta_err_sorted
            )
        )
        * np.square(
            np.median(
                np.diff(
                    beta_v_sorted
                )
            )
        )
    )

    gamma_error_integrated = np.sqrt(
        np.sum(
            np.square(
                gamma_err_sorted
            )
        )
        * np.square(
            np.median(
                np.diff(
                    gamma_v_sorted
                )
            )
        )
    )

    integrated_ratio = (
        beta_flux_integrated
        / gamma_flux_integrated
    )

    integrated_ratio_error = (
        ratio_uncertainty(
            beta_flux_integrated,
            gamma_flux_integrated,
            beta_error_integrated,
            gamma_error_integrated,
        )
    )

    print(
        f"Pa-beta integrated flux = "
        f"{beta_flux_integrated:.6f}"
    )

    print(
        f"Pa-gamma integrated flux = "
        f"{gamma_flux_integrated:.6f}"
    )

    print(
        f"Pa-beta / Pa-gamma = "
        f"{integrated_ratio:.6f}"
    )

    print(
        f"Approximate ratio uncertainty = "
        f"{integrated_ratio_error:.6f}"
    )

    # --------------------------------------------------------
    # 11. VELOCITY-SLICED FLUX
    # --------------------------------------------------------

    banner(
        "11. VELOCITY-SLICED FLUX RATIOS"
    )

    velocity_ranges = [
        (-500.0, 200.0),
        (200.0, 400.0),
        (400.0, 500.0),
        (500.0, 600.0),
        (600.0, 800.0),
        (800.0, 1200.0),
    ]

    sliced_rows = []

    for low, high in velocity_ranges:

        beta_slice = (
            (beta_v_sorted >= low)
            & (beta_v_sorted <= high)
        )

        gamma_slice = (
            (gamma_v_sorted >= low)
            & (gamma_v_sorted <= high)
        )

        if (
            np.sum(beta_slice) < 2
            or np.sum(gamma_slice) < 2
        ):
            continue

        beta_f = np.trapezoid(
            beta_flux_sorted[
                beta_slice
            ],
            beta_v_sorted[
                beta_slice
            ],
        )

        gamma_f = np.trapezoid(
            gamma_flux_sorted[
                gamma_slice
            ],
            gamma_v_sorted[
                gamma_slice
            ],
        )

        if (
            not np.isfinite(beta_f)
            or not np.isfinite(gamma_f)
            or gamma_f <= 0
        ):
            continue

        r = (
            beta_f
            / gamma_f
        )

        sliced_rows.append(
            {
                "velocity_low_kms": low,
                "velocity_high_kms": high,
                "pabeta_flux": beta_f,
                "pagamma_flux": gamma_f,
                "pabeta_pagamma_ratio": r,
            }
        )

        print(
            f"{low:+.0f} to "
            f"{high:+.0f} km/s:"
        )

        print(
            f"  Pa-beta = "
            f"{beta_f:.6f}"
        )

        print(
            f"  Pa-gamma = "
            f"{gamma_f:.6f}"
        )

        print(
            f"  Ratio = "
            f"{r:.6f}"
        )

    # --------------------------------------------------------
    # 12. PROFILE CORRELATION
    # --------------------------------------------------------

    banner(
        "12. PROFILE CORRELATION"
    )

    common_valid = (
        np.isfinite(
            beta_interp
        )
        & np.isfinite(
            gamma_interp
        )
    )

    if np.sum(common_valid) >= 3:

        correlation = np.corrcoef(
            beta_interp[
                common_valid
            ],
            gamma_interp[
                common_valid
            ],
        )[0, 1]

    else:
        correlation = np.nan

    print(
        f"Velocity-profile Pearson correlation = "
        f"{correlation:.6f}"
    )

    # --------------------------------------------------------
    # 13. SAVE VELOCITY PROFILE
    # --------------------------------------------------------

    banner(
        "13. SAVING VELOCITY-RESOLVED PROFILE"
    )

    profile_df = pd.DataFrame(
        {
            "velocity_kms":
                velocity_grid,

            "pabeta_flux":
                beta_interp,

            "pagamma_flux":
                gamma_interp,

            "pabeta_error":
                beta_err_interp,

            "pagamma_error":
                gamma_err_interp,

            "pabeta_pagamma_ratio":
                ratio,

            "ratio_error":
                ratio_error,

            "ratio_valid":
                valid_ratio,
        }
    )

    profile_path = (
        OUTPUT_DIR
        / "m51_pabeta_pagamma_velocity_resolved_ratio.csv"
    )

    save_csv(
        profile_df,
        profile_path,
    )

    print(
        f"Saved:"
    )

    print(
        f"  {profile_path}"
    )

    # --------------------------------------------------------
    # 14. SAVE VELOCITY-SLICED RESULTS
    # --------------------------------------------------------

    sliced_path = (
        OUTPUT_DIR
        / "m51_pabeta_pagamma_velocity_slices.csv"
    )

    save_csv(
        pd.DataFrame(
            sliced_rows
        ),
        sliced_path,
    )

    print(
        f"Saved:"
    )

    print(
        f"  {sliced_path}"
    )

    # --------------------------------------------------------
    # 15. SAVE SUMMARY
    # --------------------------------------------------------

    summary = {
        "aperture_pixels": len(aperture),

        "pabeta_predicted_nm":
            pabeta_predicted,

        "pagamma_predicted_nm":
            pagamma_predicted,

        "pabeta_nearest_plane":
            pabeta_index,

        "pagamma_nearest_plane":
            pagamma_index,

        "pabeta_nearest_velocity_kms":
            beta_nearest_velocity,

        "pagamma_nearest_velocity_kms":
            gamma_nearest_velocity,

        "velocity_bin_kms":
            VELOCITY_BIN_KMS,

        "valid_velocity_bins":
            int(
                np.sum(
                    valid_ratio
                )
            ),

        "pabeta_integrated_flux":
            beta_flux_integrated,

        "pagamma_integrated_flux":
            gamma_flux_integrated,

        "integrated_ratio":
            integrated_ratio,

        "integrated_ratio_error":
            integrated_ratio_error,

        "velocity_profile_correlation":
            correlation,

        "ratio_minimum":
            (
                np.nanmin(
                    ratio[valid_ratio]
                )
                if np.any(valid_ratio)
                else np.nan
            ),

        "ratio_maximum":
            (
                np.nanmax(
                    ratio[valid_ratio]
                )
                if np.any(valid_ratio)
                else np.nan
            ),

        "ratio_median":
            (
                np.nanmedian(
                    ratio[valid_ratio]
                )
                if np.any(valid_ratio)
                else np.nan
            ),
    }

    summary_path = (
        OUTPUT_DIR
        / "m51_pabeta_pagamma_velocity_resolved_ratio_summary.csv"
    )

    save_csv(
        pd.DataFrame(
            [summary]
        ),
        summary_path,
    )

    print(
        f"Saved:"
    )

    print(
        f"  {summary_path}"
    )

    # --------------------------------------------------------
    # 16. CREATE PROFILE FIGURE
    # --------------------------------------------------------

    banner(
        "16. CREATING VELOCITY PROFILE FIGURE"
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 9),
        sharex=True,
    )

    axes[0].plot(
        beta_velocity[
            beta_mask
        ],
        beta_profile[
            "residual"
        ][beta_mask],
        marker="o",
        linewidth=1,
        label="Pa-beta",
    )

    axes[0].plot(
        gamma_velocity[
            gamma_mask
        ],
        gamma_profile[
            "residual"
        ][gamma_mask],
        marker="o",
        linewidth=1,
        label="Pa-gamma",
    )

    axes[0].axvline(
        M51_VELOCITY_KMS,
        linestyle="--",
        linewidth=1,
        label="M51 reference velocity",
    )

    axes[0].set_ylabel(
        "Continuum-subtracted flux"
    )

    axes[0].set_title(
        "M51 NIRSpec Aperture Hydrogen-Line Profiles"
    )

    axes[0].legend()

    axes[1].errorbar(
        velocity_grid[
            valid_ratio
        ],
        ratio[
            valid_ratio
        ],
        yerr=ratio_error[
            valid_ratio
        ],
        marker="o",
        linestyle="none",
        capsize=3,
        label="Pa-beta / Pa-gamma",
    )

    axes[1].axvline(
        M51_VELOCITY_KMS,
        linestyle="--",
        linewidth=1,
        label="M51 reference velocity",
    )

    axes[1].axhline(
        integrated_ratio,
        linestyle=":",
        linewidth=1,
        label="Integrated ratio",
    )

    axes[1].set_xlabel(
        "Velocity (km/s)"
    )

    axes[1].set_ylabel(
        "Pa-beta / Pa-gamma"
    )

    axes[1].set_title(
        "Velocity-Resolved Hydrogen-Line Ratio"
    )

    axes[1].legend()

    plt.tight_layout()

    profile_plot = (
        PLOT_DIR
        / "m51_pabeta_pagamma_velocity_resolved_ratio.png"
    )

    plt.savefig(
        profile_plot,
        dpi=180,
    )

    plt.close()

    print(
        f"Saved:"
    )

    print(
        f"  {profile_plot}"
    )

    # --------------------------------------------------------
    # 17. FINAL RESULT
    # --------------------------------------------------------

    banner(
        "FINAL VELOCITY-RESOLVED RATIO RESULT"
    )

    print(
        f"Nominal aperture = "
        f"{len(aperture)} pixels"
    )

    print()

    print(
        f"Pa-beta integrated flux = "
        f"{beta_flux_integrated:.6f}"
    )

    print(
        f"Pa-gamma integrated flux = "
        f"{gamma_flux_integrated:.6f}"
    )

    print()

    print(
        f"Integrated Pa-beta / Pa-gamma = "
        f"{integrated_ratio:.6f} "
        f"+/- {integrated_ratio_error:.6f}"
    )

    print()

    print(
        f"Velocity-profile correlation = "
        f"{correlation:.6f}"
    )

    print()

    print(
        f"Velocity-resolved ratio range = "
        f"{np.nanmin(ratio[valid_ratio]):.6f} "
        f"to "
        f"{np.nanmax(ratio[valid_ratio]):.6f}"
        if np.any(valid_ratio)
        else "Velocity-resolved ratio unavailable."
    )

    print()

    print(
        "Interpretation:"
    )

    print(
        "The velocity-resolved ratio is a non-parametric "
        "test of whether Pa-beta and Pa-gamma trace the "
        "same kinematic emission."
    )

    print(
        "No extinction inference has been performed."
    )

    print(
        "The result should be evaluated before adopting "
        "any single Pa-beta / Pa-gamma ratio for the "
        "Storey-Hummer extinction analysis."
    )

    print()

    print(
        "Velocity-resolved Pa-beta / Pa-gamma experiment complete."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
