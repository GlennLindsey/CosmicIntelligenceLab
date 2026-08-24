#!/usr/bin/env python3

"""
M51 JWST NIRSPEC — Pa-beta / Pa-gamma DIRECT PROFILE FIT

Purpose
-------
Fit the summed 69-pixel JWST aperture spectra directly from the
S3D cube using the same basic continuum treatment for both
hydrogen lines.

The experiment:

    1. extracts the exact 69-pixel nominal aperture
    2. builds summed Pa-beta and Pa-gamma spectra
    3. fits a local linear continuum
    4. fits a Gaussian emission profile
    5. measures fitted line flux, centroid, velocity, and FWHM
    6. compares the fitted Pa-beta/Pa-gamma ratio with the
       previous fixed-window measurements
    7. compares the Pa-gamma fitted flux with the existing
       questionable Pa-gamma product

IMPORTANT
---------
No extinction calculation is performed here.

The purpose is to establish a defensible observed line ratio
before returning to the Storey-Hummer extinction analysis.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.io import fits
from scipy.optimize import curve_fit


# ============================================================
# PATHS
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

EXISTING_PAGAMMA_PATH = (
    PROJECT_DIR
    / "data"
    / "atomic_lines"
    / "m51_1096_pagamma_hydrogen_consistency_map.fits"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "data"
    / "atomic_lines"
)

OUTPUT_FIGURE = (
    PROJECT_DIR
    / "m51_pabeta_pagamma_profile_fit.png"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "m51_pabeta_pagamma_profile_fit.csv"
)

OUTPUT_SUMMARY = (
    OUTPUT_DIR
    / "m51_pabeta_pagamma_profile_fit_summary.csv"
)


# ============================================================
# CONSTANTS
# ============================================================

C_KMS = 299792.458

PA_BETA_REST_NM = 1281.807000
PA_GAMMA_REST_NM = 1093.800000

M51_VELOCITY_KMS = 463.0

RESOLVING_POWER = 2700.0

# Spectral fitting regions.
PABETA_REGION = (1274.0, 1292.0)
PAGAMMA_REGION = (1088.0, 1102.0)

# Continuum regions.
PABETA_BLUE = (1274.0, 1279.0)
PABETA_RED = (1287.0, 1292.0)

PAGAMMA_BLUE = (1088.0, 1092.5)
PAGAMMA_RED = (1098.0, 1102.0)

# Initial Gaussian search width.
INITIAL_SIGMA_NM = 0.75


# ============================================================
# DISPLAY
# ============================================================

def banner(text):
    print()
    print("=" * 70)
    print(text)
    print("=" * 70)


# ============================================================
# WAVELENGTH
# ============================================================

def build_wavelength_nm(header, n_spec):
    """
    Build wavelength array from the spectral WCS.

    The S3D wavelength axis is stored in microns.
    """

    crval3 = header["CRVAL3"]
    crpix3 = header["CRPIX3"]
    cdelt3 = header["CDELT3"]

    pixel = np.arange(n_spec, dtype=float) + 1.0

    wavelength_um = (
        crval3
        + (pixel - crpix3) * cdelt3
    )

    return wavelength_um * 1000.0


# ============================================================
# VELOCITY
# ============================================================

def velocity_from_wavelength(rest_nm, observed_nm):
    return (
        (observed_nm / rest_nm) - 1.0
    ) * C_KMS


def predicted_wavelength(rest_nm, velocity_kms):
    return (
        rest_nm
        * (1.0 + velocity_kms / C_KMS)
    )


# ============================================================
# APERTURE
# ============================================================

def load_aperture(path):
    df = pd.read_csv(path)

    required = {
        "x_pixel",
        "y_pixel",
        "inside_nominal_aperture",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"Aperture file missing columns: {sorted(missing)}"
        )

    mask = (
        df["inside_nominal_aperture"]
        .astype(bool)
    )

    aperture = df.loc[mask].copy()

    if len(aperture) != 69:
        raise RuntimeError(
            f"Expected 69 aperture pixels, found {len(aperture)}"
        )

    return aperture


# ============================================================
# APERTURE SPECTRUM
# ============================================================

def extract_aperture_spectrum(
    sci,
    err,
    aperture,
):
    """
    Sum the SCI values over the exact 69 aperture pixels.

    Errors are propagated in quadrature.
    """

    x = aperture["x_pixel"].astype(int).to_numpy()
    y = aperture["y_pixel"].astype(int).to_numpy()

    pixel_spectra = sci[:, y, x]
    pixel_errors = err[:, y, x]

    spectrum = np.nansum(
        pixel_spectra,
        axis=1,
    )

    variance = np.nansum(
        pixel_errors ** 2,
        axis=1,
    )

    error = np.sqrt(variance)

    return spectrum, error


# ============================================================
# CONTINUUM
# ============================================================

def fit_continuum(
    wavelength_nm,
    spectrum,
    error,
    blue_region,
    red_region,
):
    """
    Fit a weighted linear continuum using blue and red regions.
    """

    blue = (
        (wavelength_nm >= blue_region[0])
        & (wavelength_nm <= blue_region[1])
    )

    red = (
        (wavelength_nm >= red_region[0])
        & (wavelength_nm <= red_region[1])
    )

    mask = (
        (blue | red)
        & np.isfinite(spectrum)
        & np.isfinite(error)
        & (error > 0)
    )

    if np.sum(mask) < 4:
        raise RuntimeError(
            "Insufficient continuum points."
        )

    x = wavelength_nm[mask]
    y = spectrum[mask]
    sigma = error[mask]

    # Linear continuum:
    # y = m*x + b
    def linear(x, m, b):
        return m * x + b

    p0 = [
        0.0,
        float(np.nanmedian(y)),
    ]

    popt, pcov = curve_fit(
        linear,
        x,
        y,
        p0=p0,
        sigma=sigma,
        absolute_sigma=True,
        maxfev=20000,
    )

    m, b = popt

    continuum = (
        m * wavelength_nm + b
    )

    return (
        continuum,
        float(m),
        float(b),
        pcov,
        int(np.sum(blue)),
        int(np.sum(red)),
    )


# ============================================================
# GAUSSIAN MODEL
# ============================================================

def gaussian(x, amplitude, center, sigma):
    return (
        amplitude
        * np.exp(
            -0.5
            * ((x - center) / sigma) ** 2
        )
    )


def gaussian_plus_continuum(
    x,
    amplitude,
    center,
    sigma,
    slope,
    intercept,
    x_reference,
):
    return (
        gaussian(
            x,
            amplitude,
            center,
            sigma,
        )
        + slope * (x - x_reference)
        + intercept
    )


# ============================================================
# PROFILE FIT
# ============================================================

def fit_line_profile(
    wavelength_nm,
    spectrum,
    error,
    line_rest_nm,
    region,
    blue_region,
    red_region,
    label,
):
    """
    Fit a Gaussian emission line after fitting a local
    linear continuum.

    The continuum is fixed from the continuum windows.
    """

    region_mask = (
        (wavelength_nm >= region[0])
        & (wavelength_nm <= region[1])
        & np.isfinite(spectrum)
        & np.isfinite(error)
        & (error > 0)
    )

    if np.sum(region_mask) < 8:
        raise RuntimeError(
            f"{label}: insufficient spectral points."
        )

    x = wavelength_nm[region_mask]
    y = spectrum[region_mask]
    yerr = error[region_mask]

    (
        continuum_full,
        continuum_slope,
        continuum_intercept,
        continuum_cov,
        n_blue,
        n_red,
    ) = fit_continuum(
        wavelength_nm,
        spectrum,
        error,
        blue_region,
        red_region,
    )

    continuum = continuum_full[region_mask]

    residual = y - continuum

    # Initial center from strongest continuum-subtracted point.
    peak_index = np.nanargmax(residual)

    center0 = float(x[peak_index])
    amplitude0 = float(
        max(
            residual[peak_index],
            np.nanpercentile(residual, 90),
            1e-6,
        )
    )

    sigma0 = INITIAL_SIGMA_NM

    # Fit Gaussian only to the continuum-subtracted profile.
    def gaussian_fit(x, amplitude, center, sigma):
        return gaussian(
            x,
            amplitude,
            center,
            sigma,
        )

    lower = [
        0.0,
        region[0],
        0.05,
    ]

    upper = [
        np.inf,
        region[1],
        5.0,
    ]

    popt, pcov = curve_fit(
        gaussian_fit,
        x,
        residual,
        p0=[
            amplitude0,
            center0,
            sigma0,
        ],
        sigma=yerr,
        absolute_sigma=True,
        bounds=(lower, upper),
        maxfev=50000,
    )

    amplitude, center, sigma = popt

    parameter_errors = np.sqrt(
        np.maximum(
            np.diag(pcov),
            0.0,
        )
    )

    amplitude_err = parameter_errors[0]
    center_err = parameter_errors[1]
    sigma_err = parameter_errors[2]

    # Gaussian integrated flux:
    # amplitude * sigma * sqrt(2*pi)
    integrated_flux = (
        amplitude
        * sigma
        * np.sqrt(2.0 * np.pi)
    )

    # Propagate amplitude and sigma uncertainty.
    flux_error = integrated_flux * np.sqrt(
        (
            amplitude_err / amplitude
        ) ** 2
        +
        (
            sigma_err / sigma
        ) ** 2
    )

    fwhm = (
        2.0
        * np.sqrt(2.0 * np.log(2.0))
        * sigma
    )

    fwhm_error = (
        2.0
        * np.sqrt(2.0 * np.log(2.0))
        * sigma_err
    )

    velocity = velocity_from_wavelength(
        line_rest_nm,
        center,
    )

    velocity_err = (
        C_KMS
        * center_err
        / line_rest_nm
    )

    # Model and residuals.
    model = gaussian_fit(
        x,
        amplitude,
        center,
        sigma,
    )

    residuals = (
        residual - model
    )

    chi2 = np.sum(
        (
            residuals / yerr
        ) ** 2
    )

    dof = max(
        len(x) - 3,
        1,
    )

    reduced_chi2 = (
        chi2 / dof
    )

    peak_flux = float(
        np.nanmax(residual)
    )

    peak_velocity = velocity_from_wavelength(
        line_rest_nm,
        x[np.nanargmax(residual)],
    )

    return {
        "label": label,
        "x": x,
        "observed": y,
        "error": yerr,
        "continuum": continuum,
        "residual": residual,
        "model": model,
        "residuals": residuals,

        "continuum_slope": continuum_slope,
        "continuum_intercept": continuum_intercept,

        "n_blue": n_blue,
        "n_red": n_red,

        "amplitude": float(amplitude),
        "amplitude_error": float(amplitude_err),

        "center_nm": float(center),
        "center_error_nm": float(center_err),

        "velocity_kms": float(velocity),
        "velocity_error_kms": float(velocity_err),

        "sigma_nm": float(sigma),
        "sigma_error_nm": float(sigma_err),

        "fwhm_nm": float(fwhm),
        "fwhm_error_nm": float(fwhm_error),

        "fwhm_velocity_kms": float(
            C_KMS * fwhm / center
        ),

        "integrated_flux": float(
            integrated_flux
        ),

        "integrated_flux_error": float(
            flux_error
        ),

        "snr": float(
            integrated_flux / flux_error
        ),

        "peak_flux": peak_flux,
        "peak_velocity_kms": float(
            peak_velocity
        ),

        "chi2": float(chi2),
        "dof": int(dof),
        "reduced_chi2": float(
            reduced_chi2
        ),
    }


# ============================================================
# WINDOW COMPARISON
# ============================================================

def integrate_window(
    wavelength_nm,
    residual,
    center_nm,
    half_planes,
):
    """
    Integrate a fixed spectral window centered on the
    nearest cube plane.
    """

    nearest = int(
        np.argmin(
            np.abs(
                wavelength_nm
                - center_nm
            )
        )
    )

    lo = max(
        0,
        nearest - half_planes,
    )

    hi = min(
        len(wavelength_nm) - 1,
        nearest + half_planes,
    )

    indices = np.arange(
        lo,
        hi + 1,
    )

    values = residual[indices]

    finite = np.isfinite(values)

    if not np.any(finite):
        return np.nan

    if len(wavelength_nm) > 1:
        spacing = float(
            np.nanmedian(
                np.diff(wavelength_nm)
            )
        )
    else:
        spacing = 1.0

    return float(
        np.nansum(
            values[finite]
        )
        * spacing
    )


# ============================================================
# MAIN
# ============================================================

def main():

    banner(
        "M51 JWST NIRSPEC — "
        "Pa-beta / Pa-gamma DIRECT PROFILE FIT"
    )

    print(
        "Purpose:"
    )
    print(
        "Fit the summed 69-pixel JWST aperture spectra "
        "directly from the S3D cube."
    )
    print()
    print(
        "No extinction inference is performed."
    )
    print(
        "The existing Pa-gamma product is comparison-only."
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # 1. READ S3D
    # ========================================================

    banner("1. READING JWST S3D")

    with fits.open(
        S3D_PATH,
        memmap=True,
    ) as hdul:

        sci_hdu = hdul["SCI"]
        err_hdu = hdul["ERR"]

        sci = np.asarray(
            sci_hdu.data,
            dtype=float,
        )

        err = np.asarray(
            err_hdu.data,
            dtype=float,
        )

        header = sci_hdu.header.copy()

        print(
            f"SCI HDU = {sci_hdu.name}"
        )
        print(
            f"ERR HDU = {err_hdu.name}"
        )
        print(
            f"SCI shape = {sci.shape}"
        )
        print(
            f"ERR shape = {err.shape}"
        )

        wavelength_nm = build_wavelength_nm(
            header,
            sci.shape[0],
        )

    spacing = float(
        np.nanmedian(
            np.diff(wavelength_nm)
        )
    )

    print(
        f"Wavelength range = "
        f"{wavelength_nm[0]:.9f} - "
        f"{wavelength_nm[-1]:.9f} nm"
    )

    print(
        f"Spectral spacing = "
        f"{spacing:.9f} nm"
    )

    # ========================================================
    # 2. APERTURE
    # ========================================================

    banner(
        "2. LOADING NOMINAL APERTURE"
    )

    aperture = load_aperture(
        APERTURE_PATH
    )

    print(
        f"Aperture pixels = "
        f"{len(aperture)}"
    )

    # ========================================================
    # 3. SPECTRAL CENTERS
    # ========================================================

    banner(
        "3. PREDICTING Pa-beta / Pa-gamma"
    )

    pabeta_predicted = predicted_wavelength(
        PA_BETA_REST_NM,
        M51_VELOCITY_KMS,
    )

    pagamma_predicted = predicted_wavelength(
        PA_GAMMA_REST_NM,
        M51_VELOCITY_KMS,
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

    pabeta_observed = wavelength_nm[
        pabeta_index
    ]

    pagamma_observed = wavelength_nm[
        pagamma_index
    ]

    print(
        f"Pa-beta predicted = "
        f"{pabeta_predicted:.9f} nm"
    )

    print(
        f"Pa-gamma predicted = "
        f"{pagamma_predicted:.9f} nm"
    )

    print(
        f"Pa-beta nearest plane = "
        f"{pabeta_index}"
    )

    print(
        f"Pa-beta wavelength = "
        f"{pabeta_observed:.9f} nm"
    )

    print(
        f"Pa-gamma nearest plane = "
        f"{pagamma_index}"
    )

    print(
        f"Pa-gamma wavelength = "
        f"{pagamma_observed:.9f} nm"
    )

    pabeta_velocity = (
        velocity_from_wavelength(
            PA_BETA_REST_NM,
            pabeta_observed,
        )
    )

    pagamma_velocity = (
        velocity_from_wavelength(
            PA_GAMMA_REST_NM,
            pagamma_observed,
        )
    )

    print(
        f"Pa-beta velocity = "
        f"{pabeta_velocity:+.3f} km/s"
    )

    print(
        f"Pa-gamma velocity = "
        f"{pagamma_velocity:+.3f} km/s"
    )

    # ========================================================
    # 4. SUM APERTURE SPECTRUM
    # ========================================================

    banner(
        "4. EXTRACTING 69-PIXEL APERTURE SPECTRA"
    )

    aperture_spectrum, aperture_error = (
        extract_aperture_spectrum(
            sci,
            err,
            aperture,
        )
    )

    print(
        "Summed SCI aperture spectrum created."
    )

    print(
        "Quadrature aperture error spectrum created."
    )

    # ========================================================
    # 5. FIT Pa-beta
    # ========================================================

    banner(
        "5. FITTING Pa-beta PROFILE"
    )

    pabeta_fit = fit_line_profile(
        wavelength_nm,
        aperture_spectrum,
        aperture_error,
        PA_BETA_REST_NM,
        PABETA_REGION,
        PABETA_BLUE,
        PABETA_RED,
        "Pa-beta",
    )

    print(
        f"Continuum blue planes = "
        f"{pabeta_fit['n_blue']}"
    )

    print(
        f"Continuum red planes = "
        f"{pabeta_fit['n_red']}"
    )

    print(
        f"Centroid = "
        f"{pabeta_fit['center_nm']:.9f} nm"
    )

    print(
        f"Velocity = "
        f"{pabeta_fit['velocity_kms']:+.3f} "
        f"+/- "
        f"{pabeta_fit['velocity_error_kms']:.3f} km/s"
    )

    print(
        f"Sigma = "
        f"{pabeta_fit['sigma_nm']:.6f} nm"
    )

    print(
        f"FWHM = "
        f"{pabeta_fit['fwhm_nm']:.6f} nm"
    )

    print(
        f"FWHM velocity = "
        f"{pabeta_fit['fwhm_velocity_kms']:.3f} km/s"
    )

    print(
        f"Integrated flux = "
        f"{pabeta_fit['integrated_flux']:.8f}"
    )

    print(
        f"Flux uncertainty = "
        f"{pabeta_fit['integrated_flux_error']:.8f}"
    )

    print(
        f"S/N = "
        f"{pabeta_fit['snr']:.3f}"
    )

    print(
        f"Reduced chi2 = "
        f"{pabeta_fit['reduced_chi2']:.3f}"
    )

    # ========================================================
    # 6. FIT Pa-gamma
    # ========================================================

    banner(
        "6. FITTING Pa-gamma PROFILE"
    )

    pagamma_fit = fit_line_profile(
        wavelength_nm,
        aperture_spectrum,
        aperture_error,
        PA_GAMMA_REST_NM,
        PAGAMMA_REGION,
        PAGAMMA_BLUE,
        PAGAMMA_RED,
        "Pa-gamma",
    )

    print(
        f"Continuum blue planes = "
        f"{pagamma_fit['n_blue']}"
    )

    print(
        f"Continuum red planes = "
        f"{pagamma_fit['n_red']}"
    )

    print(
        f"Centroid = "
        f"{pagamma_fit['center_nm']:.9f} nm"
    )

    print(
        f"Velocity = "
        f"{pagamma_fit['velocity_kms']:+.3f} "
        f"+/- "
        f"{pagamma_fit['velocity_error_kms']:.3f} km/s"
    )

    print(
        f"Sigma = "
        f"{pagamma_fit['sigma_nm']:.6f} nm"
    )

    print(
        f"FWHM = "
        f"{pagamma_fit['fwhm_nm']:.6f} nm"
    )

    print(
        f"FWHM velocity = "
        f"{pagamma_fit['fwhm_velocity_kms']:.3f} km/s"
    )

    print(
        f"Integrated flux = "
        f"{pagamma_fit['integrated_flux']:.8f}"
    )

    print(
        f"Flux uncertainty = "
        f"{pagamma_fit['integrated_flux_error']:.8f}"
    )

    print(
        f"S/N = "
        f"{pagamma_fit['snr']:.3f}"
    )

    print(
        f"Reduced chi2 = "
        f"{pagamma_fit['reduced_chi2']:.3f}"
    )

    # ========================================================
    # 7. PROFILE RATIO
    # ========================================================

    banner(
        "7. PROFILE-FIT Pa-beta / Pa-gamma RATIO"
    )

    pb = pabeta_fit[
        "integrated_flux"
    ]

    pg = pagamma_fit[
        "integrated_flux"
    ]

    pb_err = pabeta_fit[
        "integrated_flux_error"
    ]

    pg_err = pagamma_fit[
        "integrated_flux_error"
    ]

    ratio = pb / pg

    ratio_error = ratio * np.sqrt(
        (pb_err / pb) ** 2
        +
        (pg_err / pg) ** 2
    )

    print(
        f"Pa-beta flux = "
        f"{pb:.8f} +/- {pb_err:.8f}"
    )

    print(
        f"Pa-gamma flux = "
        f"{pg:.8f} +/- {pg_err:.8f}"
    )

    print(
        f"Profile-fit ratio = "
        f"{ratio:.8f} +/- "
        f"{ratio_error:.8f}"
    )

    # ========================================================
    # 8. VELOCITY / WIDTH COMPARISON
    # ========================================================

    banner(
        "8. Pa-beta / Pa-gamma PROFILE CONSISTENCY"
    )

    velocity_difference = (
        pagamma_fit["velocity_kms"]
        - pabeta_fit["velocity_kms"]
    )

    velocity_difference_error = np.sqrt(
        pabeta_fit["velocity_error_kms"] ** 2
        +
        pagamma_fit["velocity_error_kms"] ** 2
    )

    fwhm_difference = (
        pagamma_fit["fwhm_nm"]
        - pabeta_fit["fwhm_nm"]
    )

    print(
        f"Velocity difference "
        f"(Pa-gamma - Pa-beta) = "
        f"{velocity_difference:+.3f} km/s"
    )

    print(
        f"Velocity difference uncertainty = "
        f"{velocity_difference_error:.3f} km/s"
    )

    print(
        f"FWHM difference "
        f"(Pa-gamma - Pa-beta) = "
        f"{fwhm_difference:+.6f} nm"
    )

    # ========================================================
    # 9. FIXED-WINDOW COMPARISON
    # ========================================================

    banner(
        "9. FIXED-WINDOW FLUX COMPARISON"
    )

    window_rows = []

    for half_planes in range(5):

        pb_residual = (
            pabeta_fit["residual"]
        )

        pg_residual = (
            pagamma_fit["residual"]
        )

        pb_flux = integrate_window(
            pabeta_fit["x"],
            pb_residual,
            pabeta_observed,
            half_planes,
        )

        pg_flux = integrate_window(
            pagamma_fit["x"],
            pg_residual,
            pagamma_observed,
            half_planes,
        )

        window_ratio = (
            pb_flux / pg_flux
            if (
                np.isfinite(pb_flux)
                and np.isfinite(pg_flux)
                and pg_flux != 0
            )
            else np.nan
        )

        print(
            f"±{half_planes}: "
            f"Pa-beta = {pb_flux:.8f}, "
            f"Pa-gamma = {pg_flux:.8f}, "
            f"ratio = {window_ratio:.8f}"
        )

        window_rows.append(
            {
                "half_planes": half_planes,
                "pabeta_flux": pb_flux,
                "pagamma_flux": pg_flux,
                "ratio": window_ratio,
            }
        )

    # ========================================================
    # 10. EXISTING Pa-gamma PRODUCT
    # ========================================================

    banner(
        "10. EXISTING Pa-gamma PRODUCT COMPARISON"
    )

    with fits.open(
        EXISTING_PAGAMMA_PATH,
        memmap=True,
    ) as hdul:

        existing_map = np.asarray(
            hdul[0].data,
            dtype=float,
        )

    x = aperture[
        "x_pixel"
    ].astype(int).to_numpy()

    y = aperture[
        "y_pixel"
    ].astype(int).to_numpy()

    existing_values = existing_map[
        y,
        x,
    ]

    existing_finite = (
        np.isfinite(existing_values)
    )

    existing_positive = (
        existing_finite
        & (existing_values > 0)
    )

    existing_sum = np.nansum(
        existing_values
    )

    print(
        f"Existing product finite = "
        f"{np.sum(existing_finite)}"
    )

    print(
        f"Existing product positive = "
        f"{np.sum(existing_positive)}"
    )

    print(
        f"Existing product aperture sum = "
        f"{existing_sum:.8f}"
    )

    print(
        f"Profile-fit Pa-gamma = "
        f"{pg:.8f}"
    )

    print(
        f"Profile-fit / existing = "
        f"{pg / existing_sum:.8f}"
    )

    # ========================================================
    # 11. SAVE RESULTS
    # ========================================================

    banner(
        "11. SAVING RESULTS"
    )

    rows = []

    for fit in (
        pabeta_fit,
        pagamma_fit,
    ):
        rows.append(
            {
                "line": fit["label"],
                "center_nm": fit["center_nm"],
                "center_error_nm":
                    fit["center_error_nm"],
                "velocity_kms":
                    fit["velocity_kms"],
                "velocity_error_kms":
                    fit["velocity_error_kms"],
                "sigma_nm":
                    fit["sigma_nm"],
                "sigma_error_nm":
                    fit["sigma_error_nm"],
                "fwhm_nm":
                    fit["fwhm_nm"],
                "fwhm_error_nm":
                    fit["fwhm_error_nm"],
                "fwhm_velocity_kms":
                    fit["fwhm_velocity_kms"],
                "integrated_flux":
                    fit["integrated_flux"],
                "integrated_flux_error":
                    fit["integrated_flux_error"],
                "snr":
                    fit["snr"],
                "peak_flux":
                    fit["peak_flux"],
                "peak_velocity_kms":
                    fit["peak_velocity_kms"],
                "reduced_chi2":
                    fit["reduced_chi2"],
            }
        )

    pd.DataFrame(
        rows
    ).to_csv(
        OUTPUT_CSV,
        index=False,
    )

    summary = {
        "aperture_pixels": 69,

        "pabeta_predicted_nm":
            pabeta_predicted,
        "pabeta_observed_nm":
            pabeta_observed,

        "pagamma_predicted_nm":
            pagamma_predicted,
        "pagamma_observed_nm":
            pagamma_observed,

        "pabeta_fit_flux":
            pb,
        "pabeta_fit_flux_error":
            pb_err,

        "pagamma_fit_flux":
            pg,
        "pagamma_fit_flux_error":
            pg_err,

        "profile_ratio":
            ratio,
        "profile_ratio_error":
            ratio_error,

        "pabeta_velocity_kms":
            pabeta_fit["velocity_kms"],
        "pagamma_velocity_kms":
            pagamma_fit["velocity_kms"],

        "velocity_difference_kms":
            velocity_difference,
        "velocity_difference_error_kms":
            velocity_difference_error,

        "pabeta_fwhm_nm":
            pabeta_fit["fwhm_nm"],
        "pagamma_fwhm_nm":
            pagamma_fit["fwhm_nm"],

        "pabeta_reduced_chi2":
            pabeta_fit["reduced_chi2"],
        "pagamma_reduced_chi2":
            pagamma_fit["reduced_chi2"],

        "existing_pagamma_flux":
            existing_sum,

        "profile_pagamma_fraction_existing":
            pg / existing_sum,
    }

    pd.DataFrame(
        [summary]
    ).to_csv(
        OUTPUT_SUMMARY,
        index=False,
    )

    window_df = pd.DataFrame(
        window_rows
    )

    window_output = (
        OUTPUT_DIR
        / "m51_pabeta_pagamma_profile_fit_windows.csv"
    )

    window_df.to_csv(
        window_output,
        index=False,
    )

    print(
        f"Profile results:\n  "
        f"{OUTPUT_CSV}"
    )

    print(
        f"Summary:\n  "
        f"{OUTPUT_SUMMARY}"
    )

    print(
        f"Window comparison:\n  "
        f"{window_output}"
    )

    # ========================================================
    # 12. FIGURE
    # ========================================================

    banner(
        "12. CREATING PROFILE FIGURE"
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11, 10),
    )

    # --------------------------------------------------------
    # Pa-beta
    # --------------------------------------------------------

    ax = axes[0]

    ax.errorbar(
        pabeta_fit["x"],
        pabeta_fit["residual"],
        yerr=pabeta_fit["error"],
        fmt=".",
        markersize=4,
        alpha=0.65,
        label="Pa-beta aperture spectrum",
    )

    ax.plot(
        pabeta_fit["x"],
        pabeta_fit["model"],
        linewidth=2,
        label="Gaussian fit",
    )

    ax.axvline(
        pabeta_predicted,
        linestyle="--",
        label="Predicted wavelength",
    )

    ax.set_title(
        "M51 JWST Pa-beta — 69-pixel aperture"
    )

    ax.set_xlabel(
        "Wavelength (nm)"
    )

    ax.set_ylabel(
        "Continuum-subtracted flux"
    )

    ax.legend()

    # --------------------------------------------------------
    # Pa-gamma
    # --------------------------------------------------------

    ax = axes[1]

    ax.errorbar(
        pagamma_fit["x"],
        pagamma_fit["residual"],
        yerr=pagamma_fit["error"],
        fmt=".",
        markersize=4,
        alpha=0.65,
        label="Pa-gamma aperture spectrum",
    )

    ax.plot(
        pagamma_fit["x"],
        pagamma_fit["model"],
        linewidth=2,
        label="Gaussian fit",
    )

    ax.axvline(
        pagamma_predicted,
        linestyle="--",
        label="Predicted wavelength",
    )

    ax.set_title(
        "M51 JWST Pa-gamma — 69-pixel aperture"
    )

    ax.set_xlabel(
        "Wavelength (nm)"
    )

    ax.set_ylabel(
        "Continuum-subtracted flux"
    )

    ax.legend()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_FIGURE,
        dpi=180,
    )

    plt.close()

    print(
        f"Saved:\n  "
        f"{OUTPUT_FIGURE}"
    )

    # ========================================================
    # 13. FINAL RESULT
    # ========================================================

    banner(
        "FINAL DIRECT PROFILE-FIT RESULT"
    )

    print(
        "Nominal aperture = 69 pixels"
    )

    print()

    print(
        f"Pa-beta:"
    )

    print(
        f"  fitted flux = "
        f"{pb:.8f} +/- {pb_err:.8f}"
    )

    print(
        f"  velocity = "
        f"{pabeta_fit['velocity_kms']:+.3f} +/- "
        f"{pabeta_fit['velocity_error_kms']:.3f} km/s"
    )

    print(
        f"  FWHM = "
        f"{pabeta_fit['fwhm_nm']:.6f} nm"
    )

    print()

    print(
        f"Pa-gamma:"
    )

    print(
        f"  fitted flux = "
        f"{pg:.8f} +/- {pg_err:.8f}"
    )

    print(
        f"  velocity = "
        f"{pagamma_fit['velocity_kms']:+.3f} +/- "
        f"{pagamma_fit['velocity_error_kms']:.3f} km/s"
    )

    print(
        f"  FWHM = "
        f"{pagamma_fit['fwhm_nm']:.6f} nm"
    )

    print()

    print(
        f"PROFILE-FIT Pa-beta / Pa-gamma = "
        f"{ratio:.8f} +/- {ratio_error:.8f}"
    )

    print()

    print(
        f"Velocity difference = "
        f"{velocity_difference:+.3f} +/- "
        f"{velocity_difference_error:.3f} km/s"
    )

    print()

    print(
        "No extinction inference has been performed."
    )

    print(
        "The profile-fit ratio should be evaluated together "
        "with the fixed-window convergence and profile residuals "
        "before being used for Storey-Hummer analysis."
    )

    print()
    print(
        "Direct Pa-beta / Pa-gamma profile-fit experiment complete."
    )


if __name__ == "__main__":
    main()
