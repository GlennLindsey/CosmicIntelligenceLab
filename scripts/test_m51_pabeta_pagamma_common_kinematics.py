#!/usr/bin/env python3

"""
M51 JWST NIRSPEC — Pa-beta / Pa-gamma COMMON-KINEMATICS FIT

Purpose
-------
Fit the summed 69-pixel JWST aperture spectra of Pa-beta and
Pa-gamma simultaneously using:

    * a common velocity centroid
    * a common velocity dispersion
    * independent line amplitudes
    * independent local linear continua

The existing Pa-gamma spatial product is comparison-only.

No extinction inference is performed.

This experiment tests whether Pa-beta and Pa-gamma are
consistent with the same emitting-gas kinematics.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.io import fits
from scipy.optimize import least_squares


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT = Path.home() / "Projects" / "cosmic_ai"

S3D_PATH = (
    PROJECT
    / "data/m51_jwst_level3/"
    / "jw03435-o006_t010_nirspec_g140m-f100lp_s3d.fits"
)

APERTURE_PATH = (
    PROJECT
    / "data/atomic_lines/"
    / "m51_jwst_extraction_aperture.csv"
)

EXISTING_PAGAMMA_PATH = (
    PROJECT
    / "data/atomic_lines/"
    / "m51_1096_pagamma_hydrogen_consistency_map.fits"
)

OUTPUT_DIR = PROJECT / "data/atomic_lines"

RESULTS_PATH = (
    OUTPUT_DIR
    / "m51_pabeta_pagamma_common_kinematics.csv"
)

PROFILE_PATH = (
    OUTPUT_DIR
    / "m51_pabeta_pagamma_common_kinematics_profiles.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "m51_pabeta_pagamma_common_kinematics_summary.csv"
)

FIGURE_PATH = (
    PROJECT
    / "m51_pabeta_pagamma_common_kinematics.png"
)

RESIDUAL_FIGURE_PATH = (
    PROJECT
    / "m51_pabeta_pagamma_common_kinematics_residuals.png"
)


# ============================================================
# HYDROGEN / VELOCITY CONSTANTS
# ============================================================

PA_BETA_REST_NM = 1281.807000
PA_GAMMA_REST_NM = 1093.800000

C_KMS = 299792.458

# Used only to initialize the fit.
M51_VELOCITY_KMS = 463.0


# ============================================================
# FIT WINDOWS
# ============================================================

PABETA_RANGE = (
    1275.0,
    1292.0,
)

PAGAMMA_RANGE = (
    1088.0,
    1102.0,
)


# Continuum windows

PABETA_BLUE = (
    1275.0,
    1279.0,
)

PABETA_RED = (
    1288.0,
    1292.0,
)

PAGAMMA_BLUE = (
    1088.0,
    1092.0,
)

PAGAMMA_RED = (
    1099.0,
    1102.0,
)


# ============================================================
# FIT BOUNDS
# ============================================================

VELOCITY_MIN = 300.0
VELOCITY_MAX = 800.0

SIGMA_V_MIN = 20.0
SIGMA_V_MAX = 500.0


# ============================================================
# UTILITY
# ============================================================

def banner(text):

    print()
    print("=" * 70)
    print(text)
    print("=" * 70)


def velocity_from_wavelength(
    rest_nm,
    observed_nm,
):

    return (
        C_KMS
        * (
            observed_nm / rest_nm
            - 1.0
        )
    )


def wavelength_from_velocity(
    rest_nm,
    velocity_kms,
):

    return (
        rest_nm
        * (
            1.0
            + velocity_kms / C_KMS
        )
    )


# ============================================================
# READ S3D
# ============================================================

def load_s3d():

    hdul = fits.open(
        S3D_PATH
    )

    sci_hdu = None
    err_hdu = None

    for hdu in hdul:

        name = (
            hdu.name
            or ""
        ).upper()

        if name == "SCI":
            sci_hdu = hdu

        elif name == "ERR":
            err_hdu = hdu

    if sci_hdu is None:
        hdul.close()
        raise RuntimeError(
            "SCI HDU not found."
        )

    if err_hdu is None:
        hdul.close()
        raise RuntimeError(
            "ERR HDU not found."
        )

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

    if sci.ndim != 3:
        hdul.close()
        raise RuntimeError(
            "SCI is not a 3-D S3D cube."
        )

    return (
        hdul,
        sci,
        err,
        header,
    )


# ============================================================
# WAVELENGTH ARRAY
# ============================================================

def build_wavelength_array(
    header,
    n_spectral,
):

    """
    Build wavelength array from the S3D spectral WCS.

    IMPORTANT:
    For this JWST S3D cube the numerical CRVAL3/CDELT3 values
    are expressed in micrometres.

    Example:

        CRVAL3 = 0.970318...
        CDELT3 = 0.000636...

    Therefore:

        0.970318 um = 970.318 nm
    """

    required = [
        "CRVAL3",
        "CRPIX3",
        "CDELT3",
    ]

    for key in required:

        if key not in header:

            raise RuntimeError(
                f"Missing spectral WCS keyword: {key}"
            )

    crval3 = float(
        header["CRVAL3"]
    )

    crpix3 = float(
        header["CRPIX3"]
    )

    cdelt3 = float(
        header["CDELT3"]
    )

    pixel = np.arange(
        n_spectral,
        dtype=float,
    )

    wavelength_um = (
        crval3
        +
        (
            pixel
            + 1.0
            - crpix3
        )
        * cdelt3
    )

    wavelength_nm = (
        wavelength_um
        * 1000.0
    )

    return wavelength_nm


# ============================================================
# APERTURE
# ============================================================

def load_aperture():

    aperture = pd.read_csv(
        APERTURE_PATH
    )

    required = {
        "x_pixel",
        "y_pixel",
        "inside_nominal_aperture",
    }

    missing = (
        required
        - set(aperture.columns)
    )

    if missing:

        raise RuntimeError(
            "Aperture file is missing columns: "
            + ", ".join(sorted(missing))
        )

    aperture = aperture[
        aperture[
            "inside_nominal_aperture"
        ].astype(bool)
    ].copy()

    if len(aperture) != 69:

        raise RuntimeError(
            "Expected exactly 69 nominal "
            f"aperture pixels; found {len(aperture)}."
        )

    print(
        f"Aperture pixels = {len(aperture)}"
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

    aperture_sci = sci[
        :,
        y,
        x,
    ]

    aperture_err = err[
        :,
        y,
        x,
    ]

    flux = np.nansum(
        aperture_sci,
        axis=1,
    )

    error = np.sqrt(
        np.nansum(
            aperture_err ** 2,
            axis=1,
        )
    )

    return (
        flux,
        error,
    )


# ============================================================
# CONTINUUM
# ============================================================

def make_continuum_mask(
    wavelength,
    blue,
    red,
):

    return (
        (
            (wavelength >= blue[0])
            &
            (wavelength <= blue[1])
        )
        |
        (
            (wavelength >= red[0])
            &
            (wavelength <= red[1])
        )
    )


def fit_continuum(
    wavelength,
    flux,
    error,
    blue,
    red,
):

    mask = make_continuum_mask(
        wavelength,
        blue,
        red,
    )

    valid = (
        mask
        &
        np.isfinite(flux)
        &
        np.isfinite(error)
        &
        (error > 0)
    )

    if np.sum(valid) < 3:

        raise RuntimeError(
            "Insufficient continuum points."
        )

    x = wavelength[valid]
    y = flux[valid]
    w = 1.0 / error[valid]

    slope, intercept = np.polyfit(
        x,
        y,
        1,
        w=w,
    )

    continuum = (
        slope
        * wavelength
        + intercept
    )

    return (
        continuum,
        slope,
        intercept,
        valid,
    )


# ============================================================
# INITIAL AMPLITUDE
# ============================================================

def initial_amplitude(
    wavelength,
    flux,
    continuum,
    rest_nm,
    velocity,
):

    center = wavelength_from_velocity(
        rest_nm,
        velocity,
    )

    sigma_v = 150.0

    sigma_lambda = (
        center
        * sigma_v
        / C_KMS
    )

    profile = np.exp(
        -0.5
        * (
            (
                wavelength
                - center
            )
            / sigma_lambda
        ) ** 2
    )

    residual = (
        flux
        - continuum
    )

    numerator = np.nansum(
        residual
        * profile
    )

    denominator = np.nansum(
        profile ** 2
    )

    if denominator <= 0:

        return 1.0

    amplitude = (
        numerator
        / denominator
    )

    return max(
        float(amplitude),
        1.0,
    )


# ============================================================
# COMMON-KINEMATIC MODEL
# ============================================================

def common_model(
    wavelength_beta,
    wavelength_gamma,
    parameters,
):

    """
    Parameters:

        0 = common velocity
        1 = common sigma_v

        2 = Pa-beta amplitude
        3 = Pa-beta continuum slope
        4 = Pa-beta continuum intercept

        5 = Pa-gamma amplitude
        6 = Pa-gamma continuum slope
        7 = Pa-gamma continuum intercept
    """

    velocity = parameters[0]
    sigma_v = parameters[1]

    beta_amplitude = parameters[2]
    beta_slope = parameters[3]
    beta_intercept = parameters[4]

    gamma_amplitude = parameters[5]
    gamma_slope = parameters[6]
    gamma_intercept = parameters[7]

    beta_center = (
        wavelength_from_velocity(
            PA_BETA_REST_NM,
            velocity,
        )
    )

    gamma_center = (
        wavelength_from_velocity(
            PA_GAMMA_REST_NM,
            velocity,
        )
    )

    beta_sigma = (
        beta_center
        * sigma_v
        / C_KMS
    )

    gamma_sigma = (
        gamma_center
        * sigma_v
        / C_KMS
    )

    beta_line = (
        beta_amplitude
        *
        np.exp(
            -0.5
            *
            (
                (
                    wavelength_beta
                    - beta_center
                )
                / beta_sigma
            ) ** 2
        )
    )

    gamma_line = (
        gamma_amplitude
        *
        np.exp(
            -0.5
            *
            (
                (
                    wavelength_gamma
                    - gamma_center
                )
                / gamma_sigma
            ) ** 2
        )
    )

    beta_continuum = (
        beta_slope
        * wavelength_beta
        + beta_intercept
    )

    gamma_continuum = (
        gamma_slope
        * wavelength_gamma
        + gamma_intercept
    )

    beta_model = (
        beta_continuum
        + beta_line
    )

    gamma_model = (
        gamma_continuum
        + gamma_line
    )

    return (
        beta_model,
        gamma_model,
        beta_center,
        gamma_center,
        beta_sigma,
        gamma_sigma,
    )


# ============================================================
# RESIDUAL
# ============================================================

def residual_function(
    parameters,
    beta_wavelength,
    beta_flux,
    beta_error,
    gamma_wavelength,
    gamma_flux,
    gamma_error,
):

    (
        beta_model,
        gamma_model,
        _,
        _,
        _,
        _,
    ) = common_model(
        beta_wavelength,
        gamma_wavelength,
        parameters,
    )

    beta_residual = (
        beta_flux
        - beta_model
    ) / beta_error

    gamma_residual = (
        gamma_flux
        - gamma_model
    ) / gamma_error

    return np.concatenate(
        [
            beta_residual,
            gamma_residual,
        ]
    )


# ============================================================
# MAIN
# ============================================================

def main():

    banner(
        "M51 JWST NIRSPEC — Pa-beta / Pa-gamma "
        "COMMON-KINEMATICS FIT"
    )

    print(
        "Purpose:"
    )

    print(
        "Fit Pa-beta and Pa-gamma simultaneously "
        "with a common velocity centroid and width."
    )

    print()
    print(
        "No extinction inference is performed."
    )

    # --------------------------------------------------------
    # 1
    # --------------------------------------------------------

    banner(
        "1. READING JWST S3D"
    )

    (
        hdul,
        sci,
        err,
        header,
    ) = load_s3d()

    wavelength_nm = (
        build_wavelength_array(
            header,
            sci.shape[0],
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
        f"{wavelength_nm[0]:.12f} - "
        f"{wavelength_nm[-1]:.12f} nm"
    )

    print(
        f"Spectral spacing = "
        f"{spacing:.12f} nm"
    )

    # Safety check
    if not (
        900.0
        < wavelength_nm[0]
        < 1100.0
    ):

        hdul.close()

        raise RuntimeError(
            "Wavelength construction failed: "
            f"first wavelength = "
            f"{wavelength_nm[0]} nm"
        )

    if not (
        1700.0
        < wavelength_nm[-1]
        < 2000.0
    ):

        hdul.close()

        raise RuntimeError(
            "Wavelength construction failed: "
            f"last wavelength = "
            f"{wavelength_nm[-1]} nm"
        )

    # --------------------------------------------------------
    # 2
    # --------------------------------------------------------

    banner(
        "2. LOADING NOMINAL APERTURE"
    )

    aperture = load_aperture()

    # --------------------------------------------------------
    # 3
    # --------------------------------------------------------

    banner(
        "3. EXTRACTING 69-PIXEL APERTURE SPECTRA"
    )

    aperture_flux, aperture_error = (
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

    # --------------------------------------------------------
    # 4
    # --------------------------------------------------------

    banner(
        "4. DEFINING Pa-beta / Pa-gamma FIT WINDOWS"
    )

    beta_mask = (
        (wavelength_nm >= PABETA_RANGE[0])
        &
        (wavelength_nm <= PABETA_RANGE[1])
    )

    gamma_mask = (
        (wavelength_nm >= PAGAMMA_RANGE[0])
        &
        (wavelength_nm <= PAGAMMA_RANGE[1])
    )

    beta_w = wavelength_nm[
        beta_mask
    ]

    gamma_w = wavelength_nm[
        gamma_mask
    ]

    beta_flux = aperture_flux[
        beta_mask
    ]

    gamma_flux = aperture_flux[
        gamma_mask
    ]

    beta_error = aperture_error[
        beta_mask
    ]

    gamma_error = aperture_error[
        gamma_mask
    ]

    print(
        f"Pa-beta planes = "
        f"{len(beta_w)}"
    )

    print(
        f"Pa-gamma planes = "
        f"{len(gamma_w)}"
    )

    if len(beta_w) == 0:

        hdul.close()

        raise RuntimeError(
            "No Pa-beta wavelength planes found."
        )

    if len(gamma_w) == 0:

        hdul.close()

        raise RuntimeError(
            "No Pa-gamma wavelength planes found."
        )

    # --------------------------------------------------------
    # 5
    # --------------------------------------------------------

    banner(
        "5. INITIAL KINEMATIC ESTIMATE"
    )

    beta_predicted = (
        wavelength_from_velocity(
            PA_BETA_REST_NM,
            M51_VELOCITY_KMS,
        )
    )

    gamma_predicted = (
        wavelength_from_velocity(
            PA_GAMMA_REST_NM,
            M51_VELOCITY_KMS,
        )
    )

    beta_index = int(
        np.argmin(
            np.abs(
                beta_w
                - beta_predicted
            )
        )
    )

    gamma_index = int(
        np.argmin(
            np.abs(
                gamma_w
                - gamma_predicted
            )
        )
    )

    beta_nearest = (
        beta_w[beta_index]
    )

    gamma_nearest = (
        gamma_w[gamma_index]
    )

    beta_nearest_velocity = (
        velocity_from_wavelength(
            PA_BETA_REST_NM,
            beta_nearest,
        )
    )

    gamma_nearest_velocity = (
        velocity_from_wavelength(
            PA_GAMMA_REST_NM,
            gamma_nearest,
        )
    )

    print(
        f"Pa-beta predicted = "
        f"{beta_predicted:.12f} nm"
    )

    print(
        f"Pa-gamma predicted = "
        f"{gamma_predicted:.12f} nm"
    )

    print(
        f"Pa-beta nearest plane = "
        f"{beta_index}"
    )

    print(
        f"Pa-beta wavelength = "
        f"{beta_nearest:.12f} nm"
    )

    print(
        f"Pa-gamma nearest plane = "
        f"{gamma_index}"
    )

    print(
        f"Pa-gamma wavelength = "
        f"{gamma_nearest:.12f} nm"
    )

    print(
        f"Pa-beta velocity = "
        f"{beta_nearest_velocity:+.3f} km/s"
    )

    print(
        f"Pa-gamma velocity = "
        f"{gamma_nearest_velocity:+.3f} km/s"
    )

    # --------------------------------------------------------
    # 6
    # --------------------------------------------------------

    banner(
        "6. INITIALIZING LOCAL CONTINUA"
    )

    (
        beta_continuum,
        beta_slope,
        beta_intercept,
        beta_continuum_mask,
    ) = fit_continuum(
        beta_w,
        beta_flux,
        beta_error,
        PABETA_BLUE,
        PABETA_RED,
    )

    (
        gamma_continuum,
        gamma_slope,
        gamma_intercept,
        gamma_continuum_mask,
    ) = fit_continuum(
        gamma_w,
        gamma_flux,
        gamma_error,
        PAGAMMA_BLUE,
        PAGAMMA_RED,
    )

    beta_amp = initial_amplitude(
        beta_w,
        beta_flux,
        beta_continuum,
        PA_BETA_REST_NM,
        beta_nearest_velocity,
    )

    gamma_amp = initial_amplitude(
        gamma_w,
        gamma_flux,
        gamma_continuum,
        PA_GAMMA_REST_NM,
        gamma_nearest_velocity,
    )

    print(
        f"Pa-beta continuum planes = "
        f"{np.sum(beta_continuum_mask)}"
    )

    print(
        f"Pa-gamma continuum planes = "
        f"{np.sum(gamma_continuum_mask)}"
    )

    # --------------------------------------------------------
    # 7
    # --------------------------------------------------------

    banner(
        "7. COMMON-KINEMATICS FIT"
    )

    initial = np.array(
        [
            np.mean(
                [
                    beta_nearest_velocity,
                    gamma_nearest_velocity,
                ]
            ),

            150.0,

            beta_amp,
            beta_slope,
            beta_intercept,

            gamma_amp,
            gamma_slope,
            gamma_intercept,
        ],
        dtype=float,
    )

    lower = np.array(
        [
            VELOCITY_MIN,
            SIGMA_V_MIN,

            0.0,
            -np.inf,
            -np.inf,

            0.0,
            -np.inf,
            -np.inf,
        ],
        dtype=float,
    )

    upper = np.array(
        [
            VELOCITY_MAX,
            SIGMA_V_MAX,

            np.inf,
            np.inf,
            np.inf,

            np.inf,
            np.inf,
            np.inf,
        ],
        dtype=float,
    )

    valid_beta = (
        np.isfinite(beta_flux)
        &
        np.isfinite(beta_error)
        &
        (beta_error > 0)
    )

    valid_gamma = (
        np.isfinite(gamma_flux)
        &
        np.isfinite(gamma_error)
        &
        (gamma_error > 0)
    )

    beta_w_fit = beta_w[
        valid_beta
    ]

    beta_flux_fit = beta_flux[
        valid_beta
    ]

    beta_error_fit = beta_error[
        valid_beta
    ]

    gamma_w_fit = gamma_w[
        valid_gamma
    ]

    gamma_flux_fit = gamma_flux[
        valid_gamma
    ]

    gamma_error_fit = gamma_error[
        valid_gamma
    ]

    fit = least_squares(
        residual_function,
        initial,
        bounds=(
            lower,
            upper,
        ),
        args=(
            beta_w_fit,
            beta_flux_fit,
            beta_error_fit,
            gamma_w_fit,
            gamma_flux_fit,
            gamma_error_fit,
        ),
        max_nfev=100000,
    )

    parameters = fit.x

    print(
        f"Optimization success = "
        f"{fit.success}"
    )

    print(
        f"Optimization message = "
        f"{fit.message}"
    )

    # --------------------------------------------------------
    # 8
    # --------------------------------------------------------

    banner(
        "8. COMMON-KINEMATIC MODEL"
    )

    (
        beta_model,
        gamma_model,
        beta_center,
        gamma_center,
        beta_sigma,
        gamma_sigma,
    ) = common_model(
        beta_w,
        gamma_w,
        parameters,
    )

    common_velocity = (
        parameters[0]
    )

    common_sigma_v = (
        parameters[1]
    )

    beta_amplitude = (
        parameters[2]
    )

    gamma_amplitude = (
        parameters[5]
    )

    beta_flux_model = (
        beta_amplitude
        * beta_sigma
        * np.sqrt(
            2.0 * np.pi
        )
    )

    gamma_flux_model = (
        gamma_amplitude
        * gamma_sigma
        * np.sqrt(
            2.0 * np.pi
        )
    )

    ratio = (
        beta_flux_model
        / gamma_flux_model
    )

    common_fwhm_v = (
        2.354820045
        * common_sigma_v
    )

    print(
        f"Common velocity = "
        f"{common_velocity:+.6f} km/s"
    )

    print(
        f"Common sigma_v = "
        f"{common_sigma_v:.6f} km/s"
    )

    print(
        f"Common FWHM velocity = "
        f"{common_fwhm_v:.6f} km/s"
    )

    print(
        f"Pa-beta centroid = "
        f"{beta_center:.9f} nm"
    )

    print(
        f"Pa-gamma centroid = "
        f"{gamma_center:.9f} nm"
    )

    print(
        f"Pa-beta sigma = "
        f"{beta_sigma:.9f} nm"
    )

    print(
        f"Pa-gamma sigma = "
        f"{gamma_sigma:.9f} nm"
    )

    print(
        f"Pa-beta fitted flux = "
        f"{beta_flux_model:.9f}"
    )

    print(
        f"Pa-gamma fitted flux = "
        f"{gamma_flux_model:.9f}"
    )

    print(
        f"Common-kinematics ratio = "
        f"{ratio:.9f}"
    )

    # --------------------------------------------------------
    # 9
    # --------------------------------------------------------

    banner(
        "9. FIT QUALITY"
    )

    beta_model_fit = (
        beta_model[valid_beta]
    )

    gamma_model_fit = (
        gamma_model[valid_gamma]
    )

    beta_residual = (
        beta_flux_fit
        - beta_model_fit
    )

    gamma_residual = (
        gamma_flux_fit
        - gamma_model_fit
    )

    beta_chi2 = np.sum(
        (
            beta_residual
            / beta_error_fit
        ) ** 2
    )

    gamma_chi2 = np.sum(
        (
            gamma_residual
            / gamma_error_fit
        ) ** 2
    )

    total_chi2 = (
        beta_chi2
        + gamma_chi2
    )

    n_data = (
        len(beta_flux_fit)
        + len(gamma_flux_fit)
    )

    n_parameters = 8

    dof = (
        n_data
        - n_parameters
    )

    reduced_chi2 = (
        total_chi2
        / dof
    )

    beta_dof = (
        len(beta_flux_fit)
        - 5
    )

    gamma_dof = (
        len(gamma_flux_fit)
        - 5
    )

    beta_reduced_chi2 = (
        beta_chi2
        / beta_dof
    )

    gamma_reduced_chi2 = (
        gamma_chi2
        / gamma_dof
    )

    print(
        f"Pa-beta chi2 = "
        f"{beta_chi2:.6f}"
    )

    print(
        f"Pa-gamma chi2 = "
        f"{gamma_chi2:.6f}"
    )

    print(
        f"Total chi2 = "
        f"{total_chi2:.6f}"
    )

    print(
        f"Degrees of freedom = "
        f"{dof}"
    )

    print(
        f"Reduced chi2 = "
        f"{reduced_chi2:.6f}"
    )

    print(
        f"Pa-beta reduced chi2 = "
        f"{beta_reduced_chi2:.6f}"
    )

    print(
        f"Pa-gamma reduced chi2 = "
        f"{gamma_reduced_chi2:.6f}"
    )

    # --------------------------------------------------------
    # 10
    # --------------------------------------------------------

    banner(
        "10. PARAMETER UNCERTAINTIES"
    )

    try:

        jacobian = fit.jac

        covariance = (
            np.linalg.inv(
                jacobian.T
                @ jacobian
            )
            * reduced_chi2
        )

        parameter_errors = np.sqrt(
            np.diag(
                covariance
            )
        )

        velocity_error = (
            parameter_errors[0]
        )

        sigma_v_error = (
            parameter_errors[1]
        )

        beta_amplitude_error = (
            parameter_errors[2]
        )

        gamma_amplitude_error = (
            parameter_errors[5]
        )

        beta_flux_error = (
            beta_sigma
            * np.sqrt(
                2.0 * np.pi
            )
            * beta_amplitude_error
        )

        gamma_flux_error = (
            gamma_sigma
            * np.sqrt(
                2.0 * np.pi
            )
            * gamma_amplitude_error
        )

        ratio_error = (
            ratio
            * np.sqrt(
                (
                    beta_flux_error
                    / beta_flux_model
                ) ** 2
                +
                (
                    gamma_flux_error
                    / gamma_flux_model
                ) ** 2
            )
        )

    except np.linalg.LinAlgError:

        velocity_error = np.nan
        sigma_v_error = np.nan

        beta_flux_error = np.nan
        gamma_flux_error = np.nan

        ratio_error = np.nan

    print(
        f"Common velocity = "
        f"{common_velocity:+.6f} +/- "
        f"{velocity_error:.6f} km/s"
    )

    print(
        f"Common sigma_v = "
        f"{common_sigma_v:.6f} +/- "
        f"{sigma_v_error:.6f} km/s"
    )

    print(
        f"Pa-beta flux = "
        f"{beta_flux_model:.9f} +/- "
        f"{beta_flux_error:.9f}"
    )

    print(
        f"Pa-gamma flux = "
        f"{gamma_flux_model:.9f} +/- "
        f"{gamma_flux_error:.9f}"
    )

    print(
        f"Ratio = "
        f"{ratio:.9f} +/- "
        f"{ratio_error:.9f}"
    )

    # --------------------------------------------------------
    # 11
    # --------------------------------------------------------

    banner(
        "11. COMPARISON WITH INDEPENDENT PROFILE FIT"
    )

    independent_beta = (
        2125.10272270
    )

    independent_gamma = (
        665.09589012
    )

    independent_ratio = (
        3.19518246
    )

    print(
        f"Independent Pa-beta fit = "
        f"{independent_beta:.6f}"
    )

    print(
        f"Independent Pa-gamma fit = "
        f"{independent_gamma:.6f}"
    )

    print(
        f"Independent ratio = "
        f"{independent_ratio:.9f}"
    )

    print(
        f"Common-kinematics ratio = "
        f"{ratio:.9f}"
    )

    print(
        f"Ratio difference = "
        f"{ratio - independent_ratio:+.9f}"
    )

    print(
        f"Relative difference = "
        f"{100.0 * (ratio / independent_ratio - 1.0):+.3f}%"
    )

    # --------------------------------------------------------
    # 12
    # --------------------------------------------------------

    banner(
        "12. EXISTING Pa-GAMMA PRODUCT COMPARISON"
    )

    existing_gamma = np.nan

    if EXISTING_PAGAMMA_PATH.exists():

        with fits.open(
            EXISTING_PAGAMMA_PATH
        ) as existing_hdul:

            existing_map = np.asarray(
                existing_hdul[0].data,
                dtype=float,
            )

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

        existing_values = (
            existing_map[y, x]
        )

        existing_gamma = np.nansum(
            existing_values
        )

        print(
            f"Existing product aperture sum = "
            f"{existing_gamma:.9f}"
        )

        print(
            f"Common-fit / existing = "
            f"{gamma_flux_model / existing_gamma:.9f}"
        )

    else:

        print(
            "Existing Pa-gamma product not found."
        )

    # --------------------------------------------------------
    # 13
    # --------------------------------------------------------

    banner(
        "13. SAVING FITTED PROFILES"
    )

    beta_profile = pd.DataFrame(
        {
            "line": "Pa-beta",
            "wavelength_nm": beta_w,
            "observed_flux": beta_flux,
            "error": beta_error,
            "model_flux": beta_model,
            "residual": (
                beta_flux
                - beta_model
            ),
        }
    )

    gamma_profile = pd.DataFrame(
        {
            "line": "Pa-gamma",
            "wavelength_nm": gamma_w,
            "observed_flux": gamma_flux,
            "error": gamma_error,
            "model_flux": gamma_model,
            "residual": (
                gamma_flux
                - gamma_model
            ),
        }
    )

    profiles = pd.concat(
        [
            beta_profile,
            gamma_profile,
        ],
        ignore_index=True,
    )

    profiles.to_csv(
        PROFILE_PATH,
        index=False,
    )

    print(
        f"Saved:\n  {PROFILE_PATH}"
    )

    # --------------------------------------------------------
    # 14
    # --------------------------------------------------------

    banner(
        "14. SAVING RESULTS"
    )

    results = pd.DataFrame(
        [
            {
                "aperture_pixels":
                    len(aperture),

                "common_velocity_kms":
                    common_velocity,

                "common_velocity_uncertainty_kms":
                    velocity_error,

                "common_sigma_velocity_kms":
                    common_sigma_v,

                "common_sigma_velocity_uncertainty_kms":
                    sigma_v_error,

                "common_fwhm_velocity_kms":
                    common_fwhm_v,

                "pabeta_centroid_nm":
                    beta_center,

                "pagamma_centroid_nm":
                    gamma_center,

                "pabeta_sigma_nm":
                    beta_sigma,

                "pagamma_sigma_nm":
                    gamma_sigma,

                "pabeta_flux":
                    beta_flux_model,

                "pabeta_flux_uncertainty":
                    beta_flux_error,

                "pagamma_flux":
                    gamma_flux_model,

                "pagamma_flux_uncertainty":
                    gamma_flux_error,

                "pabeta_pagamma_ratio":
                    ratio,

                "pabeta_pagamma_ratio_uncertainty":
                    ratio_error,

                "beta_chi2":
                    beta_chi2,

                "gamma_chi2":
                    gamma_chi2,

                "total_chi2":
                    total_chi2,

                "degrees_of_freedom":
                    dof,

                "reduced_chi2":
                    reduced_chi2,

                "beta_reduced_chi2":
                    beta_reduced_chi2,

                "gamma_reduced_chi2":
                    gamma_reduced_chi2,

                "existing_pagamma_flux":
                    existing_gamma,

                "common_fit_over_existing":
                    (
                        gamma_flux_model
                        / existing_gamma
                        if np.isfinite(
                            existing_gamma
                        )
                        else np.nan
                    ),
            }
        ]
    )

    results.to_csv(
        RESULTS_PATH,
        index=False,
    )

    print(
        f"Saved:\n  {RESULTS_PATH}"
    )

    # --------------------------------------------------------
    # 15
    # --------------------------------------------------------

    banner(
        "15. SAVING SUMMARY"
    )

    summary = pd.DataFrame(
        [
            {
                "aperture_pixels":
                    len(aperture),

                "common_velocity_kms":
                    common_velocity,

                "common_velocity_uncertainty_kms":
                    velocity_error,

                "common_fwhm_velocity_kms":
                    common_fwhm_v,

                "pabeta_flux":
                    beta_flux_model,

                "pabeta_flux_uncertainty":
                    beta_flux_error,

                "pagamma_flux":
                    gamma_flux_model,

                "pagamma_flux_uncertainty":
                    gamma_flux_error,

                "ratio":
                    ratio,

                "ratio_uncertainty":
                    ratio_error,

                "reduced_chi2":
                    reduced_chi2,

                "independent_profile_ratio":
                    independent_ratio,

                "existing_pagamma_flux":
                    existing_gamma,
            }
        ]
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    print(
        f"Saved:\n  {SUMMARY_PATH}"
    )

    # --------------------------------------------------------
    # 16
    # --------------------------------------------------------

    banner(
        "16. CREATING COMMON-KINEMATICS FIGURE"
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11, 9),
    )

    axes[0].errorbar(
        beta_w,
        beta_flux,
        yerr=beta_error,
        fmt="o",
        markersize=4,
        alpha=0.7,
        label="Observed Pa-beta",
    )

    axes[0].plot(
        beta_w,
        beta_model,
        linewidth=2,
        label="Common-kinematics model",
    )

    axes[0].axvline(
        beta_center,
        linestyle="--",
        label="Common centroid",
    )

    axes[0].set_ylabel(
        "Flux / aperture"
    )

    axes[0].set_title(
        "Pa-beta — common kinematics"
    )

    axes[0].legend()

    axes[1].errorbar(
        gamma_w,
        gamma_flux,
        yerr=gamma_error,
        fmt="o",
        markersize=4,
        alpha=0.7,
        label="Observed Pa-gamma",
    )

    axes[1].plot(
        gamma_w,
        gamma_model,
        linewidth=2,
        label="Common-kinematics model",
    )

    axes[1].axvline(
        gamma_center,
        linestyle="--",
        label="Common centroid",
    )

    axes[1].set_xlabel(
        "Wavelength (nm)"
    )

    axes[1].set_ylabel(
        "Flux / aperture"
    )

    axes[1].set_title(
        "Pa-gamma — common kinematics"
    )

    axes[1].legend()

    fig.tight_layout()

    fig.savefig(
        FIGURE_PATH,
        dpi=180,
    )

    plt.close(fig)

    print(
        f"Saved:\n  {FIGURE_PATH}"
    )

    # --------------------------------------------------------
    # 17
    # --------------------------------------------------------

    banner(
        "17. CREATING RESIDUAL FIGURE"
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11, 8),
    )

    axes[0].axhline(
        0,
        linestyle="--",
    )

    axes[0].plot(
        beta_w,
        beta_flux - beta_model,
        marker="o",
        markersize=4,
    )

    axes[0].set_ylabel(
        "Residual"
    )

    axes[0].set_title(
        "Pa-beta common-kinematics residual"
    )

    axes[1].axhline(
        0,
        linestyle="--",
    )

    axes[1].plot(
        gamma_w,
        gamma_flux - gamma_model,
        marker="o",
        markersize=4,
    )

    axes[1].set_xlabel(
        "Wavelength (nm)"
    )

    axes[1].set_ylabel(
        "Residual"
    )

    axes[1].set_title(
        "Pa-gamma common-kinematics residual"
    )

    fig.tight_layout()

    fig.savefig(
        RESIDUAL_FIGURE_PATH,
        dpi=180,
    )

    plt.close(fig)

    print(
        f"Saved:\n  {RESIDUAL_FIGURE_PATH}"
    )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    banner(
        "FINAL COMMON-KINEMATICS RESULT"
    )

    print(
        f"Nominal aperture = "
        f"{len(aperture)} pixels"
    )

    print()

    print(
        f"Common velocity = "
        f"{common_velocity:+.3f} +/- "
        f"{velocity_error:.3f} km/s"
    )

    print(
        f"Common FWHM velocity = "
        f"{common_fwhm_v:.3f} km/s"
    )

    print()

    print(
        f"Pa-beta fitted flux = "
        f"{beta_flux_model:.6f} +/- "
        f"{beta_flux_error:.6f}"
    )

    print(
        f"Pa-gamma fitted flux = "
        f"{gamma_flux_model:.6f} +/- "
        f"{gamma_flux_error:.6f}"
    )

    print()

    print(
        f"COMMON-KINEMATICS "
        f"Pa-beta / Pa-gamma = "
        f"{ratio:.6f} +/- "
        f"{ratio_error:.6f}"
    )

    print()

    print(
        f"Reduced chi2 = "
        f"{reduced_chi2:.3f}"
    )

    print(
        f"Pa-beta reduced chi2 = "
        f"{beta_reduced_chi2:.3f}"
    )

    print(
        f"Pa-gamma reduced chi2 = "
        f"{gamma_reduced_chi2:.3f}"
    )

    print()

    print(
        "No extinction inference has been performed."
    )

    print(
        "The common-kinematics ratio should be evaluated "
        "together with the independent profile fit and "
        "the residual structure before proceeding to "
        "Storey-Hummer analysis."
    )

    print()
    print(
        "Common-kinematics experiment complete."
    )

    hdul.close()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
