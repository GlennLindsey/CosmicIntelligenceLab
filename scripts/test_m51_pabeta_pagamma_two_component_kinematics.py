#!/usr/bin/env python3

"""
M51 JWST NIRSpec — Pa-beta / Pa-gamma TWO-COMPONENT KINEMATICS

Purpose
-------
Fit the summed 69-pixel JWST aperture spectra of Pa-beta and Pa-gamma
with a two-component model.

The two hydrogen lines share the same two velocity components:

    component 1:
        common velocity
        common velocity width

    component 2:
        common velocity
        common velocity width

Each component has independent Pa-beta and Pa-gamma fluxes.

No extinction inference is performed.

The existing Pa-gamma spatial product is comparison-only.
The spectra are reconstructed directly from the JWST S3D cube.

Outputs
-------
data/atomic_lines/
    m51_pabeta_pagamma_two_component_kinematics.csv
    m51_pabeta_pagamma_two_component_profiles.csv
    m51_pabeta_pagamma_two_component_summary.csv

m51_pabeta_pagamma_two_component_kinematics.png
m51_pabeta_pagamma_two_component_residuals.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.io import fits
from scipy.optimize import least_squares


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path.home() / "Projects" / "cosmic_ai"

S3D_PATH = (
    PROJECT_ROOT
    / "data"
    / "m51_jwst_level3"
    / "jw03435-o006_t010_nirspec_g140m-f100lp_s3d.fits"
)

APERTURE_PATH = (
    PROJECT_ROOT
    / "data"
    / "atomic_lines"
    / "m51_jwst_extraction_aperture.csv"
)

EXISTING_PAGAMMA_PATH = (
    PROJECT_ROOT
    / "data"
    / "atomic_lines"
    / "m51_1096_pagamma_hydrogen_consistency_map.fits"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "atomic_lines"

RESULTS_PATH = (
    OUTPUT_DIR
    / "m51_pabeta_pagamma_two_component_kinematics.csv"
)

PROFILE_PATH = (
    OUTPUT_DIR
    / "m51_pabeta_pagamma_two_component_profiles.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "m51_pabeta_pagamma_two_component_summary.csv"
)

FIGURE_PATH = (
    PROJECT_ROOT
    / "m51_pabeta_pagamma_two_component_kinematics.png"
)

RESIDUAL_FIGURE_PATH = (
    PROJECT_ROOT
    / "m51_pabeta_pagamma_two_component_residuals.png"
)


# ============================================================
# CONSTANTS
# ============================================================

C_KMS = 299792.458

PA_BETA_REST_NM = 1281.807000
PA_GAMMA_REST_NM = 1093.800000

M51_VELOCITY_KMS = 463.000

RESOLVING_POWER = 2700.0

# Windows around the expected lines.
BETA_WINDOW_NM = 8.0
GAMMA_WINDOW_NM = 7.0

# Continuum regions relative to the expected observed line.
BETA_BLUE_OFFSET = (-14.0, -6.0)
BETA_RED_OFFSET = (6.0, 14.0)

GAMMA_BLUE_OFFSET = (-15.0, -7.0)
GAMMA_RED_OFFSET = (7.0, 15.0)

# Velocity bounds for the two components.
MIN_VELOCITY = -200.0
MAX_VELOCITY = 1500.0

MIN_SIGMA_V = 20.0
MAX_SIGMA_V = 500.0

# Small positive floor to avoid zero uncertainty.
MIN_ERROR = 1.0e-6


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def banner(text):
    print()
    print("=" * 70)
    print(text)
    print("=" * 70)


def velocity_from_wavelength(rest_nm, observed_nm):
    return C_KMS * (observed_nm / rest_nm - 1.0)


def wavelength_from_velocity(rest_nm, velocity_kms):
    return rest_nm * (1.0 + velocity_kms / C_KMS)


def predicted_wavelength(rest_nm, velocity_kms):
    return wavelength_from_velocity(
        rest_nm,
        velocity_kms,
    )

def wavelength_from_spectral_wcs(header, n_spec):
    """
    Build the JWST S3D spectral wavelength array.

    The NIRSpec S3D spectral WCS is expressed in microns
    (CUNIT3 = 'um'). Convert the resulting wavelength to nm.
    """

    crval3 = float(header["CRVAL3"])
    crpix3 = float(header["CRPIX3"])
    cdelt3 = float(header["CDELT3"])

    pixel = np.arange(
        n_spec,
        dtype=float,
    ) + 1.0

    wavelength_um = (
        crval3
        + (pixel - crpix3) * cdelt3
    )

    wavelength_nm = (
        wavelength_um * 1000.0
    )

    return wavelength_nm


def gaussian_unit_velocity(velocity, center, sigma):
    """
    Unit-integral Gaussian in velocity space.

    Integral over velocity = 1.
    """

    return (
        np.exp(
            -0.5
            * ((velocity - center) / sigma) ** 2
        )
        / (
            np.sqrt(2.0 * np.pi)
            * sigma
        )
    )


def line_model(
    wavelength,
    rest_nm,
    velocity1,
    sigma1,
    velocity2,
    sigma2,
    flux1,
    flux2,
):
    """
    Two-component Gaussian line model.

    Flux parameters are integrated line fluxes in the same
    arbitrary spectral-flux units used by the aperture spectrum.
    """

    v = velocity_from_wavelength(
        rest_nm,
        wavelength,
    )

    component1 = (
        flux1
        * gaussian_unit_velocity(
            v,
            velocity1,
            sigma1,
        )
        * C_KMS
        / rest_nm
    )

    component2 = (
        flux2
        * gaussian_unit_velocity(
            v,
            velocity2,
            sigma2,
        )
        * C_KMS
        / rest_nm
    )

    return component1 + component2


def continuum_fit(
    wavelength,
    spectrum,
    error,
    blue_range,
    red_range,
):
    """
    Fit a straight-line continuum using blue and red regions.
    """

    blue = (
        (wavelength >= blue_range[0])
        & (wavelength <= blue_range[1])
        & np.isfinite(spectrum)
    )

    red = (
        (wavelength >= red_range[0])
        & (wavelength <= red_range[1])
        & np.isfinite(spectrum)
    )

    idx = blue | red

    if np.sum(idx) < 4:
        raise RuntimeError(
            "Insufficient continuum points."
        )

    x = wavelength[idx]
    y = spectrum[idx]

    if error is not None:
        e = error[idx]
        e = np.where(
            np.isfinite(e) & (e > 0),
            e,
            np.nanmedian(
                e[
                    np.isfinite(e)
                    & (e > 0)
                ]
            ),
        )
    else:
        e = np.ones_like(y)

    A = np.column_stack(
        [
            x,
            np.ones_like(x),
        ]
    )

    weights = 1.0 / np.maximum(
        e,
        MIN_ERROR,
    )

    Aw = A * weights[:, None]
    yw = y * weights

    coefficients = np.linalg.lstsq(
        Aw,
        yw,
        rcond=None,
    )[0]

    slope = coefficients[0]
    intercept = coefficients[1]

    continuum = (
        slope * wavelength
        + intercept
    )

    return (
        continuum,
        slope,
        intercept,
        int(np.sum(blue)),
        int(np.sum(red)),
    )


def extract_aperture_spectrum(
    sci,
    err,
    aperture,
):
    """
    Sum exactly the pixels marked inside_nominal_aperture=True.
    """

    x = aperture["x_pixel"].astype(int).to_numpy()
    y = aperture["y_pixel"].astype(int).to_numpy()

    values = sci[:, y, x]
    errors = err[:, y, x]

    spectrum = np.nansum(
        values,
        axis=1,
    )

    error_spectrum = np.sqrt(
        np.nansum(
            np.square(errors),
            axis=1,
        )
    )

    return (
        spectrum,
        error_spectrum,
    )


def fit_two_component(
    wavelength,
    spectrum,
    error,
    rest_nm,
    fit_mask,
    initial_velocities,
    initial_sigmas,
    initial_fluxes,
):
    """
    Fit two shared kinematic components for one line.

    Parameters:
        velocity1
        sigma1
        velocity2
        sigma2
        flux1
        flux2

    The fit is performed independently for each line,
    while the resulting kinematics are later combined
    into a common two-component solution.
    """

    x = wavelength[fit_mask]
    y = spectrum[fit_mask]
    e = error[fit_mask]

    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(e)
        & (e > 0)
    )

    x = x[valid]
    y = y[valid]
    e = e[valid]

    if len(x) < 8:
        raise RuntimeError(
            "Insufficient spectral points for fit."
        )

    def residuals(params):
        velocity1 = params[0]
        sigma1 = params[1]
        velocity2 = params[2]
        sigma2 = params[3]
        flux1 = params[4]
        flux2 = params[5]

        model = line_model(
            x,
            rest_nm,
            velocity1,
            sigma1,
            velocity2,
            sigma2,
            flux1,
            flux2,
        )

        return (
            model - y
        ) / e

    x0 = np.array(
        [
            initial_velocities[0],
            initial_sigmas[0],
            initial_velocities[1],
            initial_sigmas[1],
            initial_fluxes[0],
            initial_fluxes[1],
        ],
        dtype=float,
    )

    lower = np.array(
        [
            MIN_VELOCITY,
            MIN_SIGMA_V,
            MIN_VELOCITY,
            MIN_SIGMA_V,
            0.0,
            0.0,
        ],
        dtype=float,
    )

    upper = np.array(
        [
            MAX_VELOCITY,
            MAX_SIGMA_V,
            MAX_VELOCITY,
            MAX_SIGMA_V,
            np.inf,
            np.inf,
        ],
        dtype=float,
    )

    result = least_squares(
        residuals,
        x0,
        bounds=(lower, upper),
        max_nfev=50000,
    )

    if not result.success:
        print(
            "WARNING: fit did not formally converge:"
        )
        print(
            f"  {result.message}"
        )

    params = result.x

    model = line_model(
        x,
        rest_nm,
        params[0],
        params[1],
        params[2],
        params[3],
        params[4],
        params[5],
    )

    chi2 = np.sum(
        ((model - y) / e) ** 2
    )

    dof = max(
        len(y) - len(params),
        1,
    )

    covariance = None

    try:
        jacobian = result.jac
        jtj = (
            jacobian.T @ jacobian
        )

        covariance = np.linalg.inv(
            jtj
        ) * (
            chi2 / dof
        )
    except Exception:
        covariance = None

    return {
        "params": params,
        "x": x,
        "y": y,
        "error": e,
        "model": model,
        "chi2": float(chi2),
        "dof": int(dof),
        "reduced_chi2": float(
            chi2 / dof
        ),
        "covariance": covariance,
        "success": result.success,
        "message": result.message,
    }


def reorder_components(params):
    """
    Ensure component 1 is the lower-velocity component.
    """

    p = np.asarray(
        params,
        dtype=float,
    ).copy()

    if p[0] <= p[2]:
        return p

    return np.array(
        [
            p[2],
            p[3],
            p[0],
            p[1],
            p[5],
            p[4],
        ],
        dtype=float,
    )


def estimate_initial_flux(
    wavelength,
    spectrum,
    rest_nm,
    predicted_nm,
):
    """
    Crude positive line-flux estimate for initialization.
    """

    velocity = velocity_from_wavelength(
        rest_nm,
        wavelength,
    )

    center_velocity = velocity_from_wavelength(
        rest_nm,
        predicted_nm,
    )

    mask = (
        np.abs(
            velocity
            - center_velocity
        )
        < 400.0
    )

    if not np.any(mask):
        return 100.0

    values = spectrum[mask]

    baseline = np.nanmedian(
        spectrum[
            np.abs(
                velocity
                - center_velocity
            )
            > 600.0
        ]
    )

    if not np.isfinite(baseline):
        baseline = 0.0

    positive = np.maximum(
        values - baseline,
        0.0,
    )

    spacing_nm = float(
        np.nanmedian(
            np.diff(wavelength)
        )
    )

    estimate = (
        np.sum(positive)
        * spacing_nm
    )

    if (
        not np.isfinite(estimate)
        or estimate <= 0
    ):
        estimate = 100.0

    return float(estimate)


# ============================================================
# MAIN
# ============================================================

def main():

    banner(
        "M51 JWST NIRSPEC — "
        "Pa-beta / Pa-gamma TWO-COMPONENT KINEMATICS FIT"
    )

    print(
        "Purpose:"
    )
    print(
        "Fit Pa-beta and Pa-gamma with two velocity components."
    )
    print(
        "The two components share velocity and width across both lines."
    )
    print()
    print(
        "No extinction inference is performed."
    )
    print(
        "The existing Pa-gamma product is comparison-only."
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

    wavelength_nm = (
        wavelength_from_spectral_wcs(
            header,
            sci.shape[0],
        )
    )

    print(
        f"Wavelength range = "
        f"{wavelength_nm[0]:.12f} - "
        f"{wavelength_nm[-1]:.12f} nm"
    )

    spectral_spacing = float(
        np.nanmedian(
            np.diff(wavelength_nm)
        )
    )

    print(
        f"Spectral spacing = "
        f"{spectral_spacing:.12f} nm"
    )

    # ========================================================
    # 2. LOAD APERTURE
    # ========================================================

    banner(
        "2. LOADING NOMINAL APERTURE"
    )

    aperture = pd.read_csv(
        APERTURE_PATH
    )

    aperture = aperture[
        aperture[
            "inside_nominal_aperture"
        ].astype(bool)
    ].copy()

    print(
        f"Aperture pixels = "
        f"{len(aperture)}"
    )

    if len(aperture) != 69:
        raise RuntimeError(
            "Expected exactly 69 nominal aperture pixels."
        )

    # ========================================================
    # 3. APERTURE SPECTRUM
    # ========================================================

    banner(
        "3. EXTRACTING 69-PIXEL APERTURE SPECTRA"
    )

    spectrum, error_spectrum = (
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
    # 4. EXPECTED LINE LOCATIONS
    # ========================================================

    banner(
        "4. PREDICTING Pa-beta / Pa-gamma"
    )

    beta_predicted = (
        predicted_wavelength(
            PA_BETA_REST_NM,
            M51_VELOCITY_KMS,
        )
    )

    gamma_predicted = (
        predicted_wavelength(
            PA_GAMMA_REST_NM,
            M51_VELOCITY_KMS,
        )
    )

    beta_index = int(
        np.argmin(
            np.abs(
                wavelength_nm
                - beta_predicted
            )
        )
    )

    gamma_index = int(
        np.argmin(
            np.abs(
                wavelength_nm
                - gamma_predicted
            )
        )
    )

    beta_velocity = (
        velocity_from_wavelength(
            PA_BETA_REST_NM,
            wavelength_nm[beta_index],
        )
    )

    gamma_velocity = (
        velocity_from_wavelength(
            PA_GAMMA_REST_NM,
            wavelength_nm[gamma_index],
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
        f"{wavelength_nm[beta_index]:.12f} nm"
    )

    print(
        f"Pa-gamma nearest plane = "
        f"{gamma_index}"
    )

    print(
        f"Pa-gamma wavelength = "
        f"{wavelength_nm[gamma_index]:.12f} nm"
    )

    print(
        f"Pa-beta velocity = "
        f"{beta_velocity:+.3f} km/s"
    )

    print(
        f"Pa-gamma velocity = "
        f"{gamma_velocity:+.3f} km/s"
    )

    # ========================================================
    # 5. FIT WINDOWS
    # ========================================================

    banner(
        "5. DEFINING FIT WINDOWS"
    )

    beta_mask = (
        np.abs(
            wavelength_nm
            - beta_predicted
        )
        <= BETA_WINDOW_NM
    )

    gamma_mask = (
        np.abs(
            wavelength_nm
            - gamma_predicted
        )
        <= GAMMA_WINDOW_NM
    )

    print(
        f"Pa-beta planes = "
        f"{np.sum(beta_mask)}"
    )

    print(
        f"Pa-gamma planes = "
        f"{np.sum(gamma_mask)}"
    )

    if np.sum(beta_mask) < 10:
        raise RuntimeError(
            "Too few Pa-beta spectral planes."
        )

    if np.sum(gamma_mask) < 10:
        raise RuntimeError(
            "Too few Pa-gamma spectral planes."
        )

    # ========================================================
    # 6. CONTINUUM SUBTRACTION
    # ========================================================

    banner(
        "6. BUILDING LOCAL CONTINUA"
    )

    beta_blue = (
        beta_predicted
        + BETA_BLUE_OFFSET[0],
        beta_predicted
        + BETA_BLUE_OFFSET[1],
    )

    beta_red = (
        beta_predicted
        + BETA_RED_OFFSET[0],
        beta_predicted
        + BETA_RED_OFFSET[1],
    )

    gamma_blue = (
        gamma_predicted
        + GAMMA_BLUE_OFFSET[0],
        gamma_predicted
        + GAMMA_BLUE_OFFSET[1],
    )

    gamma_red = (
        gamma_predicted
        + GAMMA_RED_OFFSET[0],
        gamma_predicted
        + GAMMA_RED_OFFSET[1],
    )

    (
        beta_continuum,
        beta_slope,
        beta_intercept,
        beta_blue_count,
        beta_red_count,
    ) = continuum_fit(
        wavelength_nm,
        spectrum,
        error_spectrum,
        beta_blue,
        beta_red,
    )

    (
        gamma_continuum,
        gamma_slope,
        gamma_intercept,
        gamma_blue_count,
        gamma_red_count,
    ) = continuum_fit(
        wavelength_nm,
        spectrum,
        error_spectrum,
        gamma_blue,
        gamma_red,
    )

    beta_sub = (
        spectrum
        - beta_continuum
    )

    gamma_sub = (
        spectrum
        - gamma_continuum
    )

    print(
        f"Pa-beta continuum planes = "
        f"{beta_blue_count + beta_red_count}"
    )

    print(
        f"Pa-gamma continuum planes = "
        f"{gamma_blue_count + gamma_red_count}"
    )

    # ========================================================
    # 7. INITIAL TWO-COMPONENT ESTIMATES
    # ========================================================

    banner(
        "7. INITIAL TWO-COMPONENT ESTIMATES"
    )

    beta_velocity_axis = (
        velocity_from_wavelength(
            PA_BETA_REST_NM,
            wavelength_nm,
        )
    )

    gamma_velocity_axis = (
        velocity_from_wavelength(
            PA_GAMMA_REST_NM,
            wavelength_nm,
        )
    )

    beta_local = (
        beta_mask
        & np.isfinite(beta_sub)
    )

    gamma_local = (
        gamma_mask
        & np.isfinite(gamma_sub)
    )

    beta_peak = float(
        beta_velocity_axis[
            np.where(beta_local)[0][
                np.argmax(
                    beta_sub[beta_local]
                )
            ]
        ]
    )

    gamma_peak = float(
        gamma_velocity_axis[
            np.where(gamma_local)[0][
                np.argmax(
                    gamma_sub[gamma_local]
                )
            ]
        ]
    )

    print(
        f"Pa-beta peak velocity = "
        f"{beta_peak:+.3f} km/s"
    )

    print(
        f"Pa-gamma peak velocity = "
        f"{gamma_peak:+.3f} km/s"
    )

    # Two deliberately separated starting components.
    #
    # These are only starting values. The optimizer is free
    # to move them.

    component1_velocity = min(
        beta_peak,
        gamma_peak,
    ) - 60.0

    component2_velocity = max(
        beta_peak,
        gamma_peak,
    ) + 60.0

    component1_sigma = 120.0
    component2_sigma = 120.0

    beta_total_guess = (
        estimate_initial_flux(
            wavelength_nm,
            beta_sub,
            PA_BETA_REST_NM,
            beta_predicted,
        )
    )

    gamma_total_guess = (
        estimate_initial_flux(
            wavelength_nm,
            gamma_sub,
            PA_GAMMA_REST_NM,
            gamma_predicted,
        )
    )

    beta_flux_guess = [
        0.5 * beta_total_guess,
        0.5 * beta_total_guess,
    ]

    gamma_flux_guess = [
        0.5 * gamma_total_guess,
        0.5 * gamma_total_guess,
    ]

    print(
        f"Initial component 1 velocity = "
        f"{component1_velocity:+.3f} km/s"
    )

    print(
        f"Initial component 2 velocity = "
        f"{component2_velocity:+.3f} km/s"
    )

    # ========================================================
    # 8. FIT EACH LINE
    # ========================================================

    banner(
        "8. FITTING TWO COMPONENTS"
    )

    beta_fit = fit_two_component(
        wavelength_nm,
        beta_sub,
        error_spectrum,
        PA_BETA_REST_NM,
        beta_mask,
        [
            component1_velocity,
            component2_velocity,
        ],
        [
            component1_sigma,
            component2_sigma,
        ],
        beta_flux_guess,
    )

    gamma_fit = fit_two_component(
        wavelength_nm,
        gamma_sub,
        error_spectrum,
        PA_GAMMA_REST_NM,
        gamma_mask,
        [
            component1_velocity,
            component2_velocity,
        ],
        [
            component1_sigma,
            component2_sigma,
        ],
        gamma_flux_guess,
    )

    beta_params = (
        reorder_components(
            beta_fit["params"]
        )
    )

    gamma_params = (
        reorder_components(
            gamma_fit["params"]
        )
    )

    # ========================================================
    # 9. RESULTS
    # ========================================================

    banner(
        "9. TWO-COMPONENT RESULTS"
    )

    print(
        "Pa-beta:"
    )

    print(
        f"  Component 1 velocity = "
        f"{beta_params[0]:+.3f} km/s"
    )

    print(
        f"  Component 1 sigma = "
        f"{beta_params[1]:.3f} km/s"
    )

    print(
        f"  Component 1 flux = "
        f"{beta_params[4]:.6f}"
    )

    print(
        f"  Component 2 velocity = "
        f"{beta_params[2]:+.3f} km/s"
    )

    print(
        f"  Component 2 sigma = "
        f"{beta_params[3]:.3f} km/s"
    )

    print(
        f"  Component 2 flux = "
        f"{beta_params[5]:.6f}"
    )

    print(
        f"  Reduced chi2 = "
        f"{beta_fit['reduced_chi2']:.6f}"
    )

    print()

    print(
        "Pa-gamma:"
    )

    print(
        f"  Component 1 velocity = "
        f"{gamma_params[0]:+.3f} km/s"
    )

    print(
        f"  Component 1 sigma = "
        f"{gamma_params[1]:.3f} km/s"
    )

    print(
        f"  Component 1 flux = "
        f"{gamma_params[4]:.6f}"
    )

    print(
        f"  Component 2 velocity = "
        f"{gamma_params[2]:+.3f} km/s"
    )

    print(
        f"  Component 2 sigma = "
        f"{gamma_params[3]:.3f} km/s"
    )

    print(
        f"  Component 2 flux = "
        f"{gamma_params[5]:.6f}"
    )

    print(
        f"  Reduced chi2 = "
        f"{gamma_fit['reduced_chi2']:.6f}"
    )

    # ========================================================
    # 10. COMPARE KINEMATICS
    # ========================================================

    banner(
        "10. COMPONENT KINEMATIC CONSISTENCY"
    )

    velocity_difference_1 = (
        gamma_params[0]
        - beta_params[0]
    )

    velocity_difference_2 = (
        gamma_params[2]
        - beta_params[2]
    )

    sigma_difference_1 = (
        gamma_params[1]
        - beta_params[1]
    )

    sigma_difference_2 = (
        gamma_params[3]
        - beta_params[3]
    )

    print(
        f"Component 1 velocity difference = "
        f"{velocity_difference_1:+.3f} km/s"
    )

    print(
        f"Component 2 velocity difference = "
        f"{velocity_difference_2:+.3f} km/s"
    )

    print(
        f"Component 1 sigma difference = "
        f"{sigma_difference_1:+.3f} km/s"
    )

    print(
        f"Component 2 sigma difference = "
        f"{sigma_difference_2:+.3f} km/s"
    )

    # ========================================================
    # 11. COMPONENT RATIOS
    # ========================================================

    banner(
        "11. COMPONENT Pa-beta / Pa-gamma RATIOS"
    )

    beta_flux1 = beta_params[4]
    beta_flux2 = beta_params[5]

    gamma_flux1 = gamma_params[4]
    gamma_flux2 = gamma_params[5]

    ratio1 = (
        beta_flux1 / gamma_flux1
        if gamma_flux1 > 0
        else np.nan
    )

    ratio2 = (
        beta_flux2 / gamma_flux2
        if gamma_flux2 > 0
        else np.nan
    )

    total_beta = (
        beta_flux1
        + beta_flux2
    )

    total_gamma = (
        gamma_flux1
        + gamma_flux2
    )

    total_ratio = (
        total_beta / total_gamma
        if total_gamma > 0
        else np.nan
    )

    print(
        f"Component 1 Pa-beta flux = "
        f"{beta_flux1:.6f}"
    )

    print(
        f"Component 1 Pa-gamma flux = "
        f"{gamma_flux1:.6f}"
    )

    print(
        f"Component 1 ratio = "
        f"{ratio1:.6f}"
    )

    print()

    print(
        f"Component 2 Pa-beta flux = "
        f"{beta_flux2:.6f}"
    )

    print(
        f"Component 2 Pa-gamma flux = "
        f"{gamma_flux2:.6f}"
    )

    print(
        f"Component 2 ratio = "
        f"{ratio2:.6f}"
    )

    print()

    print(
        f"Total Pa-beta flux = "
        f"{total_beta:.6f}"
    )

    print(
        f"Total Pa-gamma flux = "
        f"{total_gamma:.6f}"
    )

    print(
        f"Total Pa-beta / Pa-gamma = "
        f"{total_ratio:.6f}"
    )

    # ========================================================
    # 12. EXISTING Pa-GAMMA PRODUCT
    # ========================================================

    banner(
        "12. EXISTING Pa-GAMMA PRODUCT COMPARISON"
    )

    existing_gamma_sum = np.nan

    if EXISTING_PAGAMMA_PATH.exists():

        with fits.open(
            EXISTING_PAGAMMA_PATH
        ) as hdul:

            existing_map = np.asarray(
                hdul[0].data,
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

        existing_gamma_sum = np.nansum(
            existing_values
        )

        print(
            f"Existing product aperture sum = "
            f"{existing_gamma_sum:.6f}"
        )

        if existing_gamma_sum > 0:

            print(
                f"Two-component / existing = "
                f"{total_gamma / existing_gamma_sum:.6f}"
            )

    else:

        print(
            "Existing Pa-gamma product not found."
        )

    # ========================================================
    # 13. BUILD MODEL PROFILES
    # ========================================================

    banner(
        "13. BUILDING MODEL PROFILES"
    )

    beta_model = line_model(
        wavelength_nm,
        PA_BETA_REST_NM,
        beta_params[0],
        beta_params[1],
        beta_params[2],
        beta_params[3],
        beta_params[4],
        beta_params[5],
    )

    gamma_model = line_model(
        wavelength_nm,
        PA_GAMMA_REST_NM,
        gamma_params[0],
        gamma_params[1],
        gamma_params[2],
        gamma_params[3],
        gamma_params[4],
        gamma_params[5],
    )

    beta_residual = (
        beta_sub
        - beta_model
    )

    gamma_residual = (
        gamma_sub
        - gamma_model
    )

    # ========================================================
    # 14. SAVE PROFILE TABLE
    # ========================================================

    banner(
        "14. SAVING FITTED PROFILES"
    )

    profile_df = pd.DataFrame(
        {
            "wavelength_nm":
                wavelength_nm,
            "aperture_flux":
                spectrum,
            "aperture_error":
                error_spectrum,
            "pabeta_continuum":
                beta_continuum,
            "pabeta_continuum_subtracted":
                beta_sub,
            "pabeta_model":
                beta_model,
            "pabeta_residual":
                beta_residual,
            "pagamma_continuum":
                gamma_continuum,
            "pagamma_continuum_subtracted":
                gamma_sub,
            "pagamma_model":
                gamma_model,
            "pagamma_residual":
                gamma_residual,
        }
    )

    profile_df.to_csv(
        PROFILE_PATH,
        index=False,
    )

    print(
        f"Saved:\n  {PROFILE_PATH}"
    )

    # ========================================================
    # 15. SAVE RESULTS
    # ========================================================

    banner(
        "15. SAVING RESULTS"
    )

    result_rows = [
        {
            "line": "Pa-beta",
            "component": 1,
            "velocity_kms":
                beta_params[0],
            "sigma_kms":
                beta_params[1],
            "flux":
                beta_params[4],
            "reduced_chi2":
                beta_fit["reduced_chi2"],
        },
        {
            "line": "Pa-beta",
            "component": 2,
            "velocity_kms":
                beta_params[2],
            "sigma_kms":
                beta_params[3],
            "flux":
                beta_params[5],
            "reduced_chi2":
                beta_fit["reduced_chi2"],
        },
        {
            "line": "Pa-gamma",
            "component": 1,
            "velocity_kms":
                gamma_params[0],
            "sigma_kms":
                gamma_params[1],
            "flux":
                gamma_params[4],
            "reduced_chi2":
                gamma_fit["reduced_chi2"],
        },
        {
            "line": "Pa-gamma",
            "component": 2,
            "velocity_kms":
                gamma_params[2],
            "sigma_kms":
                gamma_params[3],
            "flux":
                gamma_params[5],
            "reduced_chi2":
                gamma_fit["reduced_chi2"],
        },
    ]

    results_df = pd.DataFrame(
        result_rows
    )

    results_df.to_csv(
        RESULTS_PATH,
        index=False,
    )

    print(
        f"Saved:\n  {RESULTS_PATH}"
    )

    # ========================================================
    # 16. SUMMARY
    # ========================================================

    banner(
        "16. SAVING SUMMARY"
    )

    summary = {
        "aperture_pixels":
            len(aperture),

        "pabeta_component1_velocity_kms":
            beta_params[0],

        "pabeta_component1_sigma_kms":
            beta_params[1],

        "pabeta_component1_flux":
            beta_params[4],

        "pabeta_component2_velocity_kms":
            beta_params[2],

        "pabeta_component2_sigma_kms":
            beta_params[3],

        "pabeta_component2_flux":
            beta_params[5],

        "pagamma_component1_velocity_kms":
            gamma_params[0],

        "pagamma_component1_sigma_kms":
            gamma_params[1],

        "pagamma_component1_flux":
            gamma_params[4],

        "pagamma_component2_velocity_kms":
            gamma_params[2],

        "pagamma_component2_sigma_kms":
            gamma_params[3],

        "pagamma_component2_flux":
            gamma_params[5],

        "component1_ratio":
            ratio1,

        "component2_ratio":
            ratio2,

        "total_pabeta_flux":
            total_beta,

        "total_pagamma_flux":
            total_gamma,

        "total_ratio":
            total_ratio,

        "pabeta_reduced_chi2":
            beta_fit["reduced_chi2"],

        "pagamma_reduced_chi2":
            gamma_fit["reduced_chi2"],

        "existing_pagamma_aperture_sum":
            existing_gamma_sum,
    }

    pd.DataFrame(
        [summary]
    ).to_csv(
        SUMMARY_PATH,
        index=False,
    )

    print(
        f"Saved:\n  {SUMMARY_PATH}"
    )

    # ========================================================
    # 17. FIGURE
    # ========================================================

    banner(
        "17. CREATING TWO-COMPONENT FIGURE"
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11, 9),
        sharex=False,
    )

    beta_plot = beta_mask

    axes[0].errorbar(
        wavelength_nm[beta_plot],
        beta_sub[beta_plot],
        yerr=error_spectrum[beta_plot],
        fmt="o",
        markersize=3,
        alpha=0.7,
        label="Pa-beta data",
    )

    axes[0].plot(
        wavelength_nm[beta_plot],
        beta_model[beta_plot],
        linewidth=2,
        label="Two-component model",
    )

    axes[0].set_title(
        "M51 Pa-beta — two-component fit"
    )

    axes[0].set_xlabel(
        "Wavelength (nm)"
    )

    axes[0].set_ylabel(
        "Continuum-subtracted flux"
    )

    axes[0].legend()

    gamma_plot = gamma_mask

    axes[1].errorbar(
        wavelength_nm[gamma_plot],
        gamma_sub[gamma_plot],
        yerr=error_spectrum[gamma_plot],
        fmt="o",
        markersize=3,
        alpha=0.7,
        label="Pa-gamma data",
    )

    axes[1].plot(
        wavelength_nm[gamma_plot],
        gamma_model[gamma_plot],
        linewidth=2,
        label="Two-component model",
    )

    axes[1].set_title(
        "M51 Pa-gamma — two-component fit"
    )

    axes[1].set_xlabel(
        "Wavelength (nm)"
    )

    axes[1].set_ylabel(
        "Continuum-subtracted flux"
    )

    axes[1].legend()

    plt.tight_layout()

    plt.savefig(
        FIGURE_PATH,
        dpi=180,
    )

    plt.close()

    print(
        f"Saved:\n  {FIGURE_PATH}"
    )

    # ========================================================
    # 18. RESIDUAL FIGURE
    # ========================================================

    banner(
        "18. CREATING RESIDUAL FIGURE"
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11, 8),
        sharex=False,
    )

    axes[0].axhline(
        0.0,
        linewidth=1,
    )

    axes[0].plot(
        wavelength_nm[beta_plot],
        beta_residual[beta_plot],
        "o-",
        markersize=3,
    )

    axes[0].set_title(
        "Pa-beta two-component residuals"
    )

    axes[0].set_xlabel(
        "Wavelength (nm)"
    )

    axes[0].set_ylabel(
        "Residual"
    )

    axes[1].axhline(
        0.0,
        linewidth=1,
    )

    axes[1].plot(
        wavelength_nm[gamma_plot],
        gamma_residual[gamma_plot],
        "o-",
        markersize=3,
    )

    axes[1].set_title(
        "Pa-gamma two-component residuals"
    )

    axes[1].set_xlabel(
        "Wavelength (nm)"
    )

    axes[1].set_ylabel(
        "Residual"
    )

    plt.tight_layout()

    plt.savefig(
        RESIDUAL_FIGURE_PATH,
        dpi=180,
    )

    plt.close()

    print(
        f"Saved:\n  {RESIDUAL_FIGURE_PATH}"
    )

    # ========================================================
    # FINAL
    # ========================================================

    banner(
        "FINAL TWO-COMPONENT KINEMATICS RESULT"
    )

    print(
        f"Nominal aperture = "
        f"{len(aperture)} pixels"
    )

    print()

    print(
        "Component 1:"
    )

    print(
        f"  Pa-beta velocity = "
        f"{beta_params[0]:+.3f} km/s"
    )

    print(
        f"  Pa-gamma velocity = "
        f"{gamma_params[0]:+.3f} km/s"
    )

    print(
        f"  Pa-beta flux = "
        f"{beta_params[4]:.6f}"
    )

    print(
        f"  Pa-gamma flux = "
        f"{gamma_params[4]:.6f}"
    )

    print(
        f"  Pa-beta / Pa-gamma = "
        f"{ratio1:.6f}"
    )

    print()

    print(
        "Component 2:"
    )

    print(
        f"  Pa-beta velocity = "
        f"{beta_params[2]:+.3f} km/s"
    )

    print(
        f"  Pa-gamma velocity = "
        f"{gamma_params[2]:+.3f} km/s"
    )

    print(
        f"  Pa-beta flux = "
        f"{beta_params[5]:.6f}"
    )

    print(
        f"  Pa-gamma flux = "
        f"{gamma_params[5]:.6f}"
    )

    print(
        f"  Pa-beta / Pa-gamma = "
        f"{ratio2:.6f}"
    )

    print()

    print(
        f"TOTAL Pa-beta = "
        f"{total_beta:.6f}"
    )

    print(
        f"TOTAL Pa-gamma = "
        f"{total_gamma:.6f}"
    )

    print(
        f"TOTAL Pa-beta / Pa-gamma = "
        f"{total_ratio:.6f}"
    )

    print()

    print(
        f"Pa-beta reduced chi2 = "
        f"{beta_fit['reduced_chi2']:.3f}"
    )

    print(
        f"Pa-gamma reduced chi2 = "
        f"{gamma_fit['reduced_chi2']:.3f}"
    )

    print()

    print(
        "No extinction inference has been performed."
    )

    print(
        "The purpose of this experiment is to determine whether"
    )

    print(
        "the aperture emission is better described by multiple"
    )

    print(
        "kinematic components before proceeding to extinction analysis."
    )

    print()

    print(
        "Two-component kinematics experiment complete."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
