from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from scipy.optimize import curve_fit


# ============================================================
# Configuration
# ============================================================

X1D_PATH = Path(
    "data/m51_jwst_level3/"
    "jw03435-o006_t010_nirspec_g140m-f100lp_x1d.fits"
)

RESOLUTION_PATH = Path(
    "data/instrument/"
    "jwst_nirspec_g140m_disp.fits"
)

OBSERVED_FEATURE_NM = 1284.26130440
OBSERVED_FEATURE_ERROR_NM = 0.00134611

# Independent local M51 velocity reference.
#
# This is the velocity measured previously from the
# [Fe II] 1.257 um line.
LOCAL_M51_VELOCITY = 573.72
LOCAL_M51_VELOCITY_ERROR = 2.11

# Laboratory wavelengths.
#
# JWST wavelengths are vacuum wavelengths.
PA_BETA_REST_NM = 1281.80700000
CSII_AIR_NM = 1284.26406000
CSII_VACUUM_NM = 1284.61537587

# Local fitting region.
WINDOW_NM = 6.0

# Speed of light.
C_KM_S = 299792.458


# ============================================================
# Utility functions
# ============================================================


def predict_wavelength(rest_nm, velocity_km_s):
    """
    Predict observed wavelength using the same
    non-relativistic Doppler convention used by
    the previous M51 velocity analyses.
    """

    return rest_nm * (
        1.0 + velocity_km_s / C_KM_S
    )


def linear_gaussian(
    wavelength,
    continuum,
    slope,
    amplitude,
    center,
    sigma,
):
    """
    Linear continuum + Gaussian emission line.
    """

    continuum_model = (
        continuum
        + slope * (
            wavelength - OBSERVED_FEATURE_NM
        )
    )

    gaussian = amplitude * np.exp(
        -0.5
        * (
            (wavelength - center)
            / sigma
        ) ** 2
    )

    return continuum_model + gaussian


def fixed_center_model(
    wavelength,
    continuum,
    slope,
    amplitude,
    center,
    sigma,
):
    """
    Model with fixed Gaussian center and width.
    """

    return linear_gaussian(
        wavelength,
        continuum,
        slope,
        amplitude,
        center,
        sigma,
    )


def free_center_model(
    wavelength,
    continuum,
    slope,
    amplitude,
    center,
):
    """
    Linear continuum + Gaussian with
    free center and fixed instrumental sigma.
    """

    # sigma is supplied globally below.
    return linear_gaussian(
        wavelength,
        continuum,
        slope,
        amplitude,
        center,
        INSTRUMENT_SIGMA,
    )


def calculate_chi_square(
    observed,
    model,
    uncertainty,
):
    residuals = (
        observed - model
    )

    return np.sum(
        (residuals / uncertainty) ** 2
    )


def calculate_aic(
    chi_square,
    number_parameters,
):
    return (
        chi_square
        + 2 * number_parameters
    )


def calculate_bic(
    chi_square,
    number_parameters,
    number_points,
):
    return (
        chi_square
        + number_parameters
        * np.log(number_points)
    )


# ============================================================
# Load spectrum
# ============================================================

print("=" * 70)
print("M51: Pa BETA vs Cs II")
print("LOCAL CONTINUUM + INSTRUMENT-RESOLUTION TEST")
print("=" * 70)

print()
print("Spectrum:")
print(X1D_PATH)

with fits.open(X1D_PATH) as hdul:

    table = hdul[1].data

    wavelength_um = np.asarray(
        table["WAVELENGTH"],
        dtype=float,
    )

    flux = np.asarray(
        table["FLUX"],
        dtype=float,
    )

    uncertainty = np.asarray(
        table["FLUX_ERROR"],
        dtype=float,
    )

    dq = np.asarray(
        table["DQ"],
        dtype=np.uint32,
    )


# ============================================================
# Clean spectrum
# ============================================================

valid = (
    np.isfinite(wavelength_um)
    & np.isfinite(flux)
    & np.isfinite(uncertainty)
    & (uncertainty > 0)
    & (dq == 0)
)

wavelength_nm = (
    wavelength_um[valid] * 1000.0
)

flux = flux[valid]
uncertainty = uncertainty[valid]

print()
print("Valid spectrum points:", len(wavelength_nm))


# ============================================================
# Extract local fitting region
# ============================================================

mask = (
    np.abs(
        wavelength_nm
        - OBSERVED_FEATURE_NM
    )
    <= WINDOW_NM
)

x = wavelength_nm[mask]
y = flux[mask]
yerr = uncertainty[mask]

print()
print("Local fitting region:")
print(
    f"{x.min():.6f} - "
    f"{x.max():.6f} nm"
)

print(
    f"Points in fitting region: "
    f"{len(x)}"
)


# ============================================================
# Instrument resolution
# ============================================================

#
# The NIRSpec dispersion/resolution file has changed
# structure between calibration products, so first inspect
# the available columns.
#

with fits.open(RESOLUTION_PATH) as hdul:

    print()
    print("=" * 70)
    print("NIRSPEC RESOLUTION FILE")
    print("=" * 70)

    for i, hdu in enumerate(hdul):

        if hdu.data is None:
            continue

        print(
            f"HDU {i}: "
            f"{type(hdu).__name__}"
        )

        if hasattr(hdu.data, "names"):

            print(
                "Columns:",
                hdu.data.names
            )


# ------------------------------------------------------------
# Load resolution information.
#
# The previous analysis established an interpolated
# resolving power of R = 916.3 at 1284.3 nm.
#
# We use that established value here so that this test
# remains directly comparable with the previous result.
# ------------------------------------------------------------

R_AT_FEATURE = 916.3

instrument_fwhm = (
    OBSERVED_FEATURE_NM
    / R_AT_FEATURE
)

instrument_sigma = (
    instrument_fwhm
    / 2.354820045
)

INSTRUMENT_SIGMA = instrument_sigma

print()
print(
    f"Adopted resolving power R: "
    f"{R_AT_FEATURE:.1f}"
)

print(
    f"Instrument FWHM: "
    f"{instrument_fwhm:.6f} nm"
)

print(
    f"Instrument sigma: "
    f"{instrument_sigma:.6f} nm"
)


# ============================================================
# Hypothesis wavelengths
# ============================================================

pa_beta_predicted = predict_wavelength(
    PA_BETA_REST_NM,
    LOCAL_M51_VELOCITY,
)

csii_predicted = predict_wavelength(
    CSII_VACUUM_NM,
    LOCAL_M51_VELOCITY,
)

print()
print("=" * 70)
print("HYPOTHESIS PREDICTIONS")
print("=" * 70)

print()
print("Local M51 velocity reference:")
print(
    f"+{LOCAL_M51_VELOCITY:.2f} "
    f"+/- {LOCAL_M51_VELOCITY_ERROR:.2f} km/s"
)

print()
print("Pa beta:")
print(
    f"Predicted wavelength: "
    f"{pa_beta_predicted:.8f} nm"
)

print(
    f"Difference from feature: "
    f"{OBSERVED_FEATURE_NM - pa_beta_predicted:+.8f} nm"
)

print()
print("Cs II:")
print(
    f"Predicted wavelength: "
    f"{csii_predicted:.8f} nm"
)

print(
    f"Difference from feature: "
    f"{OBSERVED_FEATURE_NM - csii_predicted:+.8f} nm"
)


# ============================================================
# Initial parameter estimates
# ============================================================

continuum_initial = np.median(y)

slope_initial = 0.0

amplitude_initial = (
    np.max(y)
    - continuum_initial
)

# Same initial continuum for both hypotheses.
p0_pa = [
    continuum_initial,
    slope_initial,
    amplitude_initial,
]

p0_cs = [
    continuum_initial,
    slope_initial,
    amplitude_initial,
]


# ============================================================
# Constrained Pa beta fit
# ============================================================

print()
print("=" * 70)
print("CONSTRAINED Pa BETA FIT")
print("=" * 70)

def pa_model(
    wavelength,
    continuum,
    slope,
    amplitude,
):
    return linear_gaussian(
        wavelength,
        continuum,
        slope,
        amplitude,
        pa_beta_predicted,
        INSTRUMENT_SIGMA,
    )


pa_bounds = (
    [
        -np.inf,
        -np.inf,
        0.0,
    ],
    [
        np.inf,
        np.inf,
        np.inf,
    ],
)

pa_params, pa_cov = curve_fit(
    pa_model,
    x,
    y,
    p0=p0_pa,
    sigma=yerr,
    absolute_sigma=True,
    bounds=pa_bounds,
    maxfev=100000,
)

pa_model_flux = pa_model(
    x,
    *pa_params,
)

pa_chi2 = calculate_chi_square(
    y,
    pa_model_flux,
    yerr,
)

pa_k = len(pa_params)
pa_dof = len(x) - pa_k

pa_aic = calculate_aic(
    pa_chi2,
    pa_k,
)

pa_bic = calculate_bic(
    pa_chi2,
    pa_k,
    len(x),
)

pa_amplitude = pa_params[2]
pa_amplitude_error = np.sqrt(
    pa_cov[2, 2]
)

print(
    f"Fixed center: "
    f"{pa_beta_predicted:.8f} nm"
)

print(
    f"Amplitude: "
    f"{pa_amplitude:.8g}"
)

print(
    f"Amplitude error: "
    f"{pa_amplitude_error:.8g}"
)

print(
    f"Amplitude S/N: "
    f"{pa_amplitude / pa_amplitude_error:.2f}"
)

print(
    f"Chi squared: "
    f"{pa_chi2:.3f}"
)

print(
    f"Degrees of freedom: "
    f"{pa_dof}"
)

print(
    f"Reduced chi squared: "
    f"{pa_chi2 / pa_dof:.3f}"
)

print(
    f"AIC: {pa_aic:.3f}"
)

print(
    f"BIC: {pa_bic:.3f}"
)


# ============================================================
# Constrained Cs II fit
# ============================================================

print()
print("=" * 70)
print("CONSTRAINED Cs II FIT")
print("=" * 70)


def cs_model(
    wavelength,
    continuum,
    slope,
    amplitude,
):
    return linear_gaussian(
        wavelength,
        continuum,
        slope,
        amplitude,
        csii_predicted,
        INSTRUMENT_SIGMA,
    )


cs_params, cs_cov = curve_fit(
    cs_model,
    x,
    y,
    p0=p0_cs,
    sigma=yerr,
    absolute_sigma=True,
    bounds=pa_bounds,
    maxfev=100000,
)

cs_model_flux = cs_model(
    x,
    *cs_params,
)

cs_chi2 = calculate_chi_square(
    y,
    cs_model_flux,
    yerr,
)

cs_k = len(cs_params)
cs_dof = len(x) - cs_k

cs_aic = calculate_aic(
    cs_chi2,
    cs_k,
)

cs_bic = calculate_bic(
    cs_chi2,
    cs_k,
    len(x),
)

cs_amplitude = cs_params[2]
cs_amplitude_error = np.sqrt(
    cs_cov[2, 2]
)

print(
    f"Fixed center: "
    f"{csii_predicted:.8f} nm"
)

print(
    f"Amplitude: "
    f"{cs_amplitude:.8g}"
)

print(
    f"Amplitude error: "
    f"{cs_amplitude_error:.8g}"
)

print(
    f"Amplitude S/N: "
    f"{cs_amplitude / cs_amplitude_error:.2f}"
)

print(
    f"Chi squared: "
    f"{cs_chi2:.3f}"
)

print(
    f"Degrees of freedom: "
    f"{cs_dof}"
)

print(
    f"Reduced chi squared: "
    f"{cs_chi2 / cs_dof:.3f}"
)

print(
    f"AIC: {cs_aic:.3f}"
)

print(
    f"BIC: {cs_bic:.3f}"
)


# ============================================================
# Model comparison
# ============================================================

delta_chi2 = (
    cs_chi2 - pa_chi2
)

delta_aic = (
    cs_aic - pa_aic
)

delta_bic = (
    cs_bic - pa_bic
)

print()
print("=" * 70)
print("CONSTRAINED MODEL COMPARISON")
print("=" * 70)

print(
    f"Pa beta chi²: "
    f"{pa_chi2:.3f}"
)

print(
    f"Cs II chi²:   "
    f"{cs_chi2:.3f}"
)

print(
    f"Delta chi² "
    f"(Cs II - Pa beta): "
    f"{delta_chi2:+.3f}"
)

print(
    f"Delta AIC "
    f"(Cs II - Pa beta): "
    f"{delta_aic:+.3f}"
)

print(
    f"Delta BIC "
    f"(Cs II - Pa beta): "
    f"{delta_bic:+.3f}"
)


# ============================================================
# Free-center Gaussian fit
# ============================================================

print()
print("=" * 70)
print("FREE-CENTER Pa BETA-CANDIDATE FIT")
print("=" * 70)


def free_model(
    wavelength,
    continuum,
    slope,
    amplitude,
    center,
):
    return linear_gaussian(
        wavelength,
        continuum,
        slope,
        amplitude,
        center,
        INSTRUMENT_SIGMA,
    )


p0_free = [
    continuum_initial,
    slope_initial,
    amplitude_initial,
    OBSERVED_FEATURE_NM,
]

free_lower = [
    -np.inf,
    -np.inf,
    0.0,
    OBSERVED_FEATURE_NM - 3.0,
]

free_upper = [
    np.inf,
    np.inf,
    np.inf,
    OBSERVED_FEATURE_NM + 3.0,
]

free_params, free_cov = curve_fit(
    free_model,
    x,
    y,
    p0=p0_free,
    sigma=yerr,
    absolute_sigma=True,
    bounds=(
        free_lower,
        free_upper,
    ),
    maxfev=100000,
)

free_model_flux = free_model(
    x,
    *free_params,
)

free_chi2 = calculate_chi_square(
    y,
    free_model_flux,
    yerr,
)

free_k = len(free_params)
free_dof = len(x) - free_k

free_aic = calculate_aic(
    free_chi2,
    free_k,
)

free_bic = calculate_bic(
    free_chi2,
    free_k,
    len(x),
)

free_center = free_params[3]
free_center_error = np.sqrt(
    free_cov[3, 3]
)

free_amplitude = free_params[2]
free_amplitude_error = np.sqrt(
    free_cov[2, 2]
)

center_velocity = (
    (
        free_center
        / PA_BETA_REST_NM
    )
    - 1.0
) * C_KM_S

center_velocity_error = (
    free_center_error
    / PA_BETA_REST_NM
    * C_KM_S
)

print(
    f"Fitted center: "
    f"{free_center:.8f} "
    f"+/- {free_center_error:.8f} nm"
)

print(
    f"Fitted velocity assuming Pa beta: "
    f"{center_velocity:+.3f} "
    f"+/- {center_velocity_error:.3f} km/s"
)

print(
    f"Amplitude: "
    f"{free_amplitude:.8g}"
)

print(
    f"Amplitude S/N: "
    f"{free_amplitude / free_amplitude_error:.2f}"
)

print(
    f"Chi squared: "
    f"{free_chi2:.3f}"
)

print(
    f"Degrees of freedom: "
    f"{free_dof}"
)

print(
    f"Reduced chi squared: "
    f"{free_chi2 / free_dof:.3f}"
)

print(
    f"AIC: {free_aic:.3f}"
)

print(
    f"BIC: {free_bic:.3f}"
)


# ============================================================
# Residual-based noise diagnostic
# ============================================================

pa_residuals = (
    y - pa_model_flux
)

cs_residuals = (
    y - cs_model_flux
)

free_residuals = (
    y - free_model_flux
)

reported_noise = np.median(
    yerr
)

pa_residual_rms = np.std(
    pa_residuals
)

free_residual_rms = np.std(
    free_residuals
)

noise_scale_pa = (
    pa_residual_rms
    / reported_noise
)

noise_scale_free = (
    free_residual_rms
    / reported_noise
)

print()
print("=" * 70)
print("NOISE / RESIDUAL DIAGNOSTIC")
print("=" * 70)

print(
    f"Median reported flux uncertainty: "
    f"{reported_noise:.8g}"
)

print(
    f"Pa beta residual RMS: "
    f"{pa_residual_rms:.8g}"
)

print(
    f"Free-center residual RMS: "
    f"{free_residual_rms:.8g}"
)

print(
    f"Approximate Pa beta noise scale: "
    f"{noise_scale_pa:.2f}"
)

print(
    f"Approximate free-center noise scale: "
    f"{noise_scale_free:.2f}"
)


# ============================================================
# Summary
# ============================================================

print()
print("=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(
    f"Observed feature: "
    f"{OBSERVED_FEATURE_NM:.8f} nm"
)

print()
print(
    f"Local [Fe II] velocity reference: "
    f"+{LOCAL_M51_VELOCITY:.2f} km/s"
)

print()
print(
    "Pa beta predicted wavelength: "
    f"{pa_beta_predicted:.8f} nm"
)

print(
    "Cs II predicted wavelength:   "
    f"{csii_predicted:.8f} nm"
)

print()
print(
    f"Pa beta constrained chi²: "
    f"{pa_chi2:.3f}"
)

print(
    f"Cs II constrained chi²:   "
    f"{cs_chi2:.3f}"
)

print(
    f"Delta chi²: "
    f"{delta_chi2:+.3f}"
)

print()
print(
    f"Free-center wavelength: "
    f"{free_center:.8f} "
    f"+/- {free_center_error:.8f} nm"
)

print(
    f"Free-center Pa beta velocity: "
    f"{center_velocity:+.3f} "
    f"+/- {center_velocity_error:.3f} km/s"
)


# ============================================================
# Plot
# ============================================================

plot_x = np.linspace(
    x.min(),
    x.max(),
    1000,
)

plot_pa = pa_model(
    plot_x,
    *pa_params,
)

plot_cs = cs_model(
    plot_x,
    *cs_params,
)

plot_free = free_model(
    plot_x,
    *free_params,
)

plt.figure(
    figsize=(12, 7)
)

plt.errorbar(
    x,
    y,
    yerr=yerr,
    fmt="o",
    markersize=4,
    capsize=2,
    label="M51 NIRSpec",
)

plt.plot(
    plot_x,
    plot_pa,
    linewidth=2,
    label="Pa beta — fixed M51 velocity",
)

plt.plot(
    plot_x,
    plot_cs,
    linewidth=2,
    label="Cs II — fixed M51 velocity",
)

plt.plot(
    plot_x,
    plot_free,
    linewidth=2,
    linestyle="--",
    label="Free-center Gaussian",
)

plt.axvline(
    OBSERVED_FEATURE_NM,
    linestyle=":",
    linewidth=2,
    label="Observed feature",
)

plt.axvline(
    pa_beta_predicted,
    linestyle="--",
    linewidth=1.5,
    label="Pa beta predicted",
)

plt.axvline(
    csii_predicted,
    linestyle="--",
    linewidth=1.5,
    label="Cs II predicted",
)

plt.xlabel(
    "Vacuum wavelength (nm)"
)

plt.ylabel(
    "Flux (Jy)"
)

plt.title(
    "M51 1284 nm Feature — "
    "Pa beta vs Cs II"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

output = (
    "m51_pabeta_csii_local_fit.png"
)

plt.savefig(
    output,
    dpi=150,
)

print()
print(
    f"Plot saved to: {output}"
)

print()
print("=" * 70)
print("TEST COMPLETE")
print("=" * 70)
