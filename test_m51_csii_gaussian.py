from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from tools.m51_spectral_analysis import (
    load_x1d_spectrum,
    prepare_spectrum,
)


# ============================================================
# Configuration
# ============================================================

X1D_PATH = Path(
    "data/m51_jwst_level3/"
    "jw03435-o006_t010_nirspec_g140m-f100lp_x1d.fits"
)

CSII_AIR_NM = 1284.26406

# Fit window around the candidate line.
FIT_WINDOW_NM = 3.0

# Speed of light.
C_KM_S = 299792.458


# ============================================================
# NIST air -> vacuum conversion
# ============================================================

def air_to_vacuum_nist(wavelength_air_nm):
    """
    Convert NIST standard-air wavelength to vacuum wavelength.

    Uses the Peck & Reeder refractive-index formulation used
    by the NIST Atomic Spectra Database.

    Valid in the near-IR region relevant here.
    """

    wavelength_um = wavelength_air_nm / 1000.0

    sigma_squared = (
        1.0 / wavelength_um
    ) ** 2

    refractive_index = (
        1.0
        + 0.05792105
        / (238.0185 - sigma_squared)
        + 0.00167917
        / (57.362 - sigma_squared)
    )

    return (
        wavelength_air_nm
        * refractive_index
    )


# ============================================================
# Gaussian + linear continuum
# ============================================================

def gaussian_continuum(
    wavelength,
    amplitude,
    center,
    sigma,
    continuum,
    slope,
):
    """
    Gaussian emission feature plus linear continuum.
    """

    gaussian = (
        amplitude
        * np.exp(
            -0.5
            * (
                (wavelength - center)
                / sigma
            ) ** 2
        )
    )

    baseline = (
        continuum
        + slope
        * (wavelength - center)
    )

    return gaussian + baseline


# ============================================================
# Main analysis
# ============================================================

print("=" * 70)
print("M51 / Cs II GAUSSIAN FIT + AIR-VACUUM CORRECTION")
print("=" * 70)

print()
print("Spectrum:")
print(X1D_PATH)

print()
print("NIST Cs II wavelength:")
print(
    f"Air wavelength:     "
    f"{CSII_AIR_NM:.8f} nm"
)


# ============================================================
# Convert NIST air wavelength to vacuum
# ============================================================

CSII_VACUUM_NM = air_to_vacuum_nist(
    CSII_AIR_NM
)

print(
    f"Vacuum wavelength:   "
    f"{CSII_VACUUM_NM:.8f} nm"
)

print(
    f"Air-vacuum shift:    "
    f"{CSII_VACUUM_NM - CSII_AIR_NM:+.8f} nm"
)


# ============================================================
# Load spectrum
# ============================================================

print()
print("Loading M51 JWST/NIRSpec X1D spectrum...")

spectrum = load_x1d_spectrum(
    X1D_PATH
)

print("Preparing spectrum...")

clean = prepare_spectrum(
    spectrum
)

wavelength_um = clean["wavelength"]
flux = clean["flux"]
uncertainty = clean["uncertainty"]

wavelength_nm = (
    wavelength_um
    * 1000.0
)

print()
print("Spectrum loaded successfully.")

print(
    f"Valid points:       "
    f"{clean['valid_points']}"
)

print(
    f"Rejected points:    "
    f"{clean['rejected_points']}"
)

print(
    f"Wavelength range:   "
    f"{wavelength_nm.min():.3f}"
    f" - "
    f"{wavelength_nm.max():.3f} nm"
)


# ============================================================
# Extract fitting region
# ============================================================

mask = (
    np.abs(
        wavelength_nm
        - CSII_VACUUM_NM
    )
    <= FIT_WINDOW_NM
)

if np.sum(mask) < 8:

    raise RuntimeError(
        "Insufficient spectral points "
        "for Gaussian fitting."
    )

x = wavelength_nm[mask]
y = flux[mask]
yerr = uncertainty[mask]


# ============================================================
# Initial parameter estimates
# ============================================================

# Estimate continuum from the outer parts of the window.

edge_mask = (
    np.abs(
        x
        - CSII_VACUUM_NM
    )
    > FIT_WINDOW_NM * 0.65
)

if np.sum(edge_mask) >= 2:

    continuum_guess = np.median(
        y[edge_mask]
    )

else:

    continuum_guess = np.median(y)


# Strongest point as an initial center.

peak_index = np.argmax(y)

center_guess = x[peak_index]

amplitude_guess = (
    y[peak_index]
    - continuum_guess
)

# G140M has R ~ 1000, so a resolution element
# near 1284 nm is roughly 1.28 nm.
#
# A Gaussian sigma corresponding to an unresolved
# feature is approximately FWHM / 2.355.

resolution_element_nm = (
    CSII_VACUUM_NM / 1000.0
)

sigma_guess = (
    resolution_element_nm
    / 2.355
)

if sigma_guess <= 0:
    sigma_guess = 0.5


# ============================================================
# Parameter bounds
# ============================================================

lower_bounds = [
    0.0,                     # amplitude
    CSII_VACUUM_NM - 2.0,    # center
    0.05,                    # sigma
    -np.inf,                 # continuum
    -np.inf,                 # slope
]

upper_bounds = [
    np.inf,                  # amplitude
    CSII_VACUUM_NM + 2.0,    # center
    3.0,                     # sigma
    np.inf,                  # continuum
    np.inf,                  # slope
]


initial_parameters = [
    max(amplitude_guess, 0.0),
    center_guess,
    sigma_guess,
    continuum_guess,
    0.0,
]


# ============================================================
# Gaussian fit
# ============================================================

print()
print("=" * 70)
print("GAUSSIAN FIT")
print("=" * 70)

try:

    parameters, covariance = curve_fit(
        gaussian_continuum,
        x,
        y,
        p0=initial_parameters,
        sigma=yerr,
        absolute_sigma=True,
        bounds=(
            lower_bounds,
            upper_bounds,
            ),
        maxfev=50000,
    )

except RuntimeError as exc:

    raise RuntimeError(
        "Gaussian fit failed."
    ) from exc


(
    amplitude,
    center,
    sigma,
    continuum,
    slope,
) = parameters


# ============================================================
# Parameter uncertainties
# ============================================================

parameter_errors = np.sqrt(
    np.diag(covariance)
)

(
    amplitude_error,
    center_error,
    sigma_error,
    continuum_error,
    slope_error,
) = parameter_errors


# ============================================================
# Derived quantities
# ============================================================

fwhm = (
    2.354820045
    * sigma
)

fwhm_error = (
    2.354820045
    * sigma_error
)

wavelength_difference = (
    center
    - CSII_VACUUM_NM
)

velocity_km_s = (
    C_KM_S
    * wavelength_difference
    / CSII_VACUUM_NM
)

velocity_error_km_s = (
    C_KM_S
    * center_error
    / CSII_VACUUM_NM
)


# Approximate amplitude significance.

if amplitude_error > 0:

    amplitude_snr = (
        amplitude
        / amplitude_error
    )

else:

    amplitude_snr = np.nan


# ============================================================
# Goodness of fit
# ============================================================

model = gaussian_continuum(
    x,
    *parameters,
)

residuals = (
    y - model
)

chi_squared = np.sum(
    (residuals / yerr) ** 2
)

degrees_of_freedom = (
    len(x)
    - len(parameters)
)

reduced_chi_squared = (
    chi_squared
    / degrees_of_freedom
)


# ============================================================
# Report results
# ============================================================

print()
print("NIST / Cs II")
print("-" * 70)

print(
    f"Air wavelength:       "
    f"{CSII_AIR_NM:.8f} nm"
)

print(
    f"Vacuum wavelength:     "
    f"{CSII_VACUUM_NM:.8f} nm"
)

print()
print("M51 fitted feature")
print("-" * 70)

print(
    f"Gaussian center:       "
    f"{center:.8f} +/- "
    f"{center_error:.8f} nm"
)

print(
    f"Amplitude:             "
    f"{amplitude:.8g} +/- "
    f"{amplitude_error:.8g}"
)

print(
    f"Sigma:                 "
    f"{sigma:.6f} +/- "
    f"{sigma_error:.6f} nm"
)

print(
    f"FWHM:                  "
    f"{fwhm:.6f} +/- "
    f"{fwhm_error:.6f} nm"
)

print()
print("Cs II wavelength comparison")
print("-" * 70)

print(
    f"Wavelength difference: "
    f"{wavelength_difference:+.8f} nm"
)

print(
    f"Velocity difference:    "
    f"{velocity_km_s:+.3f} +/- "
    f"{velocity_error_km_s:.3f} km/s"
)

print()
print("Fit statistics")
print("-" * 70)

print(
    f"Amplitude S/N:         "
    f"{amplitude_snr:.2f}"
)

print(
    f"Chi squared:           "
    f"{chi_squared:.3f}"
)

print(
    f"Degrees of freedom:    "
    f"{degrees_of_freedom}"
)

print(
    f"Reduced chi squared:   "
    f"{reduced_chi_squared:.3f}"
)


# ============================================================
# Diagnostic plot
# ============================================================

print()
print("Generating diagnostic plot...")

x_plot = np.linspace(
    x.min(),
    x.max(),
    500,
)

y_plot = gaussian_continuum(
    x_plot,
    *parameters,
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
    x_plot,
    y_plot,
    linewidth=2,
    label="Gaussian + continuum fit",
)

plt.axvline(
    CSII_VACUUM_NM,
    linestyle="--",
    linewidth=2,
    label=(
        "Cs II vacuum "
        f"{CSII_VACUUM_NM:.5f} nm"
    ),
)

plt.axvline(
    center,
    linestyle=":",
    linewidth=2,
    label=(
        "Fitted center "
        f"{center:.5f} nm"
    ),
)

plt.xlabel(
    "Vacuum wavelength (nm)"
)

plt.ylabel(
    "Flux"
)

plt.title(
    "M51 JWST/NIRSpec — Cs II 1284 nm Candidate"
)

plt.legend()

plt.grid(
    alpha=0.25
)

plt.tight_layout()

plt.savefig(
    "m51_csii_1284_gaussian_fit.png",
    dpi=200,
)

plt.show()


print()
print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
