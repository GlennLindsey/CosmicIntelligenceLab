from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.optimize import least_squares


# ============================================================
# M51 HYDROGEN PA-BETA / PA-GAMMA PROFILE TEST
# Independent spectral-profile consistency test
# ============================================================

print("=" * 70)
print("M51 HYDROGEN PA-BETA / PA-GAMMA PROFILE TEST")
print("INDEPENDENT HYDROGEN RECOMBINATION-LINE PROFILE ANALYSIS")
print("=" * 70)


# ============================================================
# Configuration
# ============================================================

X1D_PATH = Path(
    "data/m51_jwst_level3/"
    "jw03435-o006_t010_nirspec_g140m-f100lp_x1d.fits"
)

OUTPUT_DIR = Path("data/atomic_lines")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_CSV = (
    OUTPUT_DIR /
    "m51_hydrogen_pabeta_pagamma_profiles.csv"
)

PROFILE_CSV = (
    OUTPUT_DIR /
    "m51_hydrogen_pabeta_pagamma_profile_data.csv"
)

PLOT_PATH = Path(
    "m51_hydrogen_pabeta_pagamma_profiles.png"
)

RESIDUAL_PLOT_PATH = Path(
    "m51_hydrogen_pabeta_pagamma_profile_residuals.png"
)


# ============================================================
# Physical parameters
# ============================================================

C_KMS = 299792.458

M51_VELOCITY_KMS = 573.720

PA_BETA_REST_NM = 1281.807000
PA_GAMMA_REST_NM = 1093.800000

R_NIRSPEC = 916.3

# Local fitting windows
PA_BETA_WINDOW = (1278.0, 1290.0)
PA_GAMMA_WINDOW = (1090.0, 1102.0)

# Continuum regions
PA_BETA_BLUE = (1278.0, 1281.0)
PA_BETA_RED = (1287.0, 1290.0)

PA_GAMMA_BLUE = (1090.0, 1092.0)
PA_GAMMA_RED = (1098.0, 1102.0)


# ============================================================
# Helper functions
# ============================================================

def velocity_to_wavelength(rest_nm, velocity_kms):
    """
    Non-relativistic Doppler approximation appropriate for
    these small velocities.
    """
    return rest_nm * (1.0 + velocity_kms / C_KMS)


def wavelength_to_velocity(rest_nm, wavelength_nm):
    return C_KMS * (wavelength_nm / rest_nm - 1.0)


def gaussian(x, amplitude, center, sigma):
    return amplitude * np.exp(
        -0.5 * ((x - center) / sigma) ** 2
    )


def linear_continuum(x, intercept, slope, x0):
    return intercept + slope * (x - x0)


def model_flux(
    wavelength,
    intercept,
    slope,
    amplitude,
    center,
    sigma,
    x0,
):
    return (
        linear_continuum(
            wavelength,
            intercept,
            slope,
            x0,
        )
        + gaussian(
            wavelength,
            amplitude,
            center,
            sigma,
        )
    )


def gaussian_integrated_flux(amplitude, sigma):
    return amplitude * sigma * np.sqrt(2.0 * np.pi)


def fwhm_from_sigma(sigma):
    return 2.354820045 * sigma


def velocity_width_from_sigma(rest_nm, sigma_nm):
    return C_KMS * sigma_nm / rest_nm


def robust_sigma(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size < 2:
        return np.nan

    median = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - median))

    return 1.4826 * mad


def safe_log10(value):
    if not np.isfinite(value) or value <= 0:
        return np.nan

    return np.log10(value)


# ============================================================
# Load X1D
# ============================================================

print()
print("=" * 70)
print("1. LOADING M51 X1D SPECTRUM")
print("=" * 70)

print()
print("File:")
print(f"  {X1D_PATH}")

with fits.open(X1D_PATH, memmap=False) as hdul:

    table = hdul[1].data

    wavelength_um = np.asarray(
        table["WAVELENGTH"],
        dtype=float,
    )

    flux = np.asarray(
        table["FLUX"],
        dtype=float,
    )

    flux_error = np.asarray(
        table["FLUX_ERROR"],
        dtype=float,
    )

print()
print(f"Spectral points: {len(wavelength_um)}")

wavelength_nm = wavelength_um * 1000.0

print(
    "Wavelength range: "
    f"{np.nanmin(wavelength_nm):.6f} - "
    f"{np.nanmax(wavelength_nm):.6f} nm"
)


# ============================================================
# Resolution
# ============================================================

print()
print("=" * 70)
print("2. NIRSPEC INSTRUMENT RESOLUTION")
print("=" * 70)

pa_beta_predicted = velocity_to_wavelength(
    PA_BETA_REST_NM,
    M51_VELOCITY_KMS,
)

pa_gamma_predicted = velocity_to_wavelength(
    PA_GAMMA_REST_NM,
    M51_VELOCITY_KMS,
)

pa_beta_fwhm_inst = pa_beta_predicted / R_NIRSPEC
pa_gamma_fwhm_inst = pa_gamma_predicted / R_NIRSPEC

pa_beta_sigma_inst = pa_beta_fwhm_inst / 2.354820045
pa_gamma_sigma_inst = pa_gamma_fwhm_inst / 2.354820045

print(f"Resolving power: R = {R_NIRSPEC:.1f}")

print()
print("Pa-beta:")
print(f"  Rest wavelength: {PA_BETA_REST_NM:.6f} nm")
print(f"  M51 predicted wavelength: {pa_beta_predicted:.6f} nm")
print(f"  Instrument FWHM: {pa_beta_fwhm_inst:.6f} nm")
print(f"  Instrument sigma: {pa_beta_sigma_inst:.6f} nm")

print()
print("Pa-gamma:")
print(f"  Rest wavelength: {PA_GAMMA_REST_NM:.6f} nm")
print(f"  M51 predicted wavelength: {pa_gamma_predicted:.6f} nm")
print(f"  Instrument FWHM: {pa_gamma_fwhm_inst:.6f} nm")
print(f"  Instrument sigma: {pa_gamma_sigma_inst:.6f} nm")


# ============================================================
# Profile-fitting function
# ============================================================

def fit_hydrogen_line(
    name,
    rest_nm,
    window,
    blue_region,
    red_region,
    sigma_inst,
):
    print()
    print("=" * 70)
    print(f"3. FITTING {name.upper()}")
    print("=" * 70)

    mask_window = (
        np.isfinite(wavelength_nm)
        & np.isfinite(flux)
        & np.isfinite(flux_error)
        & (wavelength_nm >= window[0])
        & (wavelength_nm <= window[1])
        & (flux_error > 0)
    )

    wl = wavelength_nm[mask_window]
    fl = flux[mask_window]
    er = flux_error[mask_window]

    print()
    print(
        f"Window: {window[0]:.3f} - "
        f"{window[1]:.3f} nm"
    )

    print(f"Usable points: {len(wl)}")

    if len(wl) < 7:
        raise RuntimeError(
            f"Insufficient spectral points for {name}."
        )

    print(
        "Actual range: "
        f"{wl.min():.6f} - "
        f"{wl.max():.6f} nm"
    )

    x0 = np.median(wl)

    # --------------------------------------------------------
    # Continuum points
    # --------------------------------------------------------

    blue_mask = (
        np.isfinite(wavelength_nm)
        & np.isfinite(flux)
        & np.isfinite(flux_error)
        & (wavelength_nm >= blue_region[0])
        & (wavelength_nm <= blue_region[1])
        & (flux_error > 0)
    )

    red_mask = (
        np.isfinite(wavelength_nm)
        & np.isfinite(flux)
        & np.isfinite(flux_error)
        & (wavelength_nm >= red_region[0])
        & (wavelength_nm <= red_region[1])
        & (flux_error > 0)
    )

    continuum_mask = blue_mask | red_mask

    continuum_wl = wavelength_nm[continuum_mask]
    continuum_flux = flux[continuum_mask]

    print()
    print("Continuum diagnostics:")
    print(f"  Blue points: {np.sum(blue_mask)}")
    print(f"  Red points: {np.sum(red_mask)}")
    print(f"  Total continuum points: {len(continuum_wl)}")

    if len(continuum_wl) < 3:
        raise RuntimeError(
            f"Insufficient continuum points for {name}."
        )

    # --------------------------------------------------------
    # Initial continuum fit
    # --------------------------------------------------------

    continuum_coeff = np.polyfit(
        continuum_wl - x0,
        continuum_flux,
        1,
    )

    continuum_initial = (
        continuum_coeff[0] * (wl - x0)
        + continuum_coeff[1]
    )

    continuum_residuals = (
        continuum_flux
        - (
            continuum_coeff[0]
            * (continuum_wl - x0)
            + continuum_coeff[1]
        )
    )

    continuum_noise = robust_sigma(
        continuum_residuals
    )

    if not np.isfinite(continuum_noise) or continuum_noise <= 0:
        continuum_noise = np.nanmedian(er)

    if not np.isfinite(continuum_noise) or continuum_noise <= 0:
        continuum_noise = 1.0

    # --------------------------------------------------------
    # Initial line estimate
    # --------------------------------------------------------

    predicted_center = velocity_to_wavelength(
        rest_nm,
        M51_VELOCITY_KMS,
    )

    continuum_at_wl = (
        continuum_coeff[0] * (wl - x0)
        + continuum_coeff[1]
    )

    residual = fl - continuum_at_wl

    peak_index = np.nanargmax(residual)

    peak_wavelength = wl[peak_index]
    peak_amplitude = residual[peak_index]

    if not np.isfinite(peak_amplitude):
        peak_amplitude = 0.0

    peak_amplitude = max(
        float(peak_amplitude),
        0.0,
    )

    print()
    print("Initial line estimate:")
    print(
        f"  M51 predicted center: "
        f"{predicted_center:.6f} nm"
    )
    print(
        f"  Local peak wavelength: "
        f"{peak_wavelength:.6f} nm"
    )
    print(
        f"  Initial amplitude: "
        f"{peak_amplitude:.6e}"
    )

    # --------------------------------------------------------
    # Initial sigma
    # --------------------------------------------------------

    sigma_initial = sigma_inst

    # Allow intrinsic broadening.
    #
    # Lower boundary is the instrumental sigma.
    # Upper boundary corresponds to a broad but still
    # physically plausible emission line.

    sigma_min = sigma_inst
    sigma_max = max(
        3.0 * sigma_inst,
        5.0,
    )

    # --------------------------------------------------------
    # Initial parameters
    # --------------------------------------------------------

    p0 = np.array(
        [
            continuum_coeff[1],
            continuum_coeff[0],
            peak_amplitude,
            predicted_center,
            sigma_initial,
        ],
        dtype=float,
    )

    # --------------------------------------------------------
    # Parameter bounds
    # --------------------------------------------------------

    lower = np.array(
        [
            -np.inf,
            -np.inf,
            0.0,
            predicted_center - 5.0,
            sigma_min,
        ]
    )

    upper = np.array(
        [
            np.inf,
            np.inf,
            np.inf,
            predicted_center + 5.0,
            sigma_max,
        ]
    )

    # --------------------------------------------------------
    # Weighted residual function
    # --------------------------------------------------------

    def residual_function(params):

        model = model_flux(
            wl,
            params[0],
            params[1],
            params[2],
            params[3],
            params[4],
            x0,
        )

        return (
            model - fl
        ) / er

    # --------------------------------------------------------
    # Fit
    # --------------------------------------------------------

    result = least_squares(
        residual_function,
        p0,
        bounds=(lower, upper),
        max_nfev=10000,
    )

    if not result.success:
        print()
        print(
            "WARNING: optimizer did not report "
            "successful convergence."
        )

    params = result.x

    intercept = params[0]
    slope = params[1]
    amplitude = params[2]
    center = params[3]
    sigma_obs = params[4]

    model = model_flux(
        wl,
        intercept,
        slope,
        amplitude,
        center,
        sigma_obs,
        x0,
    )

    residuals = fl - model

    chi2 = np.sum(
        ((fl - model) / er) ** 2
    )

    n = len(wl)
    k = 5

    aic = chi2 + 2.0 * k
    bic = (
        chi2
        + k * np.log(n)
    )

    # --------------------------------------------------------
    # Derived quantities
    # --------------------------------------------------------

    fitted_velocity = wavelength_to_velocity(
        rest_nm,
        center,
    )

    velocity_offset = (
        fitted_velocity
        - M51_VELOCITY_KMS
    )

    observed_fwhm = fwhm_from_sigma(
        sigma_obs
    )

    intrinsic_sigma_sq = (
        sigma_obs ** 2
        - sigma_inst ** 2
    )

    intrinsic_sigma = np.sqrt(
        max(
            intrinsic_sigma_sq,
            0.0,
        )
    )

    intrinsic_fwhm = fwhm_from_sigma(
        intrinsic_sigma
    )

    intrinsic_velocity_width = (
        velocity_width_from_sigma(
            rest_nm,
            intrinsic_sigma,
        )
    )

    integrated_flux = gaussian_integrated_flux(
        amplitude,
        sigma_obs,
    )

    # --------------------------------------------------------
    # Detection significance
    # --------------------------------------------------------

    amplitude_snr = (
        amplitude / continuum_noise
        if continuum_noise > 0
        else np.nan
    )

    # --------------------------------------------------------
    # Goodness-of-fit diagnostics
    # --------------------------------------------------------

    reduced_chi2 = (
        chi2 / (n - k)
        if n > k
        else np.nan
    )

    print()
    print("=" * 70)
    print(f"{name.upper()} PROFILE RESULTS")
    print("=" * 70)

    print()
    print(
        f"Fitted center: "
        f"{center:.9f} nm"
    )

    print(
        f"Fitted velocity: "
        f"{fitted_velocity:+.3f} km/s"
    )

    print(
        f"Velocity offset from M51: "
        f"{velocity_offset:+.3f} km/s"
    )

    print()
    print(
        f"Observed sigma: "
        f"{sigma_obs:.6f} nm"
    )

    print(
        f"Observed FWHM: "
        f"{observed_fwhm:.6f} nm"
    )

    print()
    print(
        f"Instrument sigma: "
        f"{sigma_inst:.6f} nm"
    )

    print(
        f"Instrument FWHM: "
        f"{2.354820045 * sigma_inst:.6f} nm"
    )

    print()
    print(
        f"Intrinsic sigma: "
        f"{intrinsic_sigma:.6f} nm"
    )

    print(
        f"Intrinsic FWHM: "
        f"{intrinsic_fwhm:.6f} nm"
    )

    print(
        f"Intrinsic velocity width: "
        f"{intrinsic_velocity_width:.3f} km/s"
    )

    print()
    print(
        f"Amplitude: "
        f"{amplitude:.6e}"
    )

    print(
        f"Amplitude / continuum noise: "
        f"{amplitude_snr:.3f}"
    )

    print(
        f"Integrated Gaussian flux: "
        f"{integrated_flux:.6e}"
    )

    print()
    print(f"Chi²: {chi2:.3f}")
    print(f"Reduced chi²: {reduced_chi2:.3f}")
    print(f"AIC: {aic:.3f}")
    print(f"BIC: {bic:.3f}")

    print()
    print("Fit status:")
    print(f"  success = {result.success}")
    print(f"  message = {result.message}")

    # --------------------------------------------------------
    # Store profile data
    # --------------------------------------------------------

    profile_df = pd.DataFrame(
        {
            "line": name,
            "wavelength_nm": wl,
            "flux": fl,
            "flux_error": er,
            "model": model,
            "residual": residuals,
            "continuum": linear_continuum(
                wl,
                intercept,
                slope,
                x0,
            ),
        }
    )

    return {
        "line": name,
        "rest_wavelength_nm": rest_nm,
        "predicted_wavelength_nm": predicted_center,
        "fitted_wavelength_nm": center,
        "fitted_velocity_kms": fitted_velocity,
        "velocity_offset_kms": velocity_offset,
        "instrument_sigma_nm": sigma_inst,
        "instrument_fwhm_nm": (
            2.354820045 * sigma_inst
        ),
        "observed_sigma_nm": sigma_obs,
        "observed_fwhm_nm": observed_fwhm,
        "intrinsic_sigma_nm": intrinsic_sigma,
        "intrinsic_fwhm_nm": intrinsic_fwhm,
        "intrinsic_velocity_width_kms": (
            intrinsic_velocity_width
        ),
        "amplitude": amplitude,
        "amplitude_snr": amplitude_snr,
        "integrated_flux": integrated_flux,
        "continuum_noise": continuum_noise,
        "chi2": chi2,
        "reduced_chi2": reduced_chi2,
        "aic": aic,
        "bic": bic,
        "n_points": n,
        "fit_success": result.success,
        "profile_df": profile_df,
    }


# ============================================================
# Fit Pa-beta
# ============================================================

pa_beta_result = fit_hydrogen_line(
    "Pa-beta",
    PA_BETA_REST_NM,
    PA_BETA_WINDOW,
    PA_BETA_BLUE,
    PA_BETA_RED,
    pa_beta_sigma_inst,
)


# ============================================================
# Fit Pa-gamma
# ============================================================

pa_gamma_result = fit_hydrogen_line(
    "Pa-gamma",
    PA_GAMMA_REST_NM,
    PA_GAMMA_WINDOW,
    PA_GAMMA_BLUE,
    PA_GAMMA_RED,
    pa_gamma_sigma_inst,
)


# ============================================================
# Compare the two hydrogen lines
# ============================================================

print()
print("=" * 70)
print("4. PA-BETA / PA-GAMMA COMPARISON")
print("=" * 70)

beta_velocity = (
    pa_beta_result["fitted_velocity_kms"]
)

gamma_velocity = (
    pa_gamma_result["fitted_velocity_kms"]
)

velocity_difference = (
    gamma_velocity
    - beta_velocity
)

beta_offset = (
    pa_beta_result["velocity_offset_kms"]
)

gamma_offset = (
    pa_gamma_result["velocity_offset_kms"]
)

flux_beta = (
    pa_beta_result["integrated_flux"]
)

flux_gamma = (
    pa_gamma_result["integrated_flux"]
)

if (
    np.isfinite(flux_beta)
    and np.isfinite(flux_gamma)
    and flux_gamma > 0
):
    flux_ratio = (
        flux_beta / flux_gamma
    )
else:
    flux_ratio = np.nan

print()
print(
    f"Independent M51 velocity: "
    f"{M51_VELOCITY_KMS:+.3f} km/s"
)

print()
print(
    f"Pa-beta velocity: "
    f"{beta_velocity:+.3f} km/s"
)

print(
    f"Pa-gamma velocity: "
    f"{gamma_velocity:+.3f} km/s"
)

print()
print(
    f"Pa-beta velocity offset: "
    f"{beta_offset:+.3f} km/s"
)

print(
    f"Pa-gamma velocity offset: "
    f"{gamma_offset:+.3f} km/s"
)

print()
print(
    f"Pa-gamma minus Pa-beta velocity: "
    f"{velocity_difference:+.3f} km/s"
)

print()
print(
    f"Pa-beta integrated flux: "
    f"{flux_beta:.6e}"
)

print(
    f"Pa-gamma integrated flux: "
    f"{flux_gamma:.6e}"
)

print(
    f"Pa-beta / Pa-gamma flux ratio: "
    f"{flux_ratio:.6f}"
)


# ============================================================
# Width comparison
# ============================================================

print()
print("=" * 70)
print("5. LINE-WIDTH COMPARISON")
print("=" * 70)

beta_intrinsic_velocity = (
    pa_beta_result[
        "intrinsic_velocity_width_kms"
    ]
)

gamma_intrinsic_velocity = (
    pa_gamma_result[
        "intrinsic_velocity_width_kms"
    ]
)

print()
print(
    "Pa-beta intrinsic FWHM: "
    f"{pa_beta_result['intrinsic_fwhm_nm']:.6f} nm"
)

print(
    "Pa-gamma intrinsic FWHM: "
    f"{pa_gamma_result['intrinsic_fwhm_nm']:.6f} nm"
)

print()
print(
    "Pa-beta intrinsic velocity width: "
    f"{beta_intrinsic_velocity:.3f} km/s"
)

print(
    "Pa-gamma intrinsic velocity width: "
    f"{gamma_intrinsic_velocity:.3f} km/s"
)


# ============================================================
# Profile consistency assessment
# ============================================================

print()
print("=" * 70)
print("6. PROFILE CONSISTENCY")
print("=" * 70)

velocity_tolerance = 50.0

beta_velocity_consistent = (
    abs(beta_offset)
    <= velocity_tolerance
)

gamma_velocity_consistent = (
    abs(gamma_offset)
    <= velocity_tolerance
)

mutual_velocity_consistent = (
    abs(velocity_difference)
    <= velocity_tolerance
)

print()
print(
    "Velocity tolerance used for diagnostic flag: "
    f"{velocity_tolerance:.1f} km/s"
)

print(
    "Pa-beta consistent with M51: "
    f"{beta_velocity_consistent}"
)

print(
    "Pa-gamma consistent with M51: "
    f"{gamma_velocity_consistent}"
)

print(
    "Pa-beta / Pa-gamma mutually consistent: "
    f"{mutual_velocity_consistent}"
)


# ============================================================
# Save results
# ============================================================

summary_rows = [
    {
        key: value
        for key, value in pa_beta_result.items()
        if key != "profile_df"
    },
    {
        key: value
        for key, value in pa_gamma_result.items()
        if key != "profile_df"
    },
]

summary_df = pd.DataFrame(summary_rows)

summary_df["independent_m51_velocity_kms"] = (
    M51_VELOCITY_KMS
)

summary_df[
    "pa_beta_pa_gamma_velocity_difference_kms"
] = velocity_difference

summary_df[
    "pa_beta_pa_gamma_flux_ratio"
] = flux_ratio

summary_df[
    "velocity_tolerance_kms"
] = velocity_tolerance

summary_df[
    "both_velocity_consistent"
] = (
    beta_velocity_consistent
    and gamma_velocity_consistent
    and mutual_velocity_consistent
)

summary_df.to_csv(
    RESULTS_CSV,
    index=False,
)


# ============================================================
# Save profile data
# ============================================================

profile_df = pd.concat(
    [
        pa_beta_result["profile_df"],
        pa_gamma_result["profile_df"],
    ],
    ignore_index=True,
)

profile_df.to_csv(
    PROFILE_CSV,
    index=False,
)


# ============================================================
# Create profile plot
# ============================================================

print()
print("=" * 70)
print("7. CREATING PROFILE PLOT")
print("=" * 70)

fig, axes = plt.subplots(
    2,
    1,
    figsize=(11, 9),
)

for ax, result in zip(
    axes,
    [pa_beta_result, pa_gamma_result],
):

    data = result["profile_df"]

    ax.errorbar(
        data["wavelength_nm"],
        data["flux"],
        yerr=data["flux_error"],
        fmt="o",
        markersize=4,
        alpha=0.7,
        label="JWST X1D",
    )

    ax.plot(
        data["wavelength_nm"],
        data["model"],
        linewidth=2,
        label="Gaussian + linear continuum",
    )

    ax.plot(
        data["wavelength_nm"],
        data["continuum"],
        linestyle="--",
        linewidth=1.5,
        label="Continuum",
    )

    ax.axvline(
        result["predicted_wavelength_nm"],
        linestyle=":",
        linewidth=1.5,
        label="M51 prediction",
    )

    ax.axvline(
        result["fitted_wavelength_nm"],
        linestyle="-.",
        linewidth=1.5,
        label="Fitted center",
    )

    ax.set_ylabel(
        "Flux"
    )

    ax.set_title(
        f"{result['line']} — "
        f"v = "
        f"{result['fitted_velocity_kms']:.2f} km/s"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend(
        fontsize=8
    )

axes[-1].set_xlabel(
    "Wavelength (nm)"
)

fig.tight_layout()

fig.savefig(
    PLOT_PATH,
    dpi=180,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# Create residual plot
# ============================================================

print()
print("=" * 70)
print("8. CREATING RESIDUAL PLOT")
print("=" * 70)

fig, axes = plt.subplots(
    2,
    1,
    figsize=(11, 7),
)

for ax, result in zip(
    axes,
    [pa_beta_result, pa_gamma_result],
):

    data = result["profile_df"]

    normalized_residual = (
        data["residual"]
        / data["flux_error"]
    )

    ax.axhline(
        0.0,
        linestyle="--",
        linewidth=1,
    )

    ax.plot(
        data["wavelength_nm"],
        normalized_residual,
        marker="o",
        linewidth=1,
    )

    ax.set_ylabel(
        "Residual / error"
    )

    ax.set_title(
        result["line"]
    )

    ax.grid(
        alpha=0.25
    )

axes[-1].set_xlabel(
    "Wavelength (nm)"
)

fig.tight_layout()

fig.savefig(
    RESIDUAL_PLOT_PATH,
    dpi=180,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# Final interpretation
# ============================================================

print()
print("=" * 70)
print("FINAL INTERPRETATION")
print("=" * 70)

print()
print(
    "This experiment independently fits the Pa-beta and "
    "Pa-gamma spectral profiles."
)

print()
print(
    "Each line is fitted with:"
)

print(
    "  • local linear continuum"
)

print(
    "  • non-negative emission amplitude"
)

print(
    "  • fitted wavelength centroid"
)

print(
    "  • Gaussian observed width"
)

print(
    "  • instrumental-resolution constraint"
)

print()
print(
    "The M51 velocity of +573.72 km/s is used as an "
    "independent reference, not as a forced fitted velocity."
)

print()
print(
    "A strong hydrogen interpretation requires:"
)

print(
    "  1. Pa-beta velocity consistent with M51;"
)

print(
    "  2. Pa-gamma velocity consistent with M51;"
)

print(
    "  3. Pa-beta and Pa-gamma velocities mutually consistent;"
)

print(
    "  4. statistically significant emission profiles;"
)

print(
    "  5. line widths compatible with the instrumental "
    "resolution or physically plausible intrinsic broadening."
)

print()
print(
    "Current velocity results:"
)

print(
    f"  Pa-beta:  {beta_velocity:+.3f} km/s"
)

print(
    f"  Pa-gamma: {gamma_velocity:+.3f} km/s"
)

print(
    f"  Difference: {velocity_difference:+.3f} km/s"
)

print()
print(
    "The fitted line velocities should be considered together "
    "with the previously measured spatial correlation of the "
    "Pa-beta and Pa-gamma emission maps."
)

print()
print(
    "IMPORTANT:"
)

print(
    "The fitted Pa-beta / Pa-gamma flux ratio is not by itself "
    "a Case-B test. A physical comparison requires consistent "
    "flux calibration, extinction treatment, aperture definition, "
    "and an appropriate hydrogen recombination model."
)

print()
print(
    "Likewise, agreement in velocity and spatial morphology "
    "does not mathematically prove that both lines are hydrogen."
)

print()
print(
    "The purpose of this experiment is to determine whether the "
    "two spectral profiles independently behave as compatible "
    "hydrogen recombination lines from the same M51 velocity field."
)

print()
print("Outputs:")
print(f"  {RESULTS_CSV}")
print(f"  {PROFILE_CSV}")
print(f"  {PLOT_PATH}")
print(f"  {RESIDUAL_PLOT_PATH}")

print()
print("=" * 70)
print("HYDROGEN PA-BETA / PA-GAMMA PROFILE TEST COMPLETE")
print("=" * 70)
