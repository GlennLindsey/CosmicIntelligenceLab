from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.io import fits
from scipy.optimize import least_squares


# ============================================================
# M51 1284 nm C I BLEND MODEL
# ============================================================

print("=" * 70)
print("M51 1284 NM C I BLEND MODEL")
print("NEXT-GENERATION SPECTROSCOPY AGENT TEST")
print("=" * 70)


# ============================================================
# Configuration
# ============================================================

X1D_PATH = Path(
    "data/m51_jwst_level3/"
    "jw03435-o006_t010_nirspec_g140m-f100lp_x1d.fits"
)

CATALOGUE_PATH = Path(
    "data/atomic_lines/"
    "m51_atomic_line_catalogue.csv"
)

OUTPUT_CSV = Path(
    "data/atomic_lines/"
    "m51_1284_ci_blend_model_results.csv"
)

OUTPUT_PLOT = Path(
    "m51_1284_ci_blend_model.png"
)

OUTPUT_COMPONENTS = Path(
    "data/atomic_lines/"
    "m51_1284_ci_blend_components.csv"
)


# ============================================================
# Scientific constants
# ============================================================

C_KMS = 299792.458

REFERENCE_VELOCITY_KMS = 573.72

OBSERVED_FEATURE_NM = 1284.26130440

# Vacuum wavelength of hydrogen Pa beta.
PA_BETA_REST_NM = 1281.8070

# Cs II vacuum wavelength used in our previous analysis.
CSII_REST_NM = 1284.61537587

# NIRSpec resolving power determined previously.
RESOLVING_POWER = 916.3

# Fit interval.
FIT_MIN_NM = 1278.0
FIT_MAX_NM = 1290.0

# C I components must be close enough to the observed feature
# to participate in the unresolved blend.
CI_COMPONENT_WINDOW_FWHM = 1.0

# Velocity search interval around the independently measured
# M51 local velocity.
VELOCITY_MIN_KMS = REFERENCE_VELOCITY_KMS - 250.0
VELOCITY_MAX_KMS = REFERENCE_VELOCITY_KMS + 250.0


# ============================================================
# Utility functions
# ============================================================

def velocity_factor(velocity_kms):
    """Relativistic Doppler factor."""

    beta = velocity_kms / C_KMS

    return np.sqrt(
        (1.0 + beta)
        / (1.0 - beta)
    )


def velocity_to_wavelength(
    rest_wavelength_nm,
    velocity_kms,
):
    """Convert rest vacuum wavelength to observed vacuum wavelength."""

    return (
        rest_wavelength_nm
        * velocity_factor(velocity_kms)
    )


def wavelength_to_velocity(
    observed_wavelength_nm,
    rest_wavelength_nm,
):
    """Relativistic wavelength-to-velocity conversion."""

    ratio = (
        observed_wavelength_nm
        / rest_wavelength_nm
    )

    beta = (
        ratio**2 - 1.0
    ) / (
        ratio**2 + 1.0
    )

    return beta * C_KMS


def gaussian(
    wavelength_nm,
    center_nm,
    sigma_nm,
):
    """Unit-height Gaussian."""

    return np.exp(
        -0.5
        * (
            (wavelength_nm - center_nm)
            / sigma_nm
        ) ** 2
    )


def linear_continuum(
    wavelength_nm,
    intercept,
    slope,
    reference_nm=OBSERVED_FEATURE_NM,
):
    """Linear local continuum."""

    return (
        intercept
        + slope
        * (
            wavelength_nm
            - reference_nm
        )
    )


def chi_squared(
    residuals,
):
    return np.sum(
        residuals**2
    )


def bic(
    chi2,
    n_parameters,
    n_data,
):
    return (
        chi2
        + n_parameters
        * np.log(n_data)
    )


def aic(
    chi2,
    n_parameters,
):
    return (
        chi2
        + 2.0
        * n_parameters
    )


# ============================================================
# Load X1D
# ============================================================

print()
print("=" * 70)
print("LOADING M51 X1D SPECTRUM")
print("=" * 70)

with fits.open(
    X1D_PATH,
    memmap=False,
) as hdul:

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

# JWST X1D wavelength is in microns.
wavelength_nm = (
    wavelength_um
    * 1000.0
)

print()
print(
    f"Spectral points: "
    f"{len(wavelength_nm)}"
)

print(
    f"Wavelength range: "
    f"{np.nanmin(wavelength_nm):.6f} - "
    f"{np.nanmax(wavelength_nm):.6f} nm"
)


# ============================================================
# Select fitting window
# ============================================================

window = (
    np.isfinite(wavelength_nm)
    & np.isfinite(flux)
    & np.isfinite(flux_error)
    & (flux_error > 0)
    & (wavelength_nm >= FIT_MIN_NM)
    & (wavelength_nm <= FIT_MAX_NM)
)

wave = wavelength_nm[window]
data = flux[window]
error = flux_error[window]

print()
print("=" * 70)
print("LOCAL SPECTRAL FIT")
print("=" * 70)

print(
    f"Window: "
    f"{FIT_MIN_NM:.3f} - "
    f"{FIT_MAX_NM:.3f} nm"
)

print(
    f"Usable spectral points: "
    f"{len(wave)}"
)

print(
    f"Actual range: "
    f"{wave.min():.6f} - "
    f"{wave.max():.6f} nm"
)


# ============================================================
# Instrumental resolution
# ============================================================

instrument_fwhm_nm = (
    OBSERVED_FEATURE_NM
    / RESOLVING_POWER
)

instrument_sigma_nm = (
    instrument_fwhm_nm
    / 2.354820045
)

print()
print("=" * 70)
print("NIRSPEC INSTRUMENT RESOLUTION")
print("=" * 70)

print(
    f"Resolving power: "
    f"R = {RESOLVING_POWER:.1f}"
)

print(
    f"Instrument FWHM: "
    f"{instrument_fwhm_nm:.6f} nm"
)

print(
    f"Instrument sigma: "
    f"{instrument_sigma_nm:.6f} nm"
)


# ============================================================
# Load atomic catalogue
# ============================================================

print()
print("=" * 70)
print("LOADING ATOMIC CATALOGUE")
print("=" * 70)

catalogue = pd.read_csv(
    CATALOGUE_PATH
)

ci = catalogue[
    catalogue["species"]
    .astype(str)
    .str.strip()
    .eq("C I")
].copy()

print(
    f"Total catalogue rows: "
    f"{len(catalogue)}"
)

print(
    f"C I rows: "
    f"{len(ci)}"
)


# ============================================================
# Select C I transitions participating in blend
# ============================================================

ci["rest_wavelength_nm"] = pd.to_numeric(
    ci["ritz_wavelength_vacuum_nm"],
    errors="coerce",
)

ci["Aki_s-1_numeric"] = pd.to_numeric(
    ci["Aki_s-1"],
    errors="coerce",
)

ci = ci[
    np.isfinite(
        ci["rest_wavelength_nm"]
    )
].copy()

ci["predicted_wavelength_nm"] = (
    velocity_to_wavelength(
        ci["rest_wavelength_nm"].to_numpy(),
        REFERENCE_VELOCITY_KMS,
    )
)

ci["velocity_offset_kms"] = (
    wavelength_to_velocity(
        OBSERVED_FEATURE_NM,
        ci["rest_wavelength_nm"].to_numpy(),
    )
    - REFERENCE_VELOCITY_KMS
)

ci["distance_nm"] = (
    np.abs(
        ci["predicted_wavelength_nm"]
        - OBSERVED_FEATURE_NM
    )
)

ci["distance_fwhm"] = (
    ci["distance_nm"]
    / instrument_fwhm_nm
)

ci_blend = ci[
    ci["distance_fwhm"]
    <= CI_COMPONENT_WINDOW_FWHM
].copy()

# Require a usable Aki for the Aki-weighted model.
ci_blend["Aki_s-1_numeric"] = pd.to_numeric(
    ci_blend["Aki_s-1_numeric"],
    errors="coerce",
)

ci_blend = ci_blend[
    np.isfinite(
        ci_blend["Aki_s-1_numeric"]
    )
    & (
        ci_blend["Aki_s-1_numeric"]
        > 0
    )
].copy()

ci_blend = ci_blend.sort_values(
    "predicted_wavelength_nm"
).reset_index(drop=True)

print()
print(
    "C I components within "
    "one instrumental FWHM:"
)

for _, row in ci_blend.iterrows():

    print(
        f"  "
        f"{row['rest_wavelength_nm']:.6f} nm"
        f" -> "
        f"{row['predicted_wavelength_nm']:.6f} nm"
        f" | "
        f"Aki = "
        f"{row['Aki_s-1_numeric']:.3g}"
    )

if len(ci_blend) == 0:
    raise RuntimeError(
        "No usable C I blend components found."
    )


# ============================================================
# Normalize C I Aki weights
# ============================================================

aki_values = (
    ci_blend[
        "Aki_s-1_numeric"
    ]
    .to_numpy(dtype=float)
)

aki_weights = (
    aki_values
    / np.max(aki_values)
)


# ============================================================
# Model definitions
# ============================================================

def model_pa_beta(
    params,
    wavelength_nm,
):
    """
    Parameters:

    0: continuum intercept
    1: continuum slope
    2: line amplitude
    3: velocity
    """

    continuum = linear_continuum(
        wavelength_nm,
        params[0],
        params[1],
    )

    center = velocity_to_wavelength(
        PA_BETA_REST_NM,
        params[3],
    )

    line = (
        params[2]
        * gaussian(
            wavelength_nm,
            center,
            instrument_sigma_nm,
        )
    )

    return (
        continuum
        + line
    )


def model_csii(
    params,
    wavelength_nm,
):
    """
    Parameters:

    0: continuum intercept
    1: continuum slope
    2: line amplitude
    3: velocity
    """

    continuum = linear_continuum(
        wavelength_nm,
        params[0],
        params[1],
    )

    center = velocity_to_wavelength(
        CSII_REST_NM,
        params[3],
    )

    line = (
        params[2]
        * gaussian(
            wavelength_nm,
            center,
            instrument_sigma_nm,
        )
    )

    return (
        continuum
        + line
    )


def model_ci_aki(
    params,
    wavelength_nm,
):
    """
    Aki-weighted C I blend.

    Parameters:

    0: continuum intercept
    1: continuum slope
    2: common C I amplitude
    3: common C I velocity
    """

    continuum = linear_continuum(
        wavelength_nm,
        params[0],
        params[1],
    )

    velocity = params[3]

    blend = np.zeros_like(
        wavelength_nm
    )

    for weight, rest in zip(
        aki_weights,
        ci_blend[
            "rest_wavelength_nm"
        ].to_numpy(),
    ):

        center = velocity_to_wavelength(
            rest,
            velocity,
        )

        blend += (
            weight
            * gaussian(
                wavelength_nm,
                center,
                instrument_sigma_nm,
            )
        )

    return (
        continuum
        + params[2]
        * blend
    )


def model_ci_free(
    params,
    wavelength_nm,
):
    """
    Free-amplitude C I blend.

    Parameters:

    0: continuum intercept
    1: continuum slope
    2: common velocity
    3+: individual C I amplitudes
    """

    continuum = linear_continuum(
        wavelength_nm,
        params[0],
        params[1],
    )

    velocity = params[2]

    model = continuum.copy()

    rests = ci_blend[
        "rest_wavelength_nm"
    ].to_numpy()

    amplitudes = params[3:]

    for amplitude, rest in zip(
        amplitudes,
        rests,
    ):

        center = velocity_to_wavelength(
            rest,
            velocity,
        )

        model += (
            amplitude
            * gaussian(
                wavelength_nm,
                center,
                instrument_sigma_nm,
            )
        )

    return model


# ============================================================
# Fit helper
# ============================================================

def fit_model(
    model_function,
    initial,
    lower,
    upper,
):
    """Weighted least-squares fit."""

    def residuals(params):

        model = model_function(
            params,
            wave,
        )

        return (
            model
            - data
        ) / error

    result = least_squares(
        residuals,
        x0=np.asarray(
            initial,
            dtype=float,
        ),
        bounds=(
            np.asarray(
                lower,
                dtype=float,
            ),
            np.asarray(
                upper,
                dtype=float,
            ),
        ),
    )

    model = model_function(
        result.x,
        wave,
    )

    residual = (
        model
        - data
    ) / error

    chi2 = chi_squared(
        residual
    )

    n = len(data)

    k = len(
        result.x
    )

    return {
        "result": result,
        "model": model,
        "chi2": chi2,
        "aic": aic(
            chi2,
            k,
        ),
        "bic": bic(
            chi2,
            k,
            n,
        ),
        "n_parameters": k,
        "success": result.success,
    }


# ============================================================
# Initial continuum
# ============================================================

continuum_guess = np.nanmedian(
    data
)

continuum_scale = (
    np.nanpercentile(
        data,
        90,
    )
    - np.nanpercentile(
        data,
        10,
    )
)

if not np.isfinite(
    continuum_scale
) or continuum_scale <= 0:

    continuum_scale = 1.0


# ============================================================
# Fit Pa beta
# ============================================================

print()
print("=" * 70)
print("FITTING PA BETA")
print("=" * 70)

pa_initial = [
    continuum_guess,
    0.0,
    continuum_scale,
    REFERENCE_VELOCITY_KMS,
]

pa_lower = [
    -np.inf,
    -np.inf,
    -np.inf,
    VELOCITY_MIN_KMS,
]

pa_upper = [
    np.inf,
    np.inf,
    np.inf,
    VELOCITY_MAX_KMS,
]

pa_fit = fit_model(
    model_pa_beta,
    pa_initial,
    pa_lower,
    pa_upper,
)

print(
    f"Velocity: "
    f"{pa_fit['result'].x[3]:+.3f} km/s"
)

print(
    f"Chi²: "
    f"{pa_fit['chi2']:.3f}"
)

print(
    f"AIC: "
    f"{pa_fit['aic']:.3f}"
)

print(
    f"BIC: "
    f"{pa_fit['bic']:.3f}"
)


# ============================================================
# Fit Cs II
# ============================================================

print()
print("=" * 70)
print("FITTING CS II")
print("=" * 70)

cs_initial = [
    continuum_guess,
    0.0,
    continuum_scale,
    REFERENCE_VELOCITY_KMS,
]

cs_lower = [
    -np.inf,
    -np.inf,
    -np.inf,
    VELOCITY_MIN_KMS,
]

cs_upper = [
    np.inf,
    np.inf,
    np.inf,
    VELOCITY_MAX_KMS,
]

cs_fit = fit_model(
    model_csii,
    cs_initial,
    cs_lower,
    cs_upper,
)

print(
    f"Velocity: "
    f"{cs_fit['result'].x[3]:+.3f} km/s"
)

print(
    f"Chi²: "
    f"{cs_fit['chi2']:.3f}"
)

print(
    f"AIC: "
    f"{cs_fit['aic']:.3f}"
)

print(
    f"BIC: "
    f"{cs_fit['bic']:.3f}"
)


# ============================================================
# Fit Aki-weighted C I blend
# ============================================================

print()
print("=" * 70)
print("FITTING AKI-WEIGHTED C I BLEND")
print("=" * 70)

ci_aki_initial = [
    continuum_guess,
    0.0,
    continuum_scale,
    REFERENCE_VELOCITY_KMS,
]

ci_aki_lower = [
    -np.inf,
    -np.inf,
    0.0,
    VELOCITY_MIN_KMS,
]

ci_aki_upper = [
    np.inf,
    np.inf,
    np.inf,
    VELOCITY_MAX_KMS,
]

ci_aki_fit = fit_model(
    model_ci_aki,
    ci_aki_initial,
    ci_aki_lower,
    ci_aki_upper,
)

print(
    f"Velocity: "
    f"{ci_aki_fit['result'].x[3]:+.3f} km/s"
)

print(
    f"Chi²: "
    f"{ci_aki_fit['chi2']:.3f}"
)

print(
    f"AIC: "
    f"{ci_aki_fit['aic']:.3f}"
)

print(
    f"BIC: "
    f"{ci_aki_fit['bic']:.3f}"
)


# ============================================================
# Fit free-amplitude C I blend
# ============================================================

print()
print("=" * 70)
print("FITTING FREE-AMPLITUDE C I BLEND")
print("=" * 70)

n_ci = len(
    ci_blend
)

ci_free_initial = [
    continuum_guess,
    0.0,
    REFERENCE_VELOCITY_KMS,
] + [
    continuum_scale
    / max(
        n_ci,
        1,
    )
] * n_ci

ci_free_lower = [
    -np.inf,
    -np.inf,
    VELOCITY_MIN_KMS,
] + [
    0.0
] * n_ci

ci_free_upper = [
    np.inf,
    np.inf,
    VELOCITY_MAX_KMS,
] + [
    np.inf
] * n_ci

ci_free_fit = fit_model(
    model_ci_free,
    ci_free_initial,
    ci_free_lower,
    ci_free_upper,
)

print(
    f"Velocity: "
    f"{ci_free_fit['result'].x[2]:+.3f} km/s"
)

print(
    f"Chi²: "
    f"{ci_free_fit['chi2']:.3f}"
)

print(
    f"AIC: "
    f"{ci_free_fit['aic']:.3f}"
)

print(
    f"BIC: "
    f"{ci_free_fit['bic']:.3f}"
)


# ============================================================
# Model comparison
# ============================================================

fits = {
    "Pa beta": pa_fit,
    "Cs II": cs_fit,
    "C I Aki-weighted": ci_aki_fit,
    "C I free-amplitude": ci_free_fit,
}

print()
print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print()
print(
    f"{'Model':<24}"
    f"{'Chi²':>14}"
    f"{'AIC':>14}"
    f"{'BIC':>14}"
)

print("-" * 66)

for name, fit in fits.items():

    print(
        f"{name:<24}"
        f"{fit['chi2']:>14.3f}"
        f"{fit['aic']:>14.3f}"
        f"{fit['bic']:>14.3f}"
    )


# ============================================================
# Δχ² relative to Pa beta
# ============================================================

print()
print("=" * 70)
print("DELTA CHI-SQUARED RELATIVE TO PA BETA")
print("=" * 70)

reference_chi2 = (
    pa_fit["chi2"]
)

for name, fit in fits.items():

    delta = (
        fit["chi2"]
        - reference_chi2
    )

    print(
        f"{name:<24}"
        f"{delta:+.3f}"
    )


# ============================================================
# Best model
# ============================================================

best_name = min(
    fits,
    key=lambda name: fits[name]["bic"],
)

print()
print("=" * 70)
print("BEST MODEL BY BIC")
print("=" * 70)

print(
    f"Best model: "
    f"{best_name}"
)


# ============================================================
# Save component table
# ============================================================

component_rows = []

free_params = (
    ci_free_fit["result"].x
)

free_velocity = (
    free_params[2]
)

free_amplitudes = (
    free_params[3:]
)

aki_params = (
    ci_aki_fit["result"].x
)

aki_velocity = (
    aki_params[3]
)

for i, (_, row) in enumerate(
    ci_blend.iterrows()
):

    rest = (
        row[
            "rest_wavelength_nm"
        ]
    )

    component_rows.append(
        {
            "species": "C I",
            "rest_wavelength_vacuum_nm":
                rest,
            "Aki_s-1":
                row[
                    "Aki_s-1_numeric"
                ],
            "reference_predicted_wavelength_nm":
                row[
                    "predicted_wavelength_nm"
                ],
            "Aki_weight":
                aki_weights[i],
            "aki_model_velocity_kms":
                aki_velocity,
            "aki_model_center_nm":
                velocity_to_wavelength(
                    rest,
                    aki_velocity,
                ),
            "free_model_velocity_kms":
                free_velocity,
            "free_model_center_nm":
                velocity_to_wavelength(
                    rest,
                    free_velocity,
                ),
            "free_model_amplitude":
                free_amplitudes[i],
        }
    )

component_table = pd.DataFrame(
    component_rows
)

component_table.to_csv(
    OUTPUT_COMPONENTS,
    index=False,
)


# ============================================================
# Save model summary
# ============================================================

summary_rows = []

for name, fit in fits.items():

    result = fit[
        "result"
    ]

    velocity = np.nan

    if name in (
        "Pa beta",
        "Cs II",
        "C I Aki-weighted",
    ):

        velocity = (
            result.x[-1]
            if name != "C I free-amplitude"
            else np.nan
        )

    elif name == "C I free-amplitude":

        velocity = result.x[2]

    summary_rows.append(
        {
            "model": name,
            "chi2": fit["chi2"],
            "AIC": fit["aic"],
            "BIC": fit["bic"],
            "n_parameters":
                fit["n_parameters"],
            "velocity_kms":
                velocity,
        }
    )

summary_table = pd.DataFrame(
    summary_rows
)

summary_table.to_csv(
    OUTPUT_CSV,
    index=False,
)


# ============================================================
# Generate model curves
# ============================================================

fine_wave = np.linspace(
    FIT_MIN_NM,
    FIT_MAX_NM,
    3000,
)

pa_curve = model_pa_beta(
    pa_fit["result"].x,
    fine_wave,
)

cs_curve = model_csii(
    cs_fit["result"].x,
    fine_wave,
)

ci_aki_curve = model_ci_aki(
    ci_aki_fit["result"].x,
    fine_wave,
)

ci_free_curve = model_ci_free(
    ci_free_fit["result"].x,
    fine_wave,
)


# ============================================================
# Plot
# ============================================================

print()
print("=" * 70)
print("CREATING MODEL COMPARISON PLOT")
print("=" * 70)

fig, ax = plt.subplots(
    figsize=(12, 7)
)

ax.errorbar(
    wave,
    data,
    yerr=error,
    fmt="o",
    markersize=4,
    alpha=0.65,
    label="M51 X1D",
)

ax.plot(
    fine_wave,
    pa_curve,
    linewidth=2,
    label="Pa beta",
)

ax.plot(
    fine_wave,
    cs_curve,
    linewidth=2,
    label="Cs II",
)

ax.plot(
    fine_wave,
    ci_aki_curve,
    linewidth=2,
    label="C I Aki-weighted",
)

ax.plot(
    fine_wave,
    ci_free_curve,
    linewidth=2,
    label="C I free-amplitude",
)

ax.axvline(
    OBSERVED_FEATURE_NM,
    linestyle="--",
    linewidth=1.5,
    label="Observed 1284.2613 nm",
)

ax.set_xlabel(
    "Wavelength (nm)"
)

ax.set_ylabel(
    "Flux"
)

ax.set_title(
    "M51 1284 nm: Pa beta vs Cs II vs C I Blend"
)

ax.legend()

ax.grid(
    alpha=0.25
)

fig.tight_layout()

fig.savefig(
    OUTPUT_PLOT,
    dpi=200,
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
    "This experiment compares four models:"
)

print(
    "  1. Pa beta"
)

print(
    "  2. Cs II"
)

print(
    "  3. Aki-weighted C I blend"
)

print(
    "  4. Free-amplitude C I blend"
)

print()
print(
    "All models use:"
)

print(
    "  - the same M51 X1D spectrum;"
)

print(
    "  - the same wavelength interval;"
)

print(
    "  - the same local linear continuum;"
)

print(
    "  - the same NIRSpec instrumental resolution;"
)

print(
    "  - a common fitted velocity around the "
    "independent M51 reference velocity."
)

print()
print(
    "IMPORTANT:"
)

print(
    "The Aki-weighted C I model is exploratory."
)

print(
    "Aki values alone do not determine observed "
    "line fluxes because excitation and level "
    "populations also matter."
)

print()
print(
    "The free-amplitude C I model therefore provides "
    "a less restrictive test of whether the observed "
    "spectral structure is spatially consistent with "
    "the C I transition wavelengths."
)

print()
print(
    f"Best model by BIC: "
    f"{best_name}"
)

print()
print(
    "Outputs:"
)

print(
    f"  {OUTPUT_CSV}"
)

print(
    f"  {OUTPUT_COMPONENTS}"
)

print(
    f"  {OUTPUT_PLOT}"
)

print()
print("=" * 70)
print("C I BLEND MODEL TEST COMPLETE")
print("=" * 70)
