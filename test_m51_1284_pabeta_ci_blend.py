from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.io import fits
from scipy.optimize import least_squares


# ============================================================
# M51 1284 NM: PA BETA + C I BLEND TEST
# ============================================================

print("=" * 70)
print("M51 1284 NM PA BETA + C I BLEND TEST")
print("NEXT-GENERATION SPECTROSCOPY AGENT")
print("=" * 70)


# ============================================================
# Paths
# ============================================================

X1D_PATH = Path(
    "data/m51_jwst_level3/"
    "jw03435-o006_t010_nirspec_g140m-f100lp_x1d.fits"
)

CATALOGUE_PATH = Path(
    "data/atomic_lines/"
    "m51_atomic_line_catalogue.csv"
)

RESULTS_PATH = Path(
    "data/atomic_lines/"
    "m51_1284_pabeta_ci_blend_results.csv"
)

COMPONENTS_PATH = Path(
    "data/atomic_lines/"
    "m51_1284_pabeta_ci_blend_components.csv"
)

PLOT_PATH = Path(
    "m51_1284_pabeta_ci_blend.png"
)


# ============================================================
# Constants
# ============================================================

C_KMS = 299792.458

REFERENCE_VELOCITY_KMS = 573.72

OBSERVED_FEATURE_NM = 1284.26130440

PA_BETA_REST_NM = 1281.8070

RESOLVING_POWER = 916.3

FIT_MIN_NM = 1278.0
FIT_MAX_NM = 1290.0

CI_FWHM_LIMIT = 1.0


# ============================================================
# Doppler functions
# ============================================================

def velocity_to_wavelength(
    rest_nm,
    velocity_kms,
):
    beta = (
        velocity_kms
        / C_KMS
    )

    return (
        rest_nm
        * np.sqrt(
            (1.0 + beta)
            / (1.0 - beta)
        )
    )


def wavelength_to_velocity(
    observed_nm,
    rest_nm,
):
    ratio = (
        observed_nm
        / rest_nm
    )

    beta = (
        ratio**2 - 1.0
    ) / (
        ratio**2 + 1.0
    )

    return (
        beta
        * C_KMS
    )


def gaussian(
    wavelength_nm,
    center_nm,
    sigma_nm,
):
    return np.exp(
        -0.5
        * (
            (
                wavelength_nm
                - center_nm
            )
            / sigma_nm
        ) ** 2
    )


def continuum(
    wavelength_nm,
    intercept,
    slope,
):
    return (
        intercept
        + slope
        * (
            wavelength_nm
            - OBSERVED_FEATURE_NM
        )
    )


def calculate_information_criteria(
    chi2,
    k,
    n,
):
    aic = (
        chi2
        + 2.0 * k
    )

    bic = (
        chi2
        + k * np.log(n)
    )

    return aic, bic


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

print(
    f"Spectral points: "
    f"{len(wavelength_nm)}"
)

print(
    f"Wavelength range: "
    f"{wavelength_nm.min():.6f}"
    f" - "
    f"{wavelength_nm.max():.6f} nm"
)


# ============================================================
# Select local fitting window
# ============================================================

mask = (
    np.isfinite(wavelength_nm)
    & np.isfinite(flux)
    & np.isfinite(flux_error)
    & (flux_error > 0)
    & (wavelength_nm >= FIT_MIN_NM)
    & (wavelength_nm <= FIT_MAX_NM)
)

wave = wavelength_nm[mask]
data = flux[mask]
error = flux_error[mask]

print()
print("=" * 70)
print("LOCAL SPECTRAL WINDOW")
print("=" * 70)

print(
    f"Window: "
    f"{FIT_MIN_NM:.3f} - "
    f"{FIT_MAX_NM:.3f} nm"
)

print(
    f"Usable points: "
    f"{len(wave)}"
)


# ============================================================
# Instrument resolution
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
    f"R = {RESOLVING_POWER:.1f}"
)

print(
    f"FWHM = "
    f"{instrument_fwhm_nm:.6f} nm"
)

print(
    f"sigma = "
    f"{instrument_sigma_nm:.6f} nm"
)


# ============================================================
# Load C I catalogue
# ============================================================

print()
print("=" * 70)
print("LOADING C I CATALOGUE")
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

ci["rest_nm"] = pd.to_numeric(
    ci["ritz_wavelength_vacuum_nm"],
    errors="coerce",
)

ci["Aki"] = pd.to_numeric(
    ci["Aki_s-1"],
    errors="coerce",
)

ci = ci[
    np.isfinite(
        ci["rest_nm"]
    )
].copy()

ci["reference_predicted_nm"] = (
    velocity_to_wavelength(
        ci["rest_nm"].to_numpy(),
        REFERENCE_VELOCITY_KMS,
    )
)

ci["distance_nm"] = np.abs(
    ci["reference_predicted_nm"]
    - OBSERVED_FEATURE_NM
)

ci["distance_fwhm"] = (
    ci["distance_nm"]
    / instrument_fwhm_nm
)

ci_blend = ci[
    ci["distance_fwhm"]
    <= CI_FWHM_LIMIT
].copy()

ci_blend = ci_blend[
    np.isfinite(
        ci_blend["Aki"]
    )
    & (
        ci_blend["Aki"] > 0
    )
].copy()

ci_blend = ci_blend.sort_values(
    "reference_predicted_nm"
).reset_index(
    drop=True
)

if len(ci_blend) == 0:
    raise RuntimeError(
        "No usable C I blend components found."
    )

print(
    f"Total catalogue rows: "
    f"{len(catalogue)}"
)

print(
    f"C I catalogue rows: "
    f"{len(ci)}"
)

print()
print(
    "C I components used:"
)

for _, row in ci_blend.iterrows():

    print(
        f"  "
        f"{row['rest_nm']:.6f}"
        f" -> "
        f"{row['reference_predicted_nm']:.6f}"
        f" nm"
        f" | Aki = "
        f"{row['Aki']:.3g}"
    )


# ============================================================
# C I Aki weights
# ============================================================

aki = (
    ci_blend["Aki"]
    .to_numpy(
        dtype=float
    )
)

aki_weights = (
    aki
    / np.max(aki)
)


# ============================================================
# Initial values
# ============================================================

continuum_guess = np.median(
    data
)

feature_scale = (
    np.percentile(
        data,
        90,
    )
    - np.percentile(
        data,
        10,
    )
)

if (
    not np.isfinite(
        feature_scale
    )
    or feature_scale <= 0
):
    feature_scale = 1.0


# ============================================================
# Model 1: Pa beta only
# ============================================================

def model_pa_beta(
    params,
    x,
):

    intercept = params[0]
    slope = params[1]
    amplitude = params[2]
    velocity = params[3]

    center = velocity_to_wavelength(
        PA_BETA_REST_NM,
        velocity,
    )

    return (
        continuum(
            x,
            intercept,
            slope,
        )
        + amplitude
        * gaussian(
            x,
            center,
            instrument_sigma_nm,
        )
    )


# ============================================================
# Model 2: Pa beta + Aki-weighted C I
# ============================================================

def model_pa_beta_ci_aki(
    params,
    x,
):

    intercept = params[0]
    slope = params[1]

    pa_amplitude = params[2]
    pa_velocity = params[3]

    ci_amplitude = params[4]
    ci_velocity = params[5]

    result = continuum(
        x,
        intercept,
        slope,
    )

    pa_center = velocity_to_wavelength(
        PA_BETA_REST_NM,
        pa_velocity,
    )

    result += (
        pa_amplitude
        * gaussian(
            x,
            pa_center,
            instrument_sigma_nm,
        )
    )

    ci_blend_model = np.zeros_like(
        x
    )

    for weight, rest in zip(
        aki_weights,
        ci_blend[
            "rest_nm"
        ].to_numpy(),
    ):

        center = velocity_to_wavelength(
            rest,
            ci_velocity,
        )

        ci_blend_model += (
            weight
            * gaussian(
                x,
                center,
                instrument_sigma_nm,
            )
        )

    result += (
        ci_amplitude
        * ci_blend_model
    )

    return result


# ============================================================
# Model 3: Pa beta + free C I blend
# ============================================================

def model_pa_beta_ci_free(
    params,
    x,
):

    intercept = params[0]
    slope = params[1]

    pa_amplitude = params[2]
    pa_velocity = params[3]

    ci_velocity = params[4]

    ci_amplitudes = params[5:]

    result = continuum(
        x,
        intercept,
        slope,
    )

    pa_center = velocity_to_wavelength(
        PA_BETA_REST_NM,
        pa_velocity,
    )

    result += (
        pa_amplitude
        * gaussian(
            x,
            pa_center,
            instrument_sigma_nm,
        )
    )

    for amplitude, rest in zip(
        ci_amplitudes,
        ci_blend[
            "rest_nm"
        ].to_numpy(),
    ):

        center = velocity_to_wavelength(
            rest,
            ci_velocity,
        )

        result += (
            amplitude
            * gaussian(
                x,
                center,
                instrument_sigma_nm,
            )
        )

    return result


# ============================================================
# Fit helper
# ============================================================

def fit_model(
    model_function,
    initial,
    lower,
    upper,
):

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
        max_nfev=50000,
    )

    model = model_function(
        result.x,
        wave,
    )

    residual = (
        model
        - data
    ) / error

    chi2 = np.sum(
        residual**2
    )

    k = len(
        result.x
    )

    n = len(
        data
    )

    aic, bic = (
        calculate_information_criteria(
            chi2,
            k,
            n,
        )
    )

    return {
        "result": result,
        "model": model,
        "chi2": chi2,
        "aic": aic,
        "bic": bic,
        "k": k,
        "success": result.success,
    }


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
    feature_scale,
    REFERENCE_VELOCITY_KMS,
]

pa_lower = [
    -np.inf,
    -np.inf,
    0.0,
    REFERENCE_VELOCITY_KMS - 250.0,
]

pa_upper = [
    np.inf,
    np.inf,
    np.inf,
    REFERENCE_VELOCITY_KMS + 250.0,
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
# Fit Pa beta + Aki-weighted C I
# ============================================================

print()
print("=" * 70)
print("FITTING PA BETA + AKI-WEIGHTED C I")
print("=" * 70)

joint_initial = [
    continuum_guess,
    0.0,
    feature_scale,
    REFERENCE_VELOCITY_KMS,
    feature_scale * 0.1,
    REFERENCE_VELOCITY_KMS,
]

joint_lower = [
    -np.inf,
    -np.inf,
    0.0,
    REFERENCE_VELOCITY_KMS - 250.0,
    0.0,
    REFERENCE_VELOCITY_KMS - 250.0,
]

joint_upper = [
    np.inf,
    np.inf,
    np.inf,
    REFERENCE_VELOCITY_KMS + 250.0,
    np.inf,
    REFERENCE_VELOCITY_KMS + 250.0,
]

joint_aki_fit = fit_model(
    model_pa_beta_ci_aki,
    joint_initial,
    joint_lower,
    joint_upper,
)

print(
    f"Pa beta velocity: "
    f"{joint_aki_fit['result'].x[3]:+.3f} km/s"
)

print(
    f"C I velocity: "
    f"{joint_aki_fit['result'].x[5]:+.3f} km/s"
)

print(
    f"C I amplitude: "
    f"{joint_aki_fit['result'].x[4]:.6g}"
)

print(
    f"Chi²: "
    f"{joint_aki_fit['chi2']:.3f}"
)

print(
    f"AIC: "
    f"{joint_aki_fit['aic']:.3f}"
)

print(
    f"BIC: "
    f"{joint_aki_fit['bic']:.3f}"
)


# ============================================================
# Fit Pa beta + free-amplitude C I
# ============================================================

print()
print("=" * 70)
print("FITTING PA BETA + FREE-AMPLITUDE C I")
print("=" * 70)

n_ci = len(
    ci_blend
)

free_initial = [
    continuum_guess,
    0.0,
    feature_scale,
    REFERENCE_VELOCITY_KMS,
    REFERENCE_VELOCITY_KMS,
] + [
    feature_scale
    / max(
        n_ci,
        1,
    )
] * n_ci

free_lower = [
    -np.inf,
    -np.inf,
    0.0,
    REFERENCE_VELOCITY_KMS - 250.0,
    REFERENCE_VELOCITY_KMS - 250.0,
] + [
    0.0
] * n_ci

free_upper = [
    np.inf,
    np.inf,
    np.inf,
    REFERENCE_VELOCITY_KMS + 250.0,
    REFERENCE_VELOCITY_KMS + 250.0,
] + [
    np.inf
] * n_ci

joint_free_fit = fit_model(
    model_pa_beta_ci_free,
    free_initial,
    free_lower,
    free_upper,
)

print(
    f"Pa beta velocity: "
    f"{joint_free_fit['result'].x[3]:+.3f} km/s"
)

print(
    f"C I velocity: "
    f"{joint_free_fit['result'].x[4]:+.3f} km/s"
)

print(
    f"Chi²: "
    f"{joint_free_fit['chi2']:.3f}"
)

print(
    f"AIC: "
    f"{joint_free_fit['aic']:.3f}"
)

print(
    f"BIC: "
    f"{joint_free_fit['bic']:.3f}"
)


# ============================================================
# Compare models
# ============================================================

fits = {
    "Pa beta": pa_fit,
    "Pa beta + C I Aki": joint_aki_fit,
    "Pa beta + C I free": joint_free_fit,
}

print()
print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print()
print(
    f"{'Model':<28}"
    f"{'Chi²':>14}"
    f"{'ΔChi²':>14}"
    f"{'AIC':>14}"
    f"{'BIC':>14}"
)

print("-" * 84)

reference_chi2 = (
    pa_fit["chi2"]
)

for name, fit in fits.items():

    delta = (
        fit["chi2"]
        - reference_chi2
    )

    print(
        f"{name:<28}"
        f"{fit['chi2']:>14.3f}"
        f"{delta:>14.3f}"
        f"{fit['aic']:>14.3f}"
        f"{fit['bic']:>14.3f}"
    )


# ============================================================
# Determine preferred model
# ============================================================

best_bic_name = min(
    fits,
    key=lambda name:
        fits[name]["bic"],
)

best_aic_name = min(
    fits,
    key=lambda name:
        fits[name]["aic"],
)

print()
print("=" * 70)
print("MODEL SELECTION")
print("=" * 70)

print(
    f"Best by BIC: "
    f"{best_bic_name}"
)

print(
    f"Best by AIC: "
    f"{best_aic_name}"
)


# ============================================================
# Save summary
# ============================================================

summary_rows = []

for name, fit in fits.items():

    params = fit[
        "result"
    ].x

    if name == "Pa beta":

        pa_velocity = params[3]
        ci_velocity = np.nan

    elif name == "Pa beta + C I Aki":

        pa_velocity = params[3]
        ci_velocity = params[5]

    else:

        pa_velocity = params[3]
        ci_velocity = params[4]

    summary_rows.append(
        {
            "model": name,
            "chi2": fit["chi2"],
            "delta_chi2_vs_pa_beta":
                fit["chi2"]
                - reference_chi2,
            "AIC": fit["aic"],
            "BIC": fit["bic"],
            "n_parameters": fit["k"],
            "pa_beta_velocity_kms":
                pa_velocity,
            "ci_velocity_kms":
                ci_velocity,
        }
    )

summary = pd.DataFrame(
    summary_rows
)

summary.to_csv(
    RESULTS_PATH,
    index=False,
)


# ============================================================
# Save C I component information
# ============================================================

component_rows = []

aki_params = (
    joint_aki_fit["result"].x
)

free_params = (
    joint_free_fit["result"].x
)

for i, (_, row) in enumerate(
    ci_blend.iterrows()
):

    rest = row["rest_nm"]

    component_rows.append(
        {
            "species": "C I",
            "rest_wavelength_vacuum_nm":
                rest,
            "Aki_s-1":
                row["Aki"],
            "reference_predicted_wavelength_nm":
                row[
                    "reference_predicted_nm"
                ],
            "Aki_weight":
                aki_weights[i],
            "Aki_model_velocity_kms":
                aki_params[5],
            "Aki_model_center_nm":
                velocity_to_wavelength(
                    rest,
                    aki_params[5],
                ),
            "Aki_model_amplitude":
                aki_params[4],
            "free_model_velocity_kms":
                free_params[4],
            "free_model_center_nm":
                velocity_to_wavelength(
                    rest,
                    free_params[4],
                ),
            "free_model_amplitude":
                free_params[5 + i],
        }
    )

components = pd.DataFrame(
    component_rows
)

components.to_csv(
    COMPONENTS_PATH,
    index=False,
)


# ============================================================
# Plot
# ============================================================

print()
print("=" * 70)
print("CREATING COMPARISON PLOT")
print("=" * 70)

fine_wave = np.linspace(
    FIT_MIN_NM,
    FIT_MAX_NM,
    3000,
)

pa_curve = model_pa_beta(
    pa_fit["result"].x,
    fine_wave,
)

aki_curve = model_pa_beta_ci_aki(
    joint_aki_fit["result"].x,
    fine_wave,
)

free_curve = model_pa_beta_ci_free(
    joint_free_fit["result"].x,
    fine_wave,
)

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
    aki_curve,
    linewidth=2,
    label="Pa beta + C I Aki",
)

ax.plot(
    fine_wave,
    free_curve,
    linewidth=2,
    label="Pa beta + C I free",
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
    "M51 1284 nm: Pa beta + C I Blend Test"
)

ax.legend()

ax.grid(
    alpha=0.25
)

fig.tight_layout()

fig.savefig(
    PLOT_PATH,
    dpi=200,
)

plt.close(fig)


# ============================================================
# Residual plot
# ============================================================

pa_residual = (
    data
    - pa_fit["model"]
) / error

aki_residual = (
    data
    - joint_aki_fit["model"]
) / error

free_residual = (
    data
    - joint_free_fit["model"]
) / error

residual_plot = Path(
    "m51_1284_pabeta_ci_residuals.png"
)

fig, ax = plt.subplots(
    figsize=(12, 7)
)

ax.axhline(
    0.0,
    linestyle="--",
    linewidth=1,
)

ax.plot(
    wave,
    pa_residual,
    "o-",
    label="Pa beta residual",
)

ax.plot(
    wave,
    aki_residual,
    "o-",
    label="Pa beta + C I Aki residual",
)

ax.plot(
    wave,
    free_residual,
    "o-",
    label="Pa beta + C I free residual",
)

ax.set_xlabel(
    "Wavelength (nm)"
)

ax.set_ylabel(
    "Normalized residual"
)

ax.set_title(
    "M51 1284 nm Model Residuals"
)

ax.legend()

ax.grid(
    alpha=0.25
)

fig.tight_layout()

fig.savefig(
    residual_plot,
    dpi=200,
)

plt.close(fig)


# ============================================================
# Final report
# ============================================================

print()
print("=" * 70)
print("FINAL INTERPRETATION")
print("=" * 70)

print()
print(
    "This experiment tests whether adding C I "
    "provides statistically useful information "
    "beyond the Pa beta model."
)

print()
print(
    "Model 1:"
)

print(
    "  continuum + Pa beta"
)

print()
print(
    "Model 2:"
)

print(
    "  continuum + Pa beta + "
    "Aki-weighted C I blend"
)

print()
print(
    "Model 3:"
)

print(
    "  continuum + Pa beta + "
    "free-amplitude C I blend"
)

print()
print(
    "The C I components share a common velocity "
    "within each blend model."
)

print()
print(
    "Aki weighting is exploratory and does not "
    "constitute a full excitation/population model."
)

print()
print(
    "The free-amplitude model is intentionally "
    "more flexible and is therefore penalized "
    "more strongly by AIC/BIC."
)

print()
print(
    f"Best model by BIC: "
    f"{best_bic_name}"
)

print(
    f"Best model by AIC: "
    f"{best_aic_name}"
)

print()
print(
    "Outputs:"
)

print(
    f"  {RESULTS_PATH}"
)

print(
    f"  {COMPONENTS_PATH}"
)

print(
    f"  {PLOT_PATH}"
)

print(
    f"  {residual_plot}"
)

print()
print("=" * 70)
print(
    "PA BETA + C I BLEND TEST COMPLETE"
)
print("=" * 70)
