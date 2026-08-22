from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# M51 HYDROGEN CASE-B FLUX-RATIO TEST
# Pa-beta / Pa-gamma
# ============================================================

print("=" * 70)
print("M51 HYDROGEN CASE-B PA-BETA / PA-GAMMA FLUX-RATIO TEST")
print("CASE-B RECOMBINATION + EXTINCTION CONSISTENCY")
print("=" * 70)


# ============================================================
# Configuration
# ============================================================

PROFILE_RESULTS = Path(
    "data/atomic_lines/"
    "m51_hydrogen_pabeta_pagamma_profiles.csv"
)

OUTPUT_DIR = Path("data/atomic_lines")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_CSV = (
    OUTPUT_DIR /
    "m51_hydrogen_caseB_flux_ratio_results.csv"
)

GRID_CSV = (
    OUTPUT_DIR /
    "m51_hydrogen_caseB_extinction_grid.csv"
)

PLOT_PATH = Path(
    "m51_hydrogen_caseB_flux_ratio.png"
)


# ============================================================
# Hydrogen wavelengths
# ============================================================

PA_BETA_NM = 1281.807000
PA_GAMMA_NM = 1093.800000


# ============================================================
# Representative Case-B reference
#
# Hummer & Storey / Storey & Hummer:
#
# Te = 10,000 K
# ne = 10,000 cm^-3
#
# Relative to H-beta:
#
# Pa-beta  = 0.162
# Pa-gamma = 0.0901
#
# Therefore:
#
# Pa-beta / Pa-gamma = 1.798...
# ============================================================

REFERENCE_TE_K = 10000.0
REFERENCE_NE_CM3 = 1.0e4

CASEB_PA_BETA_REL_HBETA = 0.162
CASEB_PA_GAMMA_REL_HBETA = 0.0901

CASEB_REFERENCE_RATIO = (
    CASEB_PA_BETA_REL_HBETA
    / CASEB_PA_GAMMA_REL_HBETA
)


# ============================================================
# Extinction-law configuration
#
# We use a simple near-IR power-law:
#
# A(lambda) ∝ lambda^-alpha
#
# This is intentionally a diagnostic model rather than
# claiming a unique extinction law for M51.
# ============================================================

EXTINCTION_ALPHA = 1.7

AV_GRID = np.linspace(
    0.0,
    10.0,
    1001,
)


# ============================================================
# Helper functions
# ============================================================

def load_profile_results():
    if not PROFILE_RESULTS.exists():
        raise FileNotFoundError(
            f"Could not find:\n{PROFILE_RESULTS}\n\n"
            "Run test_m51_hydrogen_pabeta_pagamma_profiles.py "
            "first."
        )

    df = pd.read_csv(
        PROFILE_RESULTS
    )

    return df


def get_line_row(df, line_name):
    rows = df[
        df["line"].astype(str).str.lower()
        == line_name.lower()
    ]

    if len(rows) != 1:
        raise RuntimeError(
            f"Expected exactly one {line_name} row; "
            f"found {len(rows)}."
        )

    return rows.iloc[0]


def ccm_like_nir_relative_extinction(
    wavelength_nm,
    alpha=EXTINCTION_ALPHA,
):
    """
    Simple relative near-IR extinction law.

    A(lambda) / A(V) ∝ lambda^-alpha

    The normalization is chosen so that the function is
    approximately representative of the near-IR decline.

    For the ratio test we only need the differential
    extinction between Pa-beta and Pa-gamma.
    """

    wavelength_um = wavelength_nm / 1000.0

    # Representative normalization:
    # A(lambda)/A(V) ≈ 0.55 * lambda^-alpha
    return (
        0.55
        * wavelength_um ** (-alpha)
    )


def extincted_ratio(
    intrinsic_ratio,
    av,
    alpha=EXTINCTION_ALPHA,
):
    """
    Observed ratio:

        F_beta / F_gamma
        =
        intrinsic_ratio *
        10^[ -0.4(A_beta - A_gamma) ]

    Because Pa-gamma is at the shorter wavelength,
    A_gamma > A_beta, so the observed beta/gamma
    ratio increases with extinction.
    """

    a_beta = (
        av
        * ccm_like_nir_relative_extinction(
            PA_BETA_NM,
            alpha,
        )
    )

    a_gamma = (
        av
        * ccm_like_nir_relative_extinction(
            PA_GAMMA_NM,
            alpha,
        )
    )

    return (
        intrinsic_ratio
        * 10.0 ** (
            -0.4 * (a_beta - a_gamma)
        )
    )


def required_av(
    observed_ratio,
    intrinsic_ratio,
    alpha=EXTINCTION_ALPHA,
):
    """
    Solve analytically for A_V.
    """

    if (
        not np.isfinite(observed_ratio)
        or observed_ratio <= 0
        or not np.isfinite(intrinsic_ratio)
        or intrinsic_ratio <= 0
    ):
        return np.nan

    a_beta_per_av = (
        ccm_like_nir_relative_extinction(
            PA_BETA_NM,
            alpha,
        )
    )

    a_gamma_per_av = (
        ccm_like_nir_relative_extinction(
            PA_GAMMA_NM,
            alpha,
        )
    )

    delta_a_per_av = (
        a_beta_per_av
        - a_gamma_per_av
    )

    if delta_a_per_av == 0:
        return np.nan

    return (
        -2.5
        * np.log10(
            observed_ratio
            / intrinsic_ratio
        )
        / delta_a_per_av
    )


# ============================================================
# Load measured profile results
# ============================================================

print()
print("=" * 70)
print("1. LOADING MEASURED HYDROGEN PROFILE RESULTS")
print("=" * 70)

print()
print(
    f"File:\n  {PROFILE_RESULTS}"
)

df = load_profile_results()

beta = get_line_row(
    df,
    "Pa-beta",
)

gamma = get_line_row(
    df,
    "Pa-gamma",
)

beta_flux = float(
    beta["integrated_flux"]
)

gamma_flux = float(
    gamma["integrated_flux"]
)

print()
print(
    f"Pa-beta fitted flux: "
    f"{beta_flux:.8e}"
)

print(
    f"Pa-gamma fitted flux: "
    f"{gamma_flux:.8e}"
)


# ============================================================
# Observed ratio
# ============================================================

observed_ratio = (
    beta_flux / gamma_flux
)

print()
print("=" * 70)
print("2. OBSERVED PA-BETA / PA-GAMMA RATIO")
print("=" * 70)

print()
print(
    f"Observed Pa-beta / Pa-gamma ratio: "
    f"{observed_ratio:.6f}"
)


# ============================================================
# Approximate flux uncertainty
#
# The profile-fitting script does not currently propagate
# covariance matrices. Therefore we explicitly label this
# as an approximate diagnostic.
# ============================================================

beta_amp_snr = float(
    beta["amplitude_snr"]
)

gamma_amp_snr = float(
    gamma["amplitude_snr"]
)

beta_relative_uncertainty = (
    1.0 / beta_amp_snr
    if beta_amp_snr > 0
    else np.nan
)

gamma_relative_uncertainty = (
    1.0 / gamma_amp_snr
    if gamma_amp_snr > 0
    else np.nan
)

ratio_relative_uncertainty = np.sqrt(
    beta_relative_uncertainty ** 2
    + gamma_relative_uncertainty ** 2
)

ratio_uncertainty = (
    observed_ratio
    * ratio_relative_uncertainty
)

print()
print(
    "Approximate ratio uncertainty:"
)

print(
    f"  Pa-beta amplitude S/N: "
    f"{beta_amp_snr:.3f}"
)

print(
    f"  Pa-gamma amplitude S/N: "
    f"{gamma_amp_snr:.3f}"
)

print(
    f"  Approximate ratio = "
    f"{observed_ratio:.4f} "
    f"+/- {ratio_uncertainty:.4f}"
)

print()
print(
    "NOTE:"
)

print(
    "This uncertainty is approximate because the profile "
    "fit does not yet provide the full parameter covariance."
)


# ============================================================
# Case-B reference
# ============================================================

print()
print("=" * 70)
print("3. CASE-B REFERENCE")
print("=" * 70)

print()
print(
    f"Representative electron temperature: "
    f"{REFERENCE_TE_K:.0f} K"
)

print(
    f"Representative electron density: "
    f"{REFERENCE_NE_CM3:.2e} cm^-3"
)

print()
print(
    f"Pa-beta / H-beta: "
    f"{CASEB_PA_BETA_REL_HBETA:.4f}"
)

print(
    f"Pa-gamma / H-beta: "
    f"{CASEB_PA_GAMMA_REL_HBETA:.4f}"
)

print()
print(
    f"Intrinsic Case-B Pa-beta / Pa-gamma: "
    f"{CASEB_REFERENCE_RATIO:.6f}"
)


# ============================================================
# Ratio comparison without extinction
# ============================================================

ratio_excess_factor = (
    observed_ratio
    / CASEB_REFERENCE_RATIO
)

print()
print("=" * 70)
print("4. CASE-B VS OBSERVED RATIO")
print("=" * 70)

print()
print(
    f"Observed ratio: "
    f"{observed_ratio:.6f}"
)

print(
    f"Intrinsic Case-B ratio: "
    f"{CASEB_REFERENCE_RATIO:.6f}"
)

print(
    f"Observed / Case-B ratio: "
    f"{ratio_excess_factor:.6f}"
)

if observed_ratio > CASEB_REFERENCE_RATIO:
    print()
    print(
        "The observed ratio is larger than the "
        "representative unreddened Case-B ratio."
    )

else:
    print()
    print(
        "The observed ratio is not larger than the "
        "representative unreddened Case-B ratio."
    )


# ============================================================
# Extinction calculation
# ============================================================

print()
print("=" * 70)
print("5. DIFFERENTIAL EXTINCTION TEST")
print("=" * 70)

a_beta_per_av = (
    ccm_like_nir_relative_extinction(
        PA_BETA_NM
    )
)

a_gamma_per_av = (
    ccm_like_nir_relative_extinction(
        PA_GAMMA_NM
    )
)

delta_a_per_av = (
    a_beta_per_av
    - a_gamma_per_av
)

print()
print(
    f"A(Pa-beta) / A(V): "
    f"{a_beta_per_av:.6f}"
)

print(
    f"A(Pa-gamma) / A(V): "
    f"{a_gamma_per_av:.6f}"
)

print(
    f"Delta A(beta-gamma) / A(V): "
    f"{delta_a_per_av:.6f}"
)

av_required = required_av(
    observed_ratio,
    CASEB_REFERENCE_RATIO,
)

print()

if np.isfinite(av_required):

    print(
        f"Required A(V): "
        f"{av_required:.3f} mag"
    )

    a_beta_required = (
        av_required
        * a_beta_per_av
    )

    a_gamma_required = (
        av_required
        * a_gamma_per_av
    )

    print(
        f"Required A(Pa-beta): "
        f"{a_beta_required:.3f} mag"
    )

    print(
        f"Required A(Pa-gamma): "
        f"{a_gamma_required:.3f} mag"
    )

    print(
        f"Required differential extinction: "
        f"{a_beta_required - a_gamma_required:.3f} mag"
    )

else:

    print(
        "Could not determine a finite extinction solution."
    )

    a_beta_required = np.nan
    a_gamma_required = np.nan


# ============================================================
# Generate extinction grid
# ============================================================

print()
print("=" * 70)
print("6. CASE-B EXTINCTION GRID")
print("=" * 70)

predicted_ratios = np.array(
    [
        extincted_ratio(
            CASEB_REFERENCE_RATIO,
            av,
        )
        for av in AV_GRID
    ]
)

grid_df = pd.DataFrame(
    {
        "A_V_mag": AV_GRID,
        "caseB_predicted_ratio": predicted_ratios,
        "observed_ratio": observed_ratio,
    }
)

grid_df.to_csv(
    GRID_CSV,
    index=False,
)

print()
print(
    f"Grid saved to:\n  {GRID_CSV}"
)


# ============================================================
# Residual ratio at required extinction
# ============================================================

if np.isfinite(av_required):

    ratio_at_required_av = (
        extincted_ratio(
            CASEB_REFERENCE_RATIO,
            av_required,
        )
    )

    ratio_difference = (
        observed_ratio
        - ratio_at_required_av
    )

else:

    ratio_at_required_av = np.nan
    ratio_difference = np.nan


# ============================================================
# Physical interpretation flags
# ============================================================

print()
print("=" * 70)
print("7. PHYSICAL CONSISTENCY DIAGNOSTICS")
print("=" * 70)

print()
print(
    "Case-B reference temperature and density are only "
    "one point in the Storey-Hummer parameter space."
)

print(
    "The extinction calculation is therefore a diagnostic "
    "rather than a unique physical solution."
)

if np.isfinite(av_required):

    if 0.0 <= av_required <= 10.0:

        extinction_plausible = True

        print()
        print(
            "The representative Case-B ratio can be "
            "transformed into the observed ratio with "
            f"A(V) = {av_required:.3f} mag "
            "under the adopted extinction law."
        )

    else:

        extinction_plausible = False

        print()
        print(
            "The required extinction lies outside the "
            "adopted 0-10 mag diagnostic range."
        )

else:

    extinction_plausible = False


# ============================================================
# Create summary
# ============================================================

summary = {
    "observed_pa_beta_flux": beta_flux,
    "observed_pa_gamma_flux": gamma_flux,
    "observed_pa_beta_pa_gamma_ratio": observed_ratio,
    "approx_ratio_uncertainty": ratio_uncertainty,
    "caseB_reference_Te_K": REFERENCE_TE_K,
    "caseB_reference_ne_cm3": REFERENCE_NE_CM3,
    "caseB_pa_beta_relative_Hbeta":
        CASEB_PA_BETA_REL_HBETA,
    "caseB_pa_gamma_relative_Hbeta":
        CASEB_PA_GAMMA_REL_HBETA,
    "caseB_intrinsic_ratio":
        CASEB_REFERENCE_RATIO,
    "observed_to_caseB_ratio":
        ratio_excess_factor,
    "extinction_alpha":
        EXTINCTION_ALPHA,
    "A_beta_per_Av":
        a_beta_per_av,
    "A_gamma_per_Av":
        a_gamma_per_av,
    "delta_A_beta_gamma_per_Av":
        delta_a_per_av,
    "required_Av_mag":
        av_required,
    "required_A_beta_mag":
        a_beta_required,
    "required_A_gamma_mag":
        a_gamma_required,
    "ratio_at_required_Av":
        ratio_at_required_av,
    "ratio_difference":
        ratio_difference,
    "extinction_solution_in_0_10mag":
        extinction_plausible,
}

summary_df = pd.DataFrame(
    [summary]
)

summary_df.to_csv(
    RESULTS_CSV,
    index=False,
)

print()
print(
    f"Summary saved to:\n  {RESULTS_CSV}"
)


# ============================================================
# Plot
# ============================================================

print()
print("=" * 70)
print("8. CREATING CASE-B / EXTINCTION PLOT")
print("=" * 70)

fig, ax = plt.subplots(
    figsize=(10, 6)
)

ax.plot(
    AV_GRID,
    predicted_ratios,
    linewidth=2,
    label=(
        "Case-B prediction + extinction"
    ),
)

ax.axhline(
    observed_ratio,
    linestyle="--",
    linewidth=1.8,
    label=(
        f"Observed ratio = "
        f"{observed_ratio:.3f}"
    ),
)

ax.fill_between(
    AV_GRID,
    observed_ratio - ratio_uncertainty,
    observed_ratio + ratio_uncertainty,
    alpha=0.2,
    label="Approx. observed uncertainty",
)

if np.isfinite(av_required):

    ax.axvline(
        av_required,
        linestyle=":",
        linewidth=1.8,
        label=(
            f"Required A(V) = "
            f"{av_required:.2f} mag"
        ),
    )

ax.set_xlabel(
    "A(V) [mag]"
)

ax.set_ylabel(
    "Pa-beta / Pa-gamma"
)

ax.set_title(
    "M51 Hydrogen Case-B Flux-Ratio Test"
)

ax.grid(
    alpha=0.25
)

ax.legend()

fig.tight_layout()

fig.savefig(
    PLOT_PATH,
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
    "This experiment compares the measured Pa-beta / Pa-gamma "
    "flux ratio with a representative Case-B hydrogen "
    "recombination ratio."
)

print()
print(
    "The reference Case-B calculation uses:"
)

print(
    f"  Te = {REFERENCE_TE_K:.0f} K"
)

print(
    f"  ne = {REFERENCE_NE_CM3:.2e} cm^-3"
)

print()
print(
    f"Representative intrinsic ratio: "
    f"{CASEB_REFERENCE_RATIO:.3f}"
)

print(
    f"Measured ratio: "
    f"{observed_ratio:.3f}"
)

print(
    f"Approximate uncertainty: "
    f"+/- {ratio_uncertainty:.3f}"
)

print()

if np.isfinite(av_required):

    print(
        f"Under the adopted near-IR extinction law, "
        f"the representative Case-B ratio requires "
        f"A(V) ≈ {av_required:.2f} mag "
        f"to reproduce the measured ratio."
    )

else:

    print(
        "No finite extinction solution was obtained."
    )

print()
print(
    "IMPORTANT:"
)

print(
    "This is not a definitive Case-B test."
)

print(
    "Case-B line ratios depend on electron temperature, "
    "electron density, and the validity of the Case-B "
    "assumptions. Hummer & Storey explicitly show that "
    "collisional processes can invalidate simple Case-B "
    "behavior in some regimes."
)

print(
    "The Storey-Hummer calculations cover a substantial "
    "temperature and density grid, so the next-generation "
    "version of this experiment should eventually evaluate "
    "the full grid rather than relying on one reference point."
)

print()
print(
    "Likewise, the observed ratio is based on the current "
    "Gaussian profile fits and does not yet include a full "
    "flux-calibration, covariance, aperture, or extinction "
    "uncertainty analysis."
)

print()
print(
    "Therefore the appropriate conclusion at this stage is "
    "whether the observed ratio is compatible with a plausible "
    "hydrogen recombination scenario, not whether Case B has "
    "been proven."
)

print()
print("Outputs:")
print(f"  {RESULTS_CSV}")
print(f"  {GRID_CSV}")
print(f"  {PLOT_PATH}")

print()
print("=" * 70)
print("HYDROGEN CASE-B FLUX-RATIO TEST COMPLETE")
print("=" * 70)
