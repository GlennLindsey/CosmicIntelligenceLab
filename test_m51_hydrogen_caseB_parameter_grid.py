from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# M51 HYDROGEN CASE-B PARAMETER-GRID TEST
# ============================================================

print("=" * 70)
print("M51 HYDROGEN CASE-B PA-BETA / PA-GAMMA PARAMETER GRID")
print("TEMPERATURE + DENSITY + EXTINCTION CONSISTENCY TEST")
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
    "m51_hydrogen_caseB_parameter_grid_results.csv"
)

GRID_CSV = (
    OUTPUT_DIR /
    "m51_hydrogen_caseB_parameter_grid.csv"
)

PLOT_PATH = Path(
    "m51_hydrogen_caseB_parameter_grid.png"
)


# ============================================================
# Hydrogen wavelengths
# ============================================================

PA_BETA_NM = 1281.807000
PA_GAMMA_NM = 1093.800000


# ============================================================
# Representative Case-B reference
#
# Used as the starting point until the actual
# Storey-Hummer machine-readable emissivity grid
# is incorporated.
# ============================================================

REFERENCE_TE_K = 10000.0
REFERENCE_NE_CM3 = 1.0e4

REFERENCE_PA_BETA_HBETA = 0.1620
REFERENCE_PA_GAMMA_HBETA = 0.0901

REFERENCE_RATIO = (
    REFERENCE_PA_BETA_HBETA
    / REFERENCE_PA_GAMMA_HBETA
)


# ============================================================
# Approximate temperature dependence
#
# This is deliberately a diagnostic interpolation model,
# NOT a replacement for the Storey-Hummer calculations.
#
# The scaling is weak because near-IR hydrogen recombination
# ratios vary much less strongly with Te than many other
# quantities.
# ============================================================

def approximate_caseB_ratio(
    te_k,
    ne_cm3,
):
    """
    Approximate Case-B Pa-beta / Pa-gamma ratio.

    This intentionally provides only a smooth diagnostic
    dependence around the reference value.

    The next-generation version should replace this function
    with direct Storey-Hummer emissivities.
    """

    log_te_ratio = np.log10(
        te_k / REFERENCE_TE_K
    )

    log_ne_ratio = np.log10(
        ne_cm3 / REFERENCE_NE_CM3
    )

    # Weak diagnostic dependence.
    #
    # These coefficients are NOT atomic calculations.
    # They merely allow us to explore parameter sensitivity
    # before importing the full Storey-Hummer grid.

    ratio = (
        REFERENCE_RATIO
        * (
            1.0
            - 0.025 * log_te_ratio
            + 0.008 * log_ne_ratio
        )
    )

    return ratio


# ============================================================
# Near-IR extinction law
# ============================================================

EXTINCTION_ALPHA = 1.7


def relative_extinction(
    wavelength_nm,
):
    """
    Approximate near-IR extinction:

        A(lambda)/A(V)
        = 0.55 * lambda^-alpha

    wavelength is supplied in microns internally.
    """

    wavelength_um = (
        wavelength_nm / 1000.0
    )

    return (
        0.55
        * wavelength_um ** (-EXTINCTION_ALPHA)
    )


A_BETA_PER_AV = relative_extinction(
    PA_BETA_NM
)

A_GAMMA_PER_AV = relative_extinction(
    PA_GAMMA_NM
)

DELTA_A_PER_AV = (
    A_BETA_PER_AV
    - A_GAMMA_PER_AV
)


def extincted_ratio(
    intrinsic_ratio,
    av,
):
    """
    Apply differential extinction.
    """

    return (
        intrinsic_ratio
        * 10.0 ** (
            -0.4
            * (
                av
                * DELTA_A_PER_AV
            )
        )
    )


# ============================================================
# Load measured profile results
# ============================================================

print()
print("=" * 70)
print("1. LOADING OBSERVED HYDROGEN FLUXES")
print("=" * 70)

if not PROFILE_RESULTS.exists():

    raise FileNotFoundError(
        f"\nMissing:\n{PROFILE_RESULTS}\n\n"
        "Run:\n"
        "python "
        "test_m51_hydrogen_pabeta_pagamma_profiles.py\n"
        "first."
    )


profile_df = pd.read_csv(
    PROFILE_RESULTS
)


def get_line_row(
    dataframe,
    name,
):
    rows = dataframe[
        dataframe["line"]
        .astype(str)
        .str.lower()
        == name.lower()
    ]

    if len(rows) != 1:

        raise RuntimeError(
            f"Expected one {name} row; "
            f"found {len(rows)}."
        )

    return rows.iloc[0]


beta = get_line_row(
    profile_df,
    "Pa-beta",
)

gamma = get_line_row(
    profile_df,
    "Pa-gamma",
)


beta_flux = float(
    beta["integrated_flux"]
)

gamma_flux = float(
    gamma["integrated_flux"]
)


observed_ratio = (
    beta_flux
    / gamma_flux
)


beta_snr = float(
    beta["amplitude_snr"]
)

gamma_snr = float(
    gamma["amplitude_snr"]
)


beta_rel_err = (
    1.0 / beta_snr
)

gamma_rel_err = (
    1.0 / gamma_snr
)

ratio_rel_err = np.sqrt(
    beta_rel_err ** 2
    + gamma_rel_err ** 2
)

ratio_uncertainty = (
    observed_ratio
    * ratio_rel_err
)


print()
print(
    f"Pa-beta flux: "
    f"{beta_flux:.8e}"
)

print(
    f"Pa-gamma flux: "
    f"{gamma_flux:.8e}"
)

print()
print(
    f"Observed ratio: "
    f"{observed_ratio:.6f}"
)

print(
    f"Approximate uncertainty: "
    f"+/- {ratio_uncertainty:.6f}"
)


# ============================================================
# Parameter ranges
# ============================================================

print()
print("=" * 70)
print("2. PARAMETER GRID")
print("=" * 70)


# Temperature:
# broad nebular / ionized-gas diagnostic range

TE_VALUES = np.array(
    [
        5000,
        6000,
        7000,
        8000,
        9000,
        10000,
        12000,
        15000,
        20000,
        30000,
        50000,
    ],
    dtype=float,
)


# Density:
# log-spaced representative range

LOG_NE_VALUES = np.arange(
    2.0,
    14.1,
    1.0,
)

NE_VALUES = (
    10.0 ** LOG_NE_VALUES
)


# Extinction

AV_VALUES = np.linspace(
    0.0,
    10.0,
    201,
)


print()
print(
    "Electron temperature grid:"
)

print(
    "  "
    + ", ".join(
        f"{x:.0f}"
        for x in TE_VALUES
    )
    + " K"
)

print()
print(
    "Electron density grid:"
)

print(
    "  log10(ne/cm^-3) = "
    f"{LOG_NE_VALUES[0]:.0f}"
    " ... "
    f"{LOG_NE_VALUES[-1]:.0f}"
)

print()
print(
    f"A(V): "
    f"{AV_VALUES.min():.1f}"
    " - "
    f"{AV_VALUES.max():.1f}"
    " mag"
)


# ============================================================
# Build parameter grid
# ============================================================

print()
print("=" * 70)
print("3. BUILDING CASE-B PARAMETER GRID")
print("=" * 70)


records = []


for te in TE_VALUES:

    for ne in NE_VALUES:

        intrinsic_ratio = (
            approximate_caseB_ratio(
                te,
                ne,
            )
        )

        for av in AV_VALUES:

            predicted_ratio = (
                extincted_ratio(
                    intrinsic_ratio,
                    av,
                )
            )

            residual = (
                predicted_ratio
                - observed_ratio
            )

            sigma_distance = (
                residual
                / ratio_uncertainty
            )

            records.append(
                {
                    "Te_K": te,
                    "ne_cm3": ne,
                    "log10_ne": np.log10(ne),
                    "A_V_mag": av,
                    "intrinsic_caseB_ratio":
                        intrinsic_ratio,
                    "predicted_observed_ratio":
                        predicted_ratio,
                    "observed_ratio":
                        observed_ratio,
                    "ratio_uncertainty":
                        ratio_uncertainty,
                    "ratio_residual":
                        residual,
                    "sigma_distance":
                        sigma_distance,
                    "abs_sigma_distance":
                        abs(sigma_distance),
                }
            )


grid_df = pd.DataFrame(
    records
)


print()
print(
    f"Grid points: "
    f"{len(grid_df):,}"
)


# ============================================================
# Identify best-fitting parameter combinations
# ============================================================

print()
print("=" * 70)
print("4. BEST CASE-B + EXTINCTION SOLUTIONS")
print("=" * 70)


grid_sorted = (
    grid_df
    .sort_values(
        "abs_sigma_distance"
    )
    .reset_index(
        drop=True
    )
)


best = grid_sorted.iloc[0]


print()
print(
    "Best diagnostic solution:"
)

print(
    f"  Te = "
    f"{best['Te_K']:.0f} K"
)

print(
    f"  ne = "
    f"{best['ne_cm3']:.3e} cm^-3"
)

print(
    f"  log10(ne) = "
    f"{best['log10_ne']:.2f}"
)

print(
    f"  A(V) = "
    f"{best['A_V_mag']:.3f} mag"
)

print(
    f"  Intrinsic ratio = "
    f"{best['intrinsic_caseB_ratio']:.6f}"
)

print(
    f"  Predicted observed ratio = "
    f"{best['predicted_observed_ratio']:.6f}"
)

print(
    f"  Observed ratio = "
    f"{observed_ratio:.6f}"
)

print(
    f"  Difference = "
    f"{best['ratio_residual']:.6f}"
)

print(
    f"  Difference / sigma = "
    f"{best['sigma_distance']:.3f}"
)


# ============================================================
# Solutions within 1 sigma
# ============================================================

within_1sigma = grid_df[
    grid_df["abs_sigma_distance"]
    <= 1.0
]

within_2sigma = grid_df[
    grid_df["abs_sigma_distance"]
    <= 2.0
]

within_3sigma = grid_df[
    grid_df["abs_sigma_distance"]
    <= 3.0
]


print()
print(
    "Parameter combinations compatible with "
    "the observed ratio:"
)

print(
    f"  <= 1 sigma: "
    f"{len(within_1sigma):,}"
)

print(
    f"  <= 2 sigma: "
    f"{len(within_2sigma):,}"
)

print(
    f"  <= 3 sigma: "
    f"{len(within_3sigma):,}"
)


# ============================================================
# Parameter ranges for acceptable solutions
# ============================================================

print()
print("=" * 70)
print("5. ACCEPTABLE PARAMETER RANGES")
print("=" * 70)


if len(within_1sigma) > 0:

    print()
    print("Within approximately 1 sigma:")

    print(
        f"  Te range: "
        f"{within_1sigma['Te_K'].min():.0f}"
        " - "
        f"{within_1sigma['Te_K'].max():.0f} K"
    )

    print(
        f"  log10(ne) range: "
        f"{within_1sigma['log10_ne'].min():.2f}"
        " - "
        f"{within_1sigma['log10_ne'].max():.2f}"
    )

    print(
        f"  A(V) range: "
        f"{within_1sigma['A_V_mag'].min():.3f}"
        " - "
        f"{within_1sigma['A_V_mag'].max():.3f}"
        " mag"
    )

else:

    print()
    print(
        "No grid points fall within "
        "approximately 1 sigma."
    )


# ============================================================
# Save complete grid
# ============================================================

print()
print("=" * 70)
print("6. SAVING PARAMETER GRID")
print("=" * 70)


grid_df.to_csv(
    GRID_CSV,
    index=False,
)


print()
print(
    f"Saved:\n  {GRID_CSV}"
)


# ============================================================
# Summary statistics
# ============================================================

summary = {
    "observed_ratio":
        observed_ratio,

    "ratio_uncertainty":
        ratio_uncertainty,

    "best_Te_K":
        best["Te_K"],

    "best_ne_cm3":
        best["ne_cm3"],

    "best_log10_ne":
        best["log10_ne"],

    "best_A_V_mag":
        best["A_V_mag"],

    "best_intrinsic_ratio":
        best["intrinsic_caseB_ratio"],

    "best_predicted_ratio":
        best["predicted_observed_ratio"],

    "best_sigma_distance":
        best["sigma_distance"],

    "n_grid_points":
        len(grid_df),

    "n_within_1sigma":
        len(within_1sigma),

    "n_within_2sigma":
        len(within_2sigma),

    "n_within_3sigma":
        len(within_3sigma),

    "Te_min_1sigma_K":
        (
            within_1sigma["Te_K"].min()
            if len(within_1sigma)
            else np.nan
        ),

    "Te_max_1sigma_K":
        (
            within_1sigma["Te_K"].max()
            if len(within_1sigma)
            else np.nan
        ),

    "log10_ne_min_1sigma":
        (
            within_1sigma["log10_ne"].min()
            if len(within_1sigma)
            else np.nan
        ),

    "log10_ne_max_1sigma":
        (
            within_1sigma["log10_ne"].max()
            if len(within_1sigma)
            else np.nan
        ),

    "A_V_min_1sigma":
        (
            within_1sigma["A_V_mag"].min()
            if len(within_1sigma)
            else np.nan
        ),

    "A_V_max_1sigma":
        (
            within_1sigma["A_V_mag"].max()
            if len(within_1sigma)
            else np.nan
        ),
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
    f"Summary saved:\n  {RESULTS_CSV}"
)


# ============================================================
# Plot 1 — Parameter-space solutions
# ============================================================

print()
print("=" * 70)
print("7. CREATING PARAMETER-SPACE PLOT")
print("=" * 70)


fig, ax = plt.subplots(
    figsize=(10, 7)
)

sc = ax.scatter(
    grid_df["log10_ne"],
    grid_df["Te_K"],
    c=grid_df["A_V_mag"],
    s=12,
    alpha=0.45,
)

ax.scatter(
    best["log10_ne"],
    best["Te_K"],
    marker="*",
    s=220,
    label="Best diagnostic solution",
)

ax.set_xlabel(
    r"$\log_{10}(n_e/\mathrm{cm}^{-3})$"
)

ax.set_ylabel(
    "$T_e$ [K]"
)

ax.set_title(
    "M51 Case-B Parameter Grid"
)

cbar = fig.colorbar(
    sc,
    ax=ax,
)

cbar.set_label(
    "A(V) [mag]"
)

ax.legend()

ax.grid(
    alpha=0.25
)

fig.tight_layout()

fig.savefig(
    PLOT_PATH,
    dpi=180,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# Plot 2 — Extinction curves at representative temperatures
# ============================================================

print()
print(
    "Creating extinction / temperature diagnostic plot..."
)


fig, ax = plt.subplots(
    figsize=(10, 7)
)


selected_te = [
    5000,
    10000,
    20000,
    50000,
]


for te in selected_te:

    intrinsic = (
        approximate_caseB_ratio(
            te,
            REFERENCE_NE_CM3,
        )
    )

    predicted = (
        extincted_ratio(
            intrinsic,
            AV_VALUES,
        )
    )

    ax.plot(
        AV_VALUES,
        predicted,
        linewidth=2,
        label=f"Te = {te:.0f} K",
    )


ax.axhline(
    observed_ratio,
    linestyle="--",
    linewidth=2,
    label=(
        f"Observed = "
        f"{observed_ratio:.3f}"
    ),
)


ax.fill_between(
    AV_VALUES,
    observed_ratio
    - ratio_uncertainty,
    observed_ratio
    + ratio_uncertainty,
    alpha=0.2,
)


ax.set_xlabel(
    "A(V) [mag]"
)

ax.set_ylabel(
    "Pa-beta / Pa-gamma"
)

ax.set_title(
    "Case-B + Extinction Diagnostic"
)

ax.grid(
    alpha=0.25
)

ax.legend()

fig.tight_layout()

fig.savefig(
    Path(
        "m51_hydrogen_caseB_extinction_temperature.png"
    ),
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
    "This experiment explores whether the measured "
    "Pa-beta / Pa-gamma ratio can be reproduced by a "
    "hydrogen Case-B recombination model over a range "
    "of electron temperatures, electron densities, "
    "and foreground extinction."
)

print()
print(
    f"Observed ratio:"
)

print(
    f"  {observed_ratio:.6f} "
    f"+/- {ratio_uncertainty:.6f}"
)

print()
print(
    "Best diagnostic grid solution:"
)

print(
    f"  Te = "
    f"{best['Te_K']:.0f} K"
)

print(
    f"  ne = "
    f"{best['ne_cm3']:.3e} cm^-3"
)

print(
    f"  A(V) = "
    f"{best['A_V_mag']:.3f} mag"
)

print(
    f"  Predicted ratio = "
    f"{best['predicted_observed_ratio']:.6f}"
)

print(
    f"  Residual = "
    f"{best['ratio_residual']:.6f}"
)

print()
print(
    "IMPORTANT SCIENTIFIC QUALIFICATION:"
)

print(
    "The current parameter-grid function uses an "
    "approximate smooth representation of the Case-B "
    "ratio around the reference Storey-Hummer value."
)

print(
    "It is NOT a substitute for the actual "
    "Storey-Hummer emissivity tables."
)

print()
print(
    "Storey & Hummer provide machine-readable Case-B "
    "hydrogenic calculations covering broad ranges of "
    "temperature and electron density, including "
    "log(ne) = 2 through 14."
)

print(
    "The definitive next-generation version of this "
    "experiment should therefore replace the approximate "
    "ratio function with direct Pa-beta and Pa-gamma "
    "emissivities from those tables."
)

print()
print(
    "Until that replacement is made, this result should "
    "be interpreted as a parameter-sensitivity diagnostic, "
    "not as a formal Case-B likelihood measurement."
)

print()
print(
    "Nevertheless, the measured ratio is not intrinsically "
    "incompatible with hydrogen recombination: differential "
    "extinction naturally increases Pa-beta / Pa-gamma "
    "because Pa-gamma is at the shorter wavelength."
)

print()
print("Outputs:")
print(
    f"  {RESULTS_CSV}"
)

print(
    f"  {GRID_CSV}"
)

print(
    f"  {PLOT_PATH}"
)

print(
    "  m51_hydrogen_caseB_extinction_temperature.png"
)

print()
print("=" * 70)
print(
    "HYDROGEN CASE-B PARAMETER GRID TEST COMPLETE"
)
print("=" * 70)
