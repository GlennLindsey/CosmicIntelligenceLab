#!/usr/bin/env python3

"""
M51 HYDROGEN STOREY-HUMMER REQUIRED A(V) MAP
==============================================

For every available Storey-Hummer Case-B Te/ne combination,
calculate the foreground extinction A(V) required to transform
the intrinsic Pa-beta / Pa-gamma ratio into the observed M51 ratio.

This is deliberately different from a best-fit extinction grid.

For each physical Storey-Hummer point:

    R_intrinsic = Pa-beta / Pa-gamma

    R_observed = R_intrinsic *
                 10^[0.4 * (A(Pa-gamma) - A(Pa-beta))]

Therefore:

    A(V) =
        log10(R_observed / R_intrinsic)
        / [0.4 * (k_gamma - k_beta)]

where:

    k_beta  = A(Pa-beta)  / A(V)
    k_gamma = A(Pa-gamma) / A(V)

Outputs:

    data/atomic_lines/
        m51_hydrogen_required_av.csv

    data/atomic_lines/
        m51_hydrogen_required_av_summary.csv

    m51_hydrogen_required_av_map.png

    m51_hydrogen_required_av_vs_density.png

The Storey-Hummer intrinsic ratios are taken from the previously
generated actual Storey-Hummer grid CSV.

No approximate Case-B ratio function is used.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path.home() / "Projects" / "cosmic_ai"

GRID_FILE = (
    PROJECT_ROOT
    / "data"
    / "atomic_lines"
    / "m51_hydrogen_storey_hummer_grid.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "atomic_lines"
)

OUTPUT_CSV = OUTPUT_DIR / "m51_hydrogen_required_av.csv"
SUMMARY_CSV = OUTPUT_DIR / "m51_hydrogen_required_av_summary.csv"

PLOT_MAP = PROJECT_ROOT / "m51_hydrogen_required_av_map.png"
PLOT_DENSITY = PROJECT_ROOT / "m51_hydrogen_required_av_vs_density.png"


# ------------------------------------------------------------
# Observed M51 ratio
# ------------------------------------------------------------

OBSERVED_RATIO = 3.352378
OBSERVED_RATIO_SIGMA = 0.103204


# ------------------------------------------------------------
# Extinction coefficients used in the previous experiment
# ------------------------------------------------------------

A_BETA_OVER_AV = 0.360632
A_GAMMA_OVER_AV = 0.472246

DELTA_A_OVER_AV = A_GAMMA_OVER_AV - A_BETA_OVER_AV


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def find_column(df, candidates):
    """
    Find a dataframe column using several possible names.
    """

    normalized = {
        str(col).strip().lower().replace(" ", "_"): col
        for col in df.columns
    }

    for candidate in candidates:
        key = candidate.lower().replace(" ", "_")

        if key in normalized:
            return normalized[key]

    return None


def print_columns(df):
    print("\nAvailable CSV columns:")

    for col in df.columns:
        print(f"  {col}")


def calculate_required_av(intrinsic_ratio):
    """
    Calculate A(V) required to transform intrinsic ratio into
    the observed ratio.

    R_obs / R_intrinsic =
        10^[0.4 * A(V) * (k_gamma - k_beta)]

    Hence:

    A(V) =
        log10(R_obs / R_intrinsic)
        /
        [0.4 * (k_gamma - k_beta)]
    """

    intrinsic_ratio = np.asarray(intrinsic_ratio, dtype=float)

    return (
        np.log10(OBSERVED_RATIO / intrinsic_ratio)
        / (0.4 * DELTA_A_OVER_AV)
    )


def classify_solution(required_av):
    """
    Classify whether positive foreground extinction is required.
    """

    if not np.isfinite(required_av):
        return "invalid"

    if required_av < 0:
        return "negative_Av"

    return "positive_Av"


# ============================================================
# LOAD STOREY-HUMMER GRID
# ============================================================

def load_grid():

    print("\n" + "=" * 70)
    print("1. LOADING STOREY-HUMMER GRID")
    print("=" * 70)

    print("\nFile:")
    print(f"  {GRID_FILE}")

    if not GRID_FILE.exists():

        raise FileNotFoundError(
            f"\nStorey-Hummer grid not found:\n{GRID_FILE}\n\n"
            "Run test_m51_hydrogen_storey_hummer_grid.py first."
        )

    df = pd.read_csv(GRID_FILE)

    print(f"\nRows loaded: {len(df):,}")

    print_columns(df)

    return df


# ============================================================
# EXTRACT INTRINSIC RATIOS
# ============================================================

def extract_intrinsic_grid(df):

    print("\n" + "=" * 70)
    print("2. EXTRACTING INTRINSIC STOREY-HUMMER RATIOS")
    print("=" * 70)

    te_col = find_column(
        df,
        [
            "Te_K",
            "Te",
            "temperature",
            "electron_temperature",
            "te_k",
        ],
    )

    ne_col = find_column(
        df,
        [
            "ne_cm3",
            "ne",
            "density",
            "electron_density",
            "electron_density_cm-3",
        ],
    )

    ratio_col = find_column(
        df,
        [
            "intrinsic_ratio",
            "intrinsic ratio",
            "ratio",
            "storey_hummer_ratio",
        ],
    )

    av_col = find_column(
        df,
        [
            "A_V",
            "A(V)",
            "av",
            "extinction",
        ],
    )

    if te_col is None:
        raise RuntimeError("Could not identify Te column.")

    if ne_col is None:
        raise RuntimeError("Could not identify electron-density column.")

    if ratio_col is None:
        raise RuntimeError("Could not identify intrinsic-ratio column.")

    print(f"\nTemperature column: {te_col}")
    print(f"Density column:     {ne_col}")
    print(f"Ratio column:       {ratio_col}")

    if av_col is not None:

        print(f"A(V) column:        {av_col}")

        av_numeric = pd.to_numeric(
            df[av_col],
            errors="coerce",
        )

        # Prefer the A(V)=0 rows because those represent
        # the intrinsic Storey-Hummer ratios.
        intrinsic = df[np.isclose(av_numeric, 0.0)]

        if len(intrinsic) == 0:
            print(
                "\nWARNING: No A(V)=0 rows found."
                "\nUsing unique Te/ne/ratio combinations instead."
            )

            intrinsic = df[
                [te_col, ne_col, ratio_col]
            ].drop_duplicates()

        else:

            intrinsic = intrinsic[
                [te_col, ne_col, ratio_col]
            ].drop_duplicates()

    else:

        print(
            "\nNo A(V) column detected."
            "\nUsing unique Te/ne/ratio combinations."
        )

        intrinsic = df[
            [te_col, ne_col, ratio_col]
        ].drop_duplicates()

    intrinsic = intrinsic.rename(
        columns={
            te_col: "Te",
            ne_col: "ne",
            ratio_col: "intrinsic_ratio",
        }
    )

    intrinsic["Te"] = pd.to_numeric(
        intrinsic["Te"],
        errors="coerce",
    )

    intrinsic["ne"] = pd.to_numeric(
        intrinsic["ne"],
        errors="coerce",
    )

    intrinsic["intrinsic_ratio"] = pd.to_numeric(
        intrinsic["intrinsic_ratio"],
        errors="coerce",
    )

    intrinsic = intrinsic.dropna(
        subset=[
            "Te",
            "ne",
            "intrinsic_ratio",
        ]
    )

    intrinsic = intrinsic[
        intrinsic["intrinsic_ratio"] > 0
    ]

    intrinsic = intrinsic.sort_values(
        ["Te", "ne"]
    ).reset_index(drop=True)

    print(
        "\nUnique Te/ne combinations:"
        f" {len(intrinsic)}"
    )

    print(
        "\nTemperatures:"
    )

    for te in sorted(intrinsic["Te"].unique()):

        print(
            f"  {te:,.0f} K"
        )

    print(
        "\nDensity range:"
    )

    print(
        f"  {intrinsic['ne'].min():.3e}"
        f" to "
        f"{intrinsic['ne'].max():.3e} cm^-3"
    )

    return intrinsic


# ============================================================
# CALCULATE REQUIRED A(V)
# ============================================================

def build_required_av_table(intrinsic):

    print("\n" + "=" * 70)
    print("3. CALCULATING REQUIRED A(V)")
    print("=" * 70)

    print(
        "\nObserved Pa-beta / Pa-gamma:"
    )

    print(
        f"  {OBSERVED_RATIO:.6f}"
        f" +/- {OBSERVED_RATIO_SIGMA:.6f}"
    )

    print(
        "\nExtinction coefficients:"
    )

    print(
        f"  A(Pa-beta) / A(V)  = "
        f"{A_BETA_OVER_AV:.6f}"
    )

    print(
        f"  A(Pa-gamma) / A(V) = "
        f"{A_GAMMA_OVER_AV:.6f}"
    )

    print(
        f"  Differential extinction = "
        f"{DELTA_A_OVER_AV:.6f} A(V)"
    )

    result = intrinsic.copy()

    result["observed_ratio"] = OBSERVED_RATIO

    result["ratio_sigma"] = OBSERVED_RATIO_SIGMA

    result["ratio_observed_minus_intrinsic"] = (
        OBSERVED_RATIO
        - result["intrinsic_ratio"]
    )

    result["required_Av"] = calculate_required_av(
        result["intrinsic_ratio"].values
    )

    result["A_Pa_beta"] = (
        A_BETA_OVER_AV
        * result["required_Av"]
    )

    result["A_Pa_gamma"] = (
        A_GAMMA_OVER_AV
        * result["required_Av"]
    )

    result["predicted_ratio"] = (
        result["intrinsic_ratio"]
        * 10.0
        ** (
            0.4
            * (
                result["A_Pa_gamma"]
                - result["A_Pa_beta"]
            )
        )
    )

    result["ratio_residual"] = (
        result["predicted_ratio"]
        - OBSERVED_RATIO
    )

    result["ratio_residual_sigma"] = (
        result["ratio_residual"]
        / OBSERVED_RATIO_SIGMA
    )

    result["solution_class"] = [
        classify_solution(x)
        for x in result["required_Av"]
    ]

    return result


# ============================================================
# UNCERTAINTY BOUNDS
# ============================================================

def add_ratio_uncertainty_bounds(result):

    print("\n" + "=" * 70)
    print("4. PROPAGATING OBSERVED-RATIO UNCERTAINTY")
    print("=" * 70)

    lower_ratio = OBSERVED_RATIO - OBSERVED_RATIO_SIGMA
    upper_ratio = OBSERVED_RATIO + OBSERVED_RATIO_SIGMA

    print(
        f"\n1-sigma observed ratio interval:"
    )

    print(
        f"  {lower_ratio:.6f}"
        f" to "
        f"{upper_ratio:.6f}"
    )

    intrinsic = result["intrinsic_ratio"].values

    av_lower = (
        np.log10(lower_ratio / intrinsic)
        / (0.4 * DELTA_A_OVER_AV)
    )

    av_upper = (
        np.log10(upper_ratio / intrinsic)
        / (0.4 * DELTA_A_OVER_AV)
    )

    result["required_Av_minus_1sigma"] = av_lower

    result["required_Av_plus_1sigma"] = av_upper

    result["required_Av_uncertainty"] = (
        0.5
        * (
            np.abs(av_upper - result["required_Av"])
            + np.abs(
                result["required_Av"]
                - av_lower
            )
        )
    )

    return result


# ============================================================
# SUMMARY
# ============================================================

def print_summary(result):

    print("\n" + "=" * 70)
    print("5. REQUIRED A(V) SUMMARY")
    print("=" * 70)

    print(
        "\nRequired A(V) range across the actual"
        " Storey-Hummer grid:"
    )

    positive = result[
        result["required_Av"] >= 0
    ]

    if len(positive) > 0:

        print(
            f"  Minimum: "
            f"{positive['required_Av'].min():.3f} mag"
        )

        print(
            f"  Maximum: "
            f"{positive['required_Av'].max():.3f} mag"
        )

    negative = result[
        result["required_Av"] < 0
    ]

    print(
        "\nSolutions requiring positive extinction:"
        f" {len(positive)}"
    )

    print(
        "Solutions requiring negative extinction:"
        f" {len(negative)}"
    )

    print(
        "\nTemperature-dependent ranges:"
    )

    for te in sorted(result["Te"].unique()):

        subset = result[
            result["Te"] == te
        ]

        positive_te = subset[
            subset["required_Av"] >= 0
        ]

        if len(positive_te) == 0:

            print(
                f"  Te={te:,.0f} K:"
                " no positive-A(V) solutions"
            )

            continue

        print(
            f"  Te={te:,.0f} K:"
            f" A(V) = "
            f"{positive_te['required_Av'].min():.2f}"
            f" - "
            f"{positive_te['required_Av'].max():.2f} mag"
        )

    print(
        "\nLowest required A(V) solution:"
    )

    if len(positive) > 0:

        best = positive.loc[
            positive["required_Av"].idxmin()
        ]

        print(
            f"  Te = {best['Te']:,.0f} K"
        )

        print(
            f"  ne = {best['ne']:.3e} cm^-3"
        )

        print(
            f"  log10(ne) = "
            f"{np.log10(best['ne']):.2f}"
        )

        print(
            f"  Intrinsic ratio = "
            f"{best['intrinsic_ratio']:.6f}"
        )

        print(
            f"  Required A(V) = "
            f"{best['required_Av']:.3f} mag"
        )


# ============================================================
# SAVE DATA
# ============================================================

def save_outputs(result):

    print("\n" + "=" * 70)
    print("6. SAVING REQUIRED-A(V) TABLE")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_CSV,
        index=False,
        float_format="%.8e",
    )

    print(
        "\nFull required-A(V) table:"
    )

    print(
        f"  {OUTPUT_CSV}"
    )

    # --------------------------------------------------------
    # Summary by temperature
    # --------------------------------------------------------

    summary_rows = []

    for te in sorted(result["Te"].unique()):

        subset = result[
            result["Te"] == te
        ]

        positive = subset[
            subset["required_Av"] >= 0
        ]

        if len(positive) > 0:

            summary_rows.append(
                {
                    "Te_K": te,
                    "n_density_points": len(subset),
                    "ne_min_cm-3": subset["ne"].min(),
                    "ne_max_cm-3": subset["ne"].max(),
                    "required_Av_min": positive[
                        "required_Av"
                    ].min(),
                    "required_Av_max": positive[
                        "required_Av"
                    ].max(),
                    "intrinsic_ratio_min": subset[
                        "intrinsic_ratio"
                    ].min(),
                    "intrinsic_ratio_max": subset[
                        "intrinsic_ratio"
                    ].max(),
                }
            )

        else:

            summary_rows.append(
                {
                    "Te_K": te,
                    "n_density_points": len(subset),
                    "ne_min_cm-3": subset["ne"].min(),
                    "ne_max_cm-3": subset["ne"].max(),
                    "required_Av_min": np.nan,
                    "required_Av_max": np.nan,
                    "intrinsic_ratio_min": subset[
                        "intrinsic_ratio"
                    ].min(),
                    "intrinsic_ratio_max": subset[
                        "intrinsic_ratio"
                    ].max(),
                }
            )

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        SUMMARY_CSV,
        index=False,
        float_format="%.8e",
    )

    print(
        "\nTemperature summary:"
    )

    print(
        f"  {SUMMARY_CSV}"
    )


# ============================================================
# PLOT 1 — 2D Te / density map
# ============================================================

def create_av_map(result):

    print("\n" + "=" * 70)
    print("7. CREATING REQUIRED A(V) MAP")
    print("=" * 70)

    pivot = result.pivot(
        index="Te",
        columns="ne",
        values="required_Av",
    )

    x = np.log10(
        pivot.columns.values.astype(float)
    )

    y = pivot.index.values.astype(float)

    z = pivot.values.astype(float)

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    image = ax.imshow(
        z,
        aspect="auto",
        origin="lower",
        extent=[
            x.min(),
            x.max(),
            y.min(),
            y.max(),
        ],
    )

    colorbar = fig.colorbar(
        image,
        ax=ax,
    )

    colorbar.set_label(
        r"Required $A(V)$ [mag]"
    )

    ax.set_xlabel(
        r"$\log_{10}(n_e/\mathrm{cm}^{-3})$"
    )

    ax.set_ylabel(
        r"$T_e$ [K]"
    )

    ax.set_title(
        "M51 Pa-beta / Pa-gamma: "
        "Required Extinction from Storey-Hummer Case B"
    )

    ax.set_yticks(
        y
    )

    ax.set_yticklabels(
        [
            f"{int(v):,}"
            for v in y
        ]
    )

    # Mark the observed-ratio solution exactly.
    for _, row in result.iterrows():

        if np.isfinite(row["required_Av"]):

            if row["required_Av"] >= 0:

                ax.scatter(
                    np.log10(row["ne"]),
                    row["Te"],
                    marker="o",
                    s=12,
                    edgecolors="black",
                    linewidths=0.25,
                )

    fig.tight_layout()

    fig.savefig(
        PLOT_MAP,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"\nRequired A(V) map:"
        f"\n  {PLOT_MAP}"
    )


# ============================================================
# PLOT 2 — Required A(V) vs density
# ============================================================

def create_density_plot(result):

    print("\n" + "=" * 70)
    print("8. CREATING A(V) VS DENSITY PLOT")
    print("=" * 70)

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    for te in sorted(
        result["Te"].unique()
    ):

        subset = result[
            result["Te"] == te
        ].sort_values("ne")

        positive = subset[
            subset["required_Av"] >= 0
        ]

        if len(positive) == 0:
            continue

        ax.plot(
            np.log10(
                positive["ne"]
            ),
            positive["required_Av"],
            marker="o",
            label=f"{te:,.0f} K",
        )

    ax.set_xlabel(
        r"$\log_{10}(n_e/\mathrm{cm}^{-3})$"
    )

    ax.set_ylabel(
        r"Required $A(V)$ [mag]"
    )

    ax.set_title(
        "Extinction Required to Match "
        "Observed M51 Pa-beta / Pa-gamma"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend(
        title=r"$T_e$"
    )

    fig.tight_layout()

    fig.savefig(
        PLOT_DENSITY,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"\nA(V) vs density plot:"
        f"\n  {PLOT_DENSITY}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n"
        + "=" * 70
    )

    print(
        "M51 HYDROGEN STOREY-HUMMER"
    )

    print(
        "REQUIRED A(V) MAP"
    )

    print(
        "Te + ne -> extinction required to reproduce observed ratio"
    )

    print(
        "=" * 70
    )

    df = load_grid()

    intrinsic = extract_intrinsic_grid(
        df
    )

    result = build_required_av_table(
        intrinsic
    )

    result = add_ratio_uncertainty_bounds(
        result
    )

    print_summary(
        result
    )

    save_outputs(
        result
    )

    create_av_map(
        result
    )

    create_density_plot(
        result
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FINAL INTERPRETATION"
    )

    print(
        "=" * 70
    )

    print(
        """
This experiment does not search for a single best
Te / ne / A(V) combination.

Instead, it calculates the foreground extinction A(V)
required at every actual Storey-Hummer Case-B Te/ne
grid point.

The observed M51 Pa-beta / Pa-gamma ratio is:

  R_obs = 3.352378 +/- 0.103204

The extinction calculation uses:

  A(Pa-beta) / A(V)  = 0.360632
  A(Pa-gamma) / A(V) = 0.472246

Because Pa-gamma is at the shorter wavelength, extinction
suppresses Pa-gamma more strongly than Pa-beta and therefore
increases the observed Pa-beta / Pa-gamma ratio.

The resulting map shows which Storey-Hummer physical
conditions require modest, large, or formally negative
foreground extinction to reproduce the M51 measurement.

A negative required A(V) means that the intrinsic
Storey-Hummer ratio is already larger than the observed
ratio; such a point cannot be reconciled with the observed
ratio using positive foreground extinction alone.

The map should therefore be interpreted as an extinction
requirement across physical parameter space, not as a
measurement of Te, ne, or A(V).

The next scientific question is whether the regions
requiring astrophysically reasonable extinction overlap
with independent constraints on the physical conditions
of the M51 emitting gas.
"""
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "REQUIRED A(V) MAP COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()
