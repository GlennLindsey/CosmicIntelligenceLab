#!/usr/bin/env python3

"""
M51 Hydrogen Pa-beta / Pa-gamma
Independent CHAOS Extinction Constraint

Purpose
-------
Compare the foreground A(V) required by the JWST
Pa-beta / Pa-gamma Storey-Hummer Case-B analysis
with an independent optical extinction measurement
from the CHAOS M51 HII-region database.

JWST:
    m51_hydrogen_required_av.csv

CHAOS:
    Croxall et al. (2015)
    table3b.dat
    Region: NGC5194+30.2+2.2

Important
---------
The CHAOS optical aperture and the JWST NIRSpec/IFU
0.45 arcsec extraction aperture are not identical.
Therefore this is an independent plausibility constraint,
not a direct aperture-matched measurement.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

PROJECT = Path.home() / "Projects" / "cosmic_ai"

JWST_REQUIRED_AV = (
    PROJECT
    / "data"
    / "atomic_lines"
    / "m51_hydrogen_required_av.csv"
)

CHAOS_TABLE3B = (
    PROJECT
    / "data"
    / "atomic_lines"
    / "chaos_m51"
    / "table3b.dat"
)

OUTPUT_TABLE = (
    PROJECT
    / "data"
    / "atomic_lines"
    / "m51_hydrogen_chaos_extinction_constraint.csv"
)

OUTPUT_SUMMARY = (
    PROJECT
    / "data"
    / "atomic_lines"
    / "m51_hydrogen_chaos_extinction_summary.csv"
)

OUTPUT_FIGURE = (
    PROJECT
    / "m51_hydrogen_chaos_extinction_constraint.png"
)


# ============================================================
# CHAOS REGION
# ============================================================

TARGET_ID = "NGC5194+30.2+2.2"

# From CHAOS table3b.dat
C_HBETA = 1.032
C_HBETA_SIGMA = 0.013

# Conversion used for this experiment
C_HBETA_TO_EBV = 1.43
R_V = 3.1


# ============================================================
# UTILITY
# ============================================================

def section(title):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    section(
        "M51 HYDROGEN — CHAOS INDEPENDENT EXTINCTION TEST"
    )

    # ========================================================
    # 1. CHAOS REGION
    # ========================================================

    section("1. IDENTIFYING CHAOS REGION")

    print("CHAOS table:")
    print(f"  {CHAOS_TABLE3B}")

    print("\nTarget:")
    print(f"  {TARGET_ID}")

    matches = []

    with open(
        CHAOS_TABLE3B,
        "r",
        errors="replace"
    ) as f:

        for line_number, line in enumerate(
            f,
            start=1
        ):

            if TARGET_ID in line:

                matches.append(
                    (line_number, line.rstrip())
                )

    print(
        f"\nMatches found: {len(matches)}"
    )

    for line_number, line in matches:

        print(
            f"\nLine {line_number}:"
        )

        print(
            f"  {line}"
        )

    if len(matches) != 1:

        raise RuntimeError(
            "Expected exactly one CHAOS region match."
        )

    # ========================================================
    # 2. CHAOS EXTINCTION
    # ========================================================

    section("2. CHAOS EXTINCTION")

    print(
        f"C(Hbeta) = "
        f"{C_HBETA:.3f} +/- "
        f"{C_HBETA_SIGMA:.3f}"
    )

    print(
        "\nConversion:"
    )

    print(
        "  C(Hbeta) = 1.43 E(B-V)"
    )

    print(
        "  A(V) = 3.1 E(B-V)"
    )

    ebv = (
        C_HBETA
        / C_HBETA_TO_EBV
    )

    ebv_sigma = (
        C_HBETA_SIGMA
        / C_HBETA_TO_EBV
    )

    av_chaos = (
        R_V
        * ebv
    )

    av_chaos_sigma = (
        R_V
        * ebv_sigma
    )

    print(
        f"\nE(B-V) = "
        f"{ebv:.6f} +/- "
        f"{ebv_sigma:.6f}"
    )

    print(
        f"A(V) = "
        f"{av_chaos:.6f} +/- "
        f"{av_chaos_sigma:.6f} mag"
    )

    # ========================================================
    # 3. LOAD STOREY-HUMMER DATAFRAME
    # ========================================================

    section(
        "3. LOADING STOREY-HUMMER DATAFRAME"
    )

    print("File:")
    print(
        f"  {JWST_REQUIRED_AV}"
    )

    # --------------------------------------------------------
    # THIS IS THE DATAFRAME LOADING STEP
    # --------------------------------------------------------

    df = pd.read_csv(
        JWST_REQUIRED_AV
    )

    print(
        f"\nRows loaded: {len(df)}"
    )

    print(
        "\nCSV columns:"
    )

    for column in df.columns:

        print(
            f"  {column}"
        )

    # ========================================================
    # 4. VERIFY REQUIRED COLUMN NAMES
    # ========================================================

    section(
        "4. VERIFYING STOREY-HUMMER COLUMNS"
    )

    required_columns = [
        "Te",
        "ne",
        "intrinsic_ratio",
        "required_Av",
    ]

    for column in required_columns:

        if column not in df.columns:

            raise RuntimeError(
                f"Required column not found: {column}"
            )

        print(
            f"OK: {column}"
        )

    # ========================================================
    # 5. CONVERT NUMERIC DATA
    # ========================================================

    section(
        "5. PREPARING STOREY-HUMMER DATA"
    )

    numeric_columns = [
        "Te",
        "ne",
        "intrinsic_ratio",
        "required_Av",
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    before = len(df)

    df = df.dropna(
        subset=numeric_columns
    ).copy()

    after = len(df)

    print(
        f"Rows before cleaning: {before}"
    )

    print(
        f"Rows after cleaning:  {after}"
    )

    # ========================================================
    # 6. COMPARE REQUIRED A(V) WITH CHAOS
    # ========================================================

    section(
        "6. COMPARING REQUIRED A(V) WITH CHAOS"
    )

    # IMPORTANT:
    # The CSV column is exactly:
    #
    #     required_Av
    #
    # NOT:
    #
    #     required_A_V

    df["CHAOS_Av"] = av_chaos

    df["CHAOS_Av_sigma"] = (
        av_chaos_sigma
    )

    df["delta_Av"] = (
        df["required_Av"]
        - av_chaos
    )

    df["difference_sigma"] = (
        df["delta_Av"]
        / av_chaos_sigma
    )

    print(
        f"CHAOS A(V): "
        f"{av_chaos:.6f} +/- "
        f"{av_chaos_sigma:.6f} mag"
    )

    print(
        "\nRequired A(V) statistics:"
    )

    print(
        f"  minimum = "
        f"{df['required_Av'].min():.6f} mag"
    )

    print(
        f"  maximum = "
        f"{df['required_Av'].max():.6f} mag"
    )

    print(
        f"  median  = "
        f"{df['required_Av'].median():.6f} mag"
    )

    # ========================================================
    # 7. FIND MINIMUM REQUIRED A(V)
    # ========================================================

    section(
        "7. LOWEST REQUIRED A(V)"
    )

    minimum_index = (
        df["required_Av"].idxmin()
    )

    minimum = (
        df.loc[minimum_index]
    )

    print(
        f"Te = "
        f"{minimum['Te']:.0f} K"
    )

    print(
        f"ne = "
        f"{minimum['ne']:.3e} cm^-3"
    )

    print(
        f"log10(ne) = "
        f"{np.log10(minimum['ne']):.2f}"
    )

    print(
        f"Intrinsic ratio = "
        f"{minimum['intrinsic_ratio']:.6f}"
    )

    print(
        f"Required A(V) = "
        f"{minimum['required_Av']:.6f} mag"
    )

    minimum_difference = (
        minimum["required_Av"]
        - av_chaos
    )

    minimum_difference_sigma = (
        minimum_difference
        / av_chaos_sigma
    )

    print(
        f"\nDifference from CHAOS:"
    )

    print(
        f"  {minimum_difference:.6f} mag"
    )

    print(
        f"  {minimum_difference_sigma:.3f} sigma"
    )

    # ========================================================
    # 8. CHAOS 1, 2, 3 SIGMA INTERVALS
    # ========================================================

    section(
        "8. CHAOS COMPATIBILITY INTERVALS"
    )

    compatibility_rows = []

    for n_sigma in [1, 2, 3]:

        lower = (
            av_chaos
            - n_sigma * av_chaos_sigma
        )

        upper = (
            av_chaos
            + n_sigma * av_chaos_sigma
        )

        compatible = df[
            (
                df["required_Av"]
                >= lower
            )
            &
            (
                df["required_Av"]
                <= upper
            )
        ]

        print(
            f"\n{n_sigma}-sigma:"
        )

        print(
            f"  CHAOS interval = "
            f"{lower:.3f} - "
            f"{upper:.3f} mag"
        )

        print(
            f"  Compatible grid points = "
            f"{len(compatible)}"
        )

        if len(compatible) > 0:

            print(
                f"  Te range = "
                f"{compatible['Te'].min():.0f} - "
                f"{compatible['Te'].max():.0f} K"
            )

            print(
                f"  log10(ne) range = "
                f"{np.log10(compatible['ne']).min():.2f}"
                " - "
                f"{np.log10(compatible['ne']).max():.2f}"
            )

        compatibility_rows.append(
            {
                "sigma_level": n_sigma,
                "CHAOS_Av_lower": lower,
                "CHAOS_Av_upper": upper,
                "compatible_grid_points":
                    len(compatible),
                "Te_min_K":
                    compatible["Te"].min()
                    if len(compatible)
                    else np.nan,
                "Te_max_K":
                    compatible["Te"].max()
                    if len(compatible)
                    else np.nan,
                "log10_ne_min":
                    np.log10(
                        compatible["ne"]
                    ).min()
                    if len(compatible)
                    else np.nan,
                "log10_ne_max":
                    np.log10(
                        compatible["ne"]
                    ).max()
                    if len(compatible)
                    else np.nan,
            }
        )

    compatibility = pd.DataFrame(
        compatibility_rows
    )

    # ========================================================
    # 9. TEMPERATURE SUMMARY
    # ========================================================

    section(
        "9. TEMPERATURE SUMMARY"
    )

    temperature_rows = []

    for temperature, group in (
        df.groupby("Te")
    ):

        minimum_row = group.loc[
            group["required_Av"].idxmin()
        ]

        temperature_min = (
            group["required_Av"].min()
        )

        temperature_max = (
            group["required_Av"].max()
        )

        difference = (
            temperature_min
            - av_chaos
        )

        difference_sigma = (
            difference
            / av_chaos_sigma
        )

        print(
            f"\nTe = {temperature:.0f} K"
        )

        print(
            f"  Required A(V): "
            f"{temperature_min:.3f} - "
            f"{temperature_max:.3f} mag"
        )

        print(
            f"  Minimum at ne = "
            f"{minimum_row['ne']:.3e} cm^-3"
        )

        print(
            f"  Difference from CHAOS = "
            f"{difference:.3f} mag"
        )

        print(
            f"  Difference = "
            f"{difference_sigma:.2f} sigma"
        )

        temperature_rows.append(
            {
                "Te_K": temperature,
                "required_Av_min":
                    temperature_min,
                "required_Av_max":
                    temperature_max,
                "ne_at_minimum_cm3":
                    minimum_row["ne"],
                "CHAOS_Av":
                    av_chaos,
                "CHAOS_Av_sigma":
                    av_chaos_sigma,
                "minimum_difference_Av":
                    difference,
                "minimum_difference_sigma":
                    difference_sigma,
            }
        )

    temperature_summary = pd.DataFrame(
        temperature_rows
    )

    # ========================================================
    # 10. SAVE FULL COMPARISON TABLE
    # ========================================================

    section(
        "10. SAVING COMPARISON TABLE"
    )

    df.to_csv(
        OUTPUT_TABLE,
        index=False
    )

    print(
        f"Saved:"
        f"\n  {OUTPUT_TABLE}"
    )

    # ========================================================
    # 11. SAVE SUMMARY
    # ========================================================

    section(
        "11. SAVING SUMMARY TABLE"
    )

    combined_summary = pd.concat(
        [
            temperature_summary.assign(
                summary_type="temperature"
            ),
            compatibility.assign(
                summary_type="sigma_compatibility"
            ),
        ],
        ignore_index=True,
        sort=False
    )

    combined_summary.to_csv(
        OUTPUT_SUMMARY,
        index=False
    )

    print(
        f"Saved:"
        f"\n  {OUTPUT_SUMMARY}"
    )

    # ========================================================
    # 12. CREATE FIGURE
    # ========================================================

    section(
        "12. CREATING FIGURE"
    )

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    temperatures = sorted(
        df["Te"].unique()
    )

    for temperature in temperatures:

        subset = (
            df[
                df["Te"] == temperature
            ]
            .sort_values("ne")
        )

        ax.plot(
            np.log10(subset["ne"]),
            subset["required_Av"],
            marker="o",
            linewidth=1.5,
            label=f"{temperature:.0f} K"
        )

    # CHAOS central value

    ax.axhline(
        av_chaos,
        linewidth=2,
        label=(
            f"CHAOS A(V) = "
            f"{av_chaos:.2f} +/- "
            f"{av_chaos_sigma:.2f}"
        )
    )

    # CHAOS 1-sigma band

    ax.axhspan(
        av_chaos - av_chaos_sigma,
        av_chaos + av_chaos_sigma,
        alpha=0.2
    )

    ax.set_xlabel(
        r"$\log_{10}(n_e\ /\ {\rm cm}^{-3})$"
    )

    ax.set_ylabel(
        r"Required $A(V)$ [mag]"
    )

    ax.set_title(
        "M51 Pa-beta / Pa-gamma\n"
        "Storey-Hummer Required Extinction "
        "vs. CHAOS"
    )

    ax.grid(
        True,
        alpha=0.3
    )

    ax.legend(
        fontsize=8
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_FIGURE,
        dpi=200
    )

    plt.close(fig)

    print(
        f"Saved:"
        f"\n  {OUTPUT_FIGURE}"
    )

    # ========================================================
    # 13. FINAL INTERPRETATION
    # ========================================================

    section(
        "13. FINAL INTERPRETATION"
    )

    print(
        "Independent CHAOS extinction:"
    )

    print(
        f"  A(V) = "
        f"{av_chaos:.3f} +/- "
        f"{av_chaos_sigma:.3f} mag"
    )

    print(
        "\nMinimum Storey-Hummer-required extinction:"
    )

    print(
        f"  A(V) = "
        f"{minimum['required_Av']:.3f} mag"
    )

    print(
        f"  Te = "
        f"{minimum['Te']:.0f} K"
    )

    print(
        f"  ne = "
        f"{minimum['ne']:.3e} cm^-3"
    )

    print(
        "\nDifference between minimum required A(V)"
        " and CHAOS:"
    )

    print(
        f"  {minimum_difference:.3f} mag"
    )

    print(
        f"  {minimum_difference_sigma:.2f} sigma"
    )

    print(
        "\nScientific qualification:"
    )

    print(
        "The CHAOS measurement is an independent"
        " optical constraint on an M51 HII region."
    )

    print(
        "It is not an aperture-matched measurement"
        " of the JWST 0.45-arcsec extraction."
    )

    print(
        "Agreement therefore supports astrophysical"
        " plausibility but does not by itself establish"
        " that the two measurements sample identical gas."
    )

    print(
        "\nExperiment complete."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
