#!/usr/bin/env python3

"""
M51 Pa-beta / Pa-gamma versus independent CHAOS extinction constraint.

Compares the A(V) required by the JWST hydrogen-line ratio against
the independent optical extinction measurement for the CHAOS region:

    NGC5194+30.2+2.2

Existing Storey-Hummer results:
    data/atomic_lines/m51_hydrogen_required_av.csv

CHAOS:
    C(Hbeta) = 1.032 +/- 0.013

Conversion:
    C(Hbeta) = 1.43 E(B-V)
    A(V) = 3.1 E(B-V)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# PATHS
# ============================================================

PROJECT = Path.home() / "Projects" / "cosmic_ai"

CHAOS_TABLE = PROJECT / "data/atomic_lines/chaos_m51/table3b.dat"

REQUIRED_AV_TABLE = PROJECT / "data/atomic_lines/m51_hydrogen_required_av.csv"

OUTPUT_TABLE = (
    PROJECT / "data/atomic_lines/" "m51_hydrogen_chaos_extinction_constraint.csv"
)

OUTPUT_SUMMARY = (
    PROJECT / "data/atomic_lines/" "m51_hydrogen_chaos_extinction_summary.csv"
)

OUTPUT_FIGURE = PROJECT / "m51_hydrogen_chaos_extinction_constraint.png"


# ============================================================
# CHAOS REGION
# ============================================================

TARGET_ID = "NGC5194+30.2+2.2"

C_HBETA = 1.032
C_HBETA_SIGMA = 0.013

C_TO_EBV = 1.43
RV = 3.1


def header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================


def main():

    header("M51 HYDROGEN — CHAOS INDEPENDENT EXTINCTION TEST")

    # --------------------------------------------------------
    # 1. IDENTIFY CHAOS REGION
    # --------------------------------------------------------

    header("1. CHAOS REGION")

    print("CHAOS table:")
    print(f"  {CHAOS_TABLE}")

    print("\nTarget:")
    print(f"  {TARGET_ID}")

    matches = []

    with open(CHAOS_TABLE, "r", errors="replace") as f:

        for line_number, line in enumerate(f, start=1):

            if TARGET_ID in line:
                matches.append((line_number, line.rstrip()))

    print(f"\nMatches found: {len(matches)}")

    for line_number, line in matches:
        print(f"\nLine {line_number}:")
        print(f"  {line}")

    if len(matches) != 1:
        raise RuntimeError("Expected exactly one CHAOS match.")

    # --------------------------------------------------------
    # 2. CHAOS EXTINCTION
    # --------------------------------------------------------

    header("2. CHAOS EXTINCTION")

    print(f"C(Hbeta) = " f"{C_HBETA:.3f} +/- " f"{C_HBETA_SIGMA:.3f}")

    ebv = C_HBETA / C_TO_EBV
    ebv_sigma = C_HBETA_SIGMA / C_TO_EBV

    av_chaos = RV * ebv
    av_chaos_sigma = RV * ebv_sigma

    print("\nUsing:")

    print("  C(Hbeta) = 1.43 E(B-V)")

    print("  A(V) = 3.1 E(B-V)")

    print("\nDerived:")

    print(f"  E(B-V) = " f"{ebv:.6f} +/- {ebv_sigma:.6f}")

    print(f"  A(V) = " f"{av_chaos:.6f} +/- " f"{av_chaos_sigma:.6f} mag")

    # --------------------------------------------------------
    # 3. LOAD DATAFRAME
    # --------------------------------------------------------

    header("3. LOADING STOREY-HUMMER DATAFRAME")

    print("File:")
    print(f"  {REQUIRED_AV_TABLE}")

    # THIS IS THE DATAFRAME LOADING STEP.
    df = pd.read_csv(REQUIRED_AV_TABLE)

    print(f"\nRows loaded: {len(df)}")

    print("\nColumns:")

    for column in df.columns:
        print(f"  {column}")

    # --------------------------------------------------------
    # 4. VERIFY REQUIRED COLUMNS
    # --------------------------------------------------------

    header("4. VERIFYING REQUIRED COLUMNS")

    required_columns = [
        "Te",
        "ne",
        "intrinsic_ratio",
        "required_Av",
    ]

    for column in required_columns:

        if column not in df.columns:
            raise RuntimeError(f"Required column missing: {column}")

        print(f"OK: {column}")

    # --------------------------------------------------------
    # 5. CONVERT NUMERIC COLUMNS
    # --------------------------------------------------------

    df["Te"] = pd.to_numeric(df["Te"])

    df["ne"] = pd.to_numeric(df["ne"])

    df["intrinsic_ratio"] = pd.to_numeric(df["intrinsic_ratio"])

    df["required_Av"] = pd.to_numeric(df["required_Av"])

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # From this point onward we use the ACTUAL column name:
    #
    #     required_Av
    #
    # No artificial required_A_V column is needed.
    # --------------------------------------------------------

    # --------------------------------------------------------
    # 6. COMPARE WITH CHAOS
    # --------------------------------------------------------

    header("5. COMPARING REQUIRED A(V) WITH CHAOS")

    df["CHAOS_Av"] = av_chaos

    df["CHAOS_Av_sigma"] = av_chaos_sigma

    df["delta_Av"] = df["required_Av"] - av_chaos

    df["difference_sigma"] = df["delta_Av"] / av_chaos_sigma

    # --------------------------------------------------------
    # 7. RANGE
    # --------------------------------------------------------

    header("6. REQUIRED A(V) RANGE")

    minimum = df.loc[df["required_Av"].idxmin()]

    maximum = df.loc[df["required_Av"].idxmax()]

    print(f"CHAOS A(V): " f"{av_chaos:.3f} +/- " f"{av_chaos_sigma:.3f} mag")

    print("\nStorey-Hummer required A(V):")

    print(f"  Minimum = " f"{minimum['required_Av']:.3f} mag")

    print(f"  Maximum = " f"{maximum['required_Av']:.3f} mag")

    # --------------------------------------------------------
    # 8. MINIMUM SOLUTION
    # --------------------------------------------------------

    header("7. LOWEST REQUIRED-A(V) SOLUTION")

    print(f"Te = {minimum['Te']:.0f} K")

    print(f"ne = {minimum['ne']:.3e} cm^-3")

    print(f"Intrinsic ratio = " f"{minimum['intrinsic_ratio']:.6f}")

    print(f"Required A(V) = " f"{minimum['required_Av']:.6f} mag")

    difference = minimum["required_Av"] - av_chaos

    difference_sigma = difference / av_chaos_sigma

    print(f"\nDifference from CHAOS:")

    print(f"  {difference:.6f} mag")

    print(f"  {difference_sigma:.2f} sigma")

    # --------------------------------------------------------
    # 9. COMPATIBILITY
    # --------------------------------------------------------

    header("8. CHAOS COMPATIBILITY")

    for n_sigma in [1, 2, 3]:

        lower = av_chaos - n_sigma * av_chaos_sigma

        upper = av_chaos + n_sigma * av_chaos_sigma

        compatible = df[(df["required_Av"] >= lower) & (df["required_Av"] <= upper)]

        print(f"\nWithin {n_sigma} sigma:")

        print(f"  A(V) interval = " f"{lower:.3f} - " f"{upper:.3f} mag")

        print(f"  Compatible grid points = " f"{len(compatible)}")

    # --------------------------------------------------------
    # 10. TEMPERATURE SUMMARY
    # --------------------------------------------------------

    header("9. TEMPERATURE SUMMARY")

    summary_rows = []

    for temperature, group in df.groupby("Te"):

        row = group.loc[group["required_Av"].idxmin()]

        minimum_av = row["required_Av"]

        maximum_av = group["required_Av"].max()

        delta = minimum_av - av_chaos

        delta_sigma = delta / av_chaos_sigma

        print(f"\nTe = {temperature:.0f} K")

        print(f"  Required A(V): " f"{minimum_av:.3f} - " f"{maximum_av:.3f} mag")

        print(f"  Minimum at ne = " f"{row['ne']:.3e} cm^-3")

        print(f"  Minimum difference = " f"{delta:.3f} mag")

        print(f"  Difference = " f"{delta_sigma:.2f} sigma")

        summary_rows.append(
            {
                "Te_K": temperature,
                "minimum_required_Av": minimum_av,
                "maximum_required_Av": maximum_av,
                "density_at_minimum_cm3": row["ne"],
                "CHAOS_Av": av_chaos,
                "CHAOS_Av_sigma": av_chaos_sigma,
                "minimum_difference_Av": delta,
                "minimum_difference_sigma": delta_sigma,
            }
        )

    summary = pd.DataFrame(summary_rows)

    # --------------------------------------------------------
    # 11. SAVE TABLES
    # --------------------------------------------------------

    header("10. WRITING OUTPUT TABLES")

    df.to_csv(OUTPUT_TABLE, index=False)

    summary.to_csv(OUTPUT_SUMMARY, index=False)

    print(f"Comparison table:" f"\n  {OUTPUT_TABLE}")

    print(f"\nSummary table:" f"\n  {OUTPUT_SUMMARY}")

    # --------------------------------------------------------
    # 12. FIGURE
    # --------------------------------------------------------

    header("11. CREATING FIGURE")

    fig, ax = plt.subplots(figsize=(10, 7))

    for temperature in sorted(df["Te"].unique()):

        subset = df[df["Te"] == temperature].sort_values("ne")

        ax.plot(
            np.log10(subset["ne"]),
            subset["required_Av"],
            marker="o",
            linewidth=1.5,
            label=f"{temperature:.0f} K",
        )

    ax.axhline(
        av_chaos,
        linewidth=2,
        label=(f"CHAOS: " f"{av_chaos:.2f} +/- " f"{av_chaos_sigma:.2f} mag"),
    )

    ax.axhspan(av_chaos - av_chaos_sigma, av_chaos + av_chaos_sigma, alpha=0.2)

    ax.set_xlabel(r"$\log_{10}(n_e/{\rm cm}^{-3})$")

    ax.set_ylabel(r"Required $A_V$ (mag)")

    ax.set_title("M51 Pa-beta / Pa-gamma\n" "Required Extinction vs. CHAOS")

    ax.grid(True, alpha=0.3)

    ax.legend()

    fig.tight_layout()

    fig.savefig(OUTPUT_FIGURE, dpi=200)

    plt.close(fig)

    print(f"Figure:" f"\n  {OUTPUT_FIGURE}")

    # --------------------------------------------------------
    # 13. FINAL RESULT
    # --------------------------------------------------------

    header("12. FINAL RESULT")

    print(f"CHAOS A(V): " f"{av_chaos:.3f} +/- " f"{av_chaos_sigma:.3f} mag")

    print(f"\nMinimum JWST-required A(V): " f"{minimum['required_Av']:.3f} mag")

    print(f"at Te = {minimum['Te']:.0f} K")

    print(f"and ne = {minimum['ne']:.3e} cm^-3")

    print(f"\nDifference:" f" {difference:.3f} mag")

    print(f"or {difference_sigma:.1f} CHAOS sigma")

    print("\nImportant:")

    print(
        "This is an independent physical-consistency"
        " comparison. The CHAOS optical observations"
        " and JWST 0.45-arcsec extraction do not have"
        " identical spatial apertures."
    )

    print("\nExperiment complete.")


if __name__ == "__main__":
    main()
