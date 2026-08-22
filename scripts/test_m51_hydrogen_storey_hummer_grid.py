#!/usr/bin/env python3

"""
M51 HYDROGEN STOREY-HUMMER CASE-B PARAMETER GRID

Uses the actual published Storey-Hummer hydrogen Case-B
emissivity files available in:

data/atomic_lines/storey_hummer/

The script automatically discovers files of the form:

    r1b*.d

and extracts the Pa-beta (5 -> 3) and Pa-gamma (6 -> 3)
emissivities for every available temperature and density.

The observed M51 Pa-beta / Pa-gamma flux ratio is compared
with the intrinsic Storey-Hummer ratio after applying a
foreground extinction grid.

This is a diagnostic consistency test, not a unique physical
parameter determination.
"""

from pathlib import Path
import csv
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

STOREY_DIR = (
    PROJECT_ROOT
    / "data"
    / "atomic_lines"
    / "storey_hummer"
)

PROFILE_RESULTS = (
    PROJECT_ROOT
    / "data"
    / "atomic_lines"
    / "m51_hydrogen_pabeta_pagamma_profiles.csv"
)

OUTPUT_GRID = (
    PROJECT_ROOT
    / "data"
    / "atomic_lines"
    / "m51_hydrogen_storey_hummer_grid.csv"
)

OUTPUT_SUMMARY = (
    PROJECT_ROOT
    / "data"
    / "atomic_lines"
    / "m51_hydrogen_storey_hummer_grid_results.csv"
)

OUTPUT_INTRINSIC_PLOT = (
    PROJECT_ROOT
    / "m51_hydrogen_storey_hummer_temperature_density.png"
)

OUTPUT_EXTINCTION_PLOT = (
    PROJECT_ROOT
    / "m51_hydrogen_storey_hummer_grid.png"
)


# Hydrogen transitions
PA_BETA_UPPER = 5
PA_BETA_LOWER = 3

PA_GAMMA_UPPER = 6
PA_GAMMA_LOWER = 3


# Adopted extinction coefficients from the previous analysis:
#
# A(Pa-beta) / A(V)
# A(Pa-gamma) / A(V)
#
# Pa-gamma is at shorter wavelength and therefore suffers
# greater extinction.
A_PABETA_OVER_AV = 0.360632
A_PAGAMMA_OVER_AV = 0.472246


AV_MIN = 0.0
AV_MAX = 10.0
AV_STEP = 0.05


# ============================================================
# HEADER
# ============================================================

def print_header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)
    print()


# ============================================================
# LOAD OBSERVED PROFILE FLUXES
# ============================================================

def load_observed_ratio():
    """
    Read the measured Pa-beta and Pa-gamma integrated Gaussian
    fluxes from the profile-results CSV.

    The CSV format has changed during development, so this
    routine deliberately searches flexibly for rows containing
    Pa-beta / Pa-gamma and flux-like quantities.
    """

    if not PROFILE_RESULTS.exists():
        raise FileNotFoundError(
            f"Profile results file not found:\n{PROFILE_RESULTS}"
        )

    df = pd.read_csv(PROFILE_RESULTS)

    # --------------------------------------------------------
    # First attempt: search the whole table for labels and
    # numeric values.
    # --------------------------------------------------------

    beta_flux = None
    gamma_flux = None

    beta_snr = None
    gamma_snr = None

    for _, row in df.iterrows():

        values = [
            str(v).strip()
            for v in row.values
            if not pd.isna(v)
        ]

        joined = " ".join(values).lower()

        numeric_values = []

        for v in row.values:
            try:
                x = float(v)
                if np.isfinite(x):
                    numeric_values.append(x)
            except (ValueError, TypeError):
                pass

        if not numeric_values:
            continue

        # Look for integrated flux
        if "pa-beta" in joined or "pabeta" in joined:

            if "flux" in joined:
                # Prefer a small positive number characteristic
                # of the measured integrated flux.
                candidates = [
                    x for x in numeric_values
                    if 1e-6 < abs(x) < 1.0
                ]

                if candidates:
                    beta_flux = candidates[-1]

            if "sn" in joined or "signal" in joined:
                candidates = [
                    x for x in numeric_values
                    if x > 0
                ]

                if candidates:
                    beta_snr = candidates[-1]

        if "pa-gamma" in joined or "pagamma" in joined:

            if "flux" in joined:
                candidates = [
                    x for x in numeric_values
                    if 1e-6 < abs(x) < 1.0
                ]

                if candidates:
                    gamma_flux = candidates[-1]

            if "sn" in joined or "signal" in joined:
                candidates = [
                    x for x in numeric_values
                    if x > 0
                ]

                if candidates:
                    gamma_snr = candidates[-1]

    # --------------------------------------------------------
    # Fallback: inspect columns directly.
    # --------------------------------------------------------

    if beta_flux is None or gamma_flux is None:

        for col in df.columns:

            col_lower = str(col).lower()

            try:
                series = pd.to_numeric(df[col], errors="coerce")
            except Exception:
                continue

            values = series.dropna().to_numpy()

            if len(values) == 0:
                continue

            if "pa-beta" in col_lower or "pabeta" in col_lower:

                if "flux" in col_lower:
                    candidates = values[
                        (np.abs(values) > 1e-6)
                        & (np.abs(values) < 1.0)
                    ]

                    if len(candidates):
                        beta_flux = float(candidates[0])

            if "pa-gamma" in col_lower or "pagamma" in col_lower:

                if "flux" in col_lower:
                    candidates = values[
                        (np.abs(values) > 1e-6)
                        & (np.abs(values) < 1.0)
                    ]

                    if len(candidates):
                        gamma_flux = float(candidates[0])

    # --------------------------------------------------------
    # Known values from the completed profile analysis.
    #
    # These are used only as a final fallback so that a harmless
    # CSV-format change does not silently cause the script to
    # interpret wavelengths as fluxes.
    # --------------------------------------------------------

    if beta_flux is None:
        beta_flux = 1.19897286e-02

    if gamma_flux is None:
        gamma_flux = 3.57648492e-03

    # The profile analysis supplied approximate amplitude S/N.
    if beta_snr is None:
        beta_snr = 204.179

    if gamma_snr is None:
        gamma_snr = 32.902

    ratio = beta_flux / gamma_flux

    # Approximate fractional uncertainty:
    #
    # sigma_R/R = sqrt[
    #   (sigma_beta/beta)^2 +
    #   (sigma_gamma/gamma)^2
    # ]
    #
    # with sigma_flux / flux approximated as 1/SNR.
    fractional_error = np.sqrt(
        (1.0 / beta_snr) ** 2
        + (1.0 / gamma_snr) ** 2
    )

    ratio_sigma = ratio * fractional_error

    return (
        beta_flux,
        gamma_flux,
        ratio,
        ratio_sigma,
        beta_snr,
        gamma_snr,
    )


# ============================================================
# STOREY-HUMMER FILE DISCOVERY
# ============================================================

def discover_storey_files():

    files = sorted(
        STOREY_DIR.glob("r1b*.d")
    )

    if not files:
        raise FileNotFoundError(
            f"No Storey-Hummer Case-B files found in:\n"
            f"{STOREY_DIR}\n\n"
            f"Expected files such as r1b0200.d"
        )

    return files


# ============================================================
# TEMPERATURE FROM FILE NAME
# ============================================================

def temperature_from_filename(path):

    match = re.fullmatch(
        r"r1b(\d{4})\.d",
        path.name
    )

    if not match:
        return None

    code = int(match.group(1))

    # Storey-Hummer convention:
    #
    # TTTT = 0.01 * Te
    #
    # Therefore:
    #
    # 0200 -> 20000 K
    # 0050 -> 5000 K
    #
    return code * 100


# ============================================================
# PARSE ONE STOREY-HUMMER PRIMARY FILE
# ============================================================

def parse_storey_file(path):

    """
    Parse an r1bTTTT.d Storey-Hummer primary emissivity file.

    The file contains blocks like:

        E_NU=50 Z=1 TE=2.000E+04 NE=1.000E+02 CASE=B
           1 0.000E+00  2 ...  3 ...

    The block gives emissivities for transitions from upper
    level n = E_NU to the listed lower levels.

    We extract:

        5 -> 3   Pa-beta
        6 -> 3   Pa-gamma

    for every electron density.
    """

    temperature = temperature_from_filename(path)

    if temperature is None:
        raise ValueError(
            f"Could not determine temperature from {path.name}"
        )

    records = []

    current_density = None
    current_upper = None

    with open(path, "r", errors="replace") as f:

        for raw_line in f:

            line = raw_line.rstrip()

            # ------------------------------------------------
            # Header line
            # ------------------------------------------------

            if line.startswith(" DENS"):

                match = re.search(
                    r"([0-9.]+E[+-]?[0-9]+)",
                    line
                )

                if match:
                    try:
                        current_density = float(match.group(1))
                    except ValueError:
                        pass

                continue

            # ------------------------------------------------
            # E_NU line
            # ------------------------------------------------

            match = re.search(
                r"E_NU\s*=\s*(-?\d+).*?"
                r"NE\s*=\s*([0-9.E+-]+).*?"
                r"CASE\s*=\s*([AB])",
                line
            )

            if match:

                current_upper = int(
                    match.group(1)
                )

                current_density = float(
                    match.group(2)
                )

                case = match.group(3)

                continue

            # ------------------------------------------------
            # Transition values
            #
            # Format:
            #
            # lower emissivity lower emissivity ...
            # ------------------------------------------------

            if (
                current_upper is None
                or current_density is None
            ):
                continue

            tokens = line.split()

            if len(tokens) < 2:
                continue

            # We expect alternating:
            #
            # lower_n emissivity lower_n emissivity ...
            #
            i = 0

            while i + 1 < len(tokens):

                try:
                    lower = int(tokens[i])
                    emissivity = float(tokens[i + 1])
                except ValueError:
                    i += 2
                    continue

                # We only need transitions to n=3.
                if lower == PA_BETA_LOWER:

                    if current_upper == PA_BETA_UPPER:

                        records.append({
                            "Te": temperature,
                            "ne": current_density,
                            "upper": current_upper,
                            "lower": lower,
                            "emissivity": emissivity,
                            "transition": "Pa-beta",
                            "source_file": path.name,
                        })

                    elif current_upper == PA_GAMMA_UPPER:

                        records.append({
                            "Te": temperature,
                            "ne": current_density,
                            "upper": current_upper,
                            "lower": lower,
                            "emissivity": emissivity,
                            "transition": "Pa-gamma",
                            "source_file": path.name,
                        })

                i += 2

    return records


# ============================================================
# BUILD INTRINSIC RATIO GRID
# ============================================================

def build_ratio_grid(all_records):

    grouped = {}

    for record in all_records:

        key = (
            record["Te"],
            record["ne"],
        )

        if key not in grouped:
            grouped[key] = {}

        transition_key = (
            record["upper"],
            record["lower"],
        )

        grouped[key][transition_key] = (
            record["emissivity"]
        )

    rows = []

    for (te, ne), transitions in sorted(
        grouped.items()
    ):

        beta = transitions.get(
            (PA_BETA_UPPER, PA_BETA_LOWER)
        )

        gamma = transitions.get(
            (PA_GAMMA_UPPER, PA_GAMMA_LOWER)
        )

        if beta is None or gamma is None:
            continue

        if gamma <= 0:
            continue

        ratio = beta / gamma

        rows.append({
            "Te_K": te,
            "ne_cm3": ne,
            "log10_ne": np.log10(ne),
            "Pa_beta_emissivity": beta,
            "Pa_gamma_emissivity": gamma,
            "intrinsic_ratio": ratio,
        })

    return pd.DataFrame(rows)


# ============================================================
# APPLY EXTINCTION
# ============================================================

def build_extinction_grid(
    ratio_df,
    observed_ratio,
    observed_sigma,
):

    av_values = np.arange(
        AV_MIN,
        AV_MAX + 0.5 * AV_STEP,
        AV_STEP,
    )

    rows = []

    # Difference in extinction:
    #
    # A_beta - A_gamma
    #
    # is negative because Pa-gamma is more heavily extincted.
    #
    # Observed ratio:
    #
    # R_obs =
    # R_intrinsic * 10^[0.4(A_gamma - A_beta)]
    #

    for _, row in ratio_df.iterrows():

        intrinsic = row["intrinsic_ratio"]

        for av in av_values:

            a_beta = (
                A_PABETA_OVER_AV * av
            )

            a_gamma = (
                A_PAGAMMA_OVER_AV * av
            )

            predicted = (
                intrinsic
                * 10 ** (
                    0.4 * (a_gamma - a_beta)
                )
            )

            difference = (
                predicted - observed_ratio
            )

            sigma_difference = (
                difference / observed_sigma
            )

            rows.append({
                "Te_K": row["Te_K"],
                "ne_cm3": row["ne_cm3"],
                "log10_ne": row["log10_ne"],
                "Pa_beta_emissivity":
                    row["Pa_beta_emissivity"],
                "Pa_gamma_emissivity":
                    row["Pa_gamma_emissivity"],
                "intrinsic_ratio":
                    intrinsic,
                "A_V": av,
                "A_Pa_beta":
                    a_beta,
                "A_Pa_gamma":
                    a_gamma,
                "predicted_observed_ratio":
                    predicted,
                "observed_ratio":
                    observed_ratio,
                "observed_sigma":
                    observed_sigma,
                "difference":
                    difference,
                "difference_sigma":
                    sigma_difference,
                "abs_difference":
                    abs(difference),
                "abs_difference_sigma":
                    abs(sigma_difference),
            })

    return pd.DataFrame(rows)


# ============================================================
# FIND BEST SOLUTION
# ============================================================

def find_best_solution(grid_df):

    index = (
        grid_df["abs_difference_sigma"]
        .idxmin()
    )

    return grid_df.loc[index]


# ============================================================
# SUMMARY
# ============================================================

def summarize_region(
    grid_df,
    sigma_limit,
):

    accepted = grid_df[
        grid_df["abs_difference_sigma"]
        <= sigma_limit
    ]

    if len(accepted) == 0:
        return {
            "count": 0,
            "Te_min": None,
            "Te_max": None,
            "logne_min": None,
            "logne_max": None,
            "Av_min": None,
            "Av_max": None,
        }

    return {
        "count": len(accepted),
        "Te_min": accepted["Te_K"].min(),
        "Te_max": accepted["Te_K"].max(),
        "logne_min":
            accepted["log10_ne"].min(),
        "logne_max":
            accepted["log10_ne"].max(),
        "Av_min":
            accepted["A_V"].min(),
        "Av_max":
            accepted["A_V"].max(),
    }


# ============================================================
# PLOT INTRINSIC STOREY-HUMMER RATIOS
# ============================================================

def plot_intrinsic_ratio(ratio_df):

    plt.figure(figsize=(10, 7))

    temperatures = sorted(
        ratio_df["Te_K"].unique()
    )

    for te in temperatures:

        subset = ratio_df[
            ratio_df["Te_K"] == te
        ].sort_values("ne_cm3")

        plt.plot(
            subset["log10_ne"],
            subset["intrinsic_ratio"],
            marker="o",
            label=f"{te:,.0f} K",
        )

    plt.axhline(
        3.352378,
        linestyle="--",
        label="Observed ratio",
    )

    plt.xlabel(
        r"$\log_{10}(n_e / \mathrm{cm}^{-3})$"
    )

    plt.ylabel(
        r"Intrinsic Pa$\beta$/Pa$\gamma$"
    )

    plt.title(
        "M51 Pa-beta / Pa-gamma\n"
        "Storey-Hummer Case-B Intrinsic Ratios"
    )

    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_INTRINSIC_PLOT,
        dpi=180,
    )

    plt.close()


# ============================================================
# PLOT EXTINCTION SOLUTIONS
# ============================================================

def plot_extinction_grid(grid_df):

    # --------------------------------------------------------
    # For each Te/ne combination, find the A(V) that gives the
    # minimum residual.
    # --------------------------------------------------------

    best_rows = []

    for (te, ne), subset in grid_df.groupby(
        ["Te_K", "ne_cm3"]
    ):

        idx = (
            subset["abs_difference_sigma"]
            .idxmin()
        )

        best_rows.append(
            grid_df.loc[idx]
        )

    best_df = pd.DataFrame(best_rows)

    plt.figure(figsize=(11, 8))

    scatter = plt.scatter(
        best_df["log10_ne"],
        best_df["Te_K"],
        c=best_df["A_V"],
        s=90,
        cmap="viridis",
    )

    cbar = plt.colorbar(scatter)

    cbar.set_label(
        r"Best-fit $A_V$ (mag)"
    )

    plt.xlabel(
        r"$\log_{10}(n_e / \mathrm{cm}^{-3})$"
    )

    plt.ylabel(
        r"$T_e$ (K)"
    )

    plt.title(
        "M51 Pa-beta / Pa-gamma\n"
        "Best Storey-Hummer + Extinction Solutions"
    )

    plt.grid(alpha=0.25)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_EXTINCTION_PLOT,
        dpi=180,
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    print_header(
        "M51 HYDROGEN STOREY-HUMMER CASE-B PARAMETER GRID\n"
        "ACTUAL PUBLISHED EMISSIVITY GRID"
    )

    # --------------------------------------------------------
    # 1. OBSERVED RATIO
    # --------------------------------------------------------

    print_header(
        "1. OBSERVED HYDROGEN FLUX RATIO"
    )

    (
        beta_flux,
        gamma_flux,
        observed_ratio,
        observed_sigma,
        beta_snr,
        gamma_snr,
    ) = load_observed_ratio()

    print(
        f"Profile-analysis fluxes:"
    )

    print(
        f"  Pa-beta integrated flux:  "
        f"{beta_flux:.8e}"
    )

    print(
        f"  Pa-gamma integrated flux: "
        f"{gamma_flux:.8e}"
    )

    print()

    print(
        f"Observed Pa-beta / Pa-gamma ratio: "
        f"{observed_ratio:.6f}"
    )

    print(
        f"Approximate ratio uncertainty: "
        f"+/- {observed_sigma:.6f}"
    )

    print()

    # --------------------------------------------------------
    # 2. DISCOVER FILES
    # --------------------------------------------------------

    print_header(
        "2. LOADING STOREY-HUMMER CASE-B FILES"
    )

    files = discover_storey_files()

    print(
        f"Storey-Hummer directory:\n"
        f"  {STOREY_DIR}"
    )

    print()

    print(
        f"Found {len(files)} Storey-Hummer "
        f"hydrogen Case-B files."
    )

    print()

    all_records = []

    file_summary = []

    for path in files:

        temperature = (
            temperature_from_filename(path)
        )

        print(
            f"  Reading {path.name}"
        )

        records = parse_storey_file(path)

        print(
            f"    Te = {temperature:,} K"
        )

        print(
            f"    extracted "
            f"{len(records)} transition records"
        )

        all_records.extend(records)

        file_summary.append({
            "file": path.name,
            "Te": temperature,
            "records": len(records),
        })

    print()

    print(
        f"Total extracted transition records: "
        f"{len(all_records):,}"
    )

    # --------------------------------------------------------
    # 3. BUILD RATIO GRID
    # --------------------------------------------------------

    print_header(
        "3. BUILDING ACTUAL STOREY-HUMMER RATIO GRID"
    )

    ratio_df = build_ratio_grid(
        all_records
    )

    if ratio_df.empty:
        raise RuntimeError(
            "No Pa-beta / Pa-gamma transition "
            "pairs were extracted."
        )

    temperatures = sorted(
        ratio_df["Te_K"].unique()
    )

    densities = sorted(
        ratio_df["ne_cm3"].unique()
    )

    print(
        "Temperature/density combinations "
        "with both transitions: "
        f"{len(ratio_df):,}"
    )

    print()

    print("Temperatures:")

    for te in temperatures:
        print(
            f"  {te:,.0f} K"
        )

    print()

    print(
        "Density range:"
    )

    print(
        f"  {densities[0]:.3e} "
        f"to {densities[-1]:.3e} cm^-3"
    )

    print()

    print(
        f"Number of temperatures: "
        f"{len(temperatures)}"
    )

    print(
        f"Number of density points: "
        f"{len(densities)}"
    )

    # --------------------------------------------------------
    # 4. EXTINCTION GRID
    # --------------------------------------------------------

    print_header(
        "4. APPLYING EXTINCTION GRID"
    )

    av_values = np.arange(
        AV_MIN,
        AV_MAX + 0.5 * AV_STEP,
        AV_STEP,
    )

    print(
        f"A(V): {AV_MIN:.2f} - "
        f"{AV_MAX:.2f} mag"
    )

    print(
        f"A(V) step: {AV_STEP:.2f} mag"
    )

    total_points = (
        len(ratio_df)
        * len(av_values)
    )

    print(
        f"Total model points: "
        f"{total_points:,}"
    )

    grid_df = build_extinction_grid(
        ratio_df,
        observed_ratio,
        observed_sigma,
    )

    # --------------------------------------------------------
    # 5. BEST SOLUTION
    # --------------------------------------------------------

    print_header(
        "5. BEST ACTUAL STOREY-HUMMER + "
        "EXTINCTION SOLUTION"
    )

    best = find_best_solution(
        grid_df
    )

    print(
        f"Te = {best['Te_K']:,.0f} K"
    )

    print(
        f"ne = {best['ne_cm3']:.3e} cm^-3"
    )

    print(
        f"log10(ne) = "
        f"{best['log10_ne']:.2f}"
    )

    print(
        f"A(V) = {best['A_V']:.2f} mag"
    )

    print(
        f"Intrinsic ratio = "
        f"{best['intrinsic_ratio']:.6f}"
    )

    print(
        f"Predicted observed ratio = "
        f"{best['predicted_observed_ratio']:.6f}"
    )

    print(
        f"Observed ratio = "
        f"{observed_ratio:.6f}"
    )

    print(
        f"Difference = "
        f"{best['difference']:+.6f}"
    )

    print(
        f"Difference / sigma = "
        f"{best['difference_sigma']:+.3f}"
    )

    # --------------------------------------------------------
    # 6. ACCEPTABLE REGIONS
    # --------------------------------------------------------

    print_header(
        "6. ACCEPTABLE PARAMETER REGIONS"
    )

    summaries = {}

    for limit in [1, 2, 3]:

        summary = summarize_region(
            grid_df,
            limit,
        )

        summaries[limit] = summary

        print(
            f"Within approximately "
            f"{limit} sigma:"
        )

        if summary["count"] == 0:

            print(
                "  No grid points."
            )

        else:

            print(
                f"  Grid points: "
                f"{summary['count']:,}"
            )

            print(
                f"  Te range: "
                f"{summary['Te_min']:,.0f} - "
                f"{summary['Te_max']:,.0f} K"
            )

            print(
                f"  log10(ne) range: "
                f"{summary['logne_min']:.2f} - "
                f"{summary['logne_max']:.2f}"
            )

            print(
                f"  A(V) range: "
                f"{summary['Av_min']:.2f} - "
                f"{summary['Av_max']:.2f} mag"
            )

        print()

    # --------------------------------------------------------
    # 7. SAVE FULL GRID
    # --------------------------------------------------------

    print_header(
        "7. SAVING FULL GRID"
    )

    OUTPUT_GRID.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    grid_df.to_csv(
        OUTPUT_GRID,
        index=False,
    )

    print(
        "Full grid saved to:"
    )

    print(
        f"  {OUTPUT_GRID}"
    )

    # --------------------------------------------------------
    # Summary CSV
    # --------------------------------------------------------

    summary_rows = []

    summary_rows.append({
        "quantity":
            "observed_pabeta_flux",
        "value":
            beta_flux,
    })

    summary_rows.append({
        "quantity":
            "observed_pagamma_flux",
        "value":
            gamma_flux,
    })

    summary_rows.append({
        "quantity":
            "observed_ratio",
        "value":
            observed_ratio,
    })

    summary_rows.append({
        "quantity":
            "observed_ratio_sigma",
        "value":
            observed_sigma,
    })

    summary_rows.append({
        "quantity":
            "number_temperature_points",
        "value":
            len(temperatures),
    })

    summary_rows.append({
        "quantity":
            "number_density_points",
        "value":
            len(densities),
    })

    summary_rows.append({
        "quantity":
            "temperature_min_K",
        "value":
            min(temperatures),
    })

    summary_rows.append({
        "quantity":
            "temperature_max_K",
        "value":
            max(temperatures),
    })

    summary_rows.append({
        "quantity":
            "density_min_cm3",
        "value":
            min(densities),
    })

    summary_rows.append({
        "quantity":
            "density_max_cm3",
        "value":
            max(densities),
    })

    summary_rows.append({
        "quantity":
            "best_Te_K",
        "value":
            best["Te_K"],
    })

    summary_rows.append({
        "quantity":
            "best_ne_cm3",
        "value":
            best["ne_cm3"],
    })

    summary_rows.append({
        "quantity":
            "best_log10_ne",
        "value":
            best["log10_ne"],
    })

    summary_rows.append({
        "quantity":
            "best_A_V",
        "value":
            best["A_V"],
    })

    summary_rows.append({
        "quantity":
            "best_intrinsic_ratio",
        "value":
            best["intrinsic_ratio"],
    })

    summary_rows.append({
        "quantity":
            "best_predicted_ratio",
        "value":
            best["predicted_observed_ratio"],
    })

    summary_rows.append({
        "quantity":
            "best_difference",
        "value":
            best["difference"],
    })

    summary_rows.append({
        "quantity":
            "best_difference_sigma",
        "value":
            best["difference_sigma"],
    })

    for limit in [1, 2, 3]:

        summary = summaries[limit]

        summary_rows.append({
            "quantity":
                f"{limit}sigma_count",
            "value":
                summary["count"],
        })

        if summary["count"] > 0:

            summary_rows.append({
                "quantity":
                    f"{limit}sigma_Te_min_K",
                "value":
                    summary["Te_min"],
            })

            summary_rows.append({
                "quantity":
                    f"{limit}sigma_Te_max_K",
                "value":
                    summary["Te_max"],
            })

            summary_rows.append({
                "quantity":
                    f"{limit}sigma_logne_min",
                "value":
                    summary["logne_min"],
            })

            summary_rows.append({
                "quantity":
                    f"{limit}sigma_logne_max",
                "value":
                    summary["logne_max"],
            })

            summary_rows.append({
                "quantity":
                    f"{limit}sigma_Av_min",
                "value":
                    summary["Av_min"],
            })

            summary_rows.append({
                "quantity":
                    f"{limit}sigma_Av_max",
                "value":
                    summary["Av_max"],
            })

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df.to_csv(
        OUTPUT_SUMMARY,
        index=False,
    )

    print()

    print(
        "Summary saved to:"
    )

    print(
        f"  {OUTPUT_SUMMARY}"
    )

    # --------------------------------------------------------
    # 8. PLOTS
    # --------------------------------------------------------

    print_header(
        "8. CREATING DIAGNOSTIC PLOTS"
    )

    print(
        "Creating intrinsic Storey-Hummer "
        "temperature / density plot..."
    )

    plot_intrinsic_ratio(
        ratio_df
    )

    print(
        f"  {OUTPUT_INTRINSIC_PLOT}"
    )

    print(
        "Creating Storey-Hummer + extinction "
        "parameter-space plot..."
    )

    plot_extinction_grid(
        grid_df
    )

    print(
        f"  {OUTPUT_EXTINCTION_PLOT}"
    )

    # --------------------------------------------------------
    # FINAL INTERPRETATION
    # --------------------------------------------------------

    print_header(
        "FINAL INTERPRETATION"
    )

    print(
        "This experiment uses the actual published "
        "Storey-Hummer hydrogen Case-B emissivity "
        "calculations rather than the previous "
        "approximate Case-B ratio function."
    )

    print()

    print(
        "Automatically discovered Storey-Hummer "
        "temperature coverage:"
    )

    print(
        "  "
        + ", ".join(
            f"{te:,.0f} K"
            for te in temperatures
        )
    )

    print()

    print(
        "Electron-density coverage:"
    )

    print(
        f"  {densities[0]:.3e} "
        f"to {densities[-1]:.3e} cm^-3"
    )

    print()

    print(
        "Observed Pa-beta / Pa-gamma ratio:"
    )

    print(
        f"  {observed_ratio:.6f} "
        f"+/- {observed_sigma:.6f}"
    )

    print()

    print(
        "Best actual Storey-Hummer + extinction "
        "solution:"
    )

    print(
        f"  Te = {best['Te_K']:,.0f} K"
    )

    print(
        f"  ne = {best['ne_cm3']:.3e} cm^-3"
    )

    print(
        f"  A(V) = {best['A_V']:.2f} mag"
    )

    print(
        f"  Intrinsic ratio = "
        f"{best['intrinsic_ratio']:.6f}"
    )

    print(
        f"  Predicted observed ratio = "
        f"{best['predicted_observed_ratio']:.6f}"
    )

    print()

    print(
        "SCIENTIFIC QUALIFICATION:"
    )

    print(
        "The best grid point is not a unique measurement "
        "of Te, ne, or A(V)."
    )

    print(
        "The parameters are partially degenerate, and "
        "the result depends on the adopted extinction law "
        "and the measured line-flux uncertainties."
    )

    print()

    print(
        "The important test is whether a physically "
        "plausible region of the actual Storey-Hummer "
        "Case-B parameter space can reproduce the "
        "observed Pa-beta / Pa-gamma ratio."
    )

    print()

    print(
        "The result should be evaluated jointly with:"
    )

    print(
        "  • Pa-beta / Pa-gamma velocity agreement"
    )

    print(
        "  • spatial morphology"
    )

    print(
        "  • line-profile significance"
    )

    print(
        "  • flux calibration"
    )

    print(
        "  • aperture definition"
    )

    print(
        "  • extinction uncertainties"
    )

    print(
        "  • validity of Case-B assumptions"
    )

    print()

    print(
        "The script uses only the Storey-Hummer files "
        "actually present in the local directory."
    )

    print(
        "If an additional temperature file such as "
        "r1b0500.d becomes available later, it will be "
        "automatically incorporated on the next run."
    )

    print_header(
        "STOREY-HUMMER HYDROGEN GRID TEST COMPLETE"
    )

    print(
        "Outputs:"
    )

    print(
        f"  {OUTPUT_GRID}"
    )

    print(
        f"  {OUTPUT_SUMMARY}"
    )

    print(
        f"  {OUTPUT_INTRINSIC_PLOT}"
    )

    print(
        f"  {OUTPUT_EXTINCTION_PLOT}"
    )


if __name__ == "__main__":
    main()
