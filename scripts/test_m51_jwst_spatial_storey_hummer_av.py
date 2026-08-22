#!/usr/bin/env python3

"""
M51 JWST NIRSpec/IFU — SPATIAL STOREY-HUMMER EXTINCTION TEST

Purpose
-------
Infer spatial A(V) using the full Storey-Hummer Te/ne grid.

The analysis distinguishes:

    1. individual spatial pixels
    2. the integrated JWST 0.45" aperture

The aperture-level ratio is calculated from summed Pa-beta and
Pa-gamma fluxes.

Pixel A(V) values are NOT averaged to obtain the aperture result.

Important:
---------
The nominal JWST extraction aperture is represented by the
"inside_nominal_aperture" column in:

    data/atomic_lines/m51_jwst_extraction_aperture.csv

Exactly 69 pixels are expected.

This script deliberately stops if the aperture mask does not
produce exactly 69 pixels.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales


# ======================================================================
# PATHS
# ======================================================================

PROJECT = Path(__file__).resolve().parents[1]

S3D = (
    PROJECT
    / "data/m51_jwst_level3/"
    / "jw03435-o006_t010_nirspec_g140m-f100lp_s3d.fits"
)

APERTURE_CSV = (
    PROJECT
    / "data/atomic_lines/"
    / "m51_jwst_extraction_aperture.csv"
)

PABETA_MAP = (
    PROJECT
    / "data/atomic_lines/"
    / "m51_1284_pabeta_spatial_line_map.fits"
)

PAGAMMA_MAP = (
    PROJECT
    / "data/atomic_lines/"
    / "m51_1096_pagamma_hydrogen_consistency_map.fits"
)

STOREY_HUMMER = (
    PROJECT
    / "data/atomic_lines/"
    / "m51_hydrogen_required_av.csv"
)

CHAOS_TABLE = (
    PROJECT
    / "data/atomic_lines/chaos_m51/"
    / "table3b.dat"
)

OUTPUT_APERTURE = (
    PROJECT
    / "data/atomic_lines/"
    / "m51_jwst_spatial_storey_hummer_av.csv"
)

OUTPUT_SUMMARY = (
    PROJECT
    / "data/atomic_lines/"
    / "m51_jwst_spatial_storey_hummer_av_summary.csv"
)

OUTPUT_FIGURE = PROJECT / "m51_jwst_spatial_storey_hummer_av.png"


# ======================================================================
# CONSTANTS
# ======================================================================

EXPECTED_APERTURE_PIXELS = 69

PABETA_WAVELENGTH = 1.282
PAGAMMA_WAVELENGTH = 1.094

# CHAOS target identified previously
CHAOS_TARGET = "NGC5194+30.2+2.2"

# C(Hbeta) -> E(B-V) -> A(V)
CHAOS_C_HBETA = 1.032
CHAOS_C_HBETA_SIGMA = 0.013

C_TO_EBV = 1.43
RV = 3.1

CHAOS_EBV = CHAOS_C_HBETA / C_TO_EBV
CHAOS_EBV_SIGMA = CHAOS_C_HBETA_SIGMA / C_TO_EBV

CHAOS_AV = RV * CHAOS_EBV
CHAOS_AV_SIGMA = RV * CHAOS_EBV_SIGMA


# ======================================================================
# EXTINCTION LAW
# ======================================================================

def extinction_coefficients():
    """
    Return A(lambda)/A(V) for Pa-beta and Pa-gamma.

    These are the coefficients used in the previous M51 analysis.
    """

    a_pabeta = 0.270821
    a_pagamma = 0.349594

    return a_pabeta, a_pagamma


# ======================================================================
# REQUIRED A(V)
# ======================================================================

def required_av_array(observed_ratio, intrinsic_ratio):
    """
    Calculate required A(V) for arrays or scalar/array combinations.

    Equation:

        R_obs = R_int * 10^(0.4 [A(Pa-gamma) - A(Pa-beta)])

    Therefore:

        A(V) =
            log10(R_obs / R_int)
            / [0.4 * (a_gamma - a_beta)]

    Parameters
    ----------
    observed_ratio : scalar or array
        Observed Pa-beta / Pa-gamma ratio.

    intrinsic_ratio : scalar or array
        Storey-Hummer intrinsic Pa-beta / Pa-gamma ratio.

    Returns
    -------
    numpy.ndarray
        Required A(V).
    """

    observed_ratio = np.asarray(observed_ratio, dtype=float)
    intrinsic_ratio = np.asarray(intrinsic_ratio, dtype=float)

    a_beta, a_gamma = extinction_coefficients()

    delta_a = a_gamma - a_beta

    with np.errstate(divide="ignore", invalid="ignore"):
        av = (
            np.log10(observed_ratio / intrinsic_ratio)
            / (0.4 * delta_a)
        )

    return av


# ======================================================================
# STOREY-HUMMER GRID
# ======================================================================

def load_storey_hummer():

    print("\n" + "=" * 70)
    print("6. LOADING STOREY-HUMMER GRID")
    print("=" * 70)

    print("File:")
    print(f"  {STOREY_HUMMER}")

    df = pd.read_csv(STOREY_HUMMER)

    print(f"\nRows: {len(df)}")

    required_columns = [
        "Te",
        "ne",
        "intrinsic_ratio",
    ]

    for column in required_columns:

        if column not in df.columns:
            raise RuntimeError(
                f"Required Storey-Hummer column missing: {column}"
            )

        print(f"  OK: {column}")

    df = df[
        np.isfinite(df["Te"])
        & np.isfinite(df["ne"])
        & np.isfinite(df["intrinsic_ratio"])
        & (df["intrinsic_ratio"] > 0)
    ].copy()

    print(f"\nUsable Storey-Hummer grid points: {len(df)}")

    return df


# ======================================================================
# CHAOS
# ======================================================================

def load_chaos_c_hbeta():

    """
    The exact CHAOS value was established previously from table3b.dat.

    This function verifies that the target exists in the local table,
    while using the established C(Hbeta) value above.
    """

    print("\n" + "=" * 70)
    print("CHAOS EXTINCTION")
    print("=" * 70)

    print(f"Target:")
    print(f"  {CHAOS_TARGET}")

    found = False

    if CHAOS_TABLE.exists():

        with open(CHAOS_TABLE, "r") as f:

            for line in f:

                if CHAOS_TARGET in line:

                    print("\nMatching CHAOS record:")
                    print(f"  {line.rstrip()}")

                    found = True
                    break

    if not found:

        print("\nWARNING:")
        print("CHAOS target was not located in table3b.dat.")
        print("Using the previously established C(Hbeta) value.")

    print("\nC(Hbeta):")
    print(
        f"  {CHAOS_C_HBETA:.6f} +/- "
        f"{CHAOS_C_HBETA_SIGMA:.6f}"
    )

    print("\nConversion:")
    print("  C(Hbeta) = 1.43 E(B-V)")
    print("  A(V) = 3.1 E(B-V)")

    print(
        f"\nE(B-V) = {CHAOS_EBV:.6f} +/- "
        f"{CHAOS_EBV_SIGMA:.6f}"
    )

    print(
        f"A(V) = {CHAOS_AV:.6f} +/- "
        f"{CHAOS_AV_SIGMA:.6f} mag"
    )


# ======================================================================
# LOAD APERTURE
# ======================================================================

def load_nominal_aperture():

    print("\n" + "=" * 70)
    print("2. LOADING NOMINAL JWST APERTURE")
    print("=" * 70)

    print("File:")
    print(f"  {APERTURE_CSV}")

    aperture = pd.read_csv(APERTURE_CSV)

    print("\nColumns:")
    for column in aperture.columns:
        print(f"  {column}")

    # --------------------------------------------------------------
    # REQUIRE THE CORRECT MASK COLUMN
    # --------------------------------------------------------------

    mask_column = "inside_nominal_aperture"

    if mask_column not in aperture.columns:

        raise RuntimeError(
            "\nThe aperture file does not contain the required "
            f"'{mask_column}' column.\n\n"
            "This column is required because the aperture CSV "
            "contains all 12,125 S3D pixels."
        )

    # --------------------------------------------------------------
    # APPLY MASK
    # --------------------------------------------------------------

    mask = aperture[mask_column].astype(bool)

    selected = aperture.loc[mask].copy()

    print("\nAll S3D pixels in file:")
    print(f"  {len(aperture)}")

    print("\nPixels inside nominal aperture:")
    print(f"  {len(selected)}")

    # --------------------------------------------------------------
    # HARD SAFETY CHECK
    # --------------------------------------------------------------

    if len(selected) != EXPECTED_APERTURE_PIXELS:

        raise RuntimeError(
            "\nAPERTURE MASK ERROR\n"
            f"Expected exactly {EXPECTED_APERTURE_PIXELS} pixels, "
            f"but found {len(selected)}.\n\n"
            "The analysis has been stopped deliberately so that "
            "the full S3D cube cannot accidentally be treated as "
            "the JWST extraction aperture."
        )

    print(
        f"\nConfirmed: exactly "
        f"{EXPECTED_APERTURE_PIXELS}-pixel nominal aperture."
    )

    # --------------------------------------------------------------
    # REQUIRED PIXEL COLUMNS
    # --------------------------------------------------------------

    required = [
        "x_pixel",
        "y_pixel",
        "ra_deg",
        "dec_deg",
        "dx_arcsec",
        "dy_arcsec",
        "radius_from_center_arcsec",
    ]

    for column in required:

        if column not in selected.columns:

            raise RuntimeError(
                f"Required aperture column missing: {column}"
            )

    return selected


# ======================================================================
# LOAD LINE MAPS
# ======================================================================

def load_line_maps():

    print("\n" + "=" * 70)
    print("4. LOADING PA-BETA / PA-GAMMA MAPS")
    print("=" * 70)

    print("Pa-beta:")
    print(f"  {PABETA_MAP}")

    print("Pa-gamma:")
    print(f"  {PAGAMMA_MAP}")

    with fits.open(PABETA_MAP) as hdul:

        pabeta = np.asarray(hdul[0].data, dtype=float)

    with fits.open(PAGAMMA_MAP) as hdul:

        pagamma = np.asarray(hdul[0].data, dtype=float)

    print("\nPa-beta shape:")
    print(f"  {pabeta.shape}")

    print("Pa-gamma shape:")
    print(f"  {pagamma.shape}")

    if pabeta.shape != pagamma.shape:

        raise RuntimeError(
            "Pa-beta and Pa-gamma maps have different shapes."
        )

    return pabeta, pagamma


# ======================================================================
# EXTRACT APERTURE FLUXES
# ======================================================================

def extract_aperture_fluxes(aperture, pabeta, pagamma):

    print("\n" + "=" * 70)
    print("5. EXTRACTING 69 APERTURE PIXELS")
    print("=" * 70)

    x = aperture["x_pixel"].astype(int).to_numpy()
    y = aperture["y_pixel"].astype(int).to_numpy()

    beta_flux = pabeta[y, x]
    gamma_flux = pagamma[y, x]

    aperture = aperture.copy()

    aperture["pabeta_flux"] = beta_flux
    aperture["pagamma_flux"] = gamma_flux

    return aperture


# ======================================================================
# PIXEL STOREY-HUMMER ANALYSIS
# ======================================================================

def analyze_pixels(aperture, grid):

    print("\n" + "=" * 70)
    print("8. PIXEL-LEVEL STOREY-HUMMER GRID INFERENCE")
    print("=" * 70)

    intrinsic = grid["intrinsic_ratio"].to_numpy()
    te_grid = grid["Te"].to_numpy()
    ne_grid = grid["ne"].to_numpy()

    beta = aperture["pabeta_flux"].to_numpy(float)
    gamma = aperture["pagamma_flux"].to_numpy(float)

    valid = (
        np.isfinite(beta)
        & np.isfinite(gamma)
        & (beta > 0)
        & (gamma > 0)
    )

    aperture["ratio"] = np.nan
    aperture["best_av"] = np.nan
    aperture["best_te"] = np.nan
    aperture["best_ne"] = np.nan
    aperture["best_intrinsic_ratio"] = np.nan
    aperture["grid_residual"] = np.nan

    ratio = np.full(len(aperture), np.nan)

    ratio[valid] = beta[valid] / gamma[valid]

    aperture["ratio"] = ratio

    # --------------------------------------------------------------
    # Best grid solution for each spatial pixel
    # --------------------------------------------------------------

    for i in np.where(valid)[0]:

        r = ratio[i]

        predicted_ratios = intrinsic.copy()

        # We determine the A(V) required at each grid point.
        av_grid = required_av_array(r, predicted_ratios)

        # Reconstruct the ratio exactly at the observed ratio.
        # Therefore A(V) itself is the quantity being inferred.
        #
        # For a purely extinction-derived solution every grid point
        # can produce the observed ratio. We therefore choose the
        # grid point whose inferred A(V) is most physically useful:
        # the grid point nearest the CHAOS A(V).
        #
        # This prevents an arbitrary choice among 91 equivalent
        # extinction solutions.

        distance = np.abs(av_grid - CHAOS_AV)

        j = np.nanargmin(distance)

        aperture.iloc[i, aperture.columns.get_loc("best_av")] = av_grid[j]
        aperture.iloc[i, aperture.columns.get_loc("best_te")] = te_grid[j]
        aperture.iloc[i, aperture.columns.get_loc("best_ne")] = ne_grid[j]
        aperture.iloc[
            i,
            aperture.columns.get_loc("best_intrinsic_ratio")
        ] = intrinsic[j]

        aperture.iloc[
            i,
            aperture.columns.get_loc("grid_residual")
        ] = distance[j]

    print("\nValid positive-ratio pixels:")
    print(f"  {np.sum(valid)} / {len(aperture)}")

    return aperture


# ======================================================================
# CHAOS COMPATIBILITY
# ======================================================================

def chaos_pixel_compatibility(aperture, grid):

    print("\n" + "=" * 70)
    print("10. CHAOS A(V) COMPATIBILITY")
    print("=" * 70)

    intrinsic = grid["intrinsic_ratio"].to_numpy()

    beta = aperture["pabeta_flux"].to_numpy(float)
    gamma = aperture["pagamma_flux"].to_numpy(float)

    valid = (
        np.isfinite(beta)
        & np.isfinite(gamma)
        & (beta > 0)
        & (gamma > 0)
    )

    ratio = np.full(len(aperture), np.nan)

    ratio[valid] = beta[valid] / gamma[valid]

    compatible_1 = np.zeros(len(aperture), dtype=bool)
    compatible_2 = np.zeros(len(aperture), dtype=bool)
    compatible_3 = np.zeros(len(aperture), dtype=bool)

    for i in np.where(valid)[0]:

        av_grid = required_av_array(
            ratio[i],
            intrinsic
        )

        compatible_1[i] = np.any(
            np.abs(av_grid - CHAOS_AV)
            <= CHAOS_AV_SIGMA
        )

        compatible_2[i] = np.any(
            np.abs(av_grid - CHAOS_AV)
            <= 2 * CHAOS_AV_SIGMA
        )

        compatible_3[i] = np.any(
            np.abs(av_grid - CHAOS_AV)
            <= 3 * CHAOS_AV_SIGMA
        )

    aperture["chaos_1sigma"] = compatible_1
    aperture["chaos_2sigma"] = compatible_2
    aperture["chaos_3sigma"] = compatible_3

    print("\nCHAOS A(V):")
    print(
        f"  {CHAOS_AV:.6f} +/- "
        f"{CHAOS_AV_SIGMA:.6f} mag"
    )

    for sigma, column in [
        (1, "chaos_1sigma"),
        (2, "chaos_2sigma"),
        (3, "chaos_3sigma"),
    ]:

        count = int(aperture[column].sum())

        print(f"\n{sigma}-sigma:")
        print(f"  compatible pixels = {count}")

    return aperture


# ======================================================================
# INTEGRATED APERTURE
# ======================================================================

def integrated_aperture_test(aperture, grid):

    print("\n" + "=" * 70)
    print("12. INTEGRATED APERTURE TEST")
    print("=" * 70)

    beta = aperture["pabeta_flux"].to_numpy(float)
    gamma = aperture["pagamma_flux"].to_numpy(float)

    valid = (
        np.isfinite(beta)
        & np.isfinite(gamma)
        & (beta > 0)
        & (gamma > 0)
    )

    total_beta = np.sum(beta[valid])
    total_gamma = np.sum(gamma[valid])

    integrated_ratio = total_beta / total_gamma

    print(f"Total Pa-beta  = {total_beta:.8f}")
    print(f"Total Pa-gamma = {total_gamma:.8f}")

    print(
        f"\nIntegrated Pa-beta / Pa-gamma = "
        f"{integrated_ratio:.8f}"
    )

    # --------------------------------------------------------------
    # Compare integrated ratio against ALL Storey-Hummer grid points
    # --------------------------------------------------------------

    intrinsic = grid["intrinsic_ratio"].to_numpy()
    te = grid["Te"].to_numpy()
    ne = grid["ne"].to_numpy()

    av_grid = required_av_array(
        integrated_ratio,
        intrinsic
    )

    distance = np.abs(av_grid - CHAOS_AV)

    best = np.nanargmin(distance)

    print("\nClosest Storey-Hummer solution to CHAOS A(V):")

    print(f"  Te = {te[best]:.0f} K")
    print(f"  ne = {ne[best]:.3e} cm^-3")
    print(
        f"  intrinsic ratio = "
        f"{intrinsic[best]:.8f}"
    )
    print(
        f"  required A(V) = "
        f"{av_grid[best]:.8f} mag"
    )
    print(
        f"  difference from CHAOS = "
        f"{distance[best]:.8f} mag"
    )

    compatible = (
        np.abs(av_grid - CHAOS_AV)
        <= CHAOS_AV_SIGMA
    )

    print("\n1-sigma Storey-Hummer compatibility:")
    print(
        f"  compatible grid points = "
        f"{np.sum(compatible)}"
    )

    # --------------------------------------------------------------
    # Required A(V) range across full physical grid
    # --------------------------------------------------------------

    print("\nRequired A(V) across Storey-Hummer grid:")
    print(f"  minimum = {np.nanmin(av_grid):.6f} mag")
    print(f"  maximum = {np.nanmax(av_grid):.6f} mag")
    print(f"  median  = {np.nanmedian(av_grid):.6f} mag")

    return {
        "total_beta": total_beta,
        "total_gamma": total_gamma,
        "integrated_ratio": integrated_ratio,
        "best_te": te[best],
        "best_ne": ne[best],
        "best_intrinsic_ratio": intrinsic[best],
        "best_av": av_grid[best],
        "best_av_difference": distance[best],
        "compatible_points_1sigma": int(np.sum(compatible)),
        "grid_av_min": np.nanmin(av_grid),
        "grid_av_max": np.nanmax(av_grid),
        "grid_av_median": np.nanmedian(av_grid),
    }


# ======================================================================
# FLUX WEIGHTING
# ======================================================================

def flux_weighting_analysis(aperture):

    print("\n" + "=" * 70)
    print("13. FLUX WEIGHTING OF CHAOS-COMPATIBLE PIXELS")
    print("=" * 70)

    beta = aperture["pabeta_flux"].to_numpy(float)
    gamma = aperture["pagamma_flux"].to_numpy(float)

    valid = (
        np.isfinite(beta)
        & np.isfinite(gamma)
        & (beta > 0)
        & (gamma > 0)
    )

    total_beta = np.sum(beta[valid])
    total_gamma = np.sum(gamma[valid])

    print("\nTotal aperture flux:")
    print(f"  Pa-beta  = {total_beta:.8f}")
    print(f"  Pa-gamma = {total_gamma:.8f}")

    for sigma, column in [
        (1, "chaos_1sigma"),
        (2, "chaos_2sigma"),
        (3, "chaos_3sigma"),
    ]:

        selected = valid & aperture[column].to_numpy(bool)

        beta_fraction = (
            np.sum(beta[selected]) / total_beta
            if total_beta > 0
            else np.nan
        )

        gamma_fraction = (
            np.sum(gamma[selected]) / total_gamma
            if total_gamma > 0
            else np.nan
        )

        print(f"\n{sigma}-sigma CHAOS-compatible pixels:")

        print(
            f"  pixels = "
            f"{np.sum(selected)}"
        )

        print(
            f"  Pa-beta flux fraction = "
            f"{beta_fraction:.6f}"
        )

        print(
            f"  Pa-gamma flux fraction = "
            f"{gamma_fraction:.6f}"
        )


# ======================================================================
# FIGURE
# ======================================================================

def make_figure(aperture):

    print("\n" + "=" * 70)
    print("14. CREATING SPATIAL A(V) FIGURE")
    print("=" * 70)

    x = aperture["x_pixel"].to_numpy()
    y = aperture["y_pixel"].to_numpy()

    av = aperture["best_av"].to_numpy()

    ratio = aperture["ratio"].to_numpy()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 6)
    )

    # --------------------------------------------------------------
    # A(V)
    # --------------------------------------------------------------

    ax = axes[0]

    scatter = ax.scatter(
        x,
        y,
        c=av,
        s=80,
        marker="s"
    )

    ax.set_title("M51 JWST Spatial A(V)")
    ax.set_xlabel("S3D X pixel")
    ax.set_ylabel("S3D Y pixel")

    cbar = fig.colorbar(
        scatter,
        ax=ax
    )

    cbar.set_label("Required A(V) [mag]")

    # --------------------------------------------------------------
    # Ratio
    # --------------------------------------------------------------

    ax = axes[1]

    scatter = ax.scatter(
        x,
        y,
        c=ratio,
        s=80,
        marker="s"
    )

    ax.set_title("Pa-beta / Pa-gamma")
    ax.set_xlabel("S3D X pixel")
    ax.set_ylabel("S3D Y pixel")

    cbar = fig.colorbar(
        scatter,
        ax=ax
    )

    cbar.set_label("Flux ratio")

    fig.suptitle(
        "M51 JWST NIRSpec/IFU — Storey-Hummer Spatial Analysis"
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_FIGURE,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Saved:")
    print(f"  {OUTPUT_FIGURE}")


# ======================================================================
# SUMMARY
# ======================================================================

def save_summary(aperture, integrated):

    print("\n" + "=" * 70)
    print("15. SAVING RESULTS")
    print("=" * 70)

    aperture.to_csv(
        OUTPUT_APERTURE,
        index=False
    )

    print("Spatial table:")
    print(f"  {OUTPUT_APERTURE}")

    summary = {

        "chaos_target":
            CHAOS_TARGET,

        "chaos_c_hbeta":
            CHAOS_C_HBETA,

        "chaos_c_hbeta_sigma":
            CHAOS_C_HBETA_SIGMA,

        "chaos_av":
            CHAOS_AV,

        "chaos_av_sigma":
            CHAOS_AV_SIGMA,

        "nominal_aperture_pixels":
            EXPECTED_APERTURE_PIXELS,

        "pabeta_total":
            integrated["total_beta"],

        "pagamma_total":
            integrated["total_gamma"],

        "integrated_ratio":
            integrated["integrated_ratio"],

        "best_te":
            integrated["best_te"],

        "best_ne":
            integrated["best_ne"],

        "best_intrinsic_ratio":
            integrated["best_intrinsic_ratio"],

        "integrated_required_av":
            integrated["best_av"],

        "integrated_av_difference_from_chaos":
            integrated["best_av_difference"],

        "storey_hummer_1sigma_compatible_points":
            integrated["compatible_points_1sigma"],

        "storey_hummer_av_min":
            integrated["grid_av_min"],

        "storey_hummer_av_max":
            integrated["grid_av_max"],

        "storey_hummer_av_median":
            integrated["grid_av_median"],
    }

    pd.DataFrame([summary]).to_csv(
        OUTPUT_SUMMARY,
        index=False
    )

    print("\nSummary:")
    print(f"  {OUTPUT_SUMMARY}")


# ======================================================================
# MAIN
# ======================================================================

def main():

    print("=" * 70)
    print("M51 JWST NIRSPEC/IFU — SPATIAL STOREY-HUMMER EXTINCTION TEST")
    print("=" * 70)

    print(
        """
Purpose:
Infer spatial A(V) using the full Storey-Hummer Te/ne grid.

The analysis distinguishes:
  1. individual spatial pixels
  2. the integrated JWST aperture

The aperture-level ratio is calculated from summed fluxes.
Pixel A(V) values are never averaged to obtain the aperture result.

IMPORTANT:
Only the 69 pixels explicitly marked
inside_nominal_aperture=True are analyzed.
"""
    )

    # --------------------------------------------------------------
    # 1. S3D
    # --------------------------------------------------------------

    print("\n" + "=" * 70)
    print("1. READING JWST S3D")
    print("=" * 70)

    print("File:")
    print(f"  {S3D}")

    with fits.open(S3D) as hdul:

        sci = hdul["SCI"]

        shape = sci.data.shape

        wcs = WCS(sci.header)

        scales = (
            proj_plane_pixel_scales(wcs.celestial)
            * 3600.0
        )

    print("\nSCI shape:")
    print(f"  {shape}")

    print("\nSpatial pixel scale:")
    print(
        f"  X = {scales[0]:.6f} arcsec/pixel"
    )

    print(
        f"  Y = {scales[1]:.6f} arcsec/pixel"
    )

    # --------------------------------------------------------------
    # 2. Aperture
    # --------------------------------------------------------------

    aperture = load_nominal_aperture()

    # --------------------------------------------------------------
    # 3. CHAOS
    # --------------------------------------------------------------

    print("\n" + "=" * 70)
    print("3. CHAOS EXTINCTION CONSTRAINT")
    print("=" * 70)

    load_chaos_c_hbeta()

    # --------------------------------------------------------------
    # 4. Line maps
    # --------------------------------------------------------------

    pabeta, pagamma = load_line_maps()

    # --------------------------------------------------------------
    # 5. Flux extraction
    # --------------------------------------------------------------

    aperture = extract_aperture_fluxes(
        aperture,
        pabeta,
        pagamma
    )

    # --------------------------------------------------------------
    # 6. Storey-Hummer
    # --------------------------------------------------------------

    grid = load_storey_hummer()

    # --------------------------------------------------------------
    # 7. Extinction law
    # --------------------------------------------------------------

    print("\n" + "=" * 70)
    print("7. EXTINCTION LAW")
    print("=" * 70)

    a_beta, a_gamma = extinction_coefficients()

    print(
        f"Pa-beta wavelength: "
        f"{PABETA_WAVELENGTH:.3f} um"
    )

    print(
        f"Pa-gamma wavelength: "
        f"{PAGAMMA_WAVELENGTH:.3f} um"
    )

    print(
        f"\nA(Pa-beta)/A(V) = "
        f"{a_beta:.6f}"
    )

    print(
        f"A(Pa-gamma)/A(V) = "
        f"{a_gamma:.6f}"
    )

    print(
        f"\nDelta A / A(V) = "
        f"{a_gamma - a_beta:.6f}"
    )

    # --------------------------------------------------------------
    # 8–9. Pixel inference
    # --------------------------------------------------------------

    aperture = analyze_pixels(
        aperture,
        grid
    )

    # --------------------------------------------------------------
    # 10. CHAOS compatibility
    # --------------------------------------------------------------

    aperture = chaos_pixel_compatibility(
        aperture,
        grid
    )

    # --------------------------------------------------------------
    # 11–12. Integrated aperture
    # --------------------------------------------------------------

    integrated = integrated_aperture_test(
        aperture,
        grid
    )

    # --------------------------------------------------------------
    # 13. Flux weighting
    # --------------------------------------------------------------

    flux_weighting_analysis(
        aperture
    )

    # --------------------------------------------------------------
    # 14. Figure
    # --------------------------------------------------------------

    make_figure(
        aperture
    )

    # --------------------------------------------------------------
    # 15. Save
    # --------------------------------------------------------------

    save_summary(
        aperture,
        integrated
    )

    # --------------------------------------------------------------
    # FINAL
    # --------------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    print(
        f"""
JWST nominal aperture:
  exactly {EXPECTED_APERTURE_PIXELS} S3D pixels

CHAOS:
  A(V) = {CHAOS_AV:.6f} +/- {CHAOS_AV_SIGMA:.6f} mag

Integrated JWST emission:
  Pa-beta  = {integrated["total_beta"]:.8f}
  Pa-gamma = {integrated["total_gamma"]:.8f}

Integrated Pa-beta / Pa-gamma:
  {integrated["integrated_ratio"]:.8f}

Closest Storey-Hummer solution to CHAOS A(V):
  Te = {integrated["best_te"]:.0f} K
  ne = {integrated["best_ne"]:.3e} cm^-3
  intrinsic ratio = {integrated["best_intrinsic_ratio"]:.8f}
  required A(V) = {integrated["best_av"]:.8f} mag

Difference from CHAOS:
  {integrated["best_av_difference"]:.8f} mag

1-sigma compatible Storey-Hummer grid points:
  {integrated["compatible_points_1sigma"]}

Full-grid required A(V):
  minimum = {integrated["grid_av_min"]:.6f} mag
  maximum = {integrated["grid_av_max"]:.6f} mag
  median  = {integrated["grid_av_median"]:.6f} mag

Research outputs:
  {OUTPUT_APERTURE}
  {OUTPUT_SUMMARY}
  {OUTPUT_FIGURE}
"""
    )

    print("Spatial Storey-Hummer extinction experiment complete.")


if __name__ == "__main__":
    main()
