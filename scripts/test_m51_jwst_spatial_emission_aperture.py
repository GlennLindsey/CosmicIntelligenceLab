#!/usr/bin/env python3

"""
M51 JWST NIRSpec/IFU — SPATIAL EMISSION APERTURE ANALYSIS

Purpose
-------
Map the independently identified CHAOS H II region onto the JWST
NIRSpec/IFU S3D cube and quantify the physical emission contained
within the nominal JWST 0.45 arcsec extraction aperture.

This experiment measures:

1. The 69 S3D spatial pixels in the nominal JWST aperture.
2. Pa-beta emission in those pixels.
3. Pa-gamma emission in those pixels.
4. Spatial distribution of Pa-beta / Pa-gamma.
5. Concentration of emission around the CHAOS position.
6. Whether the integrated ratio is dominated by a small subregion.
7. Whether A(V) ~= 2.24 can reproduce the observed integrated ratio.

Important
---------
The nominal aperture follows the recovered JWST CRDS EXTRACT1D
reference:

    model_type = Extract1dIFUModel
    method     = center
    radius     = 0.45 arcsec
    subpixels  = 5

For method="center", a spatial pixel is included when its center
falls inside the circular aperture.

This script does NOT claim to reproduce JWST's detailed effective
pixel weighting or aperture correction.

The line maps used here are the previously generated, continuum-
subtracted JWST spatial line maps:

    m51_1284_pabeta_spatial_line_map.fits
    m51_1096_pagamma_hydrogen_consistency_map.fits

If the line-map products are unavailable, the script stops rather
than silently substituting another product.

Outputs
-------
CSV:
    data/atomic_lines/m51_jwst_spatial_emission_aperture.csv
    data/atomic_lines/m51_jwst_spatial_emission_summary.csv

Figures:
    m51_jwst_spatial_emission_aperture.png
    m51_jwst_spatial_ratio_av.png
    m51_jwst_spatial_flux_concentration.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u


# ============================================================
# PATHS
# ============================================================

ROOT = Path("/home/glenn/Projects/cosmic_ai")

S3D_FILE = (
    ROOT
    / "data/m51_jwst_level3/"
    / "jw03435-o006_t010_nirspec_g140m-f100lp_s3d.fits"
)

PABETA_MAP = (
    ROOT
    / "data/atomic_lines/"
    / "m51_1284_pabeta_spatial_line_map.fits"
)

PAGAMMA_MAP = (
    ROOT
    / "data/atomic_lines/"
    / "m51_1096_pagamma_hydrogen_consistency_map.fits"
)

REQUIRED_AV = (
    ROOT
    / "data/atomic_lines/"
    / "m51_hydrogen_required_av.csv"
)

OUTPUT_TABLE = (
    ROOT
    / "data/atomic_lines/"
    / "m51_jwst_spatial_emission_aperture.csv"
)

OUTPUT_SUMMARY = (
    ROOT
    / "data/atomic_lines/"
    / "m51_jwst_spatial_emission_summary.csv"
)

FIG_APERTURE = ROOT / "m51_jwst_spatial_emission_aperture.png"
FIG_RATIO = ROOT / "m51_jwst_spatial_ratio_av.png"
FIG_CONCENTRATION = ROOT / "m51_jwst_spatial_flux_concentration.png"


# ============================================================
# KNOWN M51 / JWST PARAMETERS
# ============================================================

CHAOS_NAME = "NGC5194+30.2+2.2"

CHAOS_RA = 202.4820833333
CHAOS_DEC = 47.1959000000

JWST_EXTR_X = 62.0
JWST_EXTR_Y = 48.0

APERTURE_RADIUS_ARCSEC = 0.450000

CHAOS_AV = 2.237203
CHAOS_AV_SIGMA = 0.028182

# Extinction coefficients already established in the project.
A_PABETA_OVER_AV = 0.360632
A_PAGAMMA_OVER_AV = 0.472246

DIFFERENTIAL_EXTINCTION = (
    A_PAGAMMA_OVER_AV - A_PABETA_OVER_AV
)


# ============================================================
# HELPERS
# ============================================================

def print_header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def load_fits_image(path):
    """
    Load the first useful 2-D image from a FITS file.

    Returns
    -------
    data : ndarray
    header : Header
    """
    with fits.open(path) as hdul:

        for hdu in hdul:

            if hdu.data is None:
                continue

            data = np.asarray(hdu.data)

            if data.ndim == 2:
                return data.astype(float), hdu.header.copy()

        raise RuntimeError(
            f"No 2-D image found in {path}"
        )


def find_spatial_wcs(path):
    """
    Recover the celestial WCS from the S3D SCI extension.
    """
    with fits.open(path) as hdul:

        sci = hdul["SCI"]

        return (
            np.asarray(sci.data),
            WCS(sci.header),
            sci.header.copy(),
        )


def required_av_from_ratio(
    intrinsic_ratio,
    observed_ratio,
    a_beta=A_PABETA_OVER_AV,
    a_gamma=A_PAGAMMA_OVER_AV,
):
    """
    Solve:

        R_obs = R_intrinsic *
                10^(0.4 [A_gamma - A_beta])

    for A(V).

    Therefore:

        A(V) =
            log10(R_obs/R_intrinsic)
            / [0.4 (A_gamma/A_V - A_beta/A_V)]
    """

    intrinsic_ratio = np.asarray(
        intrinsic_ratio,
        dtype=float,
    )

    observed_ratio = np.asarray(
        observed_ratio,
        dtype=float,
    )

    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):

        av = (
            np.log10(
                observed_ratio / intrinsic_ratio
            )
            /
            (
                0.4
                * (
                    a_gamma
                    - a_beta
                )
            )
        )

    return av


def predicted_ratio_from_av(
    intrinsic_ratio,
    av,
    a_beta=A_PABETA_OVER_AV,
    a_gamma=A_PAGAMMA_OVER_AV,
):
    """
    Predict observed Pa-beta / Pa-gamma ratio for a given
    intrinsic ratio and foreground A(V).
    """

    return (
        intrinsic_ratio
        *
        10.0
        ** (
            0.4
            * (a_gamma - a_beta)
            * av
        )
    )


def percentile(values, p):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan

    return np.percentile(values, p)


# ============================================================
# MAIN
# ============================================================

def main():

    print_header(
        "M51 JWST NIRSPEC/IFU — "
        "SPATIAL EMISSION APERTURE ANALYSIS"
    )

    print(
        """
Purpose:
Quantify the actual Pa-beta and Pa-gamma emission contained
within the nominal JWST 0.45 arcsec extraction aperture and
test whether spatial structure can reconcile the JWST ratio
with the independent CHAOS A(V) = 2.24 mag constraint.
"""
    )

    # --------------------------------------------------------
    # 1. READ S3D
    # --------------------------------------------------------

    print_header("1. READING JWST S3D CUBE")

    cube, cube_wcs, cube_header = find_spatial_wcs(
        S3D_FILE
    )

    spectral, ny, nx = cube.shape

    print("File:")
    print(f"  {S3D_FILE}")

    print("\nSCI shape:")
    print(f"  spectral = {spectral}")
    print(f"  ny       = {ny}")
    print(f"  nx       = {nx}")

    celestial_wcs = cube_wcs.celestial

    # Pixel scale in arcsec.
    try:
        from astropy.wcs.utils import proj_plane_pixel_scales

        scales = (
            proj_plane_pixel_scales(celestial_wcs)
            * 3600.0
        )

        scale_x = float(scales[0])
        scale_y = float(scales[1])

    except Exception:

        scale_x = 0.1
        scale_y = 0.1

    print("\nSpatial pixel scale:")
    print(f"  X = {scale_x:.6f} arcsec/pixel")
    print(f"  Y = {scale_y:.6f} arcsec/pixel")

    # --------------------------------------------------------
    # 2. APERTURE MASK
    # --------------------------------------------------------

    print_header(
        "2. BUILDING NOMINAL JWST APERTURE MASK"
    )

    y_grid, x_grid = np.indices(
        (ny, nx)
    )

    radius_x = APERTURE_RADIUS_ARCSEC / scale_x
    radius_y = APERTURE_RADIUS_ARCSEC / scale_y

    # Center method:
    # include pixel if its CENTER lies inside the aperture.
    distance_arcsec = np.sqrt(
        (
            (x_grid - JWST_EXTR_X)
            * scale_x
        ) ** 2
        +
        (
            (y_grid - JWST_EXTR_Y)
            * scale_y
        ) ** 2
    )

    aperture_mask = (
        distance_arcsec
        <= APERTURE_RADIUS_ARCSEC
    )

    aperture_indices = np.where(
        aperture_mask
    )

    n_aperture_pixels = len(
        aperture_indices[0]
    )

    print("Extraction center:")
    print(
        f"  x = {JWST_EXTR_X:.3f}"
    )
    print(
        f"  y = {JWST_EXTR_Y:.3f}"
    )

    print("\nNominal radius:")
    print(
        f"  {APERTURE_RADIUS_ARCSEC:.6f} arcsec"
    )

    print("\nRadius in pixels:")
    print(
        f"  X = {radius_x:.3f}"
    )
    print(
        f"  Y = {radius_y:.3f}"
    )

    print("\nPixels inside aperture:")
    print(
        f"  {n_aperture_pixels}"
    )

    if n_aperture_pixels != 69:

        print(
            "\nWARNING:"
            f" expected 69 pixels but found "
            f"{n_aperture_pixels}."
        )

    else:

        print(
            "  Confirmed: 69-pixel nominal "
            "center-method aperture."
        )

    # --------------------------------------------------------
    # 3. SKY COORDINATES
    # --------------------------------------------------------

    print_header(
        "3. CALCULATING SPATIAL SKY COORDINATES"
    )

    ra_grid, dec_grid = celestial_wcs.pixel_to_world_values(
        x_grid,
        y_grid,
    )

    chaos_coord = SkyCoord(
        CHAOS_RA * u.deg,
        CHAOS_DEC * u.deg,
        frame="icrs",
    )

    pixel_coords = SkyCoord(
        ra_grid * u.deg,
        dec_grid * u.deg,
        frame="icrs",
    )

    chaos_separation = (
        pixel_coords.separation(
            chaos_coord
        ).arcsec
    )

    # --------------------------------------------------------
    # 4. LOAD LINE MAPS
    # --------------------------------------------------------

    print_header(
        "4. LOADING PA-BETA / PA-GAMMA LINE MAPS"
    )

    print("Pa-beta map:")
    print(f"  {PABETA_MAP}")

    pabeta, pabeta_header = load_fits_image(
        PABETA_MAP
    )

    print(
        f"  shape = {pabeta.shape}"
    )

    print("\nPa-gamma map:")
    print(f"  {PAGAMMA_MAP}")

    pagamma, pagamma_header = load_fits_image(
        PAGAMMA_MAP
    )

    print(
        f"  shape = {pagamma.shape}"
    )

    if pabeta.shape != (ny, nx):

        raise RuntimeError(
            "Pa-beta map shape does not match "
            "the S3D spatial dimensions."
        )

    if pagamma.shape != (ny, nx):

        raise RuntimeError(
            "Pa-gamma map shape does not match "
            "the S3D spatial dimensions."
        )

    # --------------------------------------------------------
    # 5. EXTRACT 69 PIXELS
    # --------------------------------------------------------

    print_header(
        "5. EXTRACTING THE 69 APERTURE PIXELS"
    )

    ys = aperture_indices[0]
    xs = aperture_indices[1]

    rows = []

    for y, x in zip(ys, xs):

        rows.append(
            {
                "x": int(x),
                "y": int(y),

                "ra_deg": float(
                    ra_grid[y, x]
                ),

                "dec_deg": float(
                    dec_grid[y, x]
                ),

                "radius_arcsec": float(
                    distance_arcsec[y, x]
                ),

                "distance_from_chaos_arcsec":
                    float(
                        chaos_separation[y, x]
                    ),

                "pabeta_flux":
                    float(pabeta[y, x]),

                "pagamma_flux":
                    float(pagamma[y, x]),
            }
        )

    df = pd.DataFrame(rows)

    # --------------------------------------------------------
    # 6. PIXEL RATIOS
    # --------------------------------------------------------

    print_header(
        "6. CALCULATING SPATIAL PA-BETA / PA-GAMMA"
    )

    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):

        df["pabeta_pagamma_ratio"] = (
            df["pabeta_flux"]
            /
            df["pagamma_flux"]
        )

    # Positive, finite emission only.
    df["valid_ratio"] = (
        np.isfinite(
            df["pabeta_pagamma_ratio"]
        )
        &
        (df["pabeta_flux"] > 0)
        &
        (df["pagamma_flux"] > 0)
    )

    print(
        "Valid positive-ratio pixels:"
    )
    print(
        f"  {df['valid_ratio'].sum()} "
        f"/ {len(df)}"
    )

    valid = df["valid_ratio"]

    if valid.any():

        ratios = df.loc[
            valid,
            "pabeta_pagamma_ratio"
        ]

        print("\nSpatial ratio statistics:")
        print(
            f"  minimum = {ratios.min():.6f}"
        )
        print(
            f"  maximum = {ratios.max():.6f}"
        )
        print(
            f"  median  = {ratios.median():.6f}"
        )

    # --------------------------------------------------------
    # 7. INTEGRATED APERTURE RATIO
    # --------------------------------------------------------

    print_header(
        "7. INTEGRATED APERTURE EMISSION"
    )

    total_pabeta = (
        df["pabeta_flux"]
        .replace([np.inf, -np.inf], np.nan)
        .sum()
    )

    total_pagamma = (
        df["pagamma_flux"]
        .replace([np.inf, -np.inf], np.nan)
        .sum()
    )

    integrated_ratio = (
        total_pabeta
        /
        total_pagamma
    )

    print(
        f"Total Pa-beta = "
        f"{total_pabeta:.8g}"
    )

    print(
        f"Total Pa-gamma = "
        f"{total_pagamma:.8g}"
    )

    print(
        f"\nIntegrated Pa-beta / Pa-gamma = "
        f"{integrated_ratio:.8f}"
    )

    # --------------------------------------------------------
    # 8. EMISSION CONCENTRATION
    # --------------------------------------------------------

    print_header(
        "8. EMISSION CONCENTRATION"
    )

    df["pabeta_flux_fraction"] = (
        df["pabeta_flux"]
        /
        total_pabeta
    )

    df["pagamma_flux_fraction"] = (
        df["pagamma_flux"]
        /
        total_pagamma
    )

    pabeta_sorted = df.sort_values(
        "pabeta_flux",
        ascending=False,
    ).reset_index(drop=True)

    pagamma_sorted = df.sort_values(
        "pagamma_flux",
        ascending=False,
    ).reset_index(drop=True)

    print("Pa-beta cumulative concentration:")

    for n in [1, 3, 5, 10, 20]:

        n = min(
            n,
            len(pabeta_sorted)
        )

        frac = (
            pabeta_sorted
            .iloc[:n]["pabeta_flux"]
            .sum()
            /
            total_pabeta
        )

        print(
            f"  brightest {n:2d} pixels:"
            f" {frac:.4f}"
            f" ({100*frac:.2f}%)"
        )

    print("\nPa-gamma cumulative concentration:")

    for n in [1, 3, 5, 10, 20]:

        n = min(
            n,
            len(pagamma_sorted)
        )

        frac = (
            pagamma_sorted
            .iloc[:n]["pagamma_flux"]
            .sum()
            /
            total_pagamma
        )

        print(
            f"  brightest {n:2d} pixels:"
            f" {frac:.4f}"
            f" ({100*frac:.2f}%)"
        )

    # --------------------------------------------------------
    # 9. CHAOS PROXIMITY / FLUX CONCENTRATION
    # --------------------------------------------------------

    print_header(
        "9. EMISSION CONCENTRATION AROUND CHAOS"
    )

    for radius in [
        0.10,
        0.20,
        0.30,
        0.45,
    ]:

        inside = (
            df["distance_from_chaos_arcsec"]
            <= radius
        )

        beta_frac = (
            df.loc[
                inside,
                "pabeta_flux"
            ].sum()
            /
            total_pabeta
        )

        gamma_frac = (
            df.loc[
                inside,
                "pagamma_flux"
            ].sum()
            /
            total_pagamma
        )

        print(
            f"\nWithin {radius:.2f} arcsec of CHAOS:"
        )

        print(
            f"  pixels = {inside.sum()}"
        )

        print(
            f"  Pa-beta flux fraction = "
            f"{beta_frac:.4f}"
            f" ({100*beta_frac:.2f}%)"
        )

        print(
            f"  Pa-gamma flux fraction = "
            f"{gamma_frac:.4f}"
            f" ({100*gamma_frac:.2f}%)"
        )

    # --------------------------------------------------------
    # 10. REQUIRED A(V) FOR EACH PIXEL
    # --------------------------------------------------------

    print_header(
        "10. SPATIAL EXTINCTION REQUIREMENTS"
    )

    df["required_Av"] = np.nan

    df.loc[valid, "required_Av"] = (
        required_av_from_ratio(
            intrinsic_ratio=2.0,
            observed_ratio=df.loc[
                valid,
                "pabeta_pagamma_ratio"
            ],
        )
    )

    # IMPORTANT:
    #
    # The above is NOT a physical Case-B calculation because
    # the intrinsic ratio depends on Te/ne.
    #
    # Therefore this column is intentionally labelled only as
    # an illustrative "relative extinction indicator" below.
    #
    # We will use the Storey-Hummer grid for the physically
    # meaningful A(V) test.

    # Relative extinction indicator:
    #
    # At A(V)=0, an intrinsic ratio of 2.0 is adopted only
    # as a diagnostic normalization.
    #
    # Rename to avoid scientific ambiguity.
    df.rename(
        columns={
            "required_Av":
                "diagnostic_Av_for_intrinsic_ratio_2p0"
        },
        inplace=True,
    )

    # --------------------------------------------------------
    # 11. STOREY-HUMMER GRID
    # --------------------------------------------------------

    print_header(
        "11. LOADING STOREY-HUMMER GRID"
    )

    sh = pd.read_csv(
        REQUIRED_AV
    )

    print("File:")
    print(f"  {REQUIRED_AV}")

    print(
        f"\nRows = {len(sh)}"
    )

    required_columns = [
        "Te",
        "ne",
        "intrinsic_ratio",
        "required_Av",
    ]

    missing = [
        c
        for c in required_columns
        if c not in sh.columns
    ]

    if missing:

        raise RuntimeError(
            "Missing Storey-Hummer columns: "
            + ", ".join(missing)
        )

    # --------------------------------------------------------
    # 12. CHAOS A(V) PREDICTION ACROSS GRID
    # --------------------------------------------------------

    print_header(
        "12. TESTING CHAOS A(V) AGAINST STOREY-HUMMER"
    )

    sh["predicted_ratio_at_chaos_Av"] = (
        predicted_ratio_from_av(
            sh["intrinsic_ratio"],
            CHAOS_AV,
        )
    )

    sh["difference_from_integrated_ratio"] = (
        sh["predicted_ratio_at_chaos_Av"]
        -
        integrated_ratio
    )

    sh["abs_difference"] = (
        np.abs(
            sh["difference_from_integrated_ratio"]
        )
    )

    best = sh.loc[
        sh["abs_difference"].idxmin()
    ]

    print(
        f"CHAOS A(V) = "
        f"{CHAOS_AV:.6f} +/- "
        f"{CHAOS_AV_SIGMA:.6f} mag"
    )

    print(
        f"\nIntegrated JWST ratio = "
        f"{integrated_ratio:.8f}"
    )

    print("\nClosest Storey-Hummer solution:")
    print(
        f"  Te = {best['Te']:.0f} K"
    )
    print(
        f"  ne = {best['ne']:.3e} cm^-3"
    )
    print(
        f"  intrinsic ratio = "
        f"{best['intrinsic_ratio']:.6f}"
    )
    print(
        f"  predicted ratio at CHAOS A(V) = "
        f"{best['predicted_ratio_at_chaos_Av']:.8f}"
    )
    print(
        f"  difference = "
        f"{best['difference_from_integrated_ratio']:.8f}"
    )

    # --------------------------------------------------------
    # 13. CAN A(V)=2.24 EXPLAIN THE OBSERVED RATIO?
    # --------------------------------------------------------

    print_header(
        "13. CAN CHAOS A(V) REPRODUCE THE JWST RATIO?"
    )

    print(
        "The question is tested across every "
        "Storey-Hummer Te/ne grid point."
    )

    compatible_1sigma = sh[
        np.abs(
            sh[
                "difference_from_integrated_ratio"
            ]
        )
        <= 0.103204
    ]

    print(
        "\nApproximate 1-sigma observed-ratio "
        "compatibility:"
    )

    print(
        f"  compatible grid points = "
        f"{len(compatible_1sigma)}"
    )

    if len(compatible_1sigma) > 0:

        print(
            "\nYes — some physical conditions "
            "can reproduce the JWST ratio "
            "with CHAOS A(V)."
        )

    else:

        print(
            "\nNo Storey-Hummer grid point "
            "reproduces the integrated JWST ratio "
            "within the observed 1-sigma ratio "
            "uncertainty at CHAOS A(V)."
        )

    # --------------------------------------------------------
    # 14. SPATIAL WEIGHTING TEST
    # --------------------------------------------------------

    print_header(
        "14. SPATIAL WEIGHTING TEST"
    )

    # Flux-weighted mean of individual ratios.
    valid_df = df.loc[
        valid
    ].copy()

    if len(valid_df) > 0:

        valid_df["beta_weight"] = (
            valid_df["pabeta_flux"]
            /
            valid_df["pabeta_flux"].sum()
        )

        flux_weighted_ratio = (
            (
                valid_df[
                    "pabeta_pagamma_ratio"
                ]
                *
                valid_df[
                    "beta_weight"
                ]
            )
            .sum()
        )

    else:

        flux_weighted_ratio = np.nan

    print(
        f"Integrated flux ratio = "
        f"{integrated_ratio:.8f}"
    )

    print(
        f"Pa-beta-flux-weighted mean "
        f"pixel ratio = "
        f"{flux_weighted_ratio:.8f}"
    )

    print(
        "\nImportant:"
    )

    print(
        "The integrated flux ratio is the "
        "physically relevant aperture ratio."
    )

    print(
        "The arithmetic mean or flux-weighted "
        "mean of pixel ratios is not equivalent "
        "to the ratio of integrated fluxes."
    )

    # --------------------------------------------------------
    # 15. WRITE PIXEL TABLE
    # --------------------------------------------------------

    print_header(
        "15. SAVING SPATIAL APERTURE TABLE"
    )

    df.to_csv(
        OUTPUT_TABLE,
        index=False,
    )

    print(
        f"Saved:\n  {OUTPUT_TABLE}"
    )

    # --------------------------------------------------------
    # 16. SUMMARY TABLE
    # --------------------------------------------------------

    print_header(
        "16. SAVING SUMMARY"
    )

    brightest_1_beta = (
        pabeta_sorted
        .iloc[0]["pabeta_flux"]
        /
        total_pabeta
    )

    brightest_5_beta = (
        pabeta_sorted
        .iloc[:5]["pabeta_flux"]
        .sum()
        /
        total_pabeta
    )

    brightest_10_beta = (
        pabeta_sorted
        .iloc[:10]["pabeta_flux"]
        .sum()
        /
        total_pabeta
    )

    brightest_1_gamma = (
        pagamma_sorted
        .iloc[0]["pagamma_flux"]
        /
        total_pagamma
    )

    brightest_5_gamma = (
        pagamma_sorted
        .iloc[:5]["pagamma_flux"]
        .sum()
        /
        total_pagamma
    )

    brightest_10_gamma = (
        pagamma_sorted
        .iloc[:10]["pagamma_flux"]
        .sum()
        /
        total_pagamma
    )

    chaos_inside = (
        df[
            "distance_from_chaos_arcsec"
        ]
        <= APERTURE_RADIUS_ARCSEC
    )

    chaos_beta_fraction = (
        df.loc[
            chaos_inside,
            "pabeta_flux"
        ].sum()
        /
        total_pabeta
    )

    chaos_gamma_fraction = (
        df.loc[
            chaos_inside,
            "pagamma_flux"
        ].sum()
        /
        total_pagamma
    )

    summary = pd.DataFrame(
        [
            {
                "chaos_region":
                    CHAOS_NAME,

                "chaos_ra_deg":
                    CHAOS_RA,

                "chaos_dec_deg":
                    CHAOS_DEC,

                "jwst_center_x":
                    JWST_EXTR_X,

                "jwst_center_y":
                    JWST_EXTR_Y,

                "aperture_radius_arcsec":
                    APERTURE_RADIUS_ARCSEC,

                "aperture_area_arcsec2":
                    np.pi
                    * APERTURE_RADIUS_ARCSEC**2,

                "n_aperture_pixels":
                    n_aperture_pixels,

                "total_pabeta":
                    total_pabeta,

                "total_pagamma":
                    total_pagamma,

                "integrated_ratio":
                    integrated_ratio,

                "pixel_ratio_median":
                    (
                        ratios.median()
                        if valid.any()
                        else np.nan
                    ),

                "pixel_ratio_min":
                    (
                        ratios.min()
                        if valid.any()
                        else np.nan
                    ),

                "pixel_ratio_max":
                    (
                        ratios.max()
                        if valid.any()
                        else np.nan
                    ),

                "flux_weighted_pixel_ratio":
                    flux_weighted_ratio,

                "brightest_1_beta_fraction":
                    brightest_1_beta,

                "brightest_5_beta_fraction":
                    brightest_5_beta,

                "brightest_10_beta_fraction":
                    brightest_10_beta,

                "brightest_1_gamma_fraction":
                    brightest_1_gamma,

                "brightest_5_gamma_fraction":
                    brightest_5_gamma,

                "brightest_10_gamma_fraction":
                    brightest_10_gamma,

                "chaos_beta_fraction":
                    chaos_beta_fraction,

                "chaos_gamma_fraction":
                    chaos_gamma_fraction,

                "chaos_Av":
                    CHAOS_AV,

                "chaos_Av_sigma":
                    CHAOS_AV_SIGMA,

                "best_Te_at_chaos_Av":
                    best["Te"],

                "best_ne_at_chaos_Av":
                    best["ne"],

                "best_predicted_ratio_at_chaos_Av":
                    best[
                        "predicted_ratio_at_chaos_Av"
                    ],

                "best_ratio_difference":
                    best[
                        "difference_from_integrated_ratio"
                    ],

                "n_1sigma_compatible_SH_points":
                    len(compatible_1sigma),
            }
        ]
    )

    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False,
    )

    print(
        f"Saved:\n  {OUTPUT_SUMMARY}"
    )

    # --------------------------------------------------------
    # 17. FIGURE 1 — APERTURE + EMISSION
    # --------------------------------------------------------

    print_header(
        "17. CREATING APERTURE / EMISSION FIGURE"
    )

    fig, ax = plt.subplots(
        figsize=(10, 8)
    )

    image = np.array(
        pabeta,
        dtype=float,
    )

    finite = np.isfinite(image)

    if finite.any():

        vmin, vmax = np.nanpercentile(
            image[finite],
            [5, 99],
        )

    else:

        vmin, vmax = 0, 1

    im = ax.imshow(
        image,
        origin="lower",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )

    # Aperture circle.
    theta = np.linspace(
        0,
        2 * np.pi,
        360,
    )

    cx = JWST_EXTR_X
    cy = JWST_EXTR_Y

    ax.plot(
        cx
        +
        (
            APERTURE_RADIUS_ARCSEC
            / scale_x
        )
        * np.cos(theta),
        cy
        +
        (
            APERTURE_RADIUS_ARCSEC
            / scale_y
        )
        * np.sin(theta),
        linewidth=2,
        label="JWST 0.45 arcsec aperture",
    )

    # CHAOS position.
    chaos_x, chaos_y = (
        celestial_wcs.world_to_pixel(
            chaos_coord
        )
    )

    ax.scatter(
        [chaos_x],
        [chaos_y],
        marker="x",
        s=100,
        linewidths=2,
        label="CHAOS NGC5194+30.2+2.2",
    )

    ax.scatter(
        [cx],
        [cy],
        marker="+",
        s=100,
        linewidths=2,
        label="JWST extraction center",
    )

    ax.set_xlabel(
        "S3D X pixel"
    )

    ax.set_ylabel(
        "S3D Y pixel"
    )

    ax.set_title(
        "M51 Pa-beta Emission and JWST Extraction Aperture"
    )

    ax.legend()

    cbar = fig.colorbar(
        im,
        ax=ax,
    )

    cbar.set_label(
        "Pa-beta line-map value"
    )

    fig.tight_layout()

    fig.savefig(
        FIG_APERTURE,
        dpi=200,
    )

    plt.close(fig)

    print(
        f"Saved:\n  {FIG_APERTURE}"
    )

    # --------------------------------------------------------
    # 18. FIGURE 2 — RATIO
    # --------------------------------------------------------

    print_header(
        "18. CREATING RATIO / A(V) FIGURE"
    )

    ratio_map = np.full(
        (ny, nx),
        np.nan,
        dtype=float,
    )

    ratio_values = (
        df[
            "pabeta_pagamma_ratio"
        ]
        .to_numpy()
    )

    ratio_map[
        ys,
        xs
    ] = ratio_values

    fig, ax = plt.subplots(
        figsize=(10, 8)
    )

    valid_map = np.isfinite(
        ratio_map
    )

    if valid_map.any():

        rvmin, rvmax = np.nanpercentile(
            ratio_map[valid_map],
            [5, 95],
        )

    else:

        rvmin, rvmax = 2.0, 4.0

    im = ax.imshow(
        ratio_map,
        origin="lower",
        vmin=rvmin,
        vmax=rvmax,
        interpolation="nearest",
    )

    ax.plot(
        cx
        +
        (
            APERTURE_RADIUS_ARCSEC
            / scale_x
        )
        * np.cos(theta),
        cy
        +
        (
            APERTURE_RADIUS_ARCSEC
            / scale_y
        )
        * np.sin(theta),
        linewidth=2,
        label="JWST aperture",
    )

    ax.scatter(
        [chaos_x],
        [chaos_y],
        marker="x",
        s=100,
        linewidths=2,
        label="CHAOS",
    )

    ax.set_xlabel(
        "S3D X pixel"
    )

    ax.set_ylabel(
        "S3D Y pixel"
    )

    ax.set_title(
        "M51 Spatial Pa-beta / Pa-gamma Ratio"
    )

    ax.legend()

    cbar = fig.colorbar(
        im,
        ax=ax,
    )

    cbar.set_label(
        "Pa-beta / Pa-gamma"
    )

    fig.tight_layout()

    fig.savefig(
        FIG_RATIO,
        dpi=200,
    )

    plt.close(fig)

    print(
        f"Saved:\n  {FIG_RATIO}"
    )

    # --------------------------------------------------------
    # 19. FIGURE 3 — CONCENTRATION
    # --------------------------------------------------------

    print_header(
        "19. CREATING FLUX CONCENTRATION FIGURE"
    )

    beta_cumulative = (
        pabeta_sorted[
            "pabeta_flux"
        ].cumsum()
        /
        total_pabeta
    )

    gamma_cumulative = (
        pagamma_sorted[
            "pagamma_flux"
        ].cumsum()
        /
        total_pagamma
    )

    n_beta = np.arange(
        1,
        len(beta_cumulative) + 1,
    )

    n_gamma = np.arange(
        1,
        len(gamma_cumulative) + 1,
    )

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    ax.plot(
        n_beta,
        beta_cumulative,
        linewidth=2,
        label="Pa-beta",
    )

    ax.plot(
        n_gamma,
        gamma_cumulative,
        linewidth=2,
        label="Pa-gamma",
    )

    ax.set_xlabel(
        "Number of brightest aperture pixels"
    )

    ax.set_ylabel(
        "Cumulative fraction of line flux"
    )

    ax.set_xlim(
        1,
        len(df),
    )

    ax.set_ylim(
        0,
        1.02,
    )

    ax.set_title(
        "M51 JWST Line-Emission Concentration"
    )

    ax.grid(
        alpha=0.3
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIG_CONCENTRATION,
        dpi=200,
    )

    plt.close(fig)

    print(
        f"Saved:\n  {FIG_CONCENTRATION}"
    )

    # --------------------------------------------------------
    # 20. FINAL RESULT
    # --------------------------------------------------------

    print_header(
        "20. FINAL SPATIAL EMISSION RESULT"
    )

    print(
        f"""
JWST nominal aperture:
  radius = {APERTURE_RADIUS_ARCSEC:.3f} arcsec
  pixels = {n_aperture_pixels}

CHAOS:
  {CHAOS_NAME}
  RA  = {CHAOS_RA:.10f} deg
  Dec = {CHAOS_DEC:.10f} deg

Integrated aperture emission:
  Pa-beta  = {total_pabeta:.8g}
  Pa-gamma = {total_pagamma:.8g}

Integrated Pa-beta / Pa-gamma:
  {integrated_ratio:.8f}

CHAOS extinction:
  A(V) = {CHAOS_AV:.6f} +/- "
  f"{CHAOS_AV_SIGMA:.6f} mag

Brightest-pixel concentration:
  Pa-beta 1 pixel  = {100*brightest_1_beta:.2f}%
  Pa-beta 5 pixels = {100*brightest_5_beta:.2f}%
  Pa-beta 10 pixels = {100*brightest_10_beta:.2f}%

  Pa-gamma 1 pixel  = {100*brightest_1_gamma:.2f}%
  Pa-gamma 5 pixels = {100*brightest_5_gamma:.2f}%
  Pa-gamma 10 pixels = {100*brightest_10_gamma:.2f}%

CHAOS-centered aperture concentration:
  Pa-beta  = {100*chaos_beta_fraction:.2f}%
  Pa-gamma = {100*chaos_gamma_fraction:.2f}%

Storey-Hummer test at CHAOS A(V):
  closest Te = {best['Te']:.0f} K
  closest ne = {best['ne']:.3e} cm^-3
  predicted ratio = "
  f"{best['predicted_ratio_at_chaos_Av']:.8f}

1-sigma compatible Storey-Hummer points:
  {len(compatible_1sigma)}

Research outputs:
  {OUTPUT_TABLE}
  {OUTPUT_SUMMARY}
  {FIG_APERTURE}
  {FIG_RATIO}
  {FIG_CONCENTRATION}
"""
    )

    print(
        "\nSpatial emission aperture experiment complete."
    )


if __name__ == "__main__":
    main()
