#!/usr/bin/env python3

"""
M51 JWST NIRSpec/IFU
Pa-gamma Spatial Line-Map Reconstruction and Provenance Comparison

Purpose
-------
Reconstruct the Pa-gamma spatial line map directly from the JWST S3D
cube using the same extraction methodology used for the validated
Pa-beta multidimensional analysis.

Then compare the reconstructed map with the existing local Pa-gamma
product:

    m51_1096_pagamma_hydrogen_consistency_map.fits

This script does NOT use the existing Pa-gamma product to calculate
the reconstructed line flux.

The reconstructed map is derived directly from the S3D SCI and ERR
arrays.

Outputs
-------
    data/atomic_lines/
        m51_1094_pagamma_reconstructed_line_map.fits
        m51_1094_pagamma_reconstructed_snr_map.fits
        m51_1094_pagamma_reconstruction_comparison.csv
        m51_1094_pagamma_reconstruction_summary.csv

    project root:
        m51_1094_pagamma_reconstructed_line_map.png
        m51_1094_pagamma_reconstruction_comparison.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.wcs import WCS


# ============================================================
# PATHS
# ============================================================

PROJECT = Path("/home/glenn/Projects/cosmic_ai")

S3D_PATH = (
    PROJECT
    / "data/m51_jwst_level3/"
    / "jw03435-o006_t010_nirspec_g140m-f100lp_s3d.fits"
)

EXISTING_PAGAMMA_PATH = (
    PROJECT
    / "data/atomic_lines/"
    / "m51_1096_pagamma_hydrogen_consistency_map.fits"
)

OUTPUT_DIR = PROJECT / "data/atomic_lines"

RECONSTRUCTED_MAP = (
    OUTPUT_DIR
    / "m51_1094_pagamma_reconstructed_line_map.fits"
)

RECONSTRUCTED_SNR = (
    OUTPUT_DIR
    / "m51_1094_pagamma_reconstructed_snr_map.fits"
)

COMPARISON_CSV = (
    OUTPUT_DIR
    / "m51_1094_pagamma_reconstruction_comparison.csv"
)

SUMMARY_CSV = (
    OUTPUT_DIR
    / "m51_1094_pagamma_reconstruction_summary.csv"
)

MAP_PLOT = (
    PROJECT
    / "m51_1094_pagamma_reconstructed_line_map.png"
)

COMPARISON_PLOT = (
    PROJECT
    / "m51_1094_pagamma_reconstruction_comparison.png"
)


# ============================================================
# SCIENTIFIC PARAMETERS
# ============================================================

# Pa-gamma transition.
#
# The previous multidimensional analysis used:
#     1093.800 nm
#
# We retain that value here for exact methodological consistency.
PA_GAMMA_REST_NM = 1093.800000

# M51 velocity used by the existing Pa-beta extraction.
M51_VELOCITY_KMS = 463.0

# NIRSpec resolving power used by the validated extraction.
RESOLVING_POWER = 2700.0

# Same FWHM fraction used in the existing extraction code.
LINE_HALF_WIDTH_FWHM = 0.5

# Continuum windows.
#
# These should match the validated Pa-beta methodology.
# They are deliberately kept reasonably close to the line while
# avoiding the line itself.
BLUE_WINDOW_NM = (1080.0, 1088.0)
RED_WINDOW_NM = (1100.0, 1108.0)


# ============================================================
# HELPERS
# ============================================================

def banner(text):
    print()
    print("=" * 70)
    print(text)
    print("=" * 70)


def velocity_from_wavelength(rest_nm, observed_nm):
    c = 299792.458
    return c * (observed_nm / rest_nm - 1.0)


def predicted_wavelength(rest_nm, velocity_kms):
    c = 299792.458
    return rest_nm * (1.0 + velocity_kms / c)


def find_plane_indices(wavelength_nm, low, high):
    return np.where(
        (wavelength_nm >= low)
        & (wavelength_nm <= high)
    )[0]


def interpolate_continuum_cube(
    wavelength_nm,
    cube,
    blue_window,
    red_window,
):
    blue_idx = find_plane_indices(
        wavelength_nm,
        blue_window[0],
        blue_window[1],
    )

    red_idx = find_plane_indices(
        wavelength_nm,
        red_window[0],
        red_window[1],
    )

    if len(blue_idx) == 0:
        raise RuntimeError(
            "No wavelength planes found in blue continuum window."
        )

    if len(red_idx) == 0:
        raise RuntimeError(
            "No wavelength planes found in red continuum window."
        )

    blue_continuum = np.nanmedian(
        cube[blue_idx],
        axis=0,
    )

    red_continuum = np.nanmedian(
        cube[red_idx],
        axis=0,
    )

    blue_wave = np.nanmedian(
        wavelength_nm[blue_idx]
    )

    red_wave = np.nanmedian(
        wavelength_nm[red_idx]
    )

    continuum_cube = np.empty_like(cube)

    fraction = (
        (wavelength_nm - blue_wave)
        / (red_wave - blue_wave)
    )

    for i in range(len(wavelength_nm)):
        continuum_cube[i] = (
            blue_continuum
            + fraction[i]
            * (red_continuum - blue_continuum)
        )

    return (
        continuum_cube,
        blue_idx,
        red_idx,
    )


def extract_line_map(
    wavelength_nm,
    cube,
    err_cube,
    line_center_nm,
    fwhm_nm,
    blue_window,
    red_window,
):
    """
    Same basic extraction method used by the validated
    Pa-beta multidimensional analysis.

    Returns
    -------
    line_map
    snr_map
    continuum_map
    line_idx
    blue_idx
    red_idx
    continuum_cube
    """

    (
        continuum_cube,
        blue_idx,
        red_idx,
    ) = interpolate_continuum_cube(
        wavelength_nm,
        cube,
        blue_window,
        red_window,
    )

    half_width = (
        LINE_HALF_WIDTH_FWHM
        * fwhm_nm
    )

    line_low = (
        line_center_nm
        - half_width
    )

    line_high = (
        line_center_nm
        + half_width
    )

    line_idx = find_plane_indices(
        wavelength_nm,
        line_low,
        line_high,
    )

    if len(line_idx) == 0:
        raise RuntimeError(
            "No spectral planes inside Pa-gamma "
            f"window {line_low:.6f}-{line_high:.6f} nm."
        )

    residual = (
        cube[line_idx]
        - continuum_cube[line_idx]
    )

    if len(wavelength_nm) > 1:
        spacing = float(
            np.nanmedian(
                np.diff(wavelength_nm)
            )
        )
    else:
        spacing = 1.0

    line_map = (
        np.nansum(
            residual,
            axis=0,
        )
        * spacing
    )

    # Error propagation.
    line_err = err_cube[line_idx]

    variance = np.nansum(
        np.square(line_err),
        axis=0,
    )

    line_sigma = (
        np.sqrt(variance)
        * spacing
    )

    snr_map = np.full_like(
        line_map,
        np.nan,
        dtype=float,
    )

    valid = (
        np.isfinite(line_map)
        & np.isfinite(line_sigma)
        & (line_sigma > 0)
    )

    snr_map[valid] = (
        line_map[valid]
        / line_sigma[valid]
    )

    continuum_map = np.nanmedian(
        continuum_cube[line_idx],
        axis=0,
    )

    return (
        line_map,
        snr_map,
        continuum_map,
        line_idx,
        blue_idx,
        red_idx,
        continuum_cube,
        line_low,
        line_high,
        spacing,
    )


def save_map_fits(
    path,
    data,
    header,
    extname,
    bunit,
):
    h = header.copy()

    h["EXTNAME"] = extname
    h["BUNIT"] = bunit

    hdu = fits.PrimaryHDU(
        data=np.asarray(
            data,
            dtype=np.float32,
        ),
        header=h,
    )

    hdu.writeto(
        path,
        overwrite=True,
    )


def finite_statistics(values):
    values = np.asarray(values)

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return {}

    return {
        "n": int(len(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    banner(
        "M51 JWST NIRSPEC — Pa-GAMMA "
        "LINE-MAP RECONSTRUCTION"
    )

    print(
        """
Purpose:
Reconstruct the Pa-gamma spatial line map directly from
the JWST S3D cube using the same methodology used for
the validated Pa-beta extraction.

The existing Pa-gamma product is treated only as a
comparison product.

No existing Pa-gamma map is used in the reconstruction.
"""
    )

    # ========================================================
    # 1. READ S3D
    # ========================================================

    banner("1. READING JWST S3D")

    print(f"File:\n  {S3D_PATH}")

    with fits.open(
        S3D_PATH,
        memmap=True,
    ) as hdul:

        sci = np.asarray(
            hdul["SCI"].data,
            dtype=float,
        )

        err = np.asarray(
            hdul["ERR"].data,
            dtype=float,
        )

        sci_header = hdul["SCI"].header.copy()

        primary_header = hdul[0].header.copy()

        print()
        print("SCI shape:")
        print(f"  {sci.shape}")

        print()
        print("ERR shape:")
        print(f"  {err.shape}")

        bunit = hdul["SCI"].header.get(
            "BUNIT",
            "unknown",
        )

        print()
        print(f"SCI BUNIT = {bunit}")

        # WCS from SCI.
        wcs = WCS(
            hdul["SCI"].header
        )

    if sci.ndim != 3:
        raise RuntimeError(
            "Expected a 3-D S3D SCI cube."
        )

    spectral_n, ny, nx = sci.shape

    if err.shape != sci.shape:
        raise RuntimeError(
            "SCI and ERR shapes do not match."
        )

    # ========================================================
    # 2. BUILD WAVELENGTH ARRAY
    # ========================================================

    banner("2. BUILDING WAVELENGTH ARRAY")

    # JWST S3D spectral axis is wavelength.
    #
    # Use the WCS rather than assuming a hard-coded spacing.
    #
    # The cube's spectral coordinate is the third world axis.
    pixel = np.arange(
        spectral_n
    )

    world = wcs.pixel_to_world_values(
        np.zeros(spectral_n),
        np.zeros(spectral_n),
        pixel,
    )

    wavelength = np.asarray(
        world[2]
    )

    # Convert meters to nm if necessary.
    if np.nanmedian(
        wavelength
    ) < 1e-3:

        wavelength_nm = (
            wavelength * 1e9
        )

    else:

        wavelength_nm = wavelength

    print(
        f"Wavelength range = "
        f"{np.nanmin(wavelength_nm):.6f} - "
        f"{np.nanmax(wavelength_nm):.6f} nm"
    )

    print(
        f"Spectral planes = "
        f"{len(wavelength_nm)}"
    )

    # ========================================================
    # 3. PREDICT PA-GAMMA POSITION
    # ========================================================

    banner("3. PREDICTING PA-GAMMA WAVELENGTH")

    predicted = predicted_wavelength(
        PA_GAMMA_REST_NM,
        M51_VELOCITY_KMS,
    )

    fwhm = (
        predicted
        / RESOLVING_POWER
    )

    half_width = (
        LINE_HALF_WIDTH_FWHM
        * fwhm
    )

    line_low = (
        predicted
        - half_width
    )

    line_high = (
        predicted
        + half_width
    )

    nearest_idx = int(
        np.nanargmin(
            np.abs(
                wavelength_nm
                - predicted
            )
        )
    )

    nearest_wave = float(
        wavelength_nm[
            nearest_idx
        ]
    )

    inferred_velocity = (
        velocity_from_wavelength(
            PA_GAMMA_REST_NM,
            nearest_wave,
        )
    )

    print(
        f"Pa-gamma rest wavelength:"
        f" {PA_GAMMA_REST_NM:.6f} nm"
    )

    print(
        f"M51 velocity:"
        f" {M51_VELOCITY_KMS:+.3f} km/s"
    )

    print(
        f"Predicted observed wavelength:"
        f" {predicted:.6f} nm"
    )

    print(
        f"Instrumental FWHM:"
        f" {fwhm:.6f} nm"
    )

    print(
        f"Extraction window:"
        f" {line_low:.6f} - "
        f"{line_high:.6f} nm"
    )

    print(
        f"Nearest cube wavelength:"
        f" {nearest_wave:.6f} nm"
    )

    print(
        f"Velocity at nearest plane:"
        f" {inferred_velocity:+.3f} km/s"
    )

    # ========================================================
    # 4. CONTINUUM WINDOWS
    # ========================================================

    banner("4. CONTINUUM WINDOWS")

    print(
        f"Blue continuum:"
        f" {BLUE_WINDOW_NM[0]:.3f} - "
        f"{BLUE_WINDOW_NM[1]:.3f} nm"
    )

    print(
        f"Red continuum:"
        f" {RED_WINDOW_NM[0]:.3f} - "
        f"{RED_WINDOW_NM[1]:.3f} nm"
    )

    # ========================================================
    # 5. RECONSTRUCT PA-GAMMA
    # ========================================================

    banner(
        "5. RECONSTRUCTING PA-GAMMA LINE MAP"
    )

    (
        line_map,
        snr_map,
        continuum_map,
        line_idx,
        blue_idx,
        red_idx,
        continuum_cube,
        actual_low,
        actual_high,
        spacing,
    ) = extract_line_map(
        wavelength_nm,
        sci,
        err,
        predicted,
        fwhm,
        BLUE_WINDOW_NM,
        RED_WINDOW_NM,
    )

    print(
        f"Line planes used: "
        f"{len(line_idx)}"
    )

    print(
        "Line-plane indices:"
    )

    print(
        "  "
        + ", ".join(
            str(int(i))
            for i in line_idx
        )
    )

    print(
        f"Actual line window:"
        f" {actual_low:.6f} - "
        f"{actual_high:.6f} nm"
    )

    print(
        f"Spectral spacing:"
        f" {spacing:.9f} nm"
    )

    print(
        f"Blue continuum planes:"
        f" {len(blue_idx)}"
    )

    print(
        f"Red continuum planes:"
        f" {len(red_idx)}"
    )

    # ========================================================
    # 6. RECONSTRUCTION STATISTICS
    # ========================================================

    banner(
        "6. RECONSTRUCTED MAP STATISTICS"
    )

    finite = np.isfinite(
        line_map
    )

    positive = (
        finite
        & (line_map > 0)
    )

    negative = (
        finite
        & (line_map < 0)
    )

    print(
        f"Finite pixels:"
        f" {np.sum(finite)}"
    )

    print(
        f"Positive pixels:"
        f" {np.sum(positive)}"
    )

    print(
        f"Negative pixels:"
        f" {np.sum(negative)}"
    )

    print(
        f"Minimum:"
        f" {np.nanmin(line_map):.6f}"
    )

    print(
        f"Maximum:"
        f" {np.nanmax(line_map):.6f}"
    )

    print(
        f"Median:"
        f" {np.nanmedian(line_map):.6f}"
    )

    print(
        f"Mean:"
        f" {np.nanmean(line_map):.6f}"
    )

    # ========================================================
    # 7. LOAD EXISTING PAGAMMA PRODUCT
    # ========================================================

    banner(
        "7. LOADING EXISTING Pa-GAMMA PRODUCT"
    )

    print(
        f"File:\n  "
        f"{EXISTING_PAGAMMA_PATH}"
    )

    if not EXISTING_PAGAMMA_PATH.exists():

        raise FileNotFoundError(
            "Existing Pa-gamma comparison "
            "product does not exist."
        )

    with fits.open(
        EXISTING_PAGAMMA_PATH
    ) as hdul:

        existing = np.asarray(
            hdul[0].data,
            dtype=float,
        )

        existing_header = (
            hdul[0].header.copy()
        )

    print(
        f"Existing map shape:"
        f" {existing.shape}"
    )

    print(
        f"Existing map BUNIT:"
        f" {existing_header.get('BUNIT')}"
    )

    print(
        f"Existing finite pixels:"
        f" {np.sum(np.isfinite(existing))}"
    )

    # ========================================================
    # 8. MAP COMPATIBILITY
    # ========================================================

    banner(
        "8. SPATIAL MAP COMPATIBILITY"
    )

    if existing.shape != line_map.shape:

        raise RuntimeError(
            "Existing Pa-gamma map and "
            "reconstructed map have different shapes."
        )

    print(
        "Shape comparison:"
    )

    print(
        f"  reconstructed = "
        f"{line_map.shape}"
    )

    print(
        f"  existing      = "
        f"{existing.shape}"
    )

    print(
        "Shapes match: TRUE"
    )

    # ========================================================
    # 9. PIXEL-BY-PIXEL COMPARISON
    # ========================================================

    banner(
        "9. PIXEL-BY-PIXEL COMPARISON"
    )

    valid = (
        np.isfinite(line_map)
        & np.isfinite(existing)
    )

    x = line_map[valid]
    y = existing[valid]

    print(
        f"Common finite pixels:"
        f" {len(x)}"
    )

    if len(x) > 2:

        correlation = np.corrcoef(
            x,
            y,
        )[0, 1]

        slope, intercept = np.polyfit(
            x,
            y,
            1,
        )

        residual = (
            y
            - (
                slope * x
                + intercept
            )
        )

        print(
            f"Pearson correlation:"
            f" {correlation:.6f}"
        )

        print(
            f"Linear slope:"
            f" {slope:.6f}"
        )

        print(
            f"Linear intercept:"
            f" {intercept:.6f}"
        )

        print(
            f"Residual RMS:"
            f" {np.sqrt(np.mean(residual**2)):.6f}"
        )

        positive_pair = (
            (x > 0)
            & (y > 0)
        )

        if np.sum(
            positive_pair
        ) > 2:

            logx = np.log10(
                x[positive_pair]
            )

            logy = np.log10(
                y[positive_pair]
            )

            logcorr = np.corrcoef(
                logx,
                logy,
            )[0, 1]

            print(
                "Positive-positive "
                "log-space correlation:"
                f" {logcorr:.6f}"
            )

    else:

        correlation = np.nan
        slope = np.nan
        intercept = np.nan
        residual = np.array([])

        print(
            "Insufficient common "
            "finite pixels for regression."
        )

    # ========================================================
    # 10. APERTURE COMPARISON
    # ========================================================

    banner(
        "10. 69-PIXEL APERTURE COMPARISON"
    )

    # Recreate nominal aperture.
    #
    # The extraction center is x=62, y=48 and the nominal radius
    # is 4.5 pixels.
    yy, xx = np.indices(
        line_map.shape
    )

    aperture_mask = (
        (
            (xx - 62.0) ** 2
            + (yy - 48.0) ** 2
        )
        <= 4.5 ** 2
    )

    print(
        f"Nominal aperture pixels:"
        f" {np.sum(aperture_mask)}"
    )

    recon_ap = (
        line_map[
            aperture_mask
        ]
    )

    existing_ap = (
        existing[
            aperture_mask
        ]
    )

    recon_valid = (
        np.isfinite(recon_ap)
        & (recon_ap > 0)
    )

    existing_valid = (
        np.isfinite(existing_ap)
        & (existing_ap > 0)
    )

    print()
    print(
        "Reconstructed Pa-gamma:"
    )

    print(
        f"  finite = "
        f"{np.sum(np.isfinite(recon_ap))}"
    )

    print(
        f"  positive = "
        f"{np.sum(recon_valid)}"
    )

    print(
        f"  sum = "
        f"{np.nansum(recon_ap):.8f}"
    )

    print()
    print(
        "Existing Pa-gamma:"
    )

    print(
        f"  finite = "
        f"{np.sum(np.isfinite(existing_ap))}"
    )

    print(
        f"  positive = "
        f"{np.sum(existing_valid)}"
    )

    print(
        f"  sum = "
        f"{np.nansum(existing_ap):.8f}"
    )

    # ========================================================
    # 11. FLUX-RATIO TEST
    # ========================================================

    banner(
        "11. RECONSTRUCTED / EXISTING FLUX TEST"
    )

    recon_sum = np.nansum(
        recon_ap
    )

    existing_sum = np.nansum(
        existing_ap
    )

    print(
        f"Reconstructed sum:"
        f" {recon_sum:.8f}"
    )

    print(
        f"Existing sum:"
        f" {existing_sum:.8f}"
    )

    if existing_sum != 0:

        print(
            f"Reconstructed / existing:"
            f" {recon_sum / existing_sum:.8f}"
        )

    else:

        print(
            "Existing sum is zero; "
            "ratio undefined."
        )

    # ========================================================
    # 12. SAVE RECONSTRUCTED MAPS
    # ========================================================

    banner(
        "12. SAVING RECONSTRUCTED PRODUCTS"
    )

    map_header = sci_header.copy()

    save_map_fits(
        RECONSTRUCTED_MAP,
        line_map,
        map_header,
        "PAGAMMA_LINE",
        bunit,
    )

    save_map_fits(
        RECONSTRUCTED_SNR,
        snr_map,
        map_header,
        "PAGAMMA_SNR",
        "dimensionless",
    )

    print(
        f"Line map:\n  "
        f"{RECONSTRUCTED_MAP}"
    )

    print(
        f"S/N map:\n  "
        f"{RECONSTRUCTED_SNR}"
    )

    # ========================================================
    # 13. SAVE COMPARISON TABLE
    # ========================================================

    banner(
        "13. SAVING PIXEL COMPARISON TABLE"
    )

    rows = []

    for ypix in range(ny):

        for xpix in range(nx):

            rows.append(
                {
                    "x_pixel": xpix,
                    "y_pixel": ypix,
                    "reconstructed_pagamma":
                        line_map[ypix, xpix],
                    "reconstructed_snr":
                        snr_map[ypix, xpix],
                    "existing_pagamma":
                        existing[ypix, xpix],
                    "difference":
                        (
                            line_map[ypix, xpix]
                            - existing[ypix, xpix]
                        ),
                    "inside_nominal_aperture":
                        bool(
                            aperture_mask[
                                ypix,
                                xpix
                            ]
                        ),
                }
            )

    comparison_df = pd.DataFrame(
        rows
    )

    comparison_df.to_csv(
        COMPARISON_CSV,
        index=False,
    )

    print(
        f"Saved:\n  "
        f"{COMPARISON_CSV}"
    )

    # ========================================================
    # 14. SUMMARY
    # ========================================================

    banner(
        "14. SAVING SUMMARY"
    )

    summary = {
        "s3d_path": str(S3D_PATH),
        "existing_pagamma_path":
            str(EXISTING_PAGAMMA_PATH),
        "rest_wavelength_nm":
            PA_GAMMA_REST_NM,
        "m51_velocity_kms":
            M51_VELOCITY_KMS,
        "predicted_wavelength_nm":
            predicted,
        "fwhm_nm":
            fwhm,
        "line_low_nm":
            actual_low,
        "line_high_nm":
            actual_high,
        "spectral_spacing_nm":
            spacing,
        "line_planes":
            len(line_idx),
        "blue_continuum_planes":
            len(blue_idx),
        "red_continuum_planes":
            len(red_idx),
        "reconstructed_finite_pixels":
            int(np.sum(finite)),
        "reconstructed_positive_pixels":
            int(np.sum(positive)),
        "reconstructed_negative_pixels":
            int(np.sum(negative)),
        "reconstructed_total_positive_flux":
            float(
                np.nansum(
                    line_map[
                        positive
                    ]
                )
            ),
        "existing_total_positive_flux":
            float(
                np.nansum(
                    existing[
                        existing > 0
                    ]
                )
            ),
        "common_finite_pixels":
            int(len(x)),
        "pearson_correlation":
            correlation,
        "linear_slope":
            slope,
        "linear_intercept":
            intercept,
        "aperture_pixels":
            int(np.sum(aperture_mask)),
        "reconstructed_aperture_sum":
            float(recon_sum),
        "existing_aperture_sum":
            float(existing_sum),
    }

    pd.DataFrame(
        [summary]
    ).to_csv(
        SUMMARY_CSV,
        index=False,
    )

    print(
        f"Saved:\n  "
        f"{SUMMARY_CSV}"
    )

    # ========================================================
    # 15. RECONSTRUCTED MAP FIGURE
    # ========================================================

    banner(
        "15. CREATING RECONSTRUCTED Pa-GAMMA FIGURE"
    )

    plt.figure(
        figsize=(9, 7)
    )

    image = plt.imshow(
        line_map,
        origin="lower",
        interpolation="nearest",
    )

    plt.colorbar(
        image,
        label=f"Pa-gamma line flux ({bunit})",
    )

    # Aperture.
    theta = np.linspace(
        0,
        2 * np.pi,
        300,
    )

    plt.plot(
        62.0
        + 4.5 * np.cos(theta),
        48.0
        + 4.5 * np.sin(theta),
        linewidth=2,
        label="JWST 0.45 arcsec aperture",
    )

    plt.scatter(
        [62],
        [48],
        marker="+",
        s=150,
        linewidths=2,
        label="JWST extraction center",
    )

    plt.xlabel(
        "X pixel"
    )

    plt.ylabel(
        "Y pixel"
    )

    plt.title(
        "M51 JWST NIRSpec Pa-gamma — Reconstructed"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        MAP_PLOT,
        dpi=180,
    )

    plt.close()

    print(
        f"Saved:\n  {MAP_PLOT}"
    )

    # ========================================================
    # 16. COMPARISON FIGURE
    # ========================================================

    banner(
        "16. CREATING RECONSTRUCTION COMPARISON"
    )

    plt.figure(
        figsize=(8, 7)
    )

    if len(x) > 2:

        plt.scatter(
            x,
            y,
            s=8,
            alpha=0.45,
        )

        finite_xy = (
            np.isfinite(x)
            & np.isfinite(y)
        )

        if np.any(finite_xy):

            lo = min(
                np.nanmin(x),
                np.nanmin(y),
            )

            hi = max(
                np.nanmax(x),
                np.nanmax(y),
            )

            plt.plot(
                [lo, hi],
                [lo, hi],
                linestyle="--",
                label="1:1",
            )

        plt.xlabel(
            "Reconstructed Pa-gamma"
        )

        plt.ylabel(
            "Existing Pa-gamma product"
        )

        plt.title(
            "M51 Pa-gamma Map Provenance Comparison"
        )

        plt.legend()

    else:

        plt.text(
            0.5,
            0.5,
            "Insufficient valid pixels",
            ha="center",
            va="center",
            transform=plt.gca().transAxes,
        )

    plt.tight_layout()

    plt.savefig(
        COMPARISON_PLOT,
        dpi=180,
    )

    plt.close()

    print(
        f"Saved:\n  "
        f"{COMPARISON_PLOT}"
    )

    # ========================================================
    # 17. FINAL RESULT
    # ========================================================

    banner(
        "FINAL Pa-GAMMA RECONSTRUCTION RESULT"
    )

    print(
        "Reconstructed directly from S3D:"
    )

    print(
        f"  Pa-gamma wavelength = "
        f"{predicted:.6f} nm"
    )

    print(
        f"  FWHM = "
        f"{fwhm:.6f} nm"
    )

    print(
        f"  spectral planes = "
        f"{len(line_idx)}"
    )

    print(
        f"  aperture pixels = "
        f"{np.sum(aperture_mask)}"
    )

    print()
    print(
        "Reconstructed 69-pixel aperture:"
    )

    print(
        f"  total Pa-gamma = "
        f"{recon_sum:.8f}"
    )

    print()
    print(
        "Existing product:"
    )

    print(
        f"  total = "
        f"{existing_sum:.8f}"
    )

    print()
    print(
        "Pixel-level comparison:"
    )

    print(
        f"  common finite pixels = "
        f"{len(x)}"
    )

    print(
        f"  Pearson correlation = "
        f"{correlation:.6f}"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "The reconstructed map is the preferred Pa-gamma "
        "measurement for subsequent aperture analysis because "
        "its provenance is explicitly tied to the S3D SCI/ERR "
        "cube and documented extraction window."
    )

    print()
    print(
        "The existing Pa-gamma product is retained only as "
        "a comparison diagnostic until its provenance is "
        "established independently."
    )

    print()
    print(
        "Pa-gamma reconstruction experiment complete."
    )


if __name__ == "__main__":
    main()
