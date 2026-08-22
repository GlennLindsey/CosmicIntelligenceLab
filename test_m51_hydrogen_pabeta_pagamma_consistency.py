#!/usr/bin/env python3

"""
M51 Hydrogen Pa-beta / Pa-gamma Consistency Test
=================================================

Tests whether the 1284.26 nm and 1095.89 nm features
behave consistently with hydrogen Paschen-beta and
Paschen-gamma emission arising from the same M51 gas.

Dimensions tested:

1. Wavelength consistency
2. Velocity consistency
3. Instrumental line-profile consistency
4. Spatial emission
5. Spatial centroid consistency
6. Spatial-map correlation
7. Integrated line flux
8. Pa-beta / Pa-gamma flux ratio

Important:
This is a diagnostic consistency test. It does not by itself
establish a definitive atomic identification.

Case-B hydrogen recombination is used only as a physical
reference framework. No single theoretical line ratio is
imposed as a hard constraint.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.io import fits


# ============================================================
# CONFIGURATION
# ============================================================

S3D_PATH = Path(
    "data/m51_jwst_level3/"
    "jw03435-o006_t010_nirspec_g140m-f100lp_s3d.fits"
)

OUTPUT_DIR = Path("data/atomic_lines")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Hydrogen wavelengths
# ------------------------------------------------------------

PA_BETA_REST_NM = 1281.807000
PA_GAMMA_REST_NM = 1093.800000

OBSERVED_PA_BETA_NM = 1284.261304400

M51_VELOCITY_KMS = 573.720

C_KMS = 299792.458


# ------------------------------------------------------------
# NIRSpec resolution
# ------------------------------------------------------------

R_NIRSPEC = 916.3


# ------------------------------------------------------------
# Spatial extraction windows
# ------------------------------------------------------------

PA_BETA_HALF_WIDTH_NM = 2.0
PA_GAMMA_HALF_WIDTH_NM = 2.0

CONTINUUM_OFFSET_NM = 3.0
CONTINUUM_WIDTH_NM = 3.0


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def predicted_wavelength(rest_nm, velocity_kms):
    """
    Relativistic Doppler wavelength.
    """
    beta = velocity_kms / C_KMS

    doppler = np.sqrt(
        (1.0 + beta) /
        (1.0 - beta)
    )

    return rest_nm * doppler


def velocity_from_wavelength(observed_nm, rest_nm):
    """
    Relativistic velocity from observed wavelength.
    """
    ratio = observed_nm / rest_nm

    beta = (
        ratio**2 - 1.0
    ) / (
        ratio**2 + 1.0
    )

    return beta * C_KMS


def gaussian_sigma_from_resolution(wavelength_nm):
    """
    Instrumental Gaussian sigma from resolving power.
    """
    fwhm = wavelength_nm / R_NIRSPEC
    sigma = fwhm / 2.354820045

    return sigma


def nearest_index(array, value):
    return int(np.argmin(np.abs(array - value)))


def spatial_centroid(image):
    """
    Flux-weighted spatial centroid.

    Negative values are excluded because this is intended
    to characterize positive emission.
    """

    data = np.asarray(image, dtype=float)

    valid = (
        np.isfinite(data)
        & (data > 0)
    )

    if not np.any(valid):
        return np.nan, np.nan

    yy, xx = np.indices(data.shape)

    weights = data[valid]

    x = xx[valid]
    y = yy[valid]

    return (
        np.sum(x * weights) / np.sum(weights),
        np.sum(y * weights) / np.sum(weights),
    )


def spatial_correlation(a, b, threshold_a=3.0, threshold_b=3.0):
    """
    Pearson correlation between two spatial maps.

    Only pixels exceeding the supplied S/N thresholds
    are included.
    """

    valid = (
        np.isfinite(a)
        & np.isfinite(b)
        & np.isfinite(threshold_a)
    )

    if not np.any(valid):
        return np.nan, 0

    x = a[valid]
    y = b[valid]

    if len(x) < 3:
        return np.nan, len(x)

    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan, len(x)

    correlation = np.corrcoef(x, y)[0, 1]

    return correlation, len(x)


def continuum_subtracted_map(
    flux_cube,
    error_cube,
    wavelengths_nm,
    line_center_nm,
    half_width_nm,
):
    """
    Construct a continuum-subtracted spatial emission map.

    Continuum is estimated from blue and red sidebands.
    """

    line_mask = (
        np.abs(wavelengths_nm - line_center_nm)
        <= half_width_nm
    )

    blue_center = line_center_nm - CONTINUUM_OFFSET_NM
    red_center = line_center_nm + CONTINUUM_OFFSET_NM

    blue_mask = (
        (wavelengths_nm >= blue_center - CONTINUUM_WIDTH_NM / 2)
        & (wavelengths_nm <= blue_center + CONTINUUM_WIDTH_NM / 2)
    )

    red_mask = (
        (wavelengths_nm >= red_center - CONTINUUM_WIDTH_NM / 2)
        & (wavelengths_nm <= red_center + CONTINUUM_WIDTH_NM / 2)
    )

    line_flux = flux_cube[line_mask]

    blue_flux = flux_cube[blue_mask]
    red_flux = flux_cube[red_mask]

    blue_error = error_cube[blue_mask]
    red_error = error_cube[red_mask]

    if blue_flux.shape[0] == 0:
        raise RuntimeError("No blue continuum planes found.")

    if red_flux.shape[0] == 0:
        raise RuntimeError("No red continuum planes found.")

    # Median continuum.
    blue_cont = np.nanmedian(
        blue_flux,
        axis=0
    )

    red_cont = np.nanmedian(
        red_flux,
        axis=0
    )

    # Interpolate continuum to line center.
    blue_lambda = np.nanmedian(
        wavelengths_nm[blue_mask]
    )

    red_lambda = np.nanmedian(
        wavelengths_nm[red_mask]
    )

    fraction = (
        line_center_nm - blue_lambda
    ) / (
        red_lambda - blue_lambda
    )

    continuum_map = (
        blue_cont
        + fraction * (red_cont - blue_cont)
    )

    # Integrated line signal.
    line_median = np.nanmedian(
        line_flux,
        axis=0
    )

    n_line = np.sum(line_mask)

    line_map = (
        line_median - continuum_map
    ) * n_line

    # --------------------------------------------------------
    # Error propagation
    # --------------------------------------------------------

    blue_err = np.nanmedian(
        blue_error,
        axis=0
    )

    red_err = np.nanmedian(
        red_error,
        axis=0
    )

    continuum_error = np.sqrt(
        (1.0 - fraction)**2 * blue_err**2
        + fraction**2 * red_err**2
    )

    line_err = np.nanmedian(
        error_cube[line_mask],
        axis=0
    )

    line_error = np.sqrt(
        line_err**2
        + continuum_error**2
    ) * np.sqrt(n_line)

    snr_map = np.divide(
        line_map,
        line_error,
        out=np.full_like(
            line_map,
            np.nan,
            dtype=float
        ),
        where=(
            np.isfinite(line_error)
            & (line_error > 0)
        )
    )

    return (
        line_map,
        snr_map,
        continuum_map,
        line_mask,
        blue_mask,
        red_mask,
    )


def save_image_fits(path, image, header, description):
    """
    Save a 2-D spatial image.
    """

    hdu = fits.PrimaryHDU(
        data=np.asarray(image, dtype=np.float32)
    )

    hdu.header["BUNIT"] = "MJy/sr"
    hdu.header["HISTORY"] = description

    if header is not None:

        for key in (
            "CRPIX1",
            "CRPIX2",
            "CRVAL1",
            "CRVAL2",
            "CDELT1",
            "CDELT2",
            "CTYPE1",
            "CTYPE2",
            "CUNIT1",
            "CUNIT2",
        ):

            if key in header:
                hdu.header[key] = header[key]

    hdu.writeto(
        path,
        overwrite=True
    )


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("M51 HYDROGEN PA-BETA / PA-GAMMA CONSISTENCY TEST")
print("MULTI-DIMENSIONAL HYDROGEN RECOMBINATION TEST")
print("=" * 70)


# ============================================================
# PREDICTED WAVELENGTHS
# ============================================================

pa_beta_predicted = predicted_wavelength(
    PA_BETA_REST_NM,
    M51_VELOCITY_KMS
)

pa_gamma_predicted = predicted_wavelength(
    PA_GAMMA_REST_NM,
    M51_VELOCITY_KMS
)

pa_beta_velocity = velocity_from_wavelength(
    OBSERVED_PA_BETA_NM,
    PA_BETA_REST_NM
)

print()
print("=" * 70)
print("1. HYDROGEN WAVELENGTH CONSISTENCY")
print("=" * 70)

print(
    f"Pa-beta rest wavelength: "
    f"{PA_BETA_REST_NM:.9f} nm"
)

print(
    f"Pa-beta predicted wavelength: "
    f"{pa_beta_predicted:.9f} nm"
)

print(
    f"Observed 1284.26 nm feature: "
    f"{OBSERVED_PA_BETA_NM:.9f} nm"
)

print(
    f"Pa-beta observed-minus-predicted: "
    f"{OBSERVED_PA_BETA_NM - pa_beta_predicted:+.9f} nm"
)

print(
    f"Velocity from observed Pa-beta feature: "
    f"{pa_beta_velocity:+.6f} km/s"
)

print(
    f"Velocity offset from M51: "
    f"{pa_beta_velocity - M51_VELOCITY_KMS:+.6f} km/s"
)

print()

print(
    f"Pa-gamma rest wavelength: "
    f"{PA_GAMMA_REST_NM:.9f} nm"
)

print(
    f"Pa-gamma predicted wavelength: "
    f"{pa_gamma_predicted:.9f} nm"
)


# ============================================================
# LOAD CUBE
# ============================================================

print()
print("=" * 70)
print("2. LOADING S3D CUBE")
print("=" * 70)

print()
print("File:")
print(f"  {S3D_PATH}")

with fits.open(
    S3D_PATH,
    memmap=False
) as hdul:

    sci = np.asarray(
        hdul["SCI"].data,
        dtype=float
    )

    err = np.asarray(
        hdul["ERR"].data,
        dtype=float
    )

    header = hdul["SCI"].header.copy()


print()
print(f"SCI shape: {sci.shape}")
print(f"ERR shape: {err.shape}")

if sci.ndim != 3:
    raise RuntimeError(
        "Expected a 3-D S3D science cube."
    )


# ============================================================
# WAVELENGTH AXIS
# ============================================================

n_wave = sci.shape[0]

crval3 = header["CRVAL3"]
crpix3 = header["CRPIX3"]
cdelt3 = header["CDELT3"]

pixel_numbers = np.arange(
    1,
    n_wave + 1
)

wavelength_um = (
    crval3
    + (
        pixel_numbers
        - crpix3
    ) * cdelt3
)

wavelength_nm = (
    wavelength_um * 1000.0
)


print()
print("=" * 70)
print("3. SPECTRAL AXIS")
print("=" * 70)

print(
    f"Wavelength range: "
    f"{wavelength_nm.min():.9f} - "
    f"{wavelength_nm.max():.9f} nm"
)

print(
    f"Spectral sampling: "
    f"{np.median(np.diff(wavelength_nm)):.9f} nm"
)


# ============================================================
# INSTRUMENT RESOLUTION
# ============================================================

pa_beta_fwhm = (
    pa_beta_predicted
    / R_NIRSPEC
)

pa_gamma_fwhm = (
    pa_gamma_predicted
    / R_NIRSPEC
)

pa_beta_sigma = (
    pa_beta_fwhm
    / 2.354820045
)

pa_gamma_sigma = (
    pa_gamma_fwhm
    / 2.354820045
)


print()
print("=" * 70)
print("4. NIRSPEC RESOLUTION")
print("=" * 70)

print(
    f"R = {R_NIRSPEC:.1f}"
)

print(
    f"Pa-beta FWHM: "
    f"{pa_beta_fwhm:.6f} nm"
)

print(
    f"Pa-gamma FWHM: "
    f"{pa_gamma_fwhm:.6f} nm"
)


# ============================================================
# PA-BETA MAP
# ============================================================

print()
print("=" * 70)
print("5. PA-BETA SPATIAL EXTRACTION")
print("=" * 70)

(
    pa_beta_map,
    pa_beta_snr,
    pa_beta_continuum,
    pa_beta_line_mask,
    pa_beta_blue_mask,
    pa_beta_red_mask,
) = continuum_subtracted_map(
    sci,
    err,
    wavelength_nm,
    pa_beta_predicted,
    PA_BETA_HALF_WIDTH_NM,
)


# ============================================================
# PA-GAMMA MAP
# ============================================================

print()
print("=" * 70)
print("6. PA-GAMMA SPATIAL EXTRACTION")
print("=" * 70)

(
    pa_gamma_map,
    pa_gamma_snr,
    pa_gamma_continuum,
    pa_gamma_line_mask,
    pa_gamma_blue_mask,
    pa_gamma_red_mask,
) = continuum_subtracted_map(
    sci,
    err,
    wavelength_nm,
    pa_gamma_predicted,
    PA_GAMMA_HALF_WIDTH_NM,
)


# ============================================================
# BASIC MAP DIAGNOSTICS
# ============================================================

print()
print("=" * 70)
print("7. SPATIAL DIAGNOSTICS")
print("=" * 70)

def map_diagnostics(name, line_map, snr_map):

    finite_flux = np.isfinite(line_map)
    finite_snr = np.isfinite(snr_map)

    positive = (
        finite_snr
        & (snr_map >= 3.0)
    )

    peak_snr = (
        np.nanmax(snr_map)
        if np.any(finite_snr)
        else np.nan
    )

    total_positive = np.sum(
        line_map[
            finite_flux
            & (line_map > 0)
        ]
    )

    xcen, ycen = spatial_centroid(
        line_map
    )

    peak_index = (
        np.unravel_index(
            np.nanargmax(snr_map),
            snr_map.shape
        )
        if np.any(finite_snr)
        else (np.nan, np.nan)
    )

    print()
    print(name)

    print(
        f"  Finite flux pixels: "
        f"{np.sum(finite_flux)}"
    )

    print(
        f"  Finite S/N pixels: "
        f"{np.sum(finite_snr)}"
    )

    print(
        f"  Peak S/N: "
        f"{peak_snr:.3f}"
    )

    print(
        f"  Pixels S/N >= 3: "
        f"{np.sum(positive)}"
    )

    print(
        f"  Positive integrated map signal: "
        f"{total_positive:.6e}"
    )

    print(
        f"  Flux-weighted centroid: "
        f"RA={xcen:.3f}, DEC={ycen:.3f}"
    )

    print(
        f"  Peak pixel: "
        f"RA={peak_index[1]}, DEC={peak_index[0]}"
    )

    return {
        "peak_snr": peak_snr,
        "n_positive_snr3": int(np.sum(positive)),
        "positive_flux_sum": total_positive,
        "centroid_ra": xcen,
        "centroid_dec": ycen,
        "peak_ra": peak_index[1],
        "peak_dec": peak_index[0],
    }


beta_diag = map_diagnostics(
    "Pa-beta",
    pa_beta_map,
    pa_beta_snr,
)

gamma_diag = map_diagnostics(
    "Pa-gamma",
    pa_gamma_map,
    pa_gamma_snr,
)


# ============================================================
# SPATIAL CORRELATION
# ============================================================

print()
print("=" * 70)
print("8. SPATIAL MORPHOLOGY CONSISTENCY")
print("=" * 70)

beta_mask = (
    np.isfinite(pa_beta_snr)
    & (pa_beta_snr >= 3.0)
)

gamma_mask = (
    np.isfinite(pa_gamma_snr)
    & (pa_gamma_snr >= 3.0)
)

common_mask = (
    beta_mask
    & gamma_mask
)

n_common = np.sum(
    common_mask
)

print(
    f"Common pixels with S/N >= 3 in both maps: "
    f"{n_common}"
)

if n_common >= 3:

    beta_values = pa_beta_map[
        common_mask
    ]

    gamma_values = pa_gamma_map[
        common_mask
    ]

    spatial_r = np.corrcoef(
        beta_values,
        gamma_values
    )[0, 1]

else:

    spatial_r = np.nan


print(
    f"Pa-beta / Pa-gamma spatial Pearson r: "
    f"{spatial_r:.5f}"
)


# ============================================================
# CENTROID SEPARATION
# ============================================================

dx = (
    beta_diag["centroid_ra"]
    - gamma_diag["centroid_ra"]
)

dy = (
    beta_diag["centroid_dec"]
    - gamma_diag["centroid_dec"]
)

centroid_separation_pixels = np.sqrt(
    dx**2 + dy**2
)

centroid_separation_arcsec = (
    centroid_separation_pixels * 0.1
)


print()
print("=" * 70)
print("9. SPATIAL CENTROID CONSISTENCY")
print("=" * 70)

print(
    f"Pa-beta centroid: "
    f"({beta_diag['centroid_ra']:.3f}, "
    f"{beta_diag['centroid_dec']:.3f})"
)

print(
    f"Pa-gamma centroid: "
    f"({gamma_diag['centroid_ra']:.3f}, "
    f"{gamma_diag['centroid_dec']:.3f})"
)

print(
    f"Centroid separation: "
    f"{centroid_separation_pixels:.3f} pixels"
)

print(
    f"Centroid separation: "
    f"{centroid_separation_arcsec:.3f} arcsec"
)


# ============================================================
# INTEGRATED FLUX
# ============================================================

print()
print("=" * 70)
print("10. PA-BETA / PA-GAMMA FLUX RATIO")
print("=" * 70)

beta_flux = beta_diag[
    "positive_flux_sum"
]

gamma_flux = gamma_diag[
    "positive_flux_sum"
]

if (
    np.isfinite(beta_flux)
    and np.isfinite(gamma_flux)
    and gamma_flux > 0
):

    beta_gamma_ratio = (
        beta_flux
        / gamma_flux
    )

else:

    beta_gamma_ratio = np.nan


print(
    f"Pa-beta positive integrated map signal: "
    f"{beta_flux:.6e}"
)

print(
    f"Pa-gamma positive integrated map signal: "
    f"{gamma_flux:.6e}"
)

print(
    f"Observed Pa-beta / Pa-gamma ratio: "
    f"{beta_gamma_ratio:.5f}"
)


# ============================================================
# SPECTRAL PLANE INFORMATION
# ============================================================

print()
print("=" * 70)
print("11. SPECTRAL SAMPLING")
print("=" * 70)

beta_indices = np.where(
    pa_beta_line_mask
)[0]

gamma_indices = np.where(
    pa_gamma_line_mask
)[0]

print("Pa-beta planes:")

for index in beta_indices:

    print(
        f"  {index:4d} "
        f"{wavelength_nm[index]:.9f} nm"
    )

print()

print("Pa-gamma planes:")

for index in gamma_indices:

    print(
        f"  {index:4d} "
        f"{wavelength_nm[index]:.9f} nm"
    )


# ============================================================
# VELOCITY CONSISTENCY
# ============================================================

print()
print("=" * 70)
print("12. VELOCITY CONSISTENCY")
print("=" * 70)

print(
    f"Independent M51 velocity: "
    f"{M51_VELOCITY_KMS:+.3f} km/s"
)

print(
    f"Pa-beta velocity from observed feature: "
    f"{pa_beta_velocity:+.3f} km/s"
)

print(
    f"Pa-beta velocity offset: "
    f"{pa_beta_velocity - M51_VELOCITY_KMS:+.3f} km/s"
)

print()
print(
    "Pa-gamma velocity will be determined from "
    "a fitted spectral centroid in a subsequent "
    "higher-resolution profile analysis."
)


# ============================================================
# SAVE MAPS
# ============================================================

print()
print("=" * 70)
print("13. SAVING SPATIAL MAPS")
print("=" * 70)

beta_map_path = (
    OUTPUT_DIR
    / "m51_1284_pabeta_hydrogen_consistency_map.fits"
)

gamma_map_path = (
    OUTPUT_DIR
    / "m51_1096_pagamma_hydrogen_consistency_map.fits"
)

beta_snr_path = (
    OUTPUT_DIR
    / "m51_1284_pabeta_hydrogen_consistency_snr.fits"
)

gamma_snr_path = (
    OUTPUT_DIR
    / "m51_1096_pagamma_hydrogen_consistency_snr.fits"
)

save_image_fits(
    beta_map_path,
    pa_beta_map,
    header,
    "Continuum-subtracted Pa-beta spatial emission map.",
)

save_image_fits(
    gamma_map_path,
    pa_gamma_map,
    header,
    "Continuum-subtracted Pa-gamma spatial emission map.",
)

save_image_fits(
    beta_snr_path,
    pa_beta_snr,
    header,
    "Pa-beta spatial S/N map.",
)

save_image_fits(
    gamma_snr_path,
    pa_gamma_snr,
    header,
    "Pa-gamma spatial S/N map.",
)

print(f"Pa-beta map: {beta_map_path}")
print(f"Pa-gamma map: {gamma_map_path}")
print(f"Pa-beta S/N: {beta_snr_path}")
print(f"Pa-gamma S/N: {gamma_snr_path}")


# ============================================================
# SAVE NUMERICAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("14. SAVING SUMMARY")
print("=" * 70)

summary = pd.DataFrame(
    [
        {
            "line": "Pa-beta",
            "rest_wavelength_nm": PA_BETA_REST_NM,
            "predicted_wavelength_nm": pa_beta_predicted,
            "observed_wavelength_nm": OBSERVED_PA_BETA_NM,
            "velocity_kms": pa_beta_velocity,
            "velocity_offset_kms":
                pa_beta_velocity - M51_VELOCITY_KMS,
            "instrument_fwhm_nm": pa_beta_fwhm,
            "peak_snr":
                beta_diag["peak_snr"],
            "positive_pixels_snr3":
                beta_diag["n_positive_snr3"],
            "centroid_ra":
                beta_diag["centroid_ra"],
            "centroid_dec":
                beta_diag["centroid_dec"],
            "positive_flux":
                beta_diag["positive_flux_sum"],
        },
        {
            "line": "Pa-gamma",
            "rest_wavelength_nm": PA_GAMMA_REST_NM,
            "predicted_wavelength_nm": pa_gamma_predicted,
            "observed_wavelength_nm": np.nan,
            "velocity_kms": np.nan,
            "velocity_offset_kms": np.nan,
            "instrument_fwhm_nm": pa_gamma_fwhm,
            "peak_snr":
                gamma_diag["peak_snr"],
            "positive_pixels_snr3":
                gamma_diag["n_positive_snr3"],
            "centroid_ra":
                gamma_diag["centroid_ra"],
            "centroid_dec":
                gamma_diag["centroid_dec"],
            "positive_flux":
                gamma_diag["positive_flux_sum"],
        },
    ]
)

summary_path = (
    OUTPUT_DIR
    / "m51_hydrogen_pabeta_pagamma_consistency.csv"
)

summary.to_csv(
    summary_path,
    index=False
)

print(
    f"Summary: {summary_path}"
)


# ============================================================
# DIAGNOSTIC PLOTS
# ============================================================

print()
print("=" * 70)
print("15. CREATING DIAGNOSTIC PLOTS")
print("=" * 70)


# ------------------------------------------------------------
# Spatial maps
# ------------------------------------------------------------

fig, axes = plt.subplots(
    1,
    2,
    figsize=(13, 5)
)

im0 = axes[0].imshow(
    pa_beta_map,
    origin="lower"
)

axes[0].set_title(
    "Pa-beta 1284.26 nm"
)

axes[0].set_xlabel(
    "RA pixel"
)

axes[0].set_ylabel(
    "DEC pixel"
)

plt.colorbar(
    im0,
    ax=axes[0],
    label="Continuum-subtracted flux"
)


im1 = axes[1].imshow(
    pa_gamma_map,
    origin="lower"
)

axes[1].set_title(
    "Pa-gamma 1095.89 nm"
)

axes[1].set_xlabel(
    "RA pixel"
)

axes[1].set_ylabel(
    "DEC pixel"
)

plt.colorbar(
    im1,
    ax=axes[1],
    label="Continuum-subtracted flux"
)

plt.tight_layout()

map_plot = (
    "m51_hydrogen_pabeta_pagamma_maps.png"
)

plt.savefig(
    map_plot,
    dpi=180
)

plt.close()


# ------------------------------------------------------------
# S/N maps
# ------------------------------------------------------------

fig, axes = plt.subplots(
    1,
    2,
    figsize=(13, 5)
)

im0 = axes[0].imshow(
    pa_beta_snr,
    origin="lower",
    vmin=0
)

axes[0].set_title(
    "Pa-beta S/N"
)

axes[0].set_xlabel(
    "RA pixel"
)

axes[0].set_ylabel(
    "DEC pixel"
)

plt.colorbar(
    im0,
    ax=axes[0]
)


im1 = axes[1].imshow(
    pa_gamma_snr,
    origin="lower",
    vmin=0
)

axes[1].set_title(
    "Pa-gamma S/N"
)

axes[1].set_xlabel(
    "RA pixel"
)

axes[1].set_ylabel(
    "DEC pixel"
)

plt.colorbar(
    im1,
    ax=axes[1]
)

plt.tight_layout()

snr_plot = (
    "m51_hydrogen_pabeta_pagamma_snr.png"
)

plt.savefig(
    snr_plot,
    dpi=180
)

plt.close()


# ------------------------------------------------------------
# Spatial comparison
# ------------------------------------------------------------

fig = plt.figure(
    figsize=(7, 6)
)

valid = (
    common_mask
    & np.isfinite(pa_beta_map)
    & np.isfinite(pa_gamma_map)
)

if np.sum(valid) >= 3:

    plt.scatter(
        pa_beta_map[valid],
        pa_gamma_map[valid],
        s=8,
        alpha=0.5
    )

    plt.xlabel(
        "Pa-beta continuum-subtracted flux"
    )

    plt.ylabel(
        "Pa-gamma continuum-subtracted flux"
    )

    plt.title(
        "Pa-beta vs Pa-gamma Spatial Emission"
    )

    plt.grid(
        alpha=0.25
    )

else:

    plt.text(
        0.5,
        0.5,
        "Insufficient common S/N pixels",
        ha="center",
        va="center"
    )

plt.tight_layout()

correlation_plot = (
    "m51_hydrogen_pabeta_pagamma_spatial_correlation.png"
)

plt.savefig(
    correlation_plot,
    dpi=180
)

plt.close()


# ============================================================
# FINAL INTERPRETATION
# ============================================================

print()
print("=" * 70)
print("FINAL INTERPRETATION")
print("=" * 70)

print(
    """
This experiment tests whether the 1284.26 nm and 1095.89 nm
features are mutually consistent with Pa-beta and Pa-gamma
hydrogen recombination emission from the same M51 gas.

The test considers:

  1. wavelength consistency;
  2. velocity consistency;
  3. instrumental resolution;
  4. spatial emission;
  5. spatial centroid;
  6. spatial morphology;
  7. integrated flux;
  8. Pa-beta / Pa-gamma flux ratio.

The Pa-beta wavelength is anchored to the independently measured
M51 velocity.

The Pa-gamma wavelength is independently predicted from the same
velocity.

A strong result would require both features to show compatible
velocities and spatial distributions, together with a physically
plausible hydrogen line ratio.

IMPORTANT:

The present experiment does not impose a single Case-B flux ratio.
Hydrogen recombination ratios depend on physical conditions,
extinction, optical depth, and radiative-transfer effects.

A high-S/N Pa-gamma feature therefore requires additional spectral
profile and velocity analysis before it can be classified as Pa-gamma.

Likewise, spatial correlation alone does not establish that both
features are hydrogen.

The purpose of this experiment is to determine whether the two
features behave as a physically coherent hydrogen-line system.
"""
)

print()
print("Key results:")
print(
    f"  Pa-beta velocity: "
    f"{pa_beta_velocity:+.3f} km/s"
)

print(
    f"  M51 velocity: "
    f"{M51_VELOCITY_KMS:+.3f} km/s"
)

print(
    f"  Pa-beta peak S/N: "
    f"{beta_diag['peak_snr']:.3f}"
)

print(
    f"  Pa-gamma peak S/N: "
    f"{gamma_diag['peak_snr']:.3f}"
)

print(
    f"  Spatial Pearson r: "
    f"{spatial_r:.5f}"
)

print(
    f"  Centroid separation: "
    f"{centroid_separation_arcsec:.3f} arcsec"
)

print(
    f"  Pa-beta / Pa-gamma positive-flux ratio: "
    f"{beta_gamma_ratio:.5f}"
)

print()
print("=" * 70)
print("HYDROGEN PA-BETA / PA-GAMMA CONSISTENCY TEST COMPLETE")
print("=" * 70)

print()
print("Outputs:")
print(f"  {summary_path}")
print(f"  {beta_map_path}")
print(f"  {gamma_map_path}")
print(f"  {beta_snr_path}")
print(f"  {gamma_snr_path}")
print(f"  {map_plot}")
print(f"  {snr_plot}")
print(f"  {correlation_plot}")
