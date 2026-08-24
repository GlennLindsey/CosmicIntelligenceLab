#!/usr/bin/env python3

"""
M51 JWST NIRSpec — Pa-beta / Pa-gamma Direct Spectral-Profile Cross-Check

Purpose
-------
Apply the same aperture-spectrum and continuum-subtraction methodology
to Pa-beta and Pa-gamma, using the exact 69-pixel nominal JWST aperture.

This is a controlled methodological cross-check.

The existing Pa-gamma consistency product is NOT used to calculate
the primary Pa-beta / Pa-gamma ratio. It is retained only as a
comparison diagnostic.

No extinction inference is performed here.

Outputs
-------
data/atomic_lines/m51_pabeta_pagamma_profile_crosscheck.csv
data/atomic_lines/m51_pabeta_pagamma_profile_crosscheck_summary.csv
data/atomic_lines/m51_pabeta_pagamma_aperture_profiles.csv
m51_pabeta_pagamma_profile_crosscheck.png
m51_pabeta_pagamma_window_convergence.png
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

PROJECT = Path.home() / "Projects" / "cosmic_ai"

S3D_PATH = (
    PROJECT
    / "data"
    / "m51_jwst_level3"
    / "jw03435-o006_t010_nirspec_g140m-f100lp_s3d.fits"
)

APERTURE_PATH = PROJECT / "data" / "atomic_lines" / "m51_jwst_extraction_aperture.csv"

EXISTING_PAGAMMA_PATH = (
    PROJECT / "data" / "atomic_lines" / "m51_1096_pagamma_hydrogen_consistency_map.fits"
)

OUTPUT_DIR = PROJECT / "data" / "atomic_lines"

RESULTS_PATH = OUTPUT_DIR / "m51_pabeta_pagamma_profile_crosscheck.csv"

SUMMARY_PATH = OUTPUT_DIR / "m51_pabeta_pagamma_profile_crosscheck_summary.csv"

PROFILE_PATH = OUTPUT_DIR / "m51_pabeta_pagamma_aperture_profiles.csv"

FIGURE_PATH = PROJECT / "m51_pabeta_pagamma_profile_crosscheck.png"

WINDOW_FIGURE_PATH = PROJECT / "m51_pabeta_pagamma_window_convergence.png"


# ============================================================
# CONSTANTS
# ============================================================

C_KMS = 299792.458

M51_VELOCITY_KMS = 463.0

PA_BETA_REST_NM = 1281.807
PA_GAMMA_REST_NM = 1093.800

RESOLVING_POWER = 3200.0

# Continuum windows deliberately kept analogous to the
# validated Pa-beta extraction methodology.

PABETA_BLUE = (1265.0, 1275.0)
PABETA_RED = (1290.0, 1300.0)

PAGAMMA_BLUE = (1080.0, 1088.0)
PAGAMMA_RED = (1100.0, 1108.0)

# Windows are expressed in numbers of spectral planes
# around the nearest observed line plane.

WINDOW_RADII = [0, 1, 2, 3, 4]


# ============================================================
# UTILITY FUNCTIONS
# ============================================================


def banner(text):
    print()
    print("=" * 70)
    print(text)
    print("=" * 70)


def predicted_wavelength(rest_nm, velocity_kms):
    return rest_nm * (1.0 + velocity_kms / C_KMS)


def velocity_from_wavelength(rest_nm, observed_nm):
    return (observed_nm / rest_nm - 1.0) * C_KMS


def find_sci_hdu(hdul):
    for index, hdu in enumerate(hdul):
        if hdu.data is None:
            continue

        if getattr(hdu, "name", "").upper() == "SCI":
            return index, hdu

    for index, hdu in enumerate(hdul):
        if hdu.data is not None:
            data = np.asarray(hdu.data)

            if data.ndim == 3:
                return index, hdu

    raise RuntimeError("Could not locate a 3-D SCI cube.")


def find_err_hdu(hdul):
    for index, hdu in enumerate(hdul):
        if getattr(hdu, "name", "").upper() == "ERR":
            return index, hdu

    return None, None


def wavelength_from_spectral_wcs(
    sci_hdu,
    n_spectral,
):
    """
    Construct wavelength array from the spectral WCS.

    JWST S3D WCS stores wavelength in metres.

    Convert explicitly to nm.
    """

    header = sci_hdu.header

    wcs = WCS(header)

    # --------------------------------------------------------
    # First try the full WCS.
    # --------------------------------------------------------

    try:
        if wcs.pixel_n_dim == 3:

            x = np.zeros(n_spectral)
            y = np.zeros(n_spectral)
            z = np.arange(n_spectral)

            world = wcs.pixel_to_world_values(
                x,
                y,
                z,
            )

            # Spectral world coordinate is normally the
            # third returned axis.

            spectral = np.asarray(
                world[2],
                dtype=float,
            )

            # WCS spectral unit is metres.
            return spectral * 1.0e9

    except Exception:
        pass

    # --------------------------------------------------------
    # Direct CRVAL3 / CRPIX3 / CDELT3 fallback.
    # --------------------------------------------------------

    required = (
        "CRVAL3",
        "CRPIX3",
        "CDELT3",
    )

    if all(key in header for key in required):

        pixels = np.arange(
            n_spectral,
            dtype=float,
        )

        wavelength_m = (
            header["CRVAL3"] + (pixels + 1.0 - header["CRPIX3"]) * header["CDELT3"]
        )

        return wavelength_m * 1.0e9

    raise RuntimeError("Unable to construct spectral wavelength array.")


def interpolate_continuum(
    wavelength_nm,
    spectrum,
    blue_window,
    red_window,
):
    """
    Fit a linear continuum using the median flux in
    each spectral plane of the two continuum regions.
    """

    blue = (wavelength_nm >= blue_window[0]) & (wavelength_nm <= blue_window[1])

    red = (wavelength_nm >= red_window[0]) & (wavelength_nm <= red_window[1])

    if not np.any(blue):
        raise RuntimeError(f"No blue continuum planes found: " f"{blue_window}")

    if not np.any(red):
        raise RuntimeError(f"No red continuum planes found: " f"{red_window}")

    x = np.concatenate(
        [
            wavelength_nm[blue],
            wavelength_nm[red],
        ]
    )

    y = np.concatenate(
        [
            spectrum[blue],
            spectrum[red],
        ]
    )

    valid = np.isfinite(x) & np.isfinite(y)

    x = x[valid]
    y = y[valid]

    if len(x) < 2:
        raise RuntimeError("Insufficient valid continuum points.")

    slope, intercept = np.polyfit(
        x,
        y,
        1,
    )

    continuum = slope * wavelength_nm + intercept

    return (
        continuum,
        slope,
        intercept,
        np.where(blue)[0],
        np.where(red)[0],
    )


def aperture_spectrum(
    cube,
    aperture_xy,
):
    """
    Sum SCI cube over the exact 69-pixel aperture.
    """

    ys = aperture_xy["y_pixel"].astype(int)
    xs = aperture_xy["x_pixel"].astype(int)

    selected = cube[:, ys, xs]

    spectrum = np.nansum(
        selected,
        axis=1,
    )

    return spectrum


def aperture_error_spectrum(
    err_cube,
    aperture_xy,
):
    """
    Propagate independent pixel errors in quadrature.
    """

    ys = aperture_xy["y_pixel"].astype(int)
    xs = aperture_xy["x_pixel"].astype(int)

    selected = err_cube[:, ys, xs]

    variance = np.nansum(
        selected**2,
        axis=1,
    )

    return np.sqrt(variance)


def extract_line_flux(
    wavelength_nm,
    continuum_subtracted,
    error_spectrum,
    line_index,
    radius,
):
    """
    Integrate a symmetric spectral-plane window around
    the nearest line plane.

    Integration uses the actual spectral spacing.
    """

    low = max(
        0,
        line_index - radius,
    )

    high = min(
        len(wavelength_nm) - 1,
        line_index + radius,
    )

    indices = np.arange(
        low,
        high + 1,
    )

    wave = wavelength_nm[indices]
    flux = continuum_subtracted[indices]

    if len(wave) == 1:
        flux_integrated = flux[0] * np.nanmedian(np.diff(wavelength_nm))
    else:
        flux_integrated = np.trapezoid(
            flux,
            wave,
        )

    errors = error_spectrum[indices]

    spacing = np.nanmedian(np.diff(wavelength_nm))

    sigma = np.sqrt(np.nansum(errors**2)) * spacing

    return (
        float(flux_integrated),
        float(sigma),
        indices,
    )


def profile_fwhm(
    wavelength_nm,
    continuum_subtracted,
    line_index,
):
    """
    Estimate FWHM from the continuum-subtracted aperture
    spectrum.
    """

    flux = np.asarray(
        continuum_subtracted,
        dtype=float,
    )

    finite = np.isfinite(flux)

    if not np.any(finite):
        return np.nan

    peak_index = line_index

    # Search locally around the line.
    local_low = max(
        0,
        line_index - 10,
    )

    local_high = min(
        len(flux),
        line_index + 11,
    )

    local = flux[local_low:local_high]

    if not np.any(np.isfinite(local)):
        return np.nan

    local_peak_offset = np.nanargmax(local)

    peak_index = local_low + local_peak_offset

    peak = flux[peak_index]

    if not np.isfinite(peak) or peak <= 0:
        return np.nan

    half = peak / 2.0

    # Find nearest left crossing.
    left = np.nan

    for i in range(
        peak_index - 1,
        -1,
        -1,
    ):
        if np.isfinite(flux[i]) and flux[i] <= half:
            x1 = wavelength_nm[i]
            x2 = wavelength_nm[i + 1]
            y1 = flux[i]
            y2 = flux[i + 1]

            if y2 != y1:
                left = x1 + (half - y1) * (x2 - x1) / (y2 - y1)
            else:
                left = x1

            break

    # Find nearest right crossing.
    right = np.nan

    for i in range(
        peak_index + 1,
        len(flux),
    ):
        if np.isfinite(flux[i]) and flux[i] <= half:
            x1 = wavelength_nm[i - 1]
            x2 = wavelength_nm[i]
            y1 = flux[i - 1]
            y2 = flux[i]

            if y2 != y1:
                right = x1 + (half - y1) * (x2 - x1) / (y2 - y1)
            else:
                right = x2

            break

    if not np.isfinite(left) or not np.isfinite(right):
        return np.nan

    return float(right - left)


# ============================================================
# MAIN
# ============================================================


def main():

    banner(
        "M51 JWST NIRSPEC — Pa-beta / Pa-gamma " "DIRECT SPECTRAL-PROFILE CROSS-CHECK"
    )

    print("Purpose:")
    print(
        "Apply the validated Pa-beta aperture-spectrum "
        "methodology consistently to Pa-beta and Pa-gamma."
    )
    print()
    print("The existing Pa-gamma product is comparison-only.")
    print("No extinction inference is performed.")

    # ========================================================
    # 1. READ S3D
    # ========================================================

    banner("1. READING JWST S3D")

    with fits.open(S3D_PATH) as hdul:

        sci_index, sci_hdu = find_sci_hdu(hdul)

        err_index, err_hdu = find_err_hdu(hdul)

        sci = np.asarray(
            sci_hdu.data,
            dtype=float,
        )

        if err_hdu is None:
            raise RuntimeError("ERR HDU not found.")

        err = np.asarray(
            err_hdu.data,
            dtype=float,
        )

        wavelength_nm = wavelength_from_spectral_wcs(
            sci_hdu,
            sci.shape[0],
        )

        print(f"SCI HDU = {sci_index}")
        print(f"ERR HDU = {err_index}")
        print(f"SCI shape = {sci.shape}")
        print(f"ERR shape = {err.shape}")

        print(
            f"Wavelength range = "
            f"{wavelength_nm[0]:.9f} - "
            f"{wavelength_nm[-1]:.9f} nm"
        )

        spacing = np.nanmedian(np.diff(wavelength_nm))

        print(f"Spectral spacing = " f"{spacing:.9f} nm")

    # ========================================================
    # 2. LOAD APERTURE
    # ========================================================

    banner("2. LOADING NOMINAL APERTURE")

    aperture = pd.read_csv(APERTURE_PATH)

    if "inside_nominal_aperture" in aperture.columns:

        aperture = aperture[aperture["inside_nominal_aperture"].astype(bool)].copy()

    if len(aperture) != 69:
        raise RuntimeError(
            f"Expected exactly 69 aperture pixels, " f"found {len(aperture)}."
        )

    print(f"Aperture pixels = {len(aperture)}")

    # ========================================================
    # 3. LINE PARAMETERS
    # ========================================================

    banner("3. PREDICTING Pa-beta / Pa-gamma")

    pabeta_center = predicted_wavelength(
        PA_BETA_REST_NM,
        M51_VELOCITY_KMS,
    )

    pagamma_center = predicted_wavelength(
        PA_GAMMA_REST_NM,
        M51_VELOCITY_KMS,
    )

    print(f"Pa-beta predicted = " f"{pabeta_center:.9f} nm")

    print(f"Pa-gamma predicted = " f"{pagamma_center:.9f} nm")

    # Find nearest spectral planes.

    pabeta_index = int(np.nanargmin(np.abs(wavelength_nm - pabeta_center)))

    pagamma_index = int(np.nanargmin(np.abs(wavelength_nm - pagamma_center)))

    print(f"Pa-beta nearest plane = " f"{pabeta_index}")

    print(f"Pa-beta wavelength = " f"{wavelength_nm[pabeta_index]:.9f} nm")

    print(f"Pa-gamma nearest plane = " f"{pagamma_index}")

    print(f"Pa-gamma wavelength = " f"{wavelength_nm[pagamma_index]:.9f} nm")

    pabeta_velocity = velocity_from_wavelength(
        PA_BETA_REST_NM,
        wavelength_nm[pabeta_index],
    )

    pagamma_velocity = velocity_from_wavelength(
        PA_GAMMA_REST_NM,
        wavelength_nm[pagamma_index],
    )

    print(f"Pa-beta velocity = " f"{pabeta_velocity:+.3f} km/s")

    print(f"Pa-gamma velocity = " f"{pagamma_velocity:+.3f} km/s")

    # ========================================================
    # 4. APERTURE SPECTRA
    # ========================================================

    banner("4. EXTRACTING 69-PIXEL APERTURE SPECTRA")

    aperture_sci = aperture_spectrum(
        sci,
        aperture,
    )

    aperture_err = aperture_error_spectrum(
        err,
        aperture,
    )

    print("Summed SCI aperture spectrum created.")

    print("Quadrature aperture error spectrum created.")

    # ========================================================
    # 5. CONTINUUM SUBTRACTION
    # ========================================================

    banner("5. BUILDING CONTINUUM-SUBTRACTED PROFILES")

    (
        pabeta_continuum,
        pabeta_slope,
        pabeta_intercept,
        pabeta_blue_idx,
        pabeta_red_idx,
    ) = interpolate_continuum(
        wavelength_nm,
        aperture_sci,
        PABETA_BLUE,
        PABETA_RED,
    )

    (
        pagamma_continuum,
        pagamma_slope,
        pagamma_intercept,
        pagamma_blue_idx,
        pagamma_red_idx,
    ) = interpolate_continuum(
        wavelength_nm,
        aperture_sci,
        PAGAMMA_BLUE,
        PAGAMMA_RED,
    )

    pabeta_residual = aperture_sci - pabeta_continuum

    pagamma_residual = aperture_sci - pagamma_continuum

    print("Pa-beta continuum:")
    print(f"  slope = {pabeta_slope:.8e}")
    print(f"  intercept = {pabeta_intercept:.8e}")
    print(f"  blue planes = " f"{len(pabeta_blue_idx)}")
    print(f"  red planes = " f"{len(pabeta_red_idx)}")

    print()
    print("Pa-gamma continuum:")
    print(f"  slope = {pagamma_slope:.8e}")
    print(f"  intercept = {pagamma_intercept:.8e}")
    print(f"  blue planes = " f"{len(pagamma_blue_idx)}")
    print(f"  red planes = " f"{len(pagamma_red_idx)}")

    # ========================================================
    # 6. WINDOW TEST
    # ========================================================

    banner("6. WINDOW-BY-WINDOW CROSS-CHECK")

    rows = []

    for radius in WINDOW_RADII:

        pabeta_flux, pabeta_sigma, pabeta_indices = extract_line_flux(
            wavelength_nm,
            pabeta_residual,
            aperture_err,
            pabeta_index,
            radius,
        )

        pagamma_flux, pagamma_sigma, pagamma_indices = extract_line_flux(
            wavelength_nm,
            pagamma_residual,
            aperture_err,
            pagamma_index,
            radius,
        )

        if pagamma_flux != 0:
            ratio = pabeta_flux / pagamma_flux
        else:
            ratio = np.nan

        if (
            np.isfinite(pabeta_sigma)
            and pabeta_sigma > 0
            and np.isfinite(pagamma_sigma)
            and pagamma_sigma > 0
            and np.isfinite(pabeta_flux)
            and np.isfinite(pagamma_flux)
        ):
            ratio_sigma = ratio * np.sqrt(
                (pabeta_sigma / pabeta_flux) ** 2 + (pagamma_sigma / pagamma_flux) ** 2
            )
        else:
            ratio_sigma = np.nan

        pabeta_wave_low = wavelength_nm[pabeta_indices[0]]

        pabeta_wave_high = wavelength_nm[pabeta_indices[-1]]

        pagamma_wave_low = wavelength_nm[pagamma_indices[0]]

        pagamma_wave_high = wavelength_nm[pagamma_indices[-1]]

        print()
        print(f"Window ±{radius} planes")

        print(f"  Pa-beta flux = " f"{pabeta_flux:.8f}")

        print(f"  Pa-gamma flux = " f"{pagamma_flux:.8f}")

        print(f"  Pa-beta / Pa-gamma = " f"{ratio:.8f}")

        print(f"  Ratio uncertainty = " f"{ratio_sigma:.8f}")

        rows.append(
            {
                "window_radius_planes": radius,
                "pabeta_planes": len(pabeta_indices),
                "pagamma_planes": len(pagamma_indices),
                "pabeta_wave_low_nm": pabeta_wave_low,
                "pabeta_wave_high_nm": pabeta_wave_high,
                "pagamma_wave_low_nm": pagamma_wave_low,
                "pagamma_wave_high_nm": pagamma_wave_high,
                "pabeta_flux": pabeta_flux,
                "pabeta_sigma": pabeta_sigma,
                "pagamma_flux": pagamma_flux,
                "pagamma_sigma": pagamma_sigma,
                "pabeta_pagamma_ratio": ratio,
                "pabeta_pagamma_ratio_sigma": ratio_sigma,
            }
        )

    results = pd.DataFrame(rows)

    # ========================================================
    # 7. PROFILE CHARACTERISTICS
    # ========================================================

    banner("7. PROFILE CHARACTERISTICS")

    pabeta_fwhm = profile_fwhm(
        wavelength_nm,
        pabeta_residual,
        pabeta_index,
    )

    pagamma_fwhm = profile_fwhm(
        wavelength_nm,
        pagamma_residual,
        pagamma_index,
    )

    print(f"Pa-beta FWHM = " f"{pabeta_fwhm:.6f} nm")

    print(
        f"Pa-beta FWHM velocity = " f"{pabeta_fwhm / PA_BETA_REST_NM * C_KMS:.3f} km/s"
    )

    print(f"Pa-gamma FWHM = " f"{pagamma_fwhm:.6f} nm")

    print(
        f"Pa-gamma FWHM velocity = "
        f"{pagamma_fwhm / PA_GAMMA_REST_NM * C_KMS:.3f} km/s"
    )

    # ========================================================
    # 8. EXISTING Pa-GAMMA COMPARISON
    # ========================================================

    banner("8. EXISTING Pa-GAMMA PRODUCT COMPARISON")

    existing_flux = np.nan

    if EXISTING_PAGAMMA_PATH.exists():

        with fits.open(EXISTING_PAGAMMA_PATH) as hdul:

            existing = np.asarray(
                hdul[0].data,
                dtype=float,
            )

        ys = aperture["y_pixel"].astype(int)

        xs = aperture["x_pixel"].astype(int)

        values = existing[
            ys,
            xs,
        ]

        finite = np.isfinite(values)

        positive = finite & (values > 0)

        existing_flux = np.nansum(values)

        print(f"Existing product finite = " f"{np.sum(finite)}")

        print(f"Existing product positive = " f"{np.sum(positive)}")

        print(f"Existing product aperture sum = " f"{existing_flux:.8f}")

    else:

        print("Existing Pa-gamma product not found.")

    # ========================================================
    # 9. CONVERGENCE TEST
    # ========================================================

    banner("9. Pa-GAMMA CONVERGENCE")

    if len(results) >= 2:

        for i in range(
            1,
            len(results),
        ):

            previous = results.iloc[i - 1]

            current = results.iloc[i]

            f0 = previous["pagamma_flux"]

            f1 = current["pagamma_flux"]

            if np.isfinite(f0) and np.isfinite(f1) and f0 != 0:

                change = (f1 - f0) / abs(f0) * 100.0

                print(
                    f"±{int(current['window_radius_planes'])}: "
                    f"{change:+.3f}% relative to "
                    f"previous window"
                )

    # ========================================================
    # 10. SAVE SPECTRAL PROFILE
    # ========================================================

    banner("10. SAVING SPECTRAL PROFILES")

    profile = pd.DataFrame(
        {
            "wavelength_nm": wavelength_nm,
            "aperture_flux": aperture_sci,
            "aperture_error": aperture_err,
            "pabeta_continuum": pabeta_continuum,
            "pabeta_residual": pabeta_residual,
            "pagamma_continuum": pagamma_continuum,
            "pagamma_residual": pagamma_residual,
        }
    )

    profile.to_csv(
        PROFILE_PATH,
        index=False,
    )

    print(f"Saved:\n  {PROFILE_PATH}")

    results.to_csv(
        RESULTS_PATH,
        index=False,
    )

    print(f"Saved:\n  {RESULTS_PATH}")

    # ========================================================
    # 11. SUMMARY
    # ========================================================

    summary_rows = []

    for _, row in results.iterrows():

        summary_rows.append(
            {
                "window_radius_planes": row["window_radius_planes"],
                "pabeta_flux": row["pabeta_flux"],
                "pagamma_flux": row["pagamma_flux"],
                "pabeta_pagamma_ratio": row["pabeta_pagamma_ratio"],
                "ratio_sigma": row["pabeta_pagamma_ratio_sigma"],
                "existing_pagamma_flux": existing_flux,
                "pagamma_fraction_existing": (
                    row["pagamma_flux"] / existing_flux
                    if np.isfinite(existing_flux) and existing_flux != 0
                    else np.nan
                ),
            }
        )

    summary = pd.DataFrame(summary_rows)

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    print(f"Saved:\n  {SUMMARY_PATH}")

    # ========================================================
    # 12. PROFILE FIGURE
    # ========================================================

    banner("12. CREATING PROFILE FIGURE")

    plt.figure(figsize=(11, 7))

    # Pa-beta
    ax1 = plt.subplot(2, 1, 1)

    ax1.plot(
        wavelength_nm,
        pabeta_residual,
        linewidth=1.5,
    )

    ax1.axvline(
        pabeta_center,
        linestyle="--",
        label="Predicted Pa-beta",
    )

    ax1.axvline(
        wavelength_nm[pabeta_index],
        linestyle=":",
        label="Nearest cube plane",
    )

    ax1.set_xlim(
        pabeta_center - 10,
        pabeta_center + 10,
    )

    ax1.set_ylabel("Continuum-subtracted flux")

    ax1.set_title("M51 JWST 69-pixel Aperture — Pa-beta")

    ax1.legend()

    # Pa-gamma
    ax2 = plt.subplot(2, 1, 2)

    ax2.plot(
        wavelength_nm,
        pagamma_residual,
        linewidth=1.5,
    )

    ax2.axvline(
        pagamma_center,
        linestyle="--",
        label="Predicted Pa-gamma",
    )

    ax2.axvline(
        wavelength_nm[pagamma_index],
        linestyle=":",
        label="Nearest cube plane",
    )

    ax2.set_xlim(
        pagamma_center - 8,
        pagamma_center + 8,
    )

    ax2.set_xlabel("Wavelength (nm)")

    ax2.set_ylabel("Continuum-subtracted flux")

    ax2.set_title("M51 JWST 69-pixel Aperture — Pa-gamma")

    ax2.legend()

    plt.tight_layout()

    plt.savefig(
        FIGURE_PATH,
        dpi=180,
    )

    plt.close()

    print(f"Saved:\n  {FIGURE_PATH}")

    # ========================================================
    # 13. WINDOW CONVERGENCE FIGURE
    # ========================================================

    banner("13. CREATING WINDOW-CONVERGENCE FIGURE")

    plt.figure(figsize=(10, 6))

    x = results["window_radius_planes"]

    plt.plot(
        x,
        results["pabeta_flux"],
        marker="o",
        label="Pa-beta",
    )

    plt.plot(
        x,
        results["pagamma_flux"],
        marker="o",
        label="Pa-gamma",
    )

    plt.xlabel("Window radius (spectral planes)")

    plt.ylabel("Integrated line flux")

    plt.title("M51 JWST Pa-beta / Pa-gamma " "Spectral-Window Convergence")

    plt.grid(alpha=0.25)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        WINDOW_FIGURE_PATH,
        dpi=180,
    )

    plt.close()

    print(f"Saved:\n  {WINDOW_FIGURE_PATH}")

    # ========================================================
    # 14. FINAL RESULT
    # ========================================================

    banner("FINAL DIRECT PROFILE CROSS-CHECK")

    print(f"Nominal aperture = 69 pixels")

    print()

    print(f"Pa-beta predicted = " f"{pabeta_center:.9f} nm")

    print(f"Pa-beta observed plane = " f"{wavelength_nm[pabeta_index]:.9f} nm")

    print()

    print(f"Pa-gamma predicted = " f"{pagamma_center:.9f} nm")

    print(f"Pa-gamma observed plane = " f"{wavelength_nm[pagamma_index]:.9f} nm")

    print()

    print("Window results:")

    for _, row in results.iterrows():

        print(
            f"  ±{int(row['window_radius_planes'])}: "
            f"Pa-beta = "
            f"{row['pabeta_flux']:.6f}, "
            f"Pa-gamma = "
            f"{row['pagamma_flux']:.6f}, "
            f"ratio = "
            f"{row['pabeta_pagamma_ratio']:.6f}"
        )

    print()

    print("IMPORTANT:")

    print(
        "The Pa-beta / Pa-gamma ratio should not be "
        "taken into extinction analysis until the "
        "spectral-window convergence is assessed."
    )

    print()

    print("Direct Pa-beta / Pa-gamma spectral-profile " "cross-check complete.")


if __name__ == "__main__":
    main()
