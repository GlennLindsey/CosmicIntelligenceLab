#!/usr/bin/env python3

"""
M51 JWST NIRSPEC — Pa-gamma aperture spectral profile

Extract the summed spectrum from the exact 69-pixel nominal
JWST aperture and characterize the Pa-gamma feature directly
from the S3D cube.

This experiment does NOT perform extinction inference.

It determines:
    - local continuum
    - Pa-gamma peak wavelength
    - peak velocity
    - continuum-subtracted line profile
    - integrated line flux
    - FWHM
    - profile symmetry
    - flux captured by several spectral windows
    - uncertainty from the ERR cube

The existing Pa-gamma products are not used to construct
the spectrum.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.io import fits


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT = Path.home() / "Projects" / "cosmic_ai"

S3D_PATH = (
    PROJECT
    / "data/m51_jwst_level3"
    / "jw03435-o006_t010_nirspec_g140m-f100lp_s3d.fits"
)

APERTURE_PATH = (
    PROJECT
    / "data/atomic_lines/m51_jwst_extraction_aperture.csv"
)

OUTPUT_CSV = (
    PROJECT
    / "data/atomic_lines/m51_pagamma_aperture_spectrum.csv"
)

OUTPUT_SUMMARY = (
    PROJECT
    / "data/atomic_lines/m51_pagamma_aperture_spectrum_summary.csv"
)

OUTPUT_FIGURE = (
    PROJECT
    / "m51_pagamma_aperture_spectrum.png"
)


# Pa-gamma
PA_GAMMA_REST_NM = 1093.800000

# M51 velocity used in previous experiments
M51_VELOCITY_KMS = 463.0

C_KMS = 299792.458


# Spectral analysis interval
SPECTRUM_LOW_NM = 1088.0
SPECTRUM_HIGH_NM = 1102.0

# Continuum regions
BLUE_CONT_LOW = 1088.0
BLUE_CONT_HIGH = 1091.0

RED_CONT_LOW = 1099.0
RED_CONT_HIGH = 1102.0

# Candidate line region
LINE_SEARCH_LOW = 1092.0
LINE_SEARCH_HIGH = 1098.5


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def banner(text):
    print()
    print("=" * 70)
    print(text)
    print("=" * 70)


def velocity_from_wavelength(rest_nm, observed_nm):
    return (
        (observed_nm / rest_nm) - 1.0
    ) * C_KMS


def wavelength_from_velocity(rest_nm, velocity_kms):
    return (
        rest_nm
        * (1.0 + velocity_kms / C_KMS)
    )


def wavelength_from_header(header, nplanes):
    """
    Construct wavelength array.

    JWST S3D spectral WCS values for this cube are in microns.
    Return wavelengths in nm.
    """

    crval3 = float(header["CRVAL3"])
    crpix3 = float(header["CRPIX3"])
    cdelt3 = float(header["CDELT3"])

    pixel = (
        np.arange(nplanes, dtype=float)
        + 1.0
    )

    wavelength_um = (
        crval3
        + (pixel - crpix3) * cdelt3
    )

    return wavelength_um * 1000.0


def linear_continuum(
    wavelength,
    flux,
    blue_low,
    blue_high,
    red_low,
    red_high,
):
    """
    Fit a linear continuum using two wavelength regions.
    """

    blue = (
        (wavelength >= blue_low)
        & (wavelength <= blue_high)
        & np.isfinite(flux)
    )

    red = (
        (wavelength >= red_low)
        & (wavelength <= red_high)
        & np.isfinite(flux)
    )

    continuum_mask = blue | red

    if np.sum(continuum_mask) < 4:
        raise RuntimeError(
            "Insufficient continuum points."
        )

    coefficients = np.polyfit(
        wavelength[continuum_mask],
        flux[continuum_mask],
        1,
    )

    continuum = np.polyval(
        coefficients,
        wavelength,
    )

    return (
        continuum,
        coefficients,
        blue,
        red,
    )


def trapezoid_integral(x, y):
    """
    Numerical integral using trapezoidal integration.
    """

    valid = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    if np.sum(valid) < 2:
        return np.nan

    return np.trapezoid(
        y[valid],
        x[valid],
    )


def fwhm_from_profile(wavelength, profile):
    """
    Estimate FWHM from the continuum-subtracted profile.

    Uses the half-maximum crossing on each side of the peak.
    """

    valid = (
        np.isfinite(wavelength)
        & np.isfinite(profile)
    )

    x = wavelength[valid]
    y = profile[valid]

    if len(x) < 3:
        return np.nan, np.nan, np.nan

    peak_index = np.argmax(y)

    peak_flux = y[peak_index]

    if peak_flux <= 0:
        return np.nan, np.nan, np.nan

    half = peak_flux / 2.0

    # --------------------------------------------------------
    # Left crossing
    # --------------------------------------------------------

    left_x = np.nan

    for i in range(
        peak_index - 1,
        -1,
        -1,
    ):
        if (
            y[i] <= half
            and y[i + 1] > half
        ):
            x1 = x[i]
            x2 = x[i + 1]
            y1 = y[i]
            y2 = y[i + 1]

            if y2 != y1:
                left_x = (
                    x1
                    + (half - y1)
                    * (x2 - x1)
                    / (y2 - y1)
                )

            break

    # --------------------------------------------------------
    # Right crossing
    # --------------------------------------------------------

    right_x = np.nan

    for i in range(
        peak_index,
        len(x) - 1,
    ):
        if (
            y[i] >= half
            and y[i + 1] < half
        ):
            x1 = x[i]
            x2 = x[i + 1]
            y1 = y[i]
            y2 = y[i + 1]

            if y2 != y1:
                right_x = (
                    x1
                    + (half - y1)
                    * (x2 - x1)
                    / (y2 - y1)
                )

            break

    if (
        np.isfinite(left_x)
        and np.isfinite(right_x)
    ):
        return (
            right_x - left_x,
            left_x,
            right_x,
        )

    return np.nan, left_x, right_x


# ============================================================
# MAIN
# ============================================================

def main():

    banner(
        "M51 JWST NIRSPEC — Pa-GAMMA APERTURE SPECTRUM"
    )

    print(
        "Purpose:"
    )
    print(
        "Extract and characterize the summed Pa-gamma spectrum"
    )
    print(
        "from the exact 69-pixel JWST nominal aperture."
    )
    print()

    # ========================================================
    # 1. READ S3D
    # ========================================================

    banner("1. READING JWST S3D")

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

        header = hdul["SCI"].header.copy()

    print("File:")
    print(f"  {S3D_PATH}")

    print()
    print("SCI shape:")
    print(f"  {sci.shape}")

    print()
    print("ERR shape:")
    print(f"  {err.shape}")

    # ========================================================
    # 2. WAVELENGTH
    # ========================================================

    banner("2. BUILDING WAVELENGTH ARRAY")

    nplanes = sci.shape[0]

    wavelength_nm = wavelength_from_header(
        header,
        nplanes,
    )

    print(
        f"Wavelength range = "
        f"{wavelength_nm[0]:.9f} - "
        f"{wavelength_nm[-1]:.9f} nm"
    )

    print(
        f"Spectral spacing = "
        f"{np.median(np.diff(wavelength_nm)):.9f} nm"
    )

    # ========================================================
    # 3. EXPECTED PA-GAMMA
    # ========================================================

    banner("3. EXPECTED PA-GAMMA")

    predicted_nm = wavelength_from_velocity(
        PA_GAMMA_REST_NM,
        M51_VELOCITY_KMS,
    )

    nearest_index = np.argmin(
        np.abs(
            wavelength_nm
            - predicted_nm
        )
    )

    nearest_nm = wavelength_nm[
        nearest_index
    ]

    nearest_velocity = velocity_from_wavelength(
        PA_GAMMA_REST_NM,
        nearest_nm,
    )

    print(
        f"Pa-gamma rest wavelength = "
        f"{PA_GAMMA_REST_NM:.6f} nm"
    )

    print(
        f"M51 velocity = "
        f"{M51_VELOCITY_KMS:+.3f} km/s"
    )

    print(
        f"Predicted wavelength = "
        f"{predicted_nm:.9f} nm"
    )

    print(
        f"Nearest cube plane = "
        f"{nearest_index}"
    )

    print(
        f"Nearest cube wavelength = "
        f"{nearest_nm:.9f} nm"
    )

    print(
        f"Nearest-plane velocity = "
        f"{nearest_velocity:+.3f} km/s"
    )

    print(
        f"Offset = "
        f"{nearest_nm - predicted_nm:+.9f} nm"
    )

    # ========================================================
    # 4. LOAD APERTURE
    # ========================================================

    banner("4. LOADING NOMINAL APERTURE")

    aperture = pd.read_csv(
        APERTURE_PATH
    )

    aperture = aperture[
        aperture[
            "inside_nominal_aperture"
        ].astype(bool)
    ].copy()

    print(
        f"Aperture pixels = "
        f"{len(aperture)}"
    )

    if len(aperture) != 69:
        raise RuntimeError(
            f"Expected 69 aperture pixels, "
            f"found {len(aperture)}."
        )

    x = (
        aperture["x_pixel"]
        .astype(int)
        .to_numpy()
    )

    y = (
        aperture["y_pixel"]
        .astype(int)
        .to_numpy()
    )

    # ========================================================
    # 5. EXTRACT SPECTRUM
    # ========================================================

    banner(
        "5. EXTRACTING 69-PIXEL SUMMED SPECTRUM"
    )

    aperture_sci = sci[
        :,
        y,
        x,
    ]

    aperture_err = err[
        :,
        y,
        x,
    ]

    # Sum spatial pixels.
    summed_flux = np.nansum(
        aperture_sci,
        axis=1,
    )

    # Independent errors summed in quadrature.
    summed_err = np.sqrt(
        np.nansum(
            aperture_err ** 2,
            axis=1,
        )
    )

    spectrum_mask = (
        (wavelength_nm >= SPECTRUM_LOW_NM)
        & (
            wavelength_nm
            <= SPECTRUM_HIGH_NM
        )
    )

    wave = wavelength_nm[
        spectrum_mask
    ]

    flux = summed_flux[
        spectrum_mask
    ]

    flux_err = summed_err[
        spectrum_mask
    ]

    print(
        f"Spectrum interval = "
        f"{SPECTRUM_LOW_NM:.1f} - "
        f"{SPECTRUM_HIGH_NM:.1f} nm"
    )

    print(
        f"Spectral planes = "
        f"{len(wave)}"
    )

    # ========================================================
    # 6. CONTINUUM
    # ========================================================

    banner("6. FITTING LOCAL CONTINUUM")

    continuum, coefficients, blue_mask, red_mask = (
        linear_continuum(
            wave,
            flux,
            BLUE_CONT_LOW,
            BLUE_CONT_HIGH,
            RED_CONT_LOW,
            RED_CONT_HIGH,
        )
    )

    line_profile = (
        flux - continuum
    )

    print(
        f"Blue continuum planes = "
        f"{np.sum(blue_mask)}"
    )

    print(
        f"Red continuum planes = "
        f"{np.sum(red_mask)}"
    )

    print()
    print(
        "Continuum:"
    )

    print(
        f"  slope = "
        f"{coefficients[0]:.8e}"
    )

    print(
        f"  intercept = "
        f"{coefficients[1]:.8e}"
    )

    # ========================================================
    # 7. PA-GAMMA PEAK
    # ========================================================

    banner("7. PA-GAMMA PEAK")

    search = (
        (wave >= LINE_SEARCH_LOW)
        & (
            wave
            <= LINE_SEARCH_HIGH
        )
        & np.isfinite(line_profile)
    )

    if not np.any(search):
        raise RuntimeError(
            "No spectral points found in Pa-gamma search region."
        )

    search_indices = np.where(
        search
    )[0]

    peak_local = search_indices[
        np.argmax(
            line_profile[
                search
            ]
        )
    ]

    peak_wave = wave[
        peak_local
    ]

    peak_flux = line_profile[
        peak_local
    ]

    peak_velocity = velocity_from_wavelength(
        PA_GAMMA_REST_NM,
        peak_wave,
    )

    print(
        f"Peak wavelength = "
        f"{peak_wave:.9f} nm"
    )

    print(
        f"Peak continuum-subtracted flux = "
        f"{peak_flux:.8f}"
    )

    print(
        f"Peak velocity = "
        f"{peak_velocity:+.3f} km/s"
    )

    # ========================================================
    # 8. FWHM
    # ========================================================

    banner("8. LINE WIDTH")

    line_wave = wave[
        search
    ]

    line_values = line_profile[
        search
    ]

    fwhm, left_half, right_half = (
        fwhm_from_profile(
            line_wave,
            line_values,
        )
    )

    print(
        f"FWHM = "
        f"{fwhm:.6f} nm"
    )

    if np.isfinite(fwhm):
        velocity_width = (
            fwhm
            / peak_wave
            * C_KMS
        )

        print(
            f"FWHM velocity width = "
            f"{velocity_width:.3f} km/s"
        )

    print(
        f"Left half-maximum = "
        f"{left_half:.6f} nm"
    )

    print(
        f"Right half-maximum = "
        f"{right_half:.6f} nm"
    )

    # ========================================================
    # 9. LINE INTEGRAL
    # ========================================================

    banner("9. INTEGRATED PA-GAMMA FLUX")

    line_flux = trapezoid_integral(
        line_wave,
        line_values,
    )

    line_error = trapezoid_integral(
        line_wave,
        np.abs(
            summed_err[
                spectrum_mask
            ][search]
        ),
    )

    print(
        f"Integrated Pa-gamma flux = "
        f"{line_flux:.8f}"
    )

    print(
        f"Approximate integrated error = "
        f"{line_error:.8f}"
    )

    if (
        np.isfinite(line_error)
        and line_error > 0
    ):
        print(
            f"Approximate S/N = "
            f"{line_flux / line_error:.3f}"
        )

    # ========================================================
    # 10. SYMMETRY
    # ========================================================

    banner("10. PROFILE SYMMETRY")

    if (
        np.isfinite(left_half)
        and np.isfinite(right_half)
    ):

        left_width = (
            peak_wave
            - left_half
        )

        right_width = (
            right_half
            - peak_wave
        )

        asymmetry = (
            (right_width - left_width)
            / (
                right_width
                + left_width
            )
        )

        print(
            f"Left half-width = "
            f"{left_width:.6f} nm"
        )

        print(
            f"Right half-width = "
            f"{right_width:.6f} nm"
        )

        print(
            f"Half-maximum asymmetry = "
            f"{asymmetry:+.6f}"
        )

    else:

        left_width = np.nan
        right_width = np.nan
        asymmetry = np.nan

        print(
            "Could not determine both half-maximum crossings."
        )

    # ========================================================
    # 11. WINDOW TEST
    # ========================================================

    banner(
        "11. FLUX CAPTURE BY SPECTRAL WINDOW"
    )

    spacing = np.median(
        np.diff(wavelength_nm)
    )

    rows = []

    for half_planes in [
        0,
        1,
        2,
        3,
        4,
    ]:

        low_index = max(
            0,
            nearest_index
            - half_planes,
        )

        high_index = min(
            nplanes - 1,
            nearest_index
            + half_planes,
        )

        indices = np.arange(
            low_index,
            high_index + 1,
        )

        local_wave = (
            wavelength_nm[
                indices
            ]
        )

        local_flux = (
            summed_flux[
                indices
            ]
        )

        local_continuum = (
            np.interp(
                local_wave,
                wave,
                continuum,
            )
        )

        local_profile = (
            local_flux
            - local_continuum
        )

        local_integrated = (
            trapezoid_integral(
                local_wave,
                local_profile,
            )
        )

        rows.append(
            {
                "half_planes":
                    half_planes,

                "number_of_planes":
                    len(indices),

                "low_wavelength_nm":
                    local_wave[0],

                "high_wavelength_nm":
                    local_wave[-1],

                "integrated_flux":
                    local_integrated,
            }
        )

        print(
            f"±{half_planes} planes "
            f"({len(indices)} planes): "
            f"{local_wave[0]:.6f} - "
            f"{local_wave[-1]:.6f} nm"
        )

        print(
            f"  Integrated flux = "
            f"{local_integrated:.8f}"
        )

    window_df = pd.DataFrame(
        rows
    )

    # ========================================================
    # 12. SAVE PROFILE
    # ========================================================

    banner(
        "12. SAVING SPECTRAL PROFILE"
    )

    profile_df = pd.DataFrame(
        {
            "wavelength_nm":
                wave,

            "summed_flux":
                flux,

            "summed_err":
                flux_err,

            "continuum":
                continuum,

            "continuum_subtracted":
                line_profile,

            "is_blue_continuum":
                blue_mask,

            "is_red_continuum":
                red_mask,

            "is_line_search_region":
                search,
        }
    )

    profile_df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print(
        f"Saved:\n  {OUTPUT_CSV}"
    )

    # ========================================================
    # 13. SAVE SUMMARY
    # ========================================================

    summary = {
        "aperture_pixels": len(aperture),

        "spectrum_low_nm":
            SPECTRUM_LOW_NM,

        "spectrum_high_nm":
            SPECTRUM_HIGH_NM,

        "pabeta_rest_nm":
            PA_GAMMA_REST_NM,

        "predicted_pagamma_nm":
            predicted_nm,

        "nearest_plane":
            nearest_index,

        "nearest_plane_wavelength_nm":
            nearest_nm,

        "nearest_plane_velocity_kms":
            nearest_velocity,

        "peak_wavelength_nm":
            peak_wave,

        "peak_velocity_kms":
            peak_velocity,

        "peak_flux":
            peak_flux,

        "fwhm_nm":
            fwhm,

        "fwhm_velocity_kms":
            (
                velocity_width
                if np.isfinite(fwhm)
                else np.nan
            ),

        "left_half_width_nm":
            left_width,

        "right_half_width_nm":
            right_width,

        "profile_asymmetry":
            asymmetry,

        "integrated_pagamma_flux":
            line_flux,

        "integrated_flux_error_estimate":
            line_error,
    }

    summary_df = pd.DataFrame(
        [summary]
    )

    summary_df.to_csv(
        OUTPUT_SUMMARY,
        index=False,
    )

    print(
        f"Saved:\n  {OUTPUT_SUMMARY}"
    )

    # ========================================================
    # 14. FIGURE
    # ========================================================

    banner(
        "13. CREATING SPECTRAL PROFILE FIGURE"
    )

    plt.figure(
        figsize=(11, 7)
    )

    plt.errorbar(
        wave,
        flux,
        yerr=flux_err,
        fmt="o-",
        markersize=4,
        linewidth=1,
        capsize=2,
        label="69-pixel aperture spectrum",
    )

    plt.plot(
        wave,
        continuum,
        linewidth=2,
        label="Linear continuum",
    )

    plt.plot(
        line_wave,
        line_values + np.interp(
            line_wave,
            wave,
            continuum,
        ),
        linewidth=3,
        label="Continuum-subtracted line + continuum",
    )

    plt.axvline(
        predicted_nm,
        linestyle="--",
        linewidth=2,
        label="Predicted Pa-gamma",
    )

    plt.axvline(
        peak_wave,
        linestyle=":",
        linewidth=2,
        label="Observed peak",
    )

    plt.axvspan(
        BLUE_CONT_LOW,
        BLUE_CONT_HIGH,
        alpha=0.15,
        label="Blue continuum",
    )

    plt.axvspan(
        RED_CONT_LOW,
        RED_CONT_HIGH,
        alpha=0.15,
        label="Red continuum",
    )

    plt.xlabel(
        "Wavelength (nm)"
    )

    plt.ylabel(
        "Summed S3D flux (MJy/sr)"
    )

    plt.title(
        "M51 JWST Pa-gamma — 69-Pixel Aperture Spectrum"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_FIGURE,
        dpi=180,
    )

    plt.close()

    print(
        f"Saved:\n  {OUTPUT_FIGURE}"
    )

    # ========================================================
    # FINAL
    # ========================================================

    banner(
        "FINAL PA-GAMMA SPECTRAL PROFILE RESULT"
    )

    print(
        f"Nominal aperture = "
        f"{len(aperture)} pixels"
    )

    print(
        f"Predicted Pa-gamma = "
        f"{predicted_nm:.9f} nm"
    )

    print(
        f"Observed peak = "
        f"{peak_wave:.9f} nm"
    )

    print(
        f"Peak velocity = "
        f"{peak_velocity:+.3f} km/s"
    )

    print(
        f"Integrated Pa-gamma = "
        f"{line_flux:.8f}"
    )

    print(
        f"FWHM = "
        f"{fwhm:.6f} nm"
    )

    if np.isfinite(fwhm):
        print(
            f"FWHM velocity = "
            f"{velocity_width:.3f} km/s"
        )

    print(
        f"Profile asymmetry = "
        f"{asymmetry:+.6f}"
    )

    print()
    print(
        "Research outputs:"
    )

    print(
        f"  {OUTPUT_CSV}"
    )

    print(
        f"  {OUTPUT_SUMMARY}"
    )

    print(
        f"  {OUTPUT_FIGURE}"
    )

    print()
    print(
        "Pa-gamma aperture spectral-profile experiment complete."
    )


if __name__ == "__main__":
    main()
