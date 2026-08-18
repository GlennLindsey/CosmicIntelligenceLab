from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits


# ============================================================
# M51 JWST/NIRSpec S3D — 1284 nm spatial velocity analysis
# ============================================================

S3D_PATH = Path(
    "data/m51_jwst_level3/"
    "jw03435-o006_t010_nirspec_g140m-f100lp_s3d.fits"
)


# ============================================================
# Laboratory wavelengths
# ============================================================

PA_BETA_REST_NM = 1281.807000

CSII_AIR_NM = 1284.264060
CSII_VACUUM_NM = 1284.61537587


# ============================================================
# Observed feature
# ============================================================

OBSERVED_FEATURE_NM = 1284.26130440
OBSERVED_FEATURE_ERROR_NM = 0.00134611


# ============================================================
# Local M51 velocity reference
# ============================================================

M51_MEDIAN_VELOCITY = 525.25
M51_VELOCITY_MAD = 45.49

FEII_REFERENCE_VELOCITY = 573.72


# ============================================================
# Instrument resolution
# ============================================================

INSTRUMENT_R = 916.3


# ============================================================
# Spectral fitting settings
# ============================================================

FIT_HALF_WIDTH_NM = 3.0

MIN_POINTS = 7

MIN_SNR = 5.0


# ============================================================
# Physical constants
# ============================================================

C_KM_S = 299792.458


# ============================================================
# Utility functions
# ============================================================


def velocity_from_wavelength(
    observed_nm,
    rest_nm,
):
    """
    Convert wavelength displacement into classical
    Doppler velocity.

    For the small velocities relevant here, this provides
    a convenient local velocity diagnostic.
    """

    return (
        C_KM_S
        * (
            observed_nm / rest_nm
            - 1.0
        )
    )


def wavelength_from_velocity(
    rest_nm,
    velocity_km_s,
):
    """
    Convert a rest wavelength to an observed wavelength
    using the classical Doppler approximation.
    """

    return (
        rest_nm
        * (
            1.0
            + velocity_km_s / C_KM_S
        )
    )


def gaussian(
    x,
    amplitude,
    center,
    sigma,
):
    """
    Gaussian emission profile.
    """

    return (
        amplitude
        * np.exp(
            -0.5
            * (
                (x - center)
                / sigma
            ) ** 2
        )
    )


def fit_fixed_gaussian(
    wavelength_nm,
    flux,
    uncertainty,
    center_nm,
    sigma_nm,
):
    """
    Fit a fixed-center, fixed-width Gaussian plus
    a local constant continuum.

    The model is:

        flux = continuum + amplitude * Gaussian

    Only continuum and amplitude are fitted.
    """

    valid = (
        np.isfinite(wavelength_nm)
        & np.isfinite(flux)
        & np.isfinite(uncertainty)
        & (uncertainty > 0)
    )

    wavelength_nm = wavelength_nm[valid]
    flux = flux[valid]
    uncertainty = uncertainty[valid]

    if len(wavelength_nm) < MIN_POINTS:
        return None

    profile = gaussian(
        wavelength_nm,
        1.0,
        center_nm,
        sigma_nm,
    )

    # --------------------------------------------------------
    # Weighted linear least squares.
    #
    # Model:
    #
    #     y = c + a * profile
    # --------------------------------------------------------

    design = np.column_stack(
        [
            np.ones(len(profile)),
            profile,
        ]
    )

    weights = 1.0 / (
        uncertainty ** 2
    )

    normal = (
        design.T
        @ (
            weights[:, None]
            * design
        )
    )

    rhs = (
        design.T
        @ (
            weights
            * flux
        )
    )

    try:

        covariance = np.linalg.inv(
            normal
        )

    except np.linalg.LinAlgError:

        return None

    parameters = (
        covariance
        @ rhs
    )

    continuum = parameters[0]
    amplitude = parameters[1]

    model = (
        continuum
        + amplitude
        * profile
    )

    residual = (
        flux
        - model
    )

    chi2 = np.sum(
        (
            residual
            / uncertainty
        ) ** 2
    )

    dof = (
        len(flux)
        - 2
    )

    if dof <= 0:
        return None

    amplitude_error = np.sqrt(
        covariance[1, 1]
    )

    amplitude_snr = (
        amplitude
        / amplitude_error
        if amplitude_error > 0
        else np.nan
    )

    return {
        "continuum": continuum,
        "amplitude": amplitude,
        "amplitude_error": amplitude_error,
        "amplitude_snr": amplitude_snr,
        "chi2": chi2,
        "dof": dof,
        "reduced_chi2": (
            chi2 / dof
        ),
        "n_points": len(flux),
    }


# ============================================================
# Load S3D cube
# ============================================================

print("=" * 70)
print("M51 1284 NM SPATIALLY RESOLVED VELOCITY ANALYSIS")
print("Pa beta versus Cs II")
print("=" * 70)

print()
print("S3D:")
print(S3D_PATH)


print()
print("=" * 70)
print("LOADING S3D CUBE")
print("=" * 70)


with fits.open(S3D_PATH) as hdul:

    cube = np.asarray(
        hdul["SCI"].data,
        dtype=float,
    )

    header = hdul["SCI"].header


print(
    f"Cube shape: {cube.shape}"
)


# ============================================================
# Construct wavelength array
# ============================================================

n_wave = cube.shape[0]

crval3 = header["CRVAL3"]
crpix3 = header["CRPIX3"]
cdelt3 = header["CDELT3"]

pixel = np.arange(
    1,
    n_wave + 1,
    dtype=float,
)

wavelength_um = (
    crval3
    + (
        pixel
        - crpix3
    )
    * cdelt3
)

wavelength_nm = (
    wavelength_um
    * 1000.0
)


print()
print(
    f"Wavelength range: "
    f"{wavelength_nm.min():.3f} - "
    f"{wavelength_nm.max():.3f} nm"
)

print(
    f"Spectral sampling: "
    f"{cdelt3 * 1000.0:.6f} nm"
)


# ============================================================
# Instrument resolution
# ============================================================

instrument_fwhm_nm = (
    OBSERVED_FEATURE_NM
    / INSTRUMENT_R
)

instrument_sigma_nm = (
    instrument_fwhm_nm
    / (
        2.0
        * np.sqrt(
            2.0
            * np.log(2.0)
        )
    )
)


print()
print("=" * 70)
print("INSTRUMENT RESOLUTION")
print("=" * 70)

print(
    f"Adopted resolving power: "
    f"R = {INSTRUMENT_R:.1f}"
)

print(
    f"Instrument FWHM: "
    f"{instrument_fwhm_nm:.6f} nm"
)

print(
    f"Instrument sigma: "
    f"{instrument_sigma_nm:.6f} nm"
)


# ============================================================
# Hypothesis centers at local reference velocities
# ============================================================

pa_beta_reference_center = (
    wavelength_from_velocity(
        PA_BETA_REST_NM,
        FEII_REFERENCE_VELOCITY,
    )
)

csii_reference_center = (
    wavelength_from_velocity(
        CSII_VACUUM_NM,
        FEII_REFERENCE_VELOCITY,
    )
)


print()
print("=" * 70)
print("HYPOTHESIS CENTERS")
print("=" * 70)

print(
    f"Pa beta at +{FEII_REFERENCE_VELOCITY:.2f} km/s:"
)

print(
    f"  {pa_beta_reference_center:.8f} nm"
)

print(
    f"Cs II vacuum at +{FEII_REFERENCE_VELOCITY:.2f} km/s:"
)

print(
    f"  {csii_reference_center:.8f} nm"
)


# ============================================================
# Determine spectral fitting window
# ============================================================

window_mask = (
    np.abs(
        wavelength_nm
        - OBSERVED_FEATURE_NM
    )
    <= FIT_HALF_WIDTH_NM
)

window_indices = np.where(
    window_mask
)[0]

fit_wavelength = wavelength_nm[
    window_indices
]

fit_cube = cube[
    window_indices,
    :,
    :
]


print()
print("=" * 70)
print("LOCAL SPECTRAL WINDOW")
print("=" * 70)

print(
    f"Window: "
    f"{OBSERVED_FEATURE_NM - FIT_HALF_WIDTH_NM:.3f}"
    f" - "
    f"{OBSERVED_FEATURE_NM + FIT_HALF_WIDTH_NM:.3f} nm"
)

print(
    f"Spectral planes: "
    f"{len(window_indices)}"
)


# ============================================================
# Estimate cube noise
# ============================================================
#
# The S3D cube does not contain a simple FLUX_ERROR array
# equivalent to the X1D product in the SCI extension.
#
# Therefore we estimate local noise for each spatial pixel
# from the high-frequency scatter of the local spectrum.
#
# This is intentionally exploratory and should be replaced
# by an ERR extension or propagated uncertainty product if
# one is available in a future analysis.
# ============================================================


def estimate_local_noise(
    flux,
):
    """
    Estimate local spectral noise from first differences.
    """

    finite = np.isfinite(flux)

    values = flux[finite]

    if len(values) < 5:
        return np.nan

    differences = np.diff(
        values
    )

    differences = differences[
        np.isfinite(differences)
    ]

    if len(differences) < 3:
        return np.nan

    return (
        np.median(
            np.abs(
                differences
            )
        )
        / 0.6745
        / np.sqrt(2.0)
    )


# ============================================================
# Allocate result maps
# ============================================================

ny = cube.shape[1]
nx = cube.shape[2]

velocity_map = np.full(
    (ny, nx),
    np.nan,
)

velocity_error_map = np.full(
    (ny, nx),
    np.nan,
)

snr_map = np.full(
    (ny, nx),
    np.nan,
)

chi2_pabeta_map = np.full(
    (ny, nx),
    np.nan,
)

chi2_csii_map = np.full(
    (ny, nx),
    np.nan,
)

delta_chi2_map = np.full(
    (ny, nx),
    np.nan,
)

amplitude_map = np.full(
    (ny, nx),
    np.nan,
)


# ============================================================
# Spatial fitting
# ============================================================

print()
print("=" * 70)
print("FITTING SPATIAL PIXELS")
print("=" * 70)


total_pixels = (
    ny * nx
)

accepted_pixels = 0


for y in range(ny):

    if y % 10 == 0:

        print(
            f"Processing row "
            f"{y + 1}/{ny}..."
        )

    for x in range(nx):

        local_flux = fit_cube[
            :,
            y,
            x
        ]

        finite = np.isfinite(
            local_flux
        )

        if np.sum(finite) < MIN_POINTS:
            continue

        local_wavelength = (
            fit_wavelength[finite]
        )

        local_flux = (
            local_flux[finite]
        )

        noise = estimate_local_noise(
            local_flux
        )

        if (
            not np.isfinite(noise)
            or noise <= 0
        ):
            continue

        local_uncertainty = np.full(
            len(local_flux),
            noise,
        )

        # ----------------------------------------------------
        # Pa beta hypothesis
        # ----------------------------------------------------

        pabeta_fit = (
            fit_fixed_gaussian(
                local_wavelength,
                local_flux,
                local_uncertainty,
                pa_beta_reference_center,
                instrument_sigma_nm,
            )
        )

        if pabeta_fit is None:
            continue

        # ----------------------------------------------------
        # Cs II hypothesis
        # ----------------------------------------------------

        csii_fit = (
            fit_fixed_gaussian(
                local_wavelength,
                local_flux,
                local_uncertainty,
                csii_reference_center,
                instrument_sigma_nm,
            )
        )

        if csii_fit is None:
            continue

        # ----------------------------------------------------
        # Require positive Pa beta emission
        # ----------------------------------------------------

        if (
            pabeta_fit["amplitude_snr"]
            < MIN_SNR
        ):
            continue

        # ----------------------------------------------------
        # Store results
        # ----------------------------------------------------

        velocity_map[y, x] = (
            FEII_REFERENCE_VELOCITY
        )

        velocity_error_map[y, x] = (
            np.nan
        )

        snr_map[y, x] = (
            pabeta_fit[
                "amplitude_snr"
            ]
        )

        chi2_pabeta_map[y, x] = (
            pabeta_fit["chi2"]
        )

        chi2_csii_map[y, x] = (
            csii_fit["chi2"]
        )

        delta_chi2_map[y, x] = (
            csii_fit["chi2"]
            - pabeta_fit["chi2"]
        )

        amplitude_map[y, x] = (
            pabeta_fit["amplitude"]
        )

        accepted_pixels += 1


# ============================================================
# Summary
# ============================================================

print()
print("=" * 70)
print("SPATIAL FIT SUMMARY")
print("=" * 70)

print(
    f"Total spatial pixels: "
    f"{total_pixels}"
)

print(
    f"Accepted pixels: "
    f"{accepted_pixels}"
)

print(
    f"Acceptance fraction: "
    f"{accepted_pixels / total_pixels:.4f}"
)


# ============================================================
# Delta-chi-square statistics
# ============================================================

valid_delta = np.isfinite(
    delta_chi2_map
)

if np.any(valid_delta):

    delta_values = (
        delta_chi2_map[
            valid_delta
        ]
    )

    print()
    print("=" * 70)
    print("PA BETA vs Cs II MODEL PREFERENCE")
    print("=" * 70)

    print(
        f"Median Δχ² "
        f"(Cs II - Pa beta): "
        f"{np.median(delta_values):.3f}"
    )

    print(
        f"Minimum Δχ²: "
        f"{np.min(delta_values):.3f}"
    )

    print(
        f"Maximum Δχ²: "
        f"{np.max(delta_values):.3f}"
    )

    pabeta_better = (
        delta_values > 0
    )

    print(
        f"Pixels favoring Pa beta: "
        f"{np.sum(pabeta_better)}"
    )

    print(
        f"Pixels favoring Cs II: "
        f"{np.sum(~pabeta_better)}"
    )


# ============================================================
# Save velocity map
# ============================================================

plt.figure(
    figsize=(10, 7)
)

plt.imshow(
    velocity_map,
    origin="lower",
    aspect="equal",
)

plt.colorbar(
    label="Velocity (km/s)"
)

plt.xlabel(
    "Spatial pixel X"
)

plt.ylabel(
    "Spatial pixel Y"
)

plt.title(
    "M51 1284 nm — Pa beta Velocity Reference"
)

plt.tight_layout()

plt.savefig(
    "m51_1284_velocity_map.png",
    dpi=150,
)

plt.close()


# ============================================================
# Save S/N map
# ============================================================

plt.figure(
    figsize=(10, 7)
)

plt.imshow(
    snr_map,
    origin="lower",
    aspect="equal",
)

plt.colorbar(
    label="Pa beta amplitude S/N"
)

plt.xlabel(
    "Spatial pixel X"
)

plt.ylabel(
    "Spatial pixel Y"
)

plt.title(
    "M51 1284 nm — Emission S/N"
)

plt.tight_layout()

plt.savefig(
    "m51_1284_snr_map.png",
    dpi=150,
)

plt.close()


# ============================================================
# Save Δχ² map
# ============================================================

plt.figure(
    figsize=(10, 7)
)

plt.imshow(
    delta_chi2_map,
    origin="lower",
    aspect="equal",
)

plt.colorbar(
    label="Δχ² = χ²(Cs II) − χ²(Pa beta)"
)

plt.xlabel(
    "Spatial pixel X"
)

plt.ylabel(
    "Spatial pixel Y"
)

plt.title(
    "M51 1284 nm — Pa beta vs Cs II"
)

plt.tight_layout()

plt.savefig(
    "m51_1284_delta_chi2_map.png",
    dpi=150,
)

plt.close()


# ============================================================
# Save amplitude map
# ============================================================

plt.figure(
    figsize=(10, 7)
)

plt.imshow(
    amplitude_map,
    origin="lower",
    aspect="equal",
)

plt.colorbar(
    label="Pa beta fitted amplitude"
)

plt.xlabel(
    "Spatial pixel X"
)

plt.ylabel(
    "Spatial pixel Y"
)

plt.title(
    "M51 1284 nm — Pa beta Emission Amplitude"
)

plt.tight_layout()

plt.savefig(
    "m51_1284_amplitude_map.png",
    dpi=150,
)

plt.close()


# ============================================================
# Final interpretation
# ============================================================

print()
print("=" * 70)
print("INTERPRETATION")
print("=" * 70)

print(
    """
This is a spatially resolved exploratory test.

The Pa beta and Cs II models are evaluated using:

  - the same local spectral window;
  - the same instrumental Gaussian width;
  - the same locally estimated noise;
  - fixed line centers based on the independently
    measured local M51 velocity.

The quantity

    Δχ² = χ²(Cs II) - χ²(Pa beta)

is positive when Pa beta provides the better fit.

A strongly positive Δχ² over the bright emission
region would provide spatially resolved support for
Pa beta over Cs II.

A region with negative Δχ² would indicate that Cs II
provides the better fixed-velocity model there and
would warrant further investigation.

IMPORTANT:

The current velocity map is a reference-velocity map,
not yet a true free-centroid velocity map. The next
stage should allow the line centroid to vary independently
at each sufficiently high-S/N spatial pixel and compare
the resulting velocity field with [Fe II] and hydrogen
recombination emission.

The local noise estimate is also exploratory because
the present analysis does not yet use a fully propagated
S3D uncertainty cube.
"""
)

print()
print("=" * 70)
print("SPATIALLY RESOLVED 1284 NM ANALYSIS COMPLETE")
print("=" * 70)
