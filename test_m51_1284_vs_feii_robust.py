from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from scipy.optimize import curve_fit
from scipy.stats import (
    pearsonr,
    spearmanr,
    theilslopes,
)


# ======================================================================
# M51 JWST/NIRSpec
# HIGH-S/N ROBUST SPATIAL VELOCITY COMPARISON
#
# 1284 nm feature / Pa beta versus [Fe II] 1.257 um
# ======================================================================


S3D_PATH = Path(
    "data/m51_jwst_level3/"
    "jw03435-o006_t010_nirspec_g140m-f100lp_s3d.fits"
)


# ======================================================================
# Laboratory wavelengths
#
# JWST wavelength arrays are vacuum wavelengths.
# ======================================================================

PA_BETA_REST_NM = 1281.80700000
FEII_REST_NM = 1256.68000000


# ======================================================================
# Reference velocity
# ======================================================================

REFERENCE_VELOCITY_KMS = 573.72


# ======================================================================
# Fitting windows
# ======================================================================

PA_BETA_WINDOW_NM = 3.0
FEII_WINDOW_NM = 3.0


# ======================================================================
# Gaussian fitting limits
# ======================================================================

SIGMA_MIN_NM = 0.30
SIGMA_MAX_NM = 1.50

MIN_POINTS = 6


# ======================================================================
# HIGH-S/N QUALITY REQUIREMENTS
# ======================================================================

MIN_SNR = 10.0

MAX_VELOCITY_ERROR_KMS = 15.0


# ----------------------------------------------------------------------
# Broad velocity range used only as a quality-control filter.
#
# This is intentionally much broader than the approximately
# 525-575 km/s reference distribution.
#
# We are NOT forcing the lines to 573.72 km/s.
# ----------------------------------------------------------------------

MIN_VELOCITY_KMS = 400.0
MAX_VELOCITY_KMS = 700.0


# ======================================================================
# Gaussian + linear continuum
# ======================================================================

def gaussian_linear(
    wavelength,
    amplitude,
    center,
    sigma,
    continuum,
    slope,
):
    gaussian = (
        amplitude
        * np.exp(
            -0.5
            * (
                (wavelength - center)
                / sigma
            ) ** 2
        )
    )

    continuum_model = (
        continuum
        + slope
        * (wavelength - center)
    )

    return gaussian + continuum_model


# ======================================================================
# Velocity conversion
# ======================================================================

def wavelength_to_velocity(
    wavelength_nm,
    rest_nm,
):
    c_kms = 299792.458

    return (
        (
            wavelength_nm
            / rest_nm
        )
        - 1.0
    ) * c_kms


# ======================================================================
# Velocity uncertainty
# ======================================================================

def wavelength_error_to_velocity_error(
    wavelength_error_nm,
    rest_nm,
):
    c_kms = 299792.458

    return (
        wavelength_error_nm
        / rest_nm
        * c_kms
    )


# ======================================================================
# Construct wavelength array
# ======================================================================

def get_wavelength_axis(
    header,
    n_spectral,
):
    crval = header.get("CRVAL3")
    cdelt = header.get("CDELT3")
    crpix = header.get(
        "CRPIX3",
        1.0,
    )

    if (
        crval is None
        or cdelt is None
    ):
        raise RuntimeError(
            "Missing CRVAL3/CDELT3 spectral WCS."
        )

    pixels = (
        np.arange(
            n_spectral,
            dtype=float,
        )
        + 1.0
    )

    wavelength = (
        crval
        + (
            pixels
            - crpix
        )
        * cdelt
    )

    unit = str(
        header.get(
            "CUNIT3",
            "",
        )
    ).lower()

    if unit in {
        "um",
        "micron",
        "microns",
    }:

        return wavelength * 1000.0

    if unit == "nm":

        return wavelength

    if unit in {
        "m",
        "meter",
        "meters",
    }:

        return wavelength * 1.0e9

    # Fallback for this JWST cube.
    if np.nanmax(wavelength) < 10.0:

        return wavelength * 1000.0

    return wavelength


# ======================================================================
# Fit one emission line
# ======================================================================

def fit_line(
    wavelength,
    flux,
    center_guess,
    window_nm,
):
    """
    Free-centroid Gaussian + linear continuum fit.

    Returns None when the fit does not satisfy basic quality checks.
    """

    mask = (
        np.isfinite(wavelength)
        & np.isfinite(flux)
        & (
            np.abs(
                wavelength
                - center_guess
            )
            <= window_nm
        )
    )

    x = wavelength[mask]
    y = flux[mask]

    if len(x) < MIN_POINTS:
        return None

    # --------------------------------------------------------------
    # Estimate continuum from outer part of window.
    # --------------------------------------------------------------

    distance = np.abs(
        x - center_guess
    )

    continuum_mask = (
        distance
        >= window_nm * 0.65
    )

    if np.sum(
        continuum_mask
    ) >= 3:

        continuum = np.median(
            y[continuum_mask]
        )

        try:

            slope = np.polyfit(
                x[continuum_mask]
                - center_guess,
                y[continuum_mask],
                1,
            )[0]

        except Exception:

            slope = 0.0

    else:

        continuum = np.median(y)
        slope = 0.0

    # --------------------------------------------------------------
    # Initial amplitude.
    # --------------------------------------------------------------

    amplitude_guess = (
        np.max(y)
        - continuum
    )

    if (
        not np.isfinite(
            amplitude_guess
        )
        or amplitude_guess <= 0
    ):
        return None

    # --------------------------------------------------------------
    # Initial parameters.
    # --------------------------------------------------------------

    p0 = [
        amplitude_guess,
        center_guess,
        0.60,
        continuum,
        slope,
    ]

    # --------------------------------------------------------------
    # Bounds.
    # --------------------------------------------------------------

    lower_bounds = [
        0.0,
        center_guess
        - window_nm * 0.50,
        SIGMA_MIN_NM,
        -np.inf,
        -np.inf,
    ]

    upper_bounds = [
        np.inf,
        center_guess
        + window_nm * 0.50,
        SIGMA_MAX_NM,
        np.inf,
        np.inf,
    ]

    try:

        popt, pcov = curve_fit(
            gaussian_linear,
            x,
            y,
            p0=p0,
            bounds=(
                lower_bounds,
                upper_bounds,
            ),
            maxfev=10000,
        )

    except Exception:

        return None

    amplitude = popt[0]
    center = popt[1]
    sigma = popt[2]

    # --------------------------------------------------------------
    # Parameter uncertainties.
    # --------------------------------------------------------------

    if (
        pcov is not None
        and np.all(
            np.isfinite(pcov)
        )
    ):

        errors = np.sqrt(
            np.diag(pcov)
        )

        amplitude_error = errors[0]
        center_error = errors[1]
        sigma_error = errors[2]

    else:

        amplitude_error = np.nan
        center_error = np.nan
        sigma_error = np.nan

    # --------------------------------------------------------------
    # Model and residual.
    # --------------------------------------------------------------

    model = gaussian_linear(
        x,
        *popt,
    )

    residual = y - model

    # --------------------------------------------------------------
    # Local empirical noise.
    # --------------------------------------------------------------

    if np.sum(
        continuum_mask
    ) >= 3:

        noise_values = residual[
            continuum_mask
        ]

    else:

        noise_values = residual

    noise = np.std(
        noise_values,
        ddof=1,
    )

    if (
        not np.isfinite(noise)
        or noise <= 0
    ):
        return None

    amplitude_snr = (
        amplitude
        / noise
    )

    # --------------------------------------------------------------
    # Fit statistics.
    # --------------------------------------------------------------

    chi2 = np.sum(
        (residual / noise) ** 2
    )

    dof = (
        len(x)
        - len(popt)
    )

    if dof > 0:

        reduced_chi2 = (
            chi2 / dof
        )

    else:

        reduced_chi2 = np.nan

    # --------------------------------------------------------------
    # Basic centroid quality.
    # --------------------------------------------------------------

    if (
        not np.isfinite(
            center_error
        )
    ):
        return None

    return {
        "center_nm": center,
        "center_error_nm": center_error,
        "amplitude": amplitude,
        "amplitude_error": amplitude_error,
        "sigma_nm": sigma,
        "sigma_error_nm": sigma_error,
        "noise": noise,
        "snr": amplitude_snr,
        "chi2": chi2,
        "dof": dof,
        "reduced_chi2": reduced_chi2,
    }


# ======================================================================
# START
# ======================================================================

print("=" * 70)
print(
    "M51 JWST/NIRSpec HIGH-S/N ROBUST SPATIAL VELOCITY COMPARISON"
)
print(
    "1284 nm feature / Pa beta versus [Fe II] 1.257 um"
)
print("=" * 70)


print()
print("S3D:")
print(S3D_PATH)


# ======================================================================
# Load cube
# ======================================================================

print()
print("=" * 70)
print("LOADING S3D CUBE")
print("=" * 70)


with fits.open(
    S3D_PATH
) as hdul:

    cube = np.asarray(
        hdul[1].data,
        dtype=float,
    )

    header = hdul[1].header

    if cube.ndim != 3:

        raise RuntimeError(
            "Expected 3-dimensional S3D cube."
        )

    n_spectral = cube.shape[0]
    ny = cube.shape[1]
    nx = cube.shape[2]

    wavelength_nm = (
        get_wavelength_axis(
            header,
            n_spectral,
        )
    )


print()
print(
    f"Cube shape: {cube.shape}"
)

print(
    f"Wavelength range: "
    f"{wavelength_nm.min():.3f} - "
    f"{wavelength_nm.max():.3f} nm"
)

print(
    f"Spectral sampling: "
    f"{np.median(np.diff(wavelength_nm)):.6f} nm"
)

print(
    f"Spatial dimensions: "
    f"{nx} x {ny}"
)


# ======================================================================
# Reference observed wavelengths
# ======================================================================

pa_beta_reference = (
    PA_BETA_REST_NM
    * (
        1.0
        + REFERENCE_VELOCITY_KMS
        / 299792.458
    )
)

feii_reference = (
    FEII_REST_NM
    * (
        1.0
        + REFERENCE_VELOCITY_KMS
        / 299792.458
    )
)


print()
print("=" * 70)
print("REFERENCE CENTERS")
print("=" * 70)

print(
    f"Reference velocity:"
    f" {REFERENCE_VELOCITY_KMS:.2f} km/s"
)

print(
    f"1284 nm / Pa beta:"
    f" {pa_beta_reference:.6f} nm"
)

print(
    f"[Fe II] 1.257 um:"
    f" {feii_reference:.6f} nm"
)


# ======================================================================
# Allocate maps
# ======================================================================

pa_velocity = np.full(
    (ny, nx),
    np.nan,
)

pa_velocity_error = np.full(
    (ny, nx),
    np.nan,
)

pa_snr = np.full(
    (ny, nx),
    np.nan,
)


fe_velocity = np.full(
    (ny, nx),
    np.nan,
)

fe_velocity_error = np.full(
    (ny, nx),
    np.nan,
)

fe_snr = np.full(
    (ny, nx),
    np.nan,
)


# ======================================================================
# Fit all spatial pixels
# ======================================================================

print()
print("=" * 70)
print("FITTING SPATIAL PIXELS")
print("=" * 70)


attempted = 0

both_fit = 0


for y in range(ny):

    if y % 10 == 0:

        print(
            f"Processing row {y + 1}/{ny}..."
        )

    for x in range(nx):

        spectrum = cube[
            :,
            y,
            x,
        ]

        if not np.any(
            np.isfinite(spectrum)
        ):
            continue

        attempted += 1

        # ----------------------------------------------------------
        # Pa beta / 1284 nm
        # ----------------------------------------------------------

        pa = fit_line(
            wavelength_nm,
            spectrum,
            pa_beta_reference,
            PA_BETA_WINDOW_NM,
        )

        if pa is None:
            continue

        # ----------------------------------------------------------
        # [Fe II]
        # ----------------------------------------------------------

        fe = fit_line(
            wavelength_nm,
            spectrum,
            feii_reference,
            FEII_WINDOW_NM,
        )

        if fe is None:
            continue

        # ----------------------------------------------------------
        # Convert Pa beta centroid to velocity.
        # ----------------------------------------------------------

        pa_v = wavelength_to_velocity(
            pa["center_nm"],
            PA_BETA_REST_NM,
        )

        pa_e = wavelength_error_to_velocity_error(
            pa["center_error_nm"],
            PA_BETA_REST_NM,
        )

        # ----------------------------------------------------------
        # Convert [Fe II] centroid to velocity.
        # ----------------------------------------------------------

        fe_v = wavelength_to_velocity(
            fe["center_nm"],
            FEII_REST_NM,
        )

        fe_e = wavelength_error_to_velocity_error(
            fe["center_error_nm"],
            FEII_REST_NM,
        )

        # ----------------------------------------------------------
        # Store everything initially.
        # ----------------------------------------------------------

        pa_velocity[y, x] = pa_v
        pa_velocity_error[y, x] = pa_e
        pa_snr[y, x] = pa["snr"]

        fe_velocity[y, x] = fe_v
        fe_velocity_error[y, x] = fe_e
        fe_snr[y, x] = fe["snr"]

        both_fit += 1


# ======================================================================
# Initial paired population
# ======================================================================

initial_valid = (
    np.isfinite(pa_velocity)
    & np.isfinite(fe_velocity)
    & np.isfinite(pa_velocity_error)
    & np.isfinite(fe_velocity_error)
    & np.isfinite(pa_snr)
    & np.isfinite(fe_snr)
)


print()
print("=" * 70)
print("INITIAL FIT POPULATION")
print("=" * 70)

print(
    f"Spatial pixels attempted: "
    f"{attempted}"
)

print(
    f"Successful fits to both lines: "
    f"{np.sum(initial_valid)}"
)


# ======================================================================
# HIGH-S/N QUALITY MASK
# ======================================================================

quality_mask = (
    initial_valid
    & (pa_snr >= MIN_SNR)
    & (fe_snr >= MIN_SNR)
    & (
        pa_velocity_error
        <= MAX_VELOCITY_ERROR_KMS
    )
    & (
        fe_velocity_error
        <= MAX_VELOCITY_ERROR_KMS
    )
    & (
        pa_velocity
        >= MIN_VELOCITY_KMS
    )
    & (
        pa_velocity
        <= MAX_VELOCITY_KMS
    )
    & (
        fe_velocity
        >= MIN_VELOCITY_KMS
    )
    & (
        fe_velocity
        <= MAX_VELOCITY_KMS
    )
)


# ======================================================================
# Extract high-quality paired velocities
# ======================================================================

pa_v = pa_velocity[
    quality_mask
]

fe_v = fe_velocity[
    quality_mask
]

pa_err = pa_velocity_error[
    quality_mask
]

fe_err = fe_velocity_error[
    quality_mask
]

pa_snr_values = pa_snr[
    quality_mask
]

fe_snr_values = fe_snr[
    quality_mask
]


print()
print("=" * 70)
print("HIGH-S/N QUALITY SELECTION")
print("=" * 70)

print(
    f"Minimum S/N: "
    f"{MIN_SNR:.1f}"
)

print(
    f"Maximum centroid velocity error:"
    f" {MAX_VELOCITY_ERROR_KMS:.1f} km/s"
)

print(
    f"Velocity range:"
    f" {MIN_VELOCITY_KMS:.0f} - "
    f"{MAX_VELOCITY_KMS:.0f} km/s"
)

print()
print(
    f"High-quality paired pixels:"
    f" {len(pa_v)}"
)

if np.sum(initial_valid) > 0:

    print(
        f"Fraction retained:"
        f" {len(pa_v) / np.sum(initial_valid):.4f}"
    )


if len(pa_v) < 10:

    raise RuntimeError(
        "Too few high-quality paired pixels "
        "for robust correlation analysis."
    )


# ======================================================================
# Velocity differences
# ======================================================================

difference = (
    pa_v
    - fe_v
)

combined_error = np.sqrt(
    pa_err ** 2
    + fe_err ** 2
)


# ======================================================================
# Velocity statistics
# ======================================================================

print()
print("=" * 70)
print("HIGH-S/N VELOCITY STATISTICS")
print("=" * 70)

print()
print("1284 nm / Pa beta")

print(
    f"  Median velocity:"
    f" {np.median(pa_v):.3f} km/s"
)

print(
    f"  Mean velocity:"
    f" {np.mean(pa_v):.3f} km/s"
)

print(
    f"  Standard deviation:"
    f" {np.std(pa_v):.3f} km/s"
)

print(
    f"  Median S/N:"
    f" {np.median(pa_snr_values):.2f}"
)

print()
print("[Fe II] 1.257 um")

print(
    f"  Median velocity:"
    f" {np.median(fe_v):.3f} km/s"
)

print(
    f"  Mean velocity:"
    f" {np.mean(fe_v):.3f} km/s"
)

print(
    f"  Standard deviation:"
    f" {np.std(fe_v):.3f} km/s"
)

print(
    f"  Median S/N:"
    f" {np.median(fe_snr_values):.2f}"
)

print()
print("Velocity difference: Pa beta - [Fe II]")

print(
    f"  Median:"
    f" {np.median(difference):+.3f} km/s"
)

print(
    f"  Mean:"
    f" {np.mean(difference):+.3f} km/s"
)

print(
    f"  Standard deviation:"
    f" {np.std(difference):.3f} km/s"
)

print(
    f"  Median combined uncertainty:"
    f" {np.median(combined_error):.3f} km/s"
)


# ======================================================================
# Pearson correlation
# ======================================================================

pearson_r, pearson_p = pearsonr(
    fe_v,
    pa_v,
)


# ======================================================================
# Spearman correlation
# ======================================================================

spearman_rho, spearman_p = spearmanr(
    fe_v,
    pa_v,
)


# ======================================================================
# Robust Theil-Sen regression
# ======================================================================

(
    robust_slope,
    robust_intercept,
    robust_low,
    robust_high,
) = theilslopes(
    pa_v,
    fe_v,
)


# ======================================================================
# Ordinary least-squares comparison
# ======================================================================

ordinary_slope, ordinary_intercept = (
    np.polyfit(
        fe_v,
        pa_v,
        1,
    )
)


print()
print("=" * 70)
print("ROBUST PIXEL-BY-PIXEL VELOCITY CORRELATION")
print("=" * 70)

print()
print(
    f"Pearson r:"
    f" {pearson_r:.6f}"
)

print(
    f"Pearson p-value:"
    f" {pearson_p:.4e}"
)

print()
print(
    f"Spearman rho:"
    f" {spearman_rho:.6f}"
)

print(
    f"Spearman p-value:"
    f" {spearman_p:.4e}"
)

print()
print(
    f"Ordinary least-squares slope:"
    f" {ordinary_slope:.6f}"
)

print(
    f"Ordinary intercept:"
    f" {ordinary_intercept:.3f} km/s"
)

print()
print(
    f"Theil-Sen robust slope:"
    f" {robust_slope:.6f}"
)

print(
    f"Theil-Sen intercept:"
    f" {robust_intercept:.3f} km/s"
)

print(
    f"Theil-Sen slope 95% interval:"
    f" {robust_low:.6f} - "
    f"{robust_high:.6f}"
)


# ======================================================================
# Agreement within formal uncertainties
# ======================================================================

agreement_sigma = (
    np.abs(difference)
    / combined_error
)


within_1 = (
    agreement_sigma <= 1.0
)

within_2 = (
    agreement_sigma <= 2.0
)

within_3 = (
    agreement_sigma <= 3.0
)


print()
print("=" * 70)
print("AGREEMENT WITHIN FIT UNCERTAINTIES")
print("=" * 70)

print(
    f"Within 1 sigma:"
    f" {np.sum(within_1)} / "
    f"{len(within_1)}"
    f" ({np.mean(within_1) * 100:.1f}%)"
)

print(
    f"Within 2 sigma:"
    f" {np.sum(within_2)} / "
    f"{len(within_2)}"
    f" ({np.mean(within_2) * 100:.1f}%)"
)

print(
    f"Within 3 sigma:"
    f" {np.sum(within_3)} / "
    f"{len(within_3)}"
    f" ({np.mean(within_3) * 100:.1f}%)"
)


# ======================================================================
# Construct high-quality maps
# ======================================================================

pa_quality_map = np.full(
    (ny, nx),
    np.nan,
)

fe_quality_map = np.full(
    (ny, nx),
    np.nan,
)

difference_map = np.full(
    (ny, nx),
    np.nan,
)


pa_quality_map[
    quality_mask
] = pa_velocity[
    quality_mask
]

fe_quality_map[
    quality_mask
] = fe_velocity[
    quality_mask
]

difference_map[
    quality_mask
] = (
    pa_velocity[
        quality_mask
    ]
    - fe_velocity[
        quality_mask
    ]
)


# ======================================================================
# Plot 1 — Pa beta velocity
# ======================================================================

plt.figure(
    figsize=(11, 8)
)

image = plt.imshow(
    pa_quality_map,
    origin="lower",
    interpolation="nearest",
)

plt.colorbar(
    image,
    label="Velocity (km/s)",
)

plt.xlabel(
    "Spatial X pixel"
)

plt.ylabel(
    "Spatial Y pixel"
)

plt.title(
    "M51 1284 nm / Pa beta\n"
    "High-S/N velocity field"
)

plt.tight_layout()

plt.savefig(
    "m51_robust_1284_velocity_map.png",
    dpi=150,
)

plt.close()


# ======================================================================
# Plot 2 — [Fe II] velocity
# ======================================================================

plt.figure(
    figsize=(11, 8)
)

image = plt.imshow(
    fe_quality_map,
    origin="lower",
    interpolation="nearest",
)

plt.colorbar(
    image,
    label="Velocity (km/s)",
)

plt.xlabel(
    "Spatial X pixel"
)

plt.ylabel(
    "Spatial Y pixel"
)

plt.title(
    "M51 [Fe II] 1.257 um\n"
    "High-S/N velocity field"
)

plt.tight_layout()

plt.savefig(
    "m51_robust_feii_velocity_map.png",
    dpi=150,
)

plt.close()


# ======================================================================
# Plot 3 — Difference map
# ======================================================================

plt.figure(
    figsize=(11, 8)
)

image = plt.imshow(
    difference_map,
    origin="lower",
    interpolation="nearest",
)

plt.colorbar(
    image,
    label="Pa beta - [Fe II] (km/s)",
)

plt.xlabel(
    "Spatial X pixel"
)

plt.ylabel(
    "Spatial Y pixel"
)

plt.title(
    "M51 Velocity Difference\n"
    "1284 nm / Pa beta minus [Fe II]"
)

plt.tight_layout()

plt.savefig(
    "m51_robust_1284_minus_feii_velocity_map.png",
    dpi=150,
)

plt.close()


# ======================================================================
# Plot 4 — Correlation
# ======================================================================

plt.figure(
    figsize=(9, 8)
)

plt.scatter(
    fe_v,
    pa_v,
    s=12,
    alpha=0.45,
)

x_line = np.linspace(
    np.min(fe_v),
    np.max(fe_v),
    300,
)

# Ordinary regression
ordinary_line = (
    ordinary_slope
    * x_line
    + ordinary_intercept
)

plt.plot(
    x_line,
    ordinary_line,
    linewidth=2,
    label="OLS",
)

# One-to-one relation
plt.plot(
    x_line,
    x_line,
    linestyle="--",
    label="1:1",
)

plt.xlabel(
    "[Fe II] velocity (km/s)"
)

plt.ylabel(
    "1284 nm velocity assuming Pa beta (km/s)"
)

plt.title(
    "M51 High-S/N Spatial Velocity Correlation\n"
    f"Pearson r = {pearson_r:.4f}, "
    f"Spearman rho = {spearman_rho:.4f}"
)

plt.legend()

plt.grid(
    alpha=0.25
)

plt.tight_layout()

plt.savefig(
    "m51_robust_1284_vs_feii_correlation.png",
    dpi=150,
)

plt.close()


# ======================================================================
# Plot 5 — Velocity difference distribution
# ======================================================================

plt.figure(
    figsize=(10, 6)
)

plt.hist(
    difference,
    bins=40,
)

plt.axvline(
    0.0,
    linestyle="--",
    label="Zero difference",
)

plt.axvline(
    np.median(difference),
    linestyle=":",
    linewidth=2,
    label="Median",
)

plt.xlabel(
    "1284 nm velocity - [Fe II] velocity (km/s)"
)

plt.ylabel(
    "Number of spatial pixels"
)

plt.title(
    "M51 High-S/N Velocity Difference Distribution"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "m51_robust_velocity_difference.png",
    dpi=150,
)

plt.close()


# ======================================================================
# Final interpretation
# ======================================================================

print()
print("=" * 70)
print("ROBUST TEST INTERPRETATION")
print("=" * 70)

print()

if pearson_r >= 0.7:

    print(
        "Pearson correlation is strong."
    )

elif pearson_r >= 0.4:

    print(
        "Pearson correlation is moderate."
    )

elif pearson_r >= 0.2:

    print(
        "Pearson correlation is weak."
    )

else:

    print(
        "Pearson correlation remains very weak."
    )


print()

if (
    spearman_rho >= 0.7
):

    print(
        "Spearman correlation is strong."
    )

elif (
    spearman_rho >= 0.4
):

    print(
        "Spearman correlation is moderate."
    )

elif (
    spearman_rho >= 0.2
):

    print(
        "Spearman correlation is weak."
    )

else:

    print(
        "Spearman correlation remains very weak."
    )


print()
print(
    "The crucial comparison is now between "
    "the high-quality Pa beta and [Fe II] "
    "spaxels only."
)

print()

print(
    "A strong correlation with a Theil-Sen slope "
    "near unity would support the interpretation "
    "that the two lines trace the same spatial "
    "kinematic structure."
)

print()

print(
    "A small median velocity difference without "
    "a strong correlation would indicate that "
    "the two lines share the same bulk velocity "
    "but do not necessarily trace identical "
    "spatial velocity structures."
)

print()

print(
    "A persistent lack of correlation after "
    "strict quality selection would mean that "
    "the spatial-kinematic evidence for a common "
    "origin is weaker than the integrated-spectrum "
    "evidence."
)

print()

print(
    "This test does NOT by itself identify the "
    "1284 nm transition."
)

print(
    "It is specifically a test of whether the "
    "1284 nm velocity field behaves like the "
    "[Fe II] velocity field."
)

print()

print(
    "The formal uncertainties remain exploratory "
    "because the analysis does not yet use a "
    "fully propagated JWST S3D uncertainty cube."
)


# ======================================================================
# Output files
# ======================================================================

print()
print("=" * 70)
print("OUTPUT FILES")
print("=" * 70)

print(
    "m51_robust_1284_velocity_map.png"
)

print(
    "m51_robust_feii_velocity_map.png"
)

print(
    "m51_robust_1284_minus_feii_velocity_map.png"
)

print(
    "m51_robust_1284_vs_feii_correlation.png"
)

print(
    "m51_robust_velocity_difference.png"
)

print()
print("=" * 70)
print("ROBUST VELOCITY COMPARISON COMPLETE")
print("=" * 70)
