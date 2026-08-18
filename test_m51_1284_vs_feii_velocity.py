from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from scipy.optimize import curve_fit


# ======================================================================
# M51 JWST/NIRSpec
# DIRECT SPATIAL VELOCITY COMPARISON
# 1284 nm feature versus [Fe II] 1.257 um
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
# Analysis parameters
# ======================================================================

# Local fitting windows
PA_BETA_WINDOW_NM = 3.0
FEII_WINDOW_NM = 3.0

# Continuum windows inside each fitting region
CONTINUUM_FRACTION = 0.25

# Gaussian width bounds.
#
# NIRSpec G140M has approximately:
#
#   FWHM ~ 1.40 nm
#   sigma ~ 0.595 nm
#
# Allowing the width to vary permits modest intrinsic broadening
# and spatial variation without allowing pathological broad fits.
SIGMA_MIN_NM = 0.30
SIGMA_MAX_NM = 1.50

# Minimum number of spectral points
MIN_POINTS = 6

# Minimum line amplitude / noise ratio
MIN_SNR = 5.0

# Maximum allowed centroid uncertainty
MAX_CENTER_ERROR_NM = 0.20


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
    """
    Gaussian emission line plus a local linear continuum.
    """

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
    """
    Non-relativistic Doppler velocity.

    Adequate here because the velocities are only
    several hundred km/s.
    """

    c_kms = 299792.458

    return (
        (wavelength_nm / rest_nm) - 1.0
    ) * c_kms


# ======================================================================
# Wavelength axis
# ======================================================================

def get_spectral_wavelengths(
    header,
    n_spectral,
):
    """
    Construct the S3D spectral wavelength axis.

    JWST S3D products normally provide the spectral
    WCS through CRVAL3/CDELT3/CRPIX3.
    """

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
            "Could not find CRVAL3/CDELT3 "
            "spectral WCS keywords."
        )

    pixels = np.arange(
        n_spectral,
        dtype=float,
    ) + 1.0

    wavelength = (
        crval
        + (pixels - crpix)
        * cdelt
    )

    unit = str(
        header.get(
            "CUNIT3",
            "",
        )
    ).lower()

    # Convert to nm.
    if unit in {"um", "micron", "microns"}:
        wavelength_nm = wavelength * 1000.0

    elif unit in {"nm"}:
        wavelength_nm = wavelength

    elif unit in {"m", "meter", "meters"}:
        wavelength_nm = wavelength * 1.0e9

    else:
        # The existing M51 cube is known to use microns.
        #
        # If CUNIT3 is missing, inspect the numerical range.
        if (
            np.nanmax(wavelength) < 10.0
        ):
            wavelength_nm = wavelength * 1000.0
        else:
            wavelength_nm = wavelength

    return wavelength_nm


# ======================================================================
# Fit one spatial spectrum
# ======================================================================

def fit_line(
    wavelength,
    flux,
    center_guess,
    window_nm,
):
    """
    Fit one emission line using:

        Gaussian + linear continuum

    with a free centroid and free Gaussian width.
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
    # Estimate local continuum from the outer portions
    # of the fitting window.
    # --------------------------------------------------------------

    distance = np.abs(
        x - center_guess
    )

    continuum_mask = (
        distance
        >= window_nm * 0.65
    )

    if np.sum(continuum_mask) >= 2:

        continuum = np.median(
            y[continuum_mask]
        )

        continuum_x = x[
            continuum_mask
        ]

        continuum_y = y[
            continuum_mask
        ]

        try:

            slope = np.polyfit(
                continuum_x
                - center_guess,
                continuum_y,
                1,
            )[0]

        except Exception:

            slope = 0.0

    else:

        continuum = np.median(y)
        slope = 0.0

    # --------------------------------------------------------------
    # Initial Gaussian amplitude
    # --------------------------------------------------------------

    amplitude_guess = (
        np.max(y)
        - continuum
    )

    if not np.isfinite(
        amplitude_guess
    ):
        return None

    if amplitude_guess <= 0:
        return None

    # --------------------------------------------------------------
    # Initial parameters
    # --------------------------------------------------------------

    p0 = [
        amplitude_guess,
        center_guess,
        0.60,
        continuum,
        slope,
    ]

    # --------------------------------------------------------------
    # Parameter bounds
    # --------------------------------------------------------------

    lower_bounds = [
        0.0,
        center_guess - window_nm * 0.50,
        SIGMA_MIN_NM,
        -np.inf,
        -np.inf,
    ]

    upper_bounds = [
        np.inf,
        center_guess + window_nm * 0.50,
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
    # Parameter uncertainties
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
    # Residuals
    # --------------------------------------------------------------

    model = gaussian_linear(
        x,
        *popt,
    )

    residual = y - model

    # --------------------------------------------------------------
    # Empirical local noise estimate.
    #
    # Use outer portions of the fitting window so that the
    # emission line does not dominate the noise estimate.
    # --------------------------------------------------------------

    if np.sum(continuum_mask) >= 3:

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
        amplitude / noise
    )

    # --------------------------------------------------------------
    # Goodness of fit
    # --------------------------------------------------------------

    chi2 = np.sum(
        (residual / noise) ** 2
    )

    dof = len(x) - len(popt)

    if dof > 0:
        reduced_chi2 = (
            chi2 / dof
        )
    else:
        reduced_chi2 = np.nan

    # --------------------------------------------------------------
    # Quality control
    # --------------------------------------------------------------

    if (
        not np.isfinite(
            center_error
        )
        or center_error
        > MAX_CENTER_ERROR_NM
    ):
        return None

    if amplitude_snr < MIN_SNR:
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
# Main analysis
# ======================================================================

print("=" * 70)
print(
    "M51 JWST/NIRSpec DIRECT SPATIAL VELOCITY COMPARISON"
)
print(
    "1284 nm feature versus [Fe II] 1.257 um"
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


with fits.open(S3D_PATH) as hdul:

    cube = np.asarray(
        hdul[1].data,
        dtype=float,
    )

    header = hdul[1].header

    primary_header = hdul[0].header

    print()
    print(
        "Raw cube shape:",
        cube.shape,
    )

    # --------------------------------------------------------------
    # Determine orientation.
    #
    # Expected JWST S3D shape:
    #
    #   spectral, y, x
    # --------------------------------------------------------------

    if cube.ndim != 3:
        raise RuntimeError(
            "Expected a 3-dimensional S3D cube."
        )

    n_spectral = cube.shape[0]
    ny = cube.shape[1]
    nx = cube.shape[2]

    wavelength_nm = (
        get_spectral_wavelengths(
            header,
            n_spectral,
        )
    )


print()
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
# Reference wavelengths
# ======================================================================

print()
print("=" * 70)
print("REFERENCE WAVELENGTHS")
print("=" * 70)

print(
    f"Pa beta / 1284 feature laboratory wavelength:"
    f" {PA_BETA_REST_NM:.8f} nm"
)

print(
    f"[Fe II] laboratory wavelength:"
    f" {FEII_REST_NM:.8f} nm"
)


# ======================================================================
# Determine observed centers from the integrated spectrum
#
# We use the previously established M51 kinematics to define
# the initial centers for the spatial fits.
# ======================================================================

REFERENCE_VELOCITY_KMS = 573.72


pa_beta_expected = (
    PA_BETA_REST_NM
    * (
        1.0
        + REFERENCE_VELOCITY_KMS
        / 299792.458
    )
)

feii_expected = (
    FEII_REST_NM
    * (
        1.0
        + REFERENCE_VELOCITY_KMS
        / 299792.458
    )
)


print()
print(
    "Reference velocity:"
    f" {REFERENCE_VELOCITY_KMS:.2f} km/s"
)

print(
    f"Expected 1284 nm center:"
    f" {pa_beta_expected:.6f} nm"
)

print(
    f"Expected [Fe II] center:"
    f" {feii_expected:.6f} nm"
)


# ======================================================================
# Verify wavelength coverage
# ======================================================================

if not (
    wavelength_nm.min()
    <= pa_beta_expected
    <= wavelength_nm.max()
):

    raise RuntimeError(
        "1284 nm feature lies outside the cube."
    )

if not (
    wavelength_nm.min()
    <= feii_expected
    <= wavelength_nm.max()
):

    raise RuntimeError(
        "[Fe II] line lies outside the cube."
    )


# ======================================================================
# Allocate output maps
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

pa_center = np.full(
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

fe_center = np.full(
    (ny, nx),
    np.nan,
)


# ======================================================================
# Fit spatial pixels
# ======================================================================

print()
print("=" * 70)
print("FITTING SPATIAL PIXELS")
print("=" * 70)


attempted = 0
accepted_both = 0


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
        # Fit 1284 nm / Pa beta
        # ----------------------------------------------------------

        pa_result = fit_line(
            wavelength_nm,
            spectrum,
            pa_beta_expected,
            PA_BETA_WINDOW_NM,
        )

        if pa_result is None:
            continue

        # ----------------------------------------------------------
        # Fit [Fe II]
        # ----------------------------------------------------------

        fe_result = fit_line(
            wavelength_nm,
            spectrum,
            feii_expected,
            FEII_WINDOW_NM,
        )

        if fe_result is None:
            continue

        # ----------------------------------------------------------
        # Store Pa beta results
        # ----------------------------------------------------------

        pa_center[
            y,
            x,
        ] = pa_result[
            "center_nm"
        ]

        pa_velocity[
            y,
            x,
        ] = wavelength_to_velocity(
            pa_result[
                "center_nm"
            ],
            PA_BETA_REST_NM,
        )

        pa_velocity_error[
            y,
            x,
        ] = (
            pa_result[
                "center_error_nm"
            ]
            / PA_BETA_REST_NM
            * 299792.458
        )

        pa_snr[
            y,
            x,
        ] = pa_result["snr"]

        # ----------------------------------------------------------
        # Store [Fe II] results
        # ----------------------------------------------------------

        fe_center[
            y,
            x,
        ] = fe_result[
            "center_nm"
        ]

        fe_velocity[
            y,
            x,
        ] = wavelength_to_velocity(
            fe_result[
                "center_nm"
            ],
            FEII_REST_NM,
        )

        fe_velocity_error[
            y,
            x,
        ] = (
            fe_result[
                "center_error_nm"
            ]
            / FEII_REST_NM
            * 299792.458
        )

        fe_snr[
            y,
            x,
        ] = fe_result["snr"]

        accepted_both += 1


# ======================================================================
# Extract valid paired pixels
# ======================================================================

valid = (
    np.isfinite(pa_velocity)
    & np.isfinite(fe_velocity)
    & np.isfinite(pa_velocity_error)
    & np.isfinite(fe_velocity_error)
)


pa_v = pa_velocity[valid]
fe_v = fe_velocity[valid]

pa_err = pa_velocity_error[valid]
fe_err = fe_velocity_error[valid]


# ======================================================================
# Summary
# ======================================================================

print()
print("=" * 70)
print("SPATIAL FIT SUMMARY")
print("=" * 70)

print(
    f"Spatial pixels attempted: "
    f"{attempted}"
)

print(
    f"Pixels with successful fits to BOTH lines: "
    f"{accepted_both}"
)

if attempted > 0:

    print(
        f"Pa beta + [Fe II] acceptance fraction: "
        f"{accepted_both / attempted:.4f}"
    )


if len(pa_v) < 3:

    raise RuntimeError(
        "Too few paired spatial pixels for "
        "a meaningful velocity comparison."
    )


# ======================================================================
# Velocity statistics
# ======================================================================

velocity_difference = (
    pa_v - fe_v
)

combined_error = np.sqrt(
    pa_err ** 2
    + fe_err ** 2
)


print()
print("=" * 70)
print("VELOCITY FIELD STATISTICS")
print("=" * 70)

print()
print("1284 nm / Pa beta:")
print(
    f"  Median: "
    f"{np.median(pa_v):.3f} km/s"
)
print(
    f"  Mean:   "
    f"{np.mean(pa_v):.3f} km/s"
)
print(
    f"  Std:    "
    f"{np.std(pa_v):.3f} km/s"
)

print()
print("[Fe II] 1.257 um:")
print(
    f"  Median: "
    f"{np.median(fe_v):.3f} km/s"
)
print(
    f"  Mean:   "
    f"{np.mean(fe_v):.3f} km/s"
)
print(
    f"  Std:    "
    f"{np.std(fe_v):.3f} km/s"
)

print()
print("Velocity difference:")
print(
    "  Pa beta - [Fe II]"
)

print(
    f"  Median: "
    f"{np.median(velocity_difference):+.3f} km/s"
)

print(
    f"  Mean:   "
    f"{np.mean(velocity_difference):+.3f} km/s"
)

print(
    f"  Std:    "
    f"{np.std(velocity_difference):.3f} km/s"
)


# ======================================================================
# Agreement within uncertainties
# ======================================================================

agreement_sigma = (
    np.abs(
        velocity_difference
    )
    / combined_error
)


within_1sigma = (
    agreement_sigma <= 1.0
)

within_2sigma = (
    agreement_sigma <= 2.0
)

within_3sigma = (
    agreement_sigma <= 3.0
)


print()
print("=" * 70)
print("VELOCITY AGREEMENT WITHIN FIT UNCERTAINTIES")
print("=" * 70)

print(
    f"Within 1 sigma: "
    f"{np.sum(within_1sigma)} / "
    f"{len(agreement_sigma)} "
    f"({np.mean(within_1sigma) * 100:.1f}%)"
)

print(
    f"Within 2 sigma: "
    f"{np.sum(within_2sigma)} / "
    f"{len(agreement_sigma)} "
    f"({np.mean(within_2sigma) * 100:.1f}%)"
)

print(
    f"Within 3 sigma: "
    f"{np.sum(within_3sigma)} / "
    f"{len(agreement_sigma)} "
    f"({np.mean(within_3sigma) * 100:.1f}%)"
)


# ======================================================================
# Pearson correlation
# ======================================================================

correlation_matrix = np.corrcoef(
    pa_v,
    fe_v,
)

pearson_r = (
    correlation_matrix[0, 1]
)


# Least-squares linear relation
slope, intercept = np.polyfit(
    fe_v,
    pa_v,
    1,
)


print()
print("=" * 70)
print("PIXEL-BY-PIXEL VELOCITY CORRELATION")
print("=" * 70)

print(
    f"Pearson r: "
    f"{pearson_r:.6f}"
)

print(
    f"Linear slope: "
    f"{slope:.6f}"
)

print(
    f"Linear intercept: "
    f"{intercept:.3f} km/s"
)


# ======================================================================
# Difference map
# ======================================================================

difference_map = (
    pa_velocity
    - fe_velocity
)


# ======================================================================
# Plot 1 — 1284 nm velocity map
# ======================================================================

plt.figure(
    figsize=(12, 8)
)

image = plt.imshow(
    pa_velocity,
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
    "M51 1284 nm Velocity Map\n"
    "Free-centroid fit assuming Pa beta"
)

plt.tight_layout()

plt.savefig(
    "m51_1284_velocity_map_comparison.png",
    dpi=150,
)

plt.close()


# ======================================================================
# Plot 2 — [Fe II] velocity map
# ======================================================================

plt.figure(
    figsize=(12, 8)
)

image = plt.imshow(
    fe_velocity,
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
    "M51 [Fe II] 1.257 um Velocity Map"
)

plt.tight_layout()

plt.savefig(
    "m51_feii_1257_velocity_map.png",
    dpi=150,
)

plt.close()


# ======================================================================
# Plot 3 — Velocity difference map
# ======================================================================

plt.figure(
    figsize=(12, 8)
)

image = plt.imshow(
    difference_map,
    origin="lower",
    interpolation="nearest",
)

plt.colorbar(
    image,
    label="Pa beta - [Fe II] velocity (km/s)",
)

plt.xlabel(
    "Spatial X pixel"
)

plt.ylabel(
    "Spatial Y pixel"
)

plt.title(
    "M51 Velocity Difference Map\n"
    "1284 nm / Pa beta minus [Fe II]"
)

plt.tight_layout()

plt.savefig(
    "m51_1284_minus_feii_velocity_map.png",
    dpi=150,
)

plt.close()


# ======================================================================
# Plot 4 — Pixel-by-pixel correlation
# ======================================================================

plt.figure(
    figsize=(9, 8)
)

plt.scatter(
    fe_v,
    pa_v,
    s=8,
    alpha=0.35,
)

x_line = np.linspace(
    np.nanmin(fe_v),
    np.nanmax(fe_v),
    200,
)

y_line = (
    slope
    * x_line
    + intercept
)

plt.plot(
    x_line,
    y_line,
    linewidth=2,
)

# One-to-one relation
plt.plot(
    x_line,
    x_line,
    linestyle="--",
)

plt.xlabel(
    "[Fe II] velocity (km/s)"
)

plt.ylabel(
    "1284 nm velocity assuming Pa beta (km/s)"
)

plt.title(
    "M51 Spatial Velocity Correlation\n"
    f"Pearson r = {pearson_r:.4f}"
)

plt.grid(
    alpha=0.25
)

plt.tight_layout()

plt.savefig(
    "m51_1284_vs_feii_velocity_correlation.png",
    dpi=150,
)

plt.close()


# ======================================================================
# Plot 5 — Velocity difference histogram
# ======================================================================

plt.figure(
    figsize=(10, 6)
)

plt.hist(
    velocity_difference,
    bins=50,
)

plt.axvline(
    0.0,
    linestyle="--",
)

plt.axvline(
    np.median(
        velocity_difference
    ),
    linestyle=":",
    linewidth=2,
)

plt.xlabel(
    "1284 nm velocity - [Fe II] velocity (km/s)"
)

plt.ylabel(
    "Number of spatial pixels"
)

plt.title(
    "M51 Spatial Velocity Difference Distribution"
)

plt.tight_layout()

plt.savefig(
    "m51_1284_vs_feii_velocity_difference_histogram.png",
    dpi=150,
)

plt.close()


# ======================================================================
# Final interpretation
# ======================================================================

print()
print("=" * 70)
print("INTERPRETATION")
print("=" * 70)

print()
print(
    "The 1284 nm and [Fe II] velocity fields were "
    "fitted independently in the same spatial pixels."
)

print()

if pearson_r >= 0.8:

    print(
        "The two velocity fields show a strong "
        "positive spatial correlation."
    )

elif pearson_r >= 0.5:

    print(
        "The two velocity fields show a moderate "
        "positive spatial correlation."
    )

else:

    print(
        "The two velocity fields show weak "
        "spatial correlation."
    )


print()
print(
    f"Median velocity difference: "
    f"{np.median(velocity_difference):+.3f} km/s"
)

print(
    f"Median combined uncertainty: "
    f"{np.median(combined_error):.3f} km/s"
)

print(
    f"Fraction agreeing within 1 sigma: "
    f"{np.mean(within_1sigma) * 100:.1f}%"
)

print(
    f"Fraction agreeing within 2 sigma: "
    f"{np.mean(within_2sigma) * 100:.1f}%"
)

print(
    f"Fraction agreeing within 3 sigma: "
    f"{np.mean(within_3sigma) * 100:.1f}%"
)


print()
print(
    "IMPORTANT:"
)

print(
    "Agreement between velocity fields does not "
    "by itself prove that the 1284 nm feature is Pa beta."
)

print(
    "However, strong spatial correlation combined "
    "with a small velocity difference would provide "
    "independent evidence that the 1284 nm emission "
    "is participating in the same nebular kinematics "
    "as [Fe II]."
)

print()
print(
    "The analysis does not yet use a fully propagated "
    "JWST S3D uncertainty cube. The uncertainty "
    "comparison should therefore be regarded as "
    "exploratory."
)


# ======================================================================
# Output files
# ======================================================================

print()
print("=" * 70)
print("OUTPUT FILES")
print("=" * 70)

print(
    "m51_1284_velocity_map_comparison.png"
)

print(
    "m51_feii_1257_velocity_map.png"
)

print(
    "m51_1284_minus_feii_velocity_map.png"
)

print(
    "m51_1284_vs_feii_velocity_correlation.png"
)

print(
    "m51_1284_vs_feii_velocity_difference_histogram.png"
)

print()
print("=" * 70)
print("TEST COMPLETE")
print("=" * 70)
