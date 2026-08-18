from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy.stats import pearsonr, spearmanr, theilslopes


# ============================================================
# Configuration
# ============================================================

S3D_PATH = Path(
    "data/m51_jwst_level3/"
    "jw03435-o006_t010_nirspec_g140m-f100lp_s3d.fits"
)

# Hydrogen recombination laboratory wavelengths used
# throughout the M51 analysis.
PA_BETA_REST_NM = 1281.80700000
PA_GAMMA_REST_NM = 1093.81000000

REFERENCE_VELOCITY = 573.72

# Local fitting windows
PA_BETA_WINDOW_NM = 6.0
PA_GAMMA_WINDOW_NM = 6.0

# Quality requirements
MIN_SN = 10.0
MAX_VELOCITY_ERROR = 15.0
MIN_VELOCITY = 400.0
MAX_VELOCITY = 700.0

# Minimum number of spectral samples required
MIN_POINTS = 6

C_KM_S = 299792.458


# ============================================================
# Gaussian model
# ============================================================


def gaussian_with_continuum(
    wavelength,
    amplitude,
    center,
    sigma,
    continuum,
):
    """
    Gaussian emission line plus constant continuum.
    """

    return (
        continuum
        + amplitude
        * np.exp(
            -0.5
            * (
                (wavelength - center)
                / sigma
            ) ** 2
        )
    )


# ============================================================
# Load S3D cube
# ============================================================


def load_s3d(path):
    """
    Load JWST/NIRSpec Level-3 S3D cube.

    Returns
    -------
    wavelength_nm
        1-D wavelength array in nm.

    cube
        Flux cube with shape:
        (spectral, y, x)
    """

    from astropy.io import fits

    print("=" * 70)
    print("LOADING S3D CUBE")
    print("=" * 70)

    with fits.open(path) as hdul:

        cube = np.asarray(
            hdul[1].data,
            dtype=float,
        )

        header = hdul[1].header

        print()
        print(
            "Raw cube shape:",
            cube.shape,
        )

        # ----------------------------------------------------
        # JWST Level-3 cube wavelength solution
        # ----------------------------------------------------

        crval3 = header["CRVAL3"]
        cdelt3 = header["CDELT3"]
        crpix3 = header["CRPIX3"]

        n_wave = cube.shape[0]

        pixel = np.arange(
            1,
            n_wave + 1,
            dtype=float,
        )

        wavelength_um = (
            crval3
            + (pixel - crpix3)
            * cdelt3
        )

        wavelength_nm = (
            wavelength_um * 1000.0
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
        f"{cube.shape[2]} x {cube.shape[1]}"
    )

    return wavelength_nm, cube


# ============================================================
# Relativistic velocity
# ============================================================


def velocity_from_wavelength(
    observed_nm,
    rest_nm,
):
    """
    Relativistic radial velocity.
    """

    ratio = observed_nm / rest_nm

    beta = (
        ratio**2 - 1.0
    ) / (
        ratio**2 + 1.0
    )

    return beta * C_KM_S


def wavelength_from_velocity(
    rest_nm,
    velocity_km_s,
):
    """
    Relativistic wavelength prediction.
    """

    beta = velocity_km_s / C_KM_S

    factor = np.sqrt(
        (1.0 + beta)
        / (1.0 - beta)
    )

    return rest_nm * factor


# ============================================================
# Fit one spatial spectrum
# ============================================================


def fit_line(
    wavelength_nm,
    spectrum,
    center_guess,
    window_nm,
):
    """
    Fit a Gaussian + constant continuum.

    Returns
    -------
    dictionary or None
    """

    mask = (
        np.isfinite(wavelength_nm)
        & np.isfinite(spectrum)
        & (
            np.abs(
                wavelength_nm
                - center_guess
            )
            <= window_nm
        )
    )

    x = wavelength_nm[mask]
    y = spectrum[mask]

    if len(x) < MIN_POINTS:
        return None

    # --------------------------------------------------------
    # Remove pathological values
    # --------------------------------------------------------

    finite = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    x = x[finite]
    y = y[finite]

    if len(x) < MIN_POINTS:
        return None

    # --------------------------------------------------------
    # Continuum estimate
    # --------------------------------------------------------

    continuum_guess = np.median(y)

    # --------------------------------------------------------
    # Emission amplitude
    # --------------------------------------------------------

    amplitude_guess = (
        np.max(y)
        - continuum_guess
    )

    if not np.isfinite(
        amplitude_guess
    ):
        return None

    if amplitude_guess <= 0:
        return None

    # --------------------------------------------------------
    # Initial Gaussian width
    # --------------------------------------------------------

    sigma_guess = 0.6

    lower_bounds = [
        0.0,
        center_guess - window_nm,
        0.15,
        -np.inf,
    ]

    upper_bounds = [
        np.inf,
        center_guess + window_nm,
        3.0,
        np.inf,
    ]

    initial = [
        amplitude_guess,
        center_guess,
        sigma_guess,
        continuum_guess,
    ]

    try:

        popt, pcov = curve_fit(
            gaussian_with_continuum,
            x,
            y,
            p0=initial,
            bounds=(
                lower_bounds,
                upper_bounds,
            ),
            maxfev=10000,
        )

    except (
        RuntimeError,
        ValueError,
        FloatingPointError,
    ):
        return None

    amplitude = popt[0]
    center = popt[1]
    sigma = abs(popt[2])
    continuum = popt[3]

    # --------------------------------------------------------
    # Parameter uncertainties
    # --------------------------------------------------------

    try:

        errors = np.sqrt(
            np.diag(pcov)
        )

        amplitude_error = errors[0]
        center_error = errors[1]

    except (
        ValueError,
        IndexError,
    ):
        return None

    if (
        not np.isfinite(
            amplitude_error
        )
        or not np.isfinite(
            center_error
        )
    ):
        return None

    if amplitude_error <= 0:
        return None

    # --------------------------------------------------------
    # Velocity
    # --------------------------------------------------------

    velocity = velocity_from_wavelength(
        center,
        center_guess
        * 0
        + (
            PA_BETA_REST_NM
            if abs(
                center_guess
                - wavelength_from_velocity(
                    PA_BETA_REST_NM,
                    REFERENCE_VELOCITY,
                )
            )
            < 10.0
            else PA_GAMMA_REST_NM
        ),
    )

    # --------------------------------------------------------
    # Convert wavelength uncertainty
    # to velocity uncertainty.
    #
    # Local non-relativistic approximation is sufficient
    # for the small centroid uncertainty.
    # --------------------------------------------------------

    velocity_error = (
        C_KM_S
        * center_error
        / center
    )

    sn = (
        amplitude
        / amplitude_error
    )

    fwhm = (
        2.354820045
        * sigma
    )

    # --------------------------------------------------------
    # Residuals
    # --------------------------------------------------------

    model = gaussian_with_continuum(
        x,
        *popt,
    )

    residual = y - model

    rms = np.sqrt(
        np.mean(
            residual**2
        )
    )

    return {
        "center": center,
        "center_error": center_error,
        "velocity": velocity,
        "velocity_error": velocity_error,
        "amplitude": amplitude,
        "amplitude_error": amplitude_error,
        "sn": sn,
        "sigma": sigma,
        "fwhm": fwhm,
        "continuum": continuum,
        "residual_rms": rms,
        "n_points": len(x),
    }


# ============================================================
# Main analysis
# ============================================================


print("=" * 70)
print(
    "M51 JWST/NIRSpec HIGH-S/N SPATIAL VELOCITY CORRELATION"
)
print(
    "Pa beta versus Pa gamma"
)
print("=" * 70)

print()
print("S3D:")
print(S3D_PATH)

wavelength_nm, cube = load_s3d(
    S3D_PATH
)


# ============================================================
# Reference wavelengths
# ============================================================

pa_beta_expected = (
    wavelength_from_velocity(
        PA_BETA_REST_NM,
        REFERENCE_VELOCITY,
    )
)

pa_gamma_expected = (
    wavelength_from_velocity(
        PA_GAMMA_REST_NM,
        REFERENCE_VELOCITY,
    )
)

print()
print("=" * 70)
print("REFERENCE WAVELENGTHS")
print("=" * 70)

print(
    f"Reference velocity: "
    f"{REFERENCE_VELOCITY:.2f} km/s"
)

print(
    f"Pa beta rest wavelength: "
    f"{PA_BETA_REST_NM:.8f} nm"
)

print(
    f"Pa gamma rest wavelength: "
    f"{PA_GAMMA_REST_NM:.8f} nm"
)

print(
    f"Expected Pa beta center: "
    f"{pa_beta_expected:.8f} nm"
)

print(
    f"Expected Pa gamma center: "
    f"{pa_gamma_expected:.8f} nm"
)


# ============================================================
# Determine spatial dimensions
# ============================================================

n_wave, ny, nx = cube.shape


# ============================================================
# Storage arrays
# ============================================================

pa_beta_velocity = np.full(
    (ny, nx),
    np.nan,
)

pa_beta_velocity_error = np.full(
    (ny, nx),
    np.nan,
)

pa_beta_sn = np.full(
    (ny, nx),
    np.nan,
)

pa_gamma_velocity = np.full(
    (ny, nx),
    np.nan,
)

pa_gamma_velocity_error = np.full(
    (ny, nx),
    np.nan,
)

pa_gamma_sn = np.full(
    (ny, nx),
    np.nan,
)


# ============================================================
# Fit all spatial pixels
# ============================================================

print()
print("=" * 70)
print("FITTING SPATIAL PIXELS")
print("=" * 70)

attempted = 0
successful_both = 0


for y in range(ny):

    if (
        y == 0
        or y % 10 == 0
    ):

        print(
            f"Processing row "
            f"{y + 1}/{ny}..."
        )

    for x in range(nx):

        spectrum = cube[
            :,
            y,
            x,
        ]

        attempted += 1

        # ----------------------------------------------------
        # Pa beta
        # ----------------------------------------------------

        beta_fit = fit_line(
            wavelength_nm,
            spectrum,
            pa_beta_expected,
            PA_BETA_WINDOW_NM,
        )

        if beta_fit is None:
            continue

        # ----------------------------------------------------
        # Pa gamma
        # ----------------------------------------------------

        gamma_fit = fit_line(
            wavelength_nm,
            spectrum,
            pa_gamma_expected,
            PA_GAMMA_WINDOW_NM,
        )

        if gamma_fit is None:
            continue

        # ----------------------------------------------------
        # Store preliminary results
        # ----------------------------------------------------

        pa_beta_velocity[
            y, x
        ] = beta_fit["velocity"]

        pa_beta_velocity_error[
            y, x
        ] = beta_fit["velocity_error"]

        pa_beta_sn[
            y, x
        ] = beta_fit["sn"]

        pa_gamma_velocity[
            y, x
        ] = gamma_fit["velocity"]

        pa_gamma_velocity_error[
            y, x
        ] = gamma_fit["velocity_error"]

        pa_gamma_sn[
            y, x
        ] = gamma_fit["sn"]

        successful_both += 1


# ============================================================
# Initial population
# ============================================================

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
    f"{successful_both}"
)


# ============================================================
# High-S/N quality selection
# ============================================================

quality = (
    np.isfinite(
        pa_beta_velocity
    )
    & np.isfinite(
        pa_gamma_velocity
    )
    & np.isfinite(
        pa_beta_velocity_error
    )
    & np.isfinite(
        pa_gamma_velocity_error
    )
    & np.isfinite(
        pa_beta_sn
    )
    & np.isfinite(
        pa_gamma_sn
    )
    & (
        pa_beta_sn
        >= MIN_SN
    )
    & (
        pa_gamma_sn
        >= MIN_SN
    )
    & (
        pa_beta_velocity_error
        <= MAX_VELOCITY_ERROR
    )
    & (
        pa_gamma_velocity_error
        <= MAX_VELOCITY_ERROR
    )
    & (
        pa_beta_velocity
        >= MIN_VELOCITY
    )
    & (
        pa_beta_velocity
        <= MAX_VELOCITY
    )
    & (
        pa_gamma_velocity
        >= MIN_VELOCITY
    )
    & (
        pa_gamma_velocity
        <= MAX_VELOCITY
    )
)


beta_v = pa_beta_velocity[
    quality
]

beta_err = pa_beta_velocity_error[
    quality
]

beta_sn = pa_beta_sn[
    quality
]

gamma_v = pa_gamma_velocity[
    quality
]

gamma_err = pa_gamma_velocity_error[
    quality
]

gamma_sn = pa_gamma_sn[
    quality
]

n_good = len(beta_v)


print()
print("=" * 70)
print("HIGH-S/N QUALITY SELECTION")
print("=" * 70)

print(
    f"Minimum S/N: {MIN_SN:.1f}"
)

print(
    f"Maximum centroid velocity error: "
    f"{MAX_VELOCITY_ERROR:.1f} km/s"
)

print(
    f"Velocity range: "
    f"{MIN_VELOCITY:.0f} - "
    f"{MAX_VELOCITY:.0f} km/s"
)

print()
print(
    f"High-quality paired pixels: "
    f"{n_good}"
)

print(
    f"Fraction retained: "
    f"{n_good / attempted:.4f}"
)


if n_good < 5:

    raise RuntimeError(
        "Too few high-quality paired "
        "pixels for correlation analysis."
    )


# ============================================================
# Velocity statistics
# ============================================================

beta_median = np.median(
    beta_v
)

gamma_median = np.median(
    gamma_v
)

beta_mean = np.mean(
    beta_v
)

gamma_mean = np.mean(
    gamma_v
)

beta_std = np.std(
    beta_v,
    ddof=1,
)

gamma_std = np.std(
    gamma_v,
    ddof=1,
)

difference = (
    beta_v
    - gamma_v
)

difference_median = np.median(
    difference
)

difference_mean = np.mean(
    difference
)

difference_std = np.std(
    difference,
    ddof=1,
)

combined_error = np.sqrt(
    beta_err**2
    + gamma_err**2
)

median_combined_error = np.median(
    combined_error
)


print()
print("=" * 70)
print("HIGH-S/N VELOCITY STATISTICS")
print("=" * 70)

print()
print("Pa beta:")
print(
    f"  Median velocity: "
    f"{beta_median:.3f} km/s"
)

print(
    f"  Mean velocity:   "
    f"{beta_mean:.3f} km/s"
)

print(
    f"  Standard deviation: "
    f"{beta_std:.3f} km/s"
)

print(
    f"  Median S/N: "
    f"{np.median(beta_sn):.2f}"
)

print()
print("Pa gamma:")
print(
    f"  Median velocity: "
    f"{gamma_median:.3f} km/s"
)

print(
    f"  Mean velocity:   "
    f"{gamma_mean:.3f} km/s"
)

print(
    f"  Standard deviation: "
    f"{gamma_std:.3f} km/s"
)

print(
    f"  Median S/N: "
    f"{np.median(gamma_sn):.2f}"
)

print()
print("Velocity difference: Pa beta - Pa gamma")

print(
    f"  Median: "
    f"{difference_median:.3f} km/s"
)

print(
    f"  Mean:   "
    f"{difference_mean:.3f} km/s"
)

print(
    f"  Standard deviation: "
    f"{difference_std:.3f} km/s"
)

print(
    f"  Median combined uncertainty: "
    f"{median_combined_error:.3f} km/s"
)


# ============================================================
# Correlation
# ============================================================

pearson_r, pearson_p = pearsonr(
    gamma_v,
    beta_v,
)

spearman_rho, spearman_p = spearmanr(
    gamma_v,
    beta_v,
)

sen_slope, sen_intercept, sen_low, sen_high = (
    theilslopes(
        beta_v,
        gamma_v,
        alpha=0.05,
    )
)


# Ordinary least squares
ols_slope, ols_intercept = np.polyfit(
    gamma_v,
    beta_v,
    1,
)


print()
print("=" * 70)
print("PA BETA vs PA GAMMA VELOCITY CORRELATION")
print("=" * 70)

print()
print(
    f"Pearson r: "
    f"{pearson_r:.6f}"
)

print(
    f"Pearson p-value: "
    f"{pearson_p:.6e}"
)

print()
print(
    f"Spearman rho: "
    f"{spearman_rho:.6f}"
)

print(
    f"Spearman p-value: "
    f"{spearman_p:.6e}"
)

print()
print(
    f"Ordinary least-squares slope: "
    f"{ols_slope:.6f}"
)

print(
    f"Ordinary intercept: "
    f"{ols_intercept:.3f} km/s"
)

print()
print(
    f"Theil-Sen robust slope: "
    f"{sen_slope:.6f}"
)

print(
    f"Theil-Sen intercept: "
    f"{sen_intercept:.3f} km/s"
)

print(
    f"Theil-Sen slope 95% interval: "
    f"{sen_low:.6f} - "
    f"{sen_high:.6f}"
)


# ============================================================
# Agreement within uncertainties
# ============================================================

for sigma_level in [
    1,
    2,
    3,
]:

    agreement = (
        np.abs(difference)
        <= sigma_level
        * combined_error
    )

    count = np.sum(
        agreement
    )

    fraction = (
        count / n_good
    )

    print(
        f"Within {sigma_level} sigma: "
        f"{count} / {n_good} "
        f"({fraction * 100:.1f}%)"
    )


# ============================================================
# Generate velocity maps
# ============================================================

beta_map = np.where(
    quality,
    pa_beta_velocity,
    np.nan,
)

gamma_map = np.where(
    quality,
    pa_gamma_velocity,
    np.nan,
)

difference_map = np.where(
    quality,
    pa_beta_velocity
    - pa_gamma_velocity,
    np.nan,
)


# ============================================================
# Plot 1 — Pa beta velocity map
# ============================================================

plt.figure(
    figsize=(11, 8)
)

im = plt.imshow(
    beta_map,
    origin="lower",
    interpolation="nearest",
)

plt.colorbar(
    im,
    label="Velocity (km/s)",
)

plt.xlabel(
    "Spatial X pixel"
)

plt.ylabel(
    "Spatial Y pixel"
)

plt.title(
    "M51 Pa beta / 1284 nm\n"
    "High-S/N Velocity Field"
)

plt.tight_layout()

plt.savefig(
    "m51_pabeta_highsn_velocity_map.png",
    dpi=150,
)

plt.close()


# ============================================================
# Plot 2 — Pa gamma velocity map
# ============================================================

plt.figure(
    figsize=(11, 8)
)

im = plt.imshow(
    gamma_map,
    origin="lower",
    interpolation="nearest",
)

plt.colorbar(
    im,
    label="Velocity (km/s)",
)

plt.xlabel(
    "Spatial X pixel"
)

plt.ylabel(
    "Spatial Y pixel"
)

plt.title(
    "M51 Pa gamma\n"
    "High-S/N Velocity Field"
)

plt.tight_layout()

plt.savefig(
    "m51_pgamma_highsn_velocity_map.png",
    dpi=150,
)

plt.close()


# ============================================================
# Plot 3 — Velocity difference map
# ============================================================

plt.figure(
    figsize=(11, 8)
)

im = plt.imshow(
    difference_map,
    origin="lower",
    interpolation="nearest",
)

plt.colorbar(
    im,
    label="Pa beta - Pa gamma (km/s)",
)

plt.xlabel(
    "Spatial X pixel"
)

plt.ylabel(
    "Spatial Y pixel"
)

plt.title(
    "M51 Pa beta - Pa gamma\n"
    "Velocity Difference"
)

plt.tight_layout()

plt.savefig(
    "m51_pabeta_pgamma_velocity_difference.png",
    dpi=150,
)

plt.close()


# ============================================================
# Plot 4 — Pixel-by-pixel correlation
# ============================================================

plt.figure(
    figsize=(10, 8)
)

plt.scatter(
    gamma_v,
    beta_v,
    s=25,
    alpha=0.65,
)

x_min = min(
    gamma_v.min(),
    beta_v.min(),
)

x_max = max(
    gamma_v.max(),
    beta_v.max(),
)

x_line = np.linspace(
    x_min,
    x_max,
    200,
)

plt.plot(
    x_line,
    ols_slope * x_line
    + ols_intercept,
    label="OLS",
)

plt.plot(
    x_line,
    x_line,
    linestyle="--",
    label="1:1",
)

plt.xlabel(
    "Pa gamma velocity (km/s)"
)

plt.ylabel(
    "Pa beta / 1284 nm velocity (km/s)"
)

plt.title(
    "M51 Pa beta vs Pa gamma\n"
    f"High-S/N Velocity Correlation\n"
    f"Pearson r = {pearson_r:.4f}, "
    f"Spearman rho = {spearman_rho:.4f}"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "m51_pabeta_pgamma_velocity_correlation.png",
    dpi=150,
)

plt.close()


# ============================================================
# Plot 5 — Velocity difference distribution
# ============================================================

plt.figure(
    figsize=(10, 7)
)

plt.hist(
    difference,
    bins=30,
)

plt.axvline(
    0.0,
    linestyle="--",
)

plt.axvline(
    difference_median,
    linestyle=":",
)

plt.xlabel(
    "Pa beta - Pa gamma velocity (km/s)"
)

plt.ylabel(
    "Number of spatial pixels"
)

plt.title(
    "M51 Pa beta - Pa gamma\n"
    "Velocity Difference Distribution"
)

plt.tight_layout()

plt.savefig(
    "m51_pabeta_pgamma_velocity_difference_histogram.png",
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
This experiment directly compares the spatially
resolved kinematics of the 1284 nm feature, treated
as Pa beta, with Pa gamma.

Both transitions are hydrogen recombination lines,
so this is a more direct physical comparison than
the previous Pa beta versus [Fe II] experiment.

The important quantities are:

  - the median velocity difference;
  - the pixel-by-pixel Pearson correlation;
  - the Spearman rank correlation;
  - the Theil-Sen slope;
  - the fraction of pixels consistent within
    their formal velocity uncertainties.

A strong positive correlation and a robust slope
near unity would indicate that the 1284 nm feature
and Pa gamma trace the same spatially resolved
kinematic structure.

A small median velocity difference with weak
correlation would indicate that the two hydrogen
lines share the same bulk velocity but do not
necessarily reproduce identical spatial velocity
structure.

A large systematic velocity difference would be
problematic for the Pa beta interpretation.

The analysis remains exploratory because the current
implementation estimates local spectral noise from
the S3D data rather than using a fully propagated
JWST uncertainty cube.
"""
)

print()
print("=" * 70)
print("FILES SAVED")
print("=" * 70)

print(
    "m51_pabeta_highsn_velocity_map.png"
)

print(
    "m51_pgamma_highsn_velocity_map.png"
)

print(
    "m51_pabeta_pgamma_velocity_difference.png"
)

print(
    "m51_pabeta_pgamma_velocity_correlation.png"
)

print(
    "m51_pabeta_pgamma_velocity_difference_histogram.png"
)

print()
print("=" * 70)
print("TEST COMPLETE")
print("=" * 70)
