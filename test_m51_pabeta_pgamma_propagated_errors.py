from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.optimize import curve_fit
from scipy.stats import pearsonr, spearmanr, theilslopes


# ============================================================
# M51 JWST/NIRSpec S3D
# Propagated-error Pa beta / Pa gamma comparison
# ============================================================

S3D_PATH = Path(
    "data/m51_jwst_level3/"
    "jw03435-o006_t010_nirspec_g140m-f100lp_s3d.fits"
)


# ============================================================
# Physical constants
# ============================================================

C_KMS = 299792.458


# ============================================================
# Laboratory wavelengths
# ============================================================

PA_BETA_REST_NM = 1281.80700
PA_GAMMA_REST_NM = 1093.81000

REFERENCE_VELOCITY = 573.72


# ============================================================
# Analysis settings
# ============================================================

PA_BETA_WINDOW_NM = 5.0
PA_GAMMA_WINDOW_NM = 5.0

MIN_SNR = 10.0
MAX_VELOCITY_ERROR = 15.0

MIN_POINTS = 7

# Require positive uncertainty and finite values
MIN_ERR = 1.0e-30


# ============================================================
# Gaussian + linear continuum
# ============================================================

def gaussian_linear(
    wavelength,
    amplitude,
    center,
    sigma,
    continuum,
    slope,
):
    """
    Gaussian emission line plus linear continuum.
    """

    return (
        continuum
        + slope * (wavelength - center)
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
# Velocity conversion
# ============================================================

def wavelength_to_velocity(
    observed_nm,
    rest_nm,
):
    """
    Non-relativistic Doppler velocity.
    """

    return (
        (observed_nm / rest_nm) - 1.0
    ) * C_KMS


def velocity_error_from_wavelength_error(
    wavelength_error,
    rest_nm,
):
    return (
        C_KMS
        * wavelength_error
        / rest_nm
    )


# ============================================================
# Locate S3D science / error arrays
# ============================================================

def find_hdu_by_name(hdul, names):

    names = {
        name.upper()
        for name in names
    }

    for hdu in hdul:

        if hdu.name.upper() in names:

            if hdu.data is not None:

                return hdu

    return None


# ============================================================
# Construct wavelength array
# ============================================================

def get_wavelength_nm(hdul, cube):

    header = hdul[1].header

    # --------------------------------------------------------
    # First try standard FITS linear spectral WCS keywords.
    # --------------------------------------------------------

    candidates = [
        (
            "CRVAL3",
            "CDELT3",
            "CRPIX3",
        ),
        (
            "CRVAL1",
            "CDELT1",
            "CRPIX1",
        ),
    ]

    for crval_key, cdelt_key, crpix_key in candidates:

        if (
            crval_key in header
            and cdelt_key in header
            and crpix_key in header
        ):

            crval = header[crval_key]
            cdelt = header[cdelt_key]
            crpix = header[crpix_key]

            # Determine which FITS axis corresponds to
            # the spectral dimension.

            naxis = cube.ndim

            for axis in range(1, naxis + 1):

                ctype = header.get(
                    f"CTYPE{axis}",
                    "",
                ).upper()

                if (
                    "WAVE" in ctype
                    or "FREQ" in ctype
                    or "AWAV" in ctype
                ):

                    crval_key = f"CRVAL{axis}"
                    cdelt_key = f"CDELT{axis}"
                    crpix_key = f"CRPIX{axis}"

                    if (
                        crval_key in header
                        and cdelt_key in header
                        and crpix_key in header
                    ):

                        crval = header[
                            crval_key
                        ]

                        cdelt = header[
                            cdelt_key
                        ]

                        crpix = header[
                            crpix_key
                        ]

                        n = cube.shape[0]

                        pixels = (
                            np.arange(n)
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

                        # S3D wavelength is normally
                        # expressed in microns.

                        if np.nanmedian(
                            wavelength
                        ) < 100:

                            wavelength_nm = (
                                wavelength
                                * 1000.0
                            )

                        else:

                            wavelength_nm = (
                                wavelength
                            )

                        return wavelength_nm

    # --------------------------------------------------------
    # Fallback to the known M51 S3D sampling.
    # --------------------------------------------------------

    print()
    print(
        "WARNING: Could not construct "
        "spectral WCS from header."
    )

    print(
        "Using fallback wavelength grid."
    )

    wavelength_um = (
        0.970318028616
        + np.arange(cube.shape[0])
        * 0.0006360000
    )

    return wavelength_um * 1000.0


# ============================================================
# Fit one emission line
# ============================================================

def fit_line(
    wavelength,
    flux,
    error,
    rest_nm,
    window_nm,
):

    mask = (
        np.isfinite(wavelength)
        & np.isfinite(flux)
        & np.isfinite(error)
        & (error > MIN_ERR)
        & (
            np.abs(
                wavelength
                - rest_nm
                * (
                    1.0
                    + REFERENCE_VELOCITY
                    / C_KMS
                )
            )
            <= window_nm
        )
    )

    if np.sum(mask) < MIN_POINTS:

        return None

    x = wavelength[mask]
    y = flux[mask]
    err = error[mask]

    # --------------------------------------------------------
    # Initial center
    # --------------------------------------------------------

    expected_center = (
        rest_nm
        * (
            1.0
            + REFERENCE_VELOCITY
            / C_KMS
        )
    )

    # --------------------------------------------------------
    # Initial continuum
    # --------------------------------------------------------

    continuum = np.median(y)

    amplitude = (
        np.max(y)
        - continuum
    )

    if not np.isfinite(amplitude):
        return None

    if amplitude <= 0:
        return None

    # Instrumental-scale initial sigma.
    sigma_initial = 0.60

    slope_initial = 0.0

    p0 = [
        amplitude,
        expected_center,
        sigma_initial,
        continuum,
        slope_initial,
    ]

    # --------------------------------------------------------
    # Bounds
    # --------------------------------------------------------

    lower = [
        0.0,
        expected_center - 3.0,
        0.10,
        -np.inf,
        -np.inf,
    ]

    upper = [
        np.inf,
        expected_center + 3.0,
        3.0,
        np.inf,
        np.inf,
    ]

    try:

        popt, pcov = curve_fit(
            gaussian_linear,
            x,
            y,
            p0=p0,
            sigma=err,
            absolute_sigma=True,
            bounds=(
                lower,
                upper,
            ),
            maxfev=20000,
        )

    except Exception:

        return None

    amplitude_fit = popt[0]
    center_fit = popt[1]
    sigma_fit = popt[2]

    amplitude_error = np.sqrt(
        np.abs(pcov[0, 0])
    )

    center_error = np.sqrt(
        np.abs(pcov[1, 1])
    )

    sigma_error = np.sqrt(
        np.abs(pcov[2, 2])
    )

    model = gaussian_linear(
        x,
        *popt,
    )

    residuals = y - model

    chi2 = np.sum(
        (residuals / err) ** 2
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

    velocity = wavelength_to_velocity(
        center_fit,
        rest_nm,
    )

    velocity_error = (
        velocity_error_from_wavelength_error(
            center_error,
            rest_nm,
        )
    )

    snr = (
        amplitude_fit
        / amplitude_error
    )

    return {
        "center_nm": center_fit,
        "center_error_nm": center_error,
        "velocity_kms": velocity,
        "velocity_error_kms": velocity_error,
        "amplitude": amplitude_fit,
        "amplitude_error": amplitude_error,
        "snr": snr,
        "sigma_nm": sigma_fit,
        "sigma_error_nm": sigma_error,
        "chi2": chi2,
        "dof": dof,
        "reduced_chi2": reduced_chi2,
    }


# ============================================================
# Load cube
# ============================================================

print("=" * 70)
print("M51 Pa BETA / Pa GAMMA")
print("PROPAGATED JWST S3D ERROR-CUBE TEST")
print("=" * 70)

print()
print("S3D:")
print(S3D_PATH)

print()
print("=" * 70)
print("LOADING S3D CUBE")
print("=" * 70)

with fits.open(
    S3D_PATH,
    memmap=False,
) as hdul:

    print()
    print("HDU structure:")

    for index, hdu in enumerate(hdul):

        shape = (
            hdu.data.shape
            if hdu.data is not None
            else None
        )

        print(
            f"  HDU {index}: "
            f"{hdu.name:12s} "
            f"{shape}"
        )

    # --------------------------------------------------------
    # Science cube
    # --------------------------------------------------------

    sci_hdu = find_hdu_by_name(
        hdul,
        [
            "SCI",
            "PRIMARY",
        ],
    )

    if sci_hdu is None:

        raise RuntimeError(
            "Could not locate S3D science cube."
        )

    cube = np.asarray(
        sci_hdu.data,
        dtype=float,
    )

    print()
    print(
        f"Science cube shape: "
        f"{cube.shape}"
    )

    # --------------------------------------------------------
    # Error cube
    # --------------------------------------------------------

    err_hdu = find_hdu_by_name(
        hdul,
        [
            "ERR",
            "ERROR",
        ],
    )

    if err_hdu is None:

        raise RuntimeError(
            "No ERR uncertainty cube was found."
        )

    error_cube = np.asarray(
        err_hdu.data,
        dtype=float,
    )

    print(
        f"Error cube shape: "
        f"{error_cube.shape}"
    )

    if error_cube.shape != cube.shape:

        raise RuntimeError(
            "SCI and ERR cube shapes differ."
        )

    # --------------------------------------------------------
    # DQ cube if available
    # --------------------------------------------------------

    dq_hdu = find_hdu_by_name(
        hdul,
        [
            "DQ",
        ],
    )

    if dq_hdu is not None:

        dq_cube = np.asarray(
            dq_hdu.data
        )

        print(
            f"DQ cube shape: "
            f"{dq_cube.shape}"
        )

    else:

        dq_cube = None

        print(
            "No DQ cube found."
        )

    # --------------------------------------------------------
    # Wavelength
    # --------------------------------------------------------

    wavelength_nm = (
        get_wavelength_nm(
            hdul,
            cube,
        )
    )


# ============================================================
# Cube diagnostics
# ============================================================

print()
print("=" * 70)
print("CUBE DIAGNOSTICS")
print("=" * 70)

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

finite_error_fraction = (
    np.count_nonzero(
        np.isfinite(error_cube)
        & (error_cube > 0)
    )
    / error_cube.size
)

print(
    f"Valid propagated-error fraction: "
    f"{finite_error_fraction:.4f}"
)


# ============================================================
# Verify target wavelength coverage
# ============================================================

for name, rest_nm in [
    (
        "Pa beta",
        PA_BETA_REST_NM,
    ),
    (
        "Pa gamma",
        PA_GAMMA_REST_NM,
    ),
]:

    expected = (
        rest_nm
        * (
            1.0
            + REFERENCE_VELOCITY
            / C_KMS
        )
    )

    print()
    print(
        f"{name}:"
    )

    print(
        f"  Rest wavelength: "
        f"{rest_nm:.6f} nm"
    )

    print(
        f"  Expected observed wavelength "
        f"at {REFERENCE_VELOCITY:.2f} km/s: "
        f"{expected:.6f} nm"
    )


# ============================================================
# Spatial arrays
# ============================================================

nwave, ny, nx = cube.shape

beta_velocity = np.full(
    (ny, nx),
    np.nan,
)

beta_velocity_error = np.full(
    (ny, nx),
    np.nan,
)

beta_snr = np.full(
    (ny, nx),
    np.nan,
)

gamma_velocity = np.full(
    (ny, nx),
    np.nan,
)

gamma_velocity_error = np.full(
    (ny, nx),
    np.nan,
)

gamma_snr = np.full(
    (ny, nx),
    np.nan,
)


# ============================================================
# Fit spatial pixels
# ============================================================

print()
print("=" * 70)
print("FITTING SPATIAL PIXELS")
print("=" * 70)

attempted = 0
successful_both = 0

for y in range(ny):

    if y % 10 == 0:

        print(
            f"Processing row "
            f"{y + 1}/{ny}..."
        )

    for x in range(nx):

        attempted += 1

        flux = cube[:, y, x]
        err = error_cube[:, y, x]

        # ----------------------------------------------------
        # Mask DQ-flagged values if available.
        # ----------------------------------------------------

        if dq_cube is not None:

            good_dq = (
                dq_cube[:, y, x] == 0
            )

            flux_for_fit = flux.copy()
            err_for_fit = err.copy()

            flux_for_fit[
                ~good_dq
            ] = np.nan

            err_for_fit[
                ~good_dq
            ] = np.nan

        else:

            flux_for_fit = flux
            err_for_fit = err

        beta = fit_line(
            wavelength_nm,
            flux_for_fit,
            err_for_fit,
            PA_BETA_REST_NM,
            PA_BETA_WINDOW_NM,
        )

        gamma = fit_line(
            wavelength_nm,
            flux_for_fit,
            err_for_fit,
            PA_GAMMA_REST_NM,
            PA_GAMMA_WINDOW_NM,
        )

        if (
            beta is None
            or gamma is None
        ):

            continue

        beta_velocity[y, x] = (
            beta["velocity_kms"]
        )

        beta_velocity_error[y, x] = (
            beta["velocity_error_kms"]
        )

        beta_snr[y, x] = (
            beta["snr"]
        )

        gamma_velocity[y, x] = (
            gamma["velocity_kms"]
        )

        gamma_velocity_error[y, x] = (
            gamma["velocity_error_kms"]
        )

        gamma_snr[y, x] = (
            gamma["snr"]
        )

        successful_both += 1


# ============================================================
# High-S/N selection
# ============================================================

valid = (
    np.isfinite(beta_velocity)
    & np.isfinite(gamma_velocity)
    & np.isfinite(beta_velocity_error)
    & np.isfinite(gamma_velocity_error)
    & np.isfinite(beta_snr)
    & np.isfinite(gamma_snr)

    & (beta_snr >= MIN_SNR)
    & (gamma_snr >= MIN_SNR)

    & (
        beta_velocity_error
        <= MAX_VELOCITY_ERROR
    )

    & (
        gamma_velocity_error
        <= MAX_VELOCITY_ERROR
    )

    & (
        beta_velocity >= 400
    )

    & (
        beta_velocity <= 700
    )

    & (
        gamma_velocity >= 400
    )

    & (
        gamma_velocity <= 700
    )
)


# ============================================================
# Extract paired measurements
# ============================================================

beta_v = beta_velocity[valid]
gamma_v = gamma_velocity[valid]

beta_err = beta_velocity_error[valid]
gamma_err = gamma_velocity_error[valid]

beta_sn = beta_snr[valid]
gamma_sn = gamma_snr[valid]


# ============================================================
# Statistics
# ============================================================

if len(beta_v) < 5:

    raise RuntimeError(
        "Too few high-quality paired pixels."
    )

difference = (
    beta_v
    - gamma_v
)

combined_error = np.sqrt(
    beta_err ** 2
    + gamma_err ** 2
)


# ============================================================
# Correlations
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


# ============================================================
# Agreement fractions
# ============================================================

within_1 = (
    np.abs(difference)
    <= combined_error
)

within_2 = (
    np.abs(difference)
    <= 2.0 * combined_error
)

within_3 = (
    np.abs(difference)
    <= 3.0 * combined_error
)


# ============================================================
# Print results
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


print()
print("=" * 70)
print("PROPAGATED-ERROR HIGH-S/N SELECTION")
print("=" * 70)

print(
    f"Minimum S/N: "
    f"{MIN_SNR:.1f}"
)

print(
    f"Maximum centroid velocity error: "
    f"{MAX_VELOCITY_ERROR:.1f} km/s"
)

print(
    f"High-quality paired pixels: "
    f"{len(beta_v)}"
)

print(
    f"Fraction retained: "
    f"{len(beta_v) / attempted:.4f}"
)


print()
print("=" * 70)
print("PA BETA")
print("=" * 70)

print(
    f"Median velocity: "
    f"{np.median(beta_v):.3f} km/s"
)

print(
    f"Mean velocity: "
    f"{np.mean(beta_v):.3f} km/s"
)

print(
    f"Standard deviation: "
    f"{np.std(beta_v):.3f} km/s"
)

print(
    f"Median velocity uncertainty: "
    f"{np.median(beta_err):.3f} km/s"
)

print(
    f"Median S/N: "
    f"{np.median(beta_sn):.2f}"
)


print()
print("=" * 70)
print("PA GAMMA")
print("=" * 70)

print(
    f"Median velocity: "
    f"{np.median(gamma_v):.3f} km/s"
)

print(
    f"Mean velocity: "
    f"{np.mean(gamma_v):.3f} km/s"
)

print(
    f"Standard deviation: "
    f"{np.std(gamma_v):.3f} km/s"
)

print(
    f"Median velocity uncertainty: "
    f"{np.median(gamma_err):.3f} km/s"
)

print(
    f"Median S/N: "
    f"{np.median(gamma_sn):.2f}"
)


print()
print("=" * 70)
print("PA BETA - PA GAMMA VELOCITY DIFFERENCE")
print("=" * 70)

print(
    f"Median: "
    f"{np.median(difference):.3f} km/s"
)

print(
    f"Mean: "
    f"{np.mean(difference):.3f} km/s"
)

print(
    f"Standard deviation: "
    f"{np.std(difference):.3f} km/s"
)

print(
    f"Median combined uncertainty: "
    f"{np.median(combined_error):.3f} km/s"
)


print()
print("=" * 70)
print("PROPAGATED-ERROR VELOCITY CORRELATION")
print("=" * 70)

print(
    f"Pearson r: "
    f"{pearson_r:.6f}"
)

print(
    f"Pearson p-value: "
    f"{pearson_p:.6e}"
)

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
    f"Theil-Sen slope: "
    f"{sen_slope:.6f}"
)

print(
    f"Theil-Sen intercept: "
    f"{sen_intercept:.3f} km/s"
)

print(
    f"Theil-Sen 95% interval: "
    f"{sen_low:.6f} - "
    f"{sen_high:.6f}"
)


print()
print("=" * 70)
print("AGREEMENT WITHIN PROPAGATED UNCERTAINTIES")
print("=" * 70)

print(
    f"Within 1 sigma: "
    f"{np.sum(within_1)} / "
    f"{len(difference)} "
    f"({100*np.mean(within_1):.1f}%)"
)

print(
    f"Within 2 sigma: "
    f"{np.sum(within_2)} / "
    f"{len(difference)} "
    f"({100*np.mean(within_2):.1f}%)"
)

print(
    f"Within 3 sigma: "
    f"{np.sum(within_3)} / "
    f"{len(difference)} "
    f"({100*np.mean(within_3):.1f}%)"
)


# ============================================================
# Build maps
# ============================================================

beta_map = np.where(
    valid,
    beta_velocity,
    np.nan,
)

gamma_map = np.where(
    valid,
    gamma_velocity,
    np.nan,
)

difference_map = np.where(
    valid,
    beta_velocity
    - gamma_velocity,
    np.nan,
)


# ============================================================
# Plot 1 — Pa beta velocity map
# ============================================================

plt.figure(
    figsize=(10, 7)
)

plt.imshow(
    beta_map,
    origin="lower",
    interpolation="nearest",
)

plt.colorbar(
    label="Pa beta velocity (km/s)"
)

plt.xlabel(
    "Spatial X pixel"
)

plt.ylabel(
    "Spatial Y pixel"
)

plt.title(
    "M51 Pa beta Velocity Map\n"
    "Propagated S3D uncertainties"
)

plt.tight_layout()

plt.savefig(
    "m51_pabeta_propagated_velocity_map.png",
    dpi=150,
)

plt.close()


# ============================================================
# Plot 2 — Pa gamma velocity map
# ============================================================

plt.figure(
    figsize=(10, 7)
)

plt.imshow(
    gamma_map,
    origin="lower",
    interpolation="nearest",
)

plt.colorbar(
    label="Pa gamma velocity (km/s)"
)

plt.xlabel(
    "Spatial X pixel"
)

plt.ylabel(
    "Spatial Y pixel"
)

plt.title(
    "M51 Pa gamma Velocity Map\n"
    "Propagated S3D uncertainties"
)

plt.tight_layout()

plt.savefig(
    "m51_pgamma_propagated_velocity_map.png",
    dpi=150,
)

plt.close()


# ============================================================
# Plot 3 — Difference map
# ============================================================

plt.figure(
    figsize=(10, 7)
)

plt.imshow(
    difference_map,
    origin="lower",
    interpolation="nearest",
)

plt.colorbar(
    label="Pa beta - Pa gamma (km/s)"
)

plt.xlabel(
    "Spatial X pixel"
)

plt.ylabel(
    "Spatial Y pixel"
)

plt.title(
    "M51 Pa beta - Pa gamma Velocity Difference"
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
    figsize=(9, 8)
)

plt.scatter(
    gamma_v,
    beta_v,
    s=18,
    alpha=0.55,
)

lims = [
    min(
        gamma_v.min(),
        beta_v.min(),
    ),
    max(
        gamma_v.max(),
        beta_v.max(),
    ),
]

plt.plot(
    lims,
    lims,
    linestyle="--",
    label="1:1",
)

x_line = np.linspace(
    lims[0],
    lims[1],
    200,
)

plt.plot(
    x_line,
    sen_slope * x_line
    + sen_intercept,
    label="Theil-Sen",
)

plt.xlabel(
    "Pa gamma velocity (km/s)"
)

plt.ylabel(
    "Pa beta velocity (km/s)"
)

plt.title(
    "M51 Pa beta vs Pa gamma\n"
    f"Pearson r = {pearson_r:.3f}, "
    f"Spearman rho = {spearman_rho:.3f}"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "m51_pabeta_pgamma_propagated_correlation.png",
    dpi=150,
)

plt.close()


# ============================================================
# Plot 5 — Velocity difference histogram
# ============================================================

plt.figure(
    figsize=(9, 6)
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
    np.median(difference),
    linestyle="-",
    label=(
        f"Median = "
        f"{np.median(difference):.2f} km/s"
    ),
)

plt.xlabel(
    "Pa beta - Pa gamma velocity (km/s)"
)

plt.ylabel(
    "Number of spatial pixels"
)

plt.title(
    "M51 Pa beta / Pa gamma Velocity Difference"
)

plt.legend()

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
This experiment uses the propagated uncertainty cube
provided with the JWST/NIRSpec S3D data rather than
estimating noise from the local spectrum.

Pa beta and Pa gamma are fitted independently in the
same spatial pixels with free Gaussian centroids.

The resulting velocities are compared directly.

The most important diagnostics are:

  * Pearson correlation
  * Spearman rank correlation
  * Theil-Sen robust slope
  * median velocity difference
  * propagated velocity uncertainties
  * fraction of pixels agreeing within 1, 2 and 3 sigma

A strong Pa beta / Pa gamma correlation together with
a small median velocity difference would provide strong
independent evidence that the 1284 nm feature participates
in the same hydrogen recombination kinematics as Pa gamma.

If the result differs substantially from the earlier
noise-estimated experiment, the propagated JWST uncertainties
will be the preferred result.

This remains an identification test rather than a formal
atomic-identification proof.
"""
)

print()
print("=" * 70)
print("OUTPUT FILES")
print("=" * 70)

print(
    "m51_pabeta_propagated_velocity_map.png"
)

print(
    "m51_pgamma_propagated_velocity_map.png"
)

print(
    "m51_pabeta_pgamma_velocity_difference.png"
)

print(
    "m51_pabeta_pgamma_propagated_correlation.png"
)

print(
    "m51_pabeta_pgamma_velocity_difference_histogram.png"
)

print()
print("=" * 70)
print("TEST COMPLETE")
print("=" * 70)
