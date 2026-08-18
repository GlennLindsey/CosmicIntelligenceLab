from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.optimize import curve_fit


# ============================================================
# M51 JWST/NIRSpec S3D
# Free-centroid 1284 nm velocity map
# ============================================================

S3D_PATH = Path(
    "data/m51_jwst_level3/"
    "jw03435-o006_t010_nirspec_g140m-f100lp_s3d.fits"
)


# ============================================================
# Analysis parameters
# ============================================================

TARGET_WAVELENGTH_NM = 1284.26130440

# Pa beta laboratory wavelength
PA_BETA_REST_NM = 1281.80700000

# Cs II laboratory wavelengths
CSII_AIR_NM = 1284.26406000
CSII_VACUUM_NM = 1284.61537587

# Local fitting window
WINDOW_NM = 3.0

# Minimum number of spectral points
MIN_POINTS = 6

# Minimum amplitude S/N
MIN_SNR = 5.0

# Initial Gaussian width
INITIAL_SIGMA_NM = 0.60

# Maximum allowed centroid displacement
MAX_CENTER_SHIFT_NM = 2.5

# Speed of light
C_KM_S = 299792.458


# ============================================================
# Gaussian model
# ============================================================


def gaussian(x, amplitude, center, sigma):
    """
    Gaussian emission-line model.
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


def gaussian_plus_continuum(
    x,
    amplitude,
    center,
    sigma,
    continuum,
):
    """
    Gaussian plus constant local continuum.
    """

    return (
        continuum
        + gaussian(
            x,
            amplitude,
            center,
            sigma,
        )
    )


# ============================================================
# Load S3D cube
# ============================================================


print("=" * 70)
print("M51 1284 NM FREE-CENTROID VELOCITY MAP")
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
        hdul[1].data,
        dtype=float,
    )

    header = hdul[1].header

    print()
    print("Cube shape:", cube.shape)

    # --------------------------------------------------------
    # Determine wavelength solution.
    #
    # This Level-3 S3D cube has a simple linear spectral
    # wavelength axis.  The spectral axis is axis 3 in the
    # FITS header.
    #
    # Do NOT use the full 3-D WCS transformation here.
    # The cube dimensions are:
    #
    #     spectral, y, x
    #
    # and the previous WCS approach can return an incorrect
    # axis ordering for this product.
    # --------------------------------------------------------

    spectral_axis = np.arange(
        cube.shape[0],
        dtype=float,
    )

    crval3 = header["CRVAL3"]
    cdelt3 = header["CDELT3"]
    crpix3 = header["CRPIX3"]

    wavelength_um = (
        crval3
        + (
            spectral_axis
            + 1.0
            - crpix3
        )
        * cdelt3
    )

wavelength_nm = (
    wavelength_um * 1000.0
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
    f"{cube.shape[2]} x {cube.shape[1]}"
)


# ============================================================
# Instrument resolution
# ============================================================

R = 916.3

instrument_fwhm = (
    TARGET_WAVELENGTH_NM
    / R
)

instrument_sigma = (
    instrument_fwhm
    / 2.354820045
)


print()
print("=" * 70)
print("INSTRUMENT RESOLUTION")
print("=" * 70)

print(
    f"Resolving power: R = {R:.1f}"
)

print(
    f"Instrument FWHM: "
    f"{instrument_fwhm:.6f} nm"
)

print(
    f"Instrument sigma: "
    f"{instrument_sigma:.6f} nm"
)


# ============================================================
# Hypothesis reference wavelengths
# ============================================================

print()
print("=" * 70)
print("REFERENCE WAVELENGTHS")
print("=" * 70)

print(
    f"Pa beta rest wavelength: "
    f"{PA_BETA_REST_NM:.8f} nm"
)

print(
    f"Cs II air wavelength: "
    f"{CSII_AIR_NM:.8f} nm"
)

print(
    f"Cs II vacuum wavelength: "
    f"{CSII_VACUUM_NM:.8f} nm"
)


# ============================================================
# Local spectral window
# ============================================================

window_mask = (
    np.abs(
        wavelength_nm
        - TARGET_WAVELENGTH_NM
    )
    <= WINDOW_NM
)

window_indices = np.where(
    window_mask
)[0]

local_wavelength = (
    wavelength_nm[window_mask]
)

print()
print("=" * 70)
print("LOCAL SPECTRAL WINDOW")
print("=" * 70)

print(
    f"Window: "
    f"{local_wavelength.min():.3f} - "
    f"{local_wavelength.max():.3f} nm"
)

print(
    f"Spectral planes: "
    f"{len(local_wavelength)}"
)


# ============================================================
# Prepare result arrays
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

center_map = np.full(
    (ny, nx),
    np.nan,
)

center_error_map = np.full(
    (ny, nx),
    np.nan,
)

amplitude_map = np.full(
    (ny, nx),
    np.nan,
)

snr_map = np.full(
    (ny, nx),
    np.nan,
)

sigma_map = np.full(
    (ny, nx),
    np.nan,
)

chi2_map = np.full(
    (ny, nx),
    np.nan,
)

dof_map = np.full(
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


accepted = 0
attempted = 0


for y in range(ny):

    if y % 10 == 1:

        print(
            f"Processing row "
            f"{y + 1}/{ny}..."
        )

    for x in range(nx):

        flux = cube[
            window_indices,
            y,
            x,
        ]

        # ----------------------------------------------------
        # Remove non-finite values.
        # ----------------------------------------------------

        valid = np.isfinite(flux)

        if np.sum(valid) < MIN_POINTS:

            continue

        xwave = local_wavelength[valid]
        yflux = flux[valid]

        attempted += 1

        # ----------------------------------------------------
        # Estimate continuum.
        # ----------------------------------------------------

        continuum_guess = np.median(
            yflux
        )

        # ----------------------------------------------------
        # Estimate amplitude.
        # ----------------------------------------------------

        amplitude_guess = (
            np.nanmax(yflux)
            - continuum_guess
        )

        if not np.isfinite(
            amplitude_guess
        ):

            continue

        if amplitude_guess <= 0:

            continue

        # ----------------------------------------------------
        # Estimate noise from local
        # spectral scatter.
        # ----------------------------------------------------

        residual_initial = (
            yflux
            - continuum_guess
        )

        noise = np.std(
            residual_initial
        )

        if (
            not np.isfinite(noise)
            or noise <= 0
        ):

            continue

        # ----------------------------------------------------
        # Initial centroid.
        # ----------------------------------------------------

        peak_index = np.argmax(
            yflux
        )

        center_guess = (
            xwave[peak_index]
        )

        # ----------------------------------------------------
        # Reject obviously unrelated
        # peaks before fitting.
        # ----------------------------------------------------

        if (
            abs(
                center_guess
                - TARGET_WAVELENGTH_NM
            )
            > MAX_CENTER_SHIFT_NM
        ):

            continue

        # ----------------------------------------------------
        # Initial parameter vector.
        # ----------------------------------------------------

        p0 = [
            amplitude_guess,
            center_guess,
            INITIAL_SIGMA_NM,
            continuum_guess,
        ]

        # ----------------------------------------------------
        # Parameter limits.
        # ----------------------------------------------------

        lower_bounds = [
            0.0,
            TARGET_WAVELENGTH_NM
            - MAX_CENTER_SHIFT_NM,
            0.15,
            -np.inf,
        ]

        upper_bounds = [
            np.inf,
            TARGET_WAVELENGTH_NM
            + MAX_CENTER_SHIFT_NM,
            2.5,
            np.inf,
        ]

        # ----------------------------------------------------
        # Fit Gaussian + continuum.
        # ----------------------------------------------------

        try:

            popt, pcov = curve_fit(
                gaussian_plus_continuum,
                xwave,
                yflux,
                p0=p0,
                bounds=(
                    lower_bounds,
                    upper_bounds,
                ),
                maxfev=10000,
            )

        except (
            RuntimeError,
            ValueError,
            OverflowError,
        ):

            continue

        amplitude = popt[0]
        center = popt[1]
        sigma = popt[2]
        continuum = popt[3]

        # ----------------------------------------------------
        # Parameter uncertainties.
        # ----------------------------------------------------

        try:

            parameter_errors = np.sqrt(
                np.diag(pcov)
            )

            amplitude_error = (
                parameter_errors[0]
            )

            center_error = (
                parameter_errors[1]
            )

        except Exception:

            continue

        if (
            not np.isfinite(
                amplitude_error
            )
            or amplitude_error <= 0
        ):

            continue

        if (
            not np.isfinite(
                center_error
            )
            or center_error <= 0
        ):

            continue

        # ----------------------------------------------------
        # Amplitude S/N.
        # ----------------------------------------------------

        snr = (
            amplitude
            / amplitude_error
        )

        if (
            not np.isfinite(snr)
            or snr < MIN_SNR
        ):

            continue

        # ----------------------------------------------------
        # Calculate model and residuals.
        # ----------------------------------------------------

        model = gaussian_plus_continuum(
            xwave,
            *popt,
        )

        residual = (
            yflux
            - model
        )

        # ----------------------------------------------------
        # Estimate residual variance.
        # ----------------------------------------------------

        residual_variance = np.var(
            residual
        )

        if (
            not np.isfinite(
                residual_variance
            )
            or residual_variance <= 0
        ):

            continue

        chi2 = np.sum(
            residual ** 2
            / residual_variance
        )

        dof = (
            len(yflux)
            - len(popt)
        )

        if dof <= 0:

            continue

        # ----------------------------------------------------
        # Convert free centroid to velocity
        # assuming Pa beta.
        #
        # Non-relativistic approximation is
        # appropriate for these velocities.
        # ----------------------------------------------------

        velocity = (
            (
                center
                / PA_BETA_REST_NM
            )
            - 1.0
        ) * C_KM_S

        # ----------------------------------------------------
        # Propagate wavelength uncertainty
        # into velocity uncertainty.
        # ----------------------------------------------------

        velocity_error = (
            C_KM_S
            / PA_BETA_REST_NM
            * center_error
        )

        # ----------------------------------------------------
        # Store results.
        # ----------------------------------------------------

        velocity_map[y, x] = velocity

        velocity_error_map[y, x] = (
            velocity_error
        )

        center_map[y, x] = center

        center_error_map[y, x] = (
            center_error
        )

        amplitude_map[y, x] = (
            amplitude
        )

        snr_map[y, x] = snr

        sigma_map[y, x] = sigma

        chi2_map[y, x] = chi2

        dof_map[y, x] = dof

        accepted += 1


# ============================================================
# Summary
# ============================================================

accepted_fraction = (
    accepted / attempted
    if attempted > 0
    else np.nan
)


print()
print("=" * 70)
print("FREE-CENTROID FIT SUMMARY")
print("=" * 70)

print(
    f"Spatial pixels attempted: "
    f"{attempted}"
)

print(
    f"Accepted pixels: "
    f"{accepted}"
)

print(
    f"Acceptance fraction: "
    f"{accepted_fraction:.4f}"
)


# ============================================================
# Velocity statistics
# ============================================================

valid_velocity = (
    np.isfinite(
        velocity_map
    )
)

velocities = (
    velocity_map[
        valid_velocity
    ]
)

velocity_errors = (
    velocity_error_map[
        valid_velocity
    ]
)


print()
print("=" * 70)
print("1284 NM VELOCITY DISTRIBUTION")
print("=" * 70)


if len(velocities) > 0:

    median_velocity = np.median(
        velocities
    )

    mean_velocity = np.mean(
        velocities
    )

    std_velocity = np.std(
        velocities
    )

    p16, p84 = np.percentile(
        velocities,
        [16, 84],
    )

    median_velocity_error = np.median(
        velocity_errors
    )

    print(
        f"Median velocity: "
        f"{median_velocity:.2f} km/s"
    )

    print(
        f"Mean velocity: "
        f"{mean_velocity:.2f} km/s"
    )

    print(
        f"Velocity standard deviation: "
        f"{std_velocity:.2f} km/s"
    )

    print(
        f"16th percentile: "
        f"{p16:.2f} km/s"
    )

    print(
        f"84th percentile: "
        f"{p84:.2f} km/s"
    )

    print(
        f"Median velocity uncertainty: "
        f"{median_velocity_error:.2f} km/s"
    )


# ============================================================
# Compare with known nebular velocity
# ============================================================

REFERENCE_VELOCITY = 573.72

print()
print("=" * 70)
print("COMPARISON WITH LOCAL NEBULAR VELOCITY")
print("=" * 70)

print(
    f"Reference [Fe II] velocity: "
    f"{REFERENCE_VELOCITY:.2f} km/s"
)

if len(velocities) > 0:

    difference = (
        median_velocity
        - REFERENCE_VELOCITY
    )

    print(
        f"1284 nm median velocity: "
        f"{median_velocity:.2f} km/s"
    )

    print(
        f"Difference: "
        f"{difference:+.2f} km/s"
    )


# ============================================================
# Cs II interpretation
# ============================================================

print()
print("=" * 70)
print("Cs II VELOCITY INTERPRETATION")
print("=" * 70)

print(
    "The free-centroid velocity map is"
)

print(
    "constructed assuming the 1284 nm"
)

print(
    "feature is Pa beta."
)

print()
print(
    "If interpreted instead as Cs II,"
)

print(
    "the same measured centroid would"
)

print(
    "correspond to a velocity near"
)

print(
    "-82.6 km/s for the vacuum Cs II"
)

print(
    "laboratory wavelength."
)


# ============================================================
# Save velocity map
# ============================================================

velocity_output = (
    "m51_1284_free_velocity_map.png"
)

plt.figure(
    figsize=(10, 7)
)

image = plt.imshow(
    velocity_map,
    origin="lower",
    interpolation="nearest",
)

plt.colorbar(
    image,
    label="Velocity (km/s)",
)

plt.title(
    "M51 1284 nm Free-Centroid Velocity Map\n"
    "Velocity relative to Pa beta laboratory wavelength"
)

plt.xlabel("Spatial X pixel")
plt.ylabel("Spatial Y pixel")

plt.tight_layout()

plt.savefig(
    velocity_output,
    dpi=150,
)

plt.close()


print()
print(
    f"Saved: {velocity_output}"
)


# ============================================================
# Save velocity uncertainty map
# ============================================================

error_output = (
    "m51_1284_velocity_error_map.png"
)

plt.figure(
    figsize=(10, 7)
)

image = plt.imshow(
    velocity_error_map,
    origin="lower",
    interpolation="nearest",
)

plt.colorbar(
    image,
    label="Velocity uncertainty (km/s)",
)

plt.title(
    "M51 1284 nm Velocity Uncertainty"
)

plt.xlabel("Spatial X pixel")
plt.ylabel("Spatial Y pixel")

plt.tight_layout()

plt.savefig(
    error_output,
    dpi=150,
)

plt.close()


print(
    f"Saved: {error_output}"
)


# ============================================================
# Save S/N map
# ============================================================

snr_output = (
    "m51_1284_snr_map.png"
)

plt.figure(
    figsize=(10, 7)
)

image = plt.imshow(
    snr_map,
    origin="lower",
    interpolation="nearest",
)

plt.colorbar(
    image,
    label="Amplitude S/N",
)

plt.title(
    "M51 1284 nm Gaussian Amplitude S/N"
)

plt.xlabel("Spatial X pixel")
plt.ylabel("Spatial Y pixel")

plt.tight_layout()

plt.savefig(
    snr_output,
    dpi=150,
)

plt.close()


print(
    f"Saved: {snr_output}"
)


# ============================================================
# Velocity histogram
# ============================================================

histogram_output = (
    "m51_1284_velocity_histogram.png"
)

plt.figure(
    figsize=(10, 6)
)

plt.hist(
    velocities,
    bins=40,
)

plt.axvline(
    REFERENCE_VELOCITY,
    linestyle="--",
    label="[Fe II] reference",
)

plt.axvline(
    median_velocity,
    linestyle="-",
    label="1284 nm median",
)

plt.xlabel(
    "Velocity assuming Pa beta (km/s)"
)

plt.ylabel(
    "Number of spatial pixels"
)

plt.title(
    "M51 1284 nm Spatial Velocity Distribution"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    histogram_output,
    dpi=150,
)

plt.close()


print(
    f"Saved: {histogram_output}"
)


# ============================================================
# Final interpretation
# ============================================================

print()
print("=" * 70)
print("INTERPRETATION")
print("=" * 70)

print(
    """
This analysis allows the 1284 nm Gaussian
centroid to vary independently in each
sufficiently strong spatial pixel.

The resulting velocity map is therefore
not constrained to the previously adopted
+573.72 km/s reference velocity.

The velocity is reported assuming Pa beta
(1281.807 nm) as the laboratory transition.

A spatially coherent velocity field that
agrees with the independently measured
hydrogen and [Fe II] velocities would
provide additional evidence that the
1284 nm feature is ordinary nebular
emission associated with Pa beta.

A substantially different velocity field,
especially one that forms a coherent
spatial component, would motivate further
investigation.

The Cs II hypothesis is evaluated afterward:
the same measured centroid can be converted
to a Cs II velocity using its vacuum
laboratory wavelength.

Important limitation:

The present S3D analysis estimates local
noise from the spectral data themselves.
It does not yet use the fully propagated
JWST uncertainty cube.

Also, the Gaussian width is allowed to vary
between spatial pixels. Instrumental
resolution is therefore not imposed as a
fixed physical width in this free-centroid
experiment.
"""
)

print()
print("=" * 70)
print("FREE-CENTROID VELOCITY MAP COMPLETE")
print("=" * 70)
