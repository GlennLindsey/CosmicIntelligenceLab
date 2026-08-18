from pathlib import Path

import numpy as np
from astropy.io import fits

# ============================================================
# Configuration
# ============================================================

S3D_PATH = Path(
    "data/m51_jwst_level3/" "jw03435-o006_t010_nirspec_g140m-f100lp_s3d.fits"
)

RESOLUTION_PATH = Path("data/instrument/" "jwst_nirspec_g140m_disp.fits")

# Observed integrated-spectrum feature
OBSERVED_WAVELENGTH_NM = 1284.26130440

# Independently measured local [Fe II] velocity
REFERENCE_VELOCITY_KMS = 573.72

# Laboratory wavelengths
PA_BETA_REST_NM = 1281.80700000
CSII_AIR_NM = 1284.26406000
CSII_VACUUM_NM = 1284.61537587

# Local fitting window
WINDOW_NM = 6.0

# Quality requirements
MIN_SNR = 5.0
MAX_VELOCITY_ERROR_KMS = 30.0

C_KMS = 299792.458


# ============================================================
# Utility functions
# ============================================================


def velocity_to_wavelength(rest_nm, velocity_kms):
    """
    Convert rest wavelength to observed wavelength using the
    relativistic Doppler relation.
    """

    beta = velocity_kms / C_KMS

    return rest_nm * np.sqrt((1.0 + beta) / (1.0 - beta))


def wavelength_to_velocity(observed_nm, rest_nm):
    """
    Relativistic velocity corresponding to an observed
    wavelength and laboratory wavelength.
    """

    ratio = observed_nm / rest_nm

    beta = (ratio**2 - 1.0) / (ratio**2 + 1.0)

    return beta * C_KMS


def gaussian(x, amplitude, center, sigma):
    return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def weighted_linear_amplitude(
    x,
    y,
    uncertainty,
    center,
    sigma,
):
    """
    Fit amplitude for a fixed-center Gaussian plus a constant
    local continuum.

    Model:

        y = continuum + amplitude * Gaussian

    Returns:

        amplitude
        amplitude_error
        continuum
        continuum_error
        chi2
    """

    g = np.exp(-0.5 * ((x - center) / sigma) ** 2)

    valid = (
        np.isfinite(x) & np.isfinite(y) & np.isfinite(uncertainty) & (uncertainty > 0)
    )

    x = x[valid]
    y = y[valid]
    uncertainty = uncertainty[valid]
    g = g[valid]

    if len(y) < 4:
        return None

    design = np.column_stack(
        [
            np.ones(len(y)),
            g,
        ]
    )

    weights = 1.0 / uncertainty**2

    normal = design.T @ (weights[:, None] * design)

    rhs = design.T @ (weights * y)

    try:
        covariance = np.linalg.inv(normal)
    except np.linalg.LinAlgError:
        return None

    parameters = covariance @ rhs

    continuum = parameters[0]
    amplitude = parameters[1]

    continuum_error = np.sqrt(covariance[0, 0])

    amplitude_error = np.sqrt(covariance[1, 1])

    model = continuum + amplitude * g

    residual = y - model

    chi2 = np.sum((residual / uncertainty) ** 2)

    return {
        "continuum": continuum,
        "continuum_error": continuum_error,
        "amplitude": amplitude,
        "amplitude_error": amplitude_error,
        "chi2": chi2,
        "n_points": len(y),
    }


# ============================================================
# Load S3D cube and propagated errors
# ============================================================

print("=" * 70)
print("M51 FINAL PROPAGATED-ERROR Pa BETA vs Cs II COMPARISON")
print("=" * 70)

print()
print("S3D:")
print(S3D_PATH)

print()
print("=" * 70)
print("LOADING S3D SCIENCE + PROPAGATED ERROR CUBE")
print("=" * 70)

# memmap=False is important because the FITS file contains
# BSCALE/BZERO information.

with fits.open(
    S3D_PATH,
    memmap=False,
) as hdul:

    print()
    print("HDU structure:")

    for index, hdu in enumerate(hdul):
        shape = None if hdu.data is None else hdu.data.shape

        print(f"  HDU {index}: " f"{hdu.name:12s} " f"{shape}")

    sci = np.asarray(
        hdul[1].data,
        dtype=float,
    )

    err = np.asarray(
        hdul[2].data,
        dtype=float,
    )

    header = hdul[1].header


print()
print("Science cube shape:")
print(sci.shape)

print()
print("Error cube shape:")
print(err.shape)


# ============================================================
# Wavelength construction
# ============================================================

print()
print("=" * 70)
print("CONSTRUCTING S3D SPECTRAL WAVELENGTH AXIS")
print("=" * 70)

# The Level-3 S3D SCI extension explicitly defines the
# spectral axis with CRPIX3, CRVAL3 and CDELT3.
#
# Verified from the actual FITS header:
#
#   CRPIX3 = 1.0
#   CRVAL3 = 0.9703180286160205 um
#   CDELT3 = 0.000636000011581927 um
#   CTYPE3 = WAVE
#   CUNIT3 = um

n_spectral = sci.shape[0]

crpix3 = header["CRPIX3"]
crval3 = header["CRVAL3"]
cdelt3 = header["CDELT3"]

spectral_axis = np.arange(
    n_spectral,
    dtype=float,
)

wavelength_um = crval3 + (spectral_axis + 1.0 - crpix3) * cdelt3

wavelength_nm = wavelength_um * 1000.0

sampling = np.median(np.diff(wavelength_nm))

print()
print(f"CRPIX3: {crpix3}")
print(f"CRVAL3: {crval3:.15f} um")
print(f"CDELT3: {cdelt3:.15f} um")

print()
print(
    f"Wavelength range: "
    f"{wavelength_nm.min():.3f} - "
    f"{wavelength_nm.max():.3f} nm"
)

print(f"Spectral sampling: " f"{sampling:.6f} nm")

if not (900.0 < wavelength_nm.min() < 1100.0):
    raise RuntimeError("Invalid S3D wavelength minimum.")

if not (1800.0 < wavelength_nm.max() < 2000.0):
    raise RuntimeError("Invalid S3D wavelength maximum.")

if not (0.5 < sampling < 0.8):
    raise RuntimeError("Invalid S3D spectral sampling.")

print()
print("Wavelength-axis validation: PASSED")

# ============================================================
# Instrument resolution
# ============================================================

print()
print("=" * 70)
print("NIRSPEC INSTRUMENT RESOLUTION")
print("=" * 70)

# The resolution reference file has wavelength and resolving-power
# columns. Inspect the available columns rather than assuming names.

with fits.open(
    RESOLUTION_PATH,
    memmap=False,
) as hdul:

    resolution_table = None

    for hdu in hdul:

        if hdu.data is None:
            continue

        if hasattr(hdu.data, "names"):
            names = [name.upper() for name in hdu.data.names]

            if "WAVELENGTH" in names and "R" in names:
                resolution_table = hdu.data
                break

            if "WAVELENGTH" in names and "RESOLVING_POWER" in names:
                resolution_table = hdu.data
                break

    if resolution_table is None:
        raise RuntimeError(
            "Could not identify wavelength/resolution "
            "columns in the NIRSpec resolution file."
        )

    names = [name.upper() for name in resolution_table.names]

    wave_column = resolution_table.names[names.index("WAVELENGTH")]

    if "R" in names:
        r_column = resolution_table.names[names.index("R")]
    else:
        r_column = resolution_table.names[names.index("RESOLVING_POWER")]

    resolution_wave = np.asarray(
        resolution_table[wave_column],
        dtype=float,
    )

    resolving_power = np.asarray(
        resolution_table[r_column],
        dtype=float,
    )

# Determine whether wavelength is in microns or nm.

if np.nanmedian(resolution_wave) < 10:

    resolution_wave_nm = resolution_wave * 1000.0

else:

    resolution_wave_nm = resolution_wave


R = np.interp(
    OBSERVED_WAVELENGTH_NM,
    resolution_wave_nm,
    resolving_power,
)

instrument_fwhm = OBSERVED_WAVELENGTH_NM / R

instrument_sigma = instrument_fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))

print(f"Resolving power: R = {R:.1f}")

print(f"Instrument FWHM: " f"{instrument_fwhm:.6f} nm")

print(f"Instrument sigma: " f"{instrument_sigma:.6f} nm")


# ============================================================
# Hypothesis centers
# ============================================================

pa_beta_center = velocity_to_wavelength(
    PA_BETA_REST_NM,
    REFERENCE_VELOCITY_KMS,
)

csii_center = velocity_to_wavelength(
    CSII_VACUUM_NM,
    REFERENCE_VELOCITY_KMS,
)

print()
print("=" * 70)
print("HYPOTHESIS CENTERS")
print("=" * 70)

print()
print(f"Pa beta at " f"+{REFERENCE_VELOCITY_KMS:.2f} km/s:")

print(f"  {pa_beta_center:.8f} nm")

print()
print(f"Cs II vacuum at " f"+{REFERENCE_VELOCITY_KMS:.2f} km/s:")

print(f"  {csii_center:.8f} nm")

print()
print(f"Observed feature:")

print(f"  {OBSERVED_WAVELENGTH_NM:.8f} nm")


# ============================================================
# Required velocities
# ============================================================

pa_beta_required_velocity = wavelength_to_velocity(
    OBSERVED_WAVELENGTH_NM,
    PA_BETA_REST_NM,
)

csii_required_velocity = wavelength_to_velocity(
    OBSERVED_WAVELENGTH_NM,
    CSII_VACUUM_NM,
)

print()
print("=" * 70)
print("VELOCITY REQUIRED TO MATCH OBSERVED FEATURE")
print("=" * 70)

print()
print(f"Pa beta:")

print(f"  {pa_beta_required_velocity:+.3f} km/s")

print()
print(f"Cs II:")

print(f"  {csii_required_velocity:+.3f} km/s")


# ============================================================
# Spectral window
# ============================================================

half_window = WINDOW_NM / 2.0

window_mask = (wavelength_nm >= OBSERVED_WAVELENGTH_NM - half_window) & (
    wavelength_nm <= OBSERVED_WAVELENGTH_NM + half_window
)

local_wavelength = wavelength_nm[window_mask]

local_cube = sci[window_mask, :, :]

local_error = err[window_mask, :, :]

print()
print("=" * 70)
print("LOCAL SPECTRAL WINDOW")
print("=" * 70)

print(f"Window: " f"{local_wavelength.min():.3f} - " f"{local_wavelength.max():.3f} nm")

print(f"Spectral planes: " f"{len(local_wavelength)}")


# ============================================================
# Spatial analysis
# ============================================================

ny = sci.shape[1]
nx = sci.shape[2]

pa_beta_results = []
csii_results = []


# Store useful diagnostics.

records = []


print()
print("=" * 70)
print("SPATIAL PROPAGATED-ERROR FITTING")
print("=" * 70)


for y in range(ny):

    if y % 10 == 0:

        print(f"Processing row " f"{y + 1}/{ny}...")

    for x in range(nx):

        flux = local_cube[:, y, x]
        uncertainty = local_error[:, y, x]

        valid = (
            np.isfinite(local_wavelength)
            & np.isfinite(flux)
            & np.isfinite(uncertainty)
            & (uncertainty > 0)
        )

        if np.sum(valid) < 6:
            continue

        xwave = local_wavelength[valid]
        yflux = flux[valid]
        yerr = uncertainty[valid]

        # ----------------------------------------------------
        # Continuum + Pa beta
        # ----------------------------------------------------

        pa_beta_fit = weighted_linear_amplitude(
            xwave,
            yflux,
            yerr,
            pa_beta_center,
            instrument_sigma,
        )

        # ----------------------------------------------------
        # Continuum + Cs II
        # ----------------------------------------------------

        csii_fit = weighted_linear_amplitude(
            xwave,
            yflux,
            yerr,
            csii_center,
            instrument_sigma,
        )

        if pa_beta_fit is None or csii_fit is None:
            continue

        # ----------------------------------------------------
        # Quality estimate
        # ----------------------------------------------------

        pa_snr = pa_beta_fit["amplitude"] / pa_beta_fit["amplitude_error"]

        cs_snr = csii_fit["amplitude"] / csii_fit["amplitude_error"]

        # ----------------------------------------------------
        # Model comparison
        # ----------------------------------------------------

        delta_chi2 = csii_fit["chi2"] - pa_beta_fit["chi2"]

        # ----------------------------------------------------
        # Require positive Pa beta emission
        # ----------------------------------------------------

        if pa_snr < MIN_SNR:
            continue

        # ----------------------------------------------------
        # Record
        # ----------------------------------------------------

        record = {
            "x": x,
            "y": y,
            "pa_amplitude": pa_beta_fit["amplitude"],
            "pa_amplitude_error": pa_beta_fit["amplitude_error"],
            "pa_snr": pa_snr,
            "cs_amplitude": csii_fit["amplitude"],
            "cs_amplitude_error": csii_fit["amplitude_error"],
            "cs_snr": cs_snr,
            "pa_chi2": pa_beta_fit["chi2"],
            "cs_chi2": csii_fit["chi2"],
            "delta_chi2": delta_chi2,
            "n_points": pa_beta_fit["n_points"],
        }

        records.append(record)


# ============================================================
# Convert results
# ============================================================

print()
print("=" * 70)
print("PROPAGATED-ERROR MODEL COMPARISON")
print("=" * 70)

print()
print(f"Accepted high-S/N spatial pixels: " f"{len(records)}")

if not records:

    raise RuntimeError(
        "No spatial pixels passed the propagated-error " "quality selection."
    )


delta_values = np.array([record["delta_chi2"] for record in records])

pa_chi_values = np.array([record["pa_chi2"] for record in records])

cs_chi_values = np.array([record["cs_chi2"] for record in records])

pa_snr_values = np.array([record["pa_snr"] for record in records])

cs_snr_values = np.array([record["cs_snr"] for record in records])


# ============================================================
# Summary statistics
# ============================================================

pa_preferred = delta_values > 0

cs_preferred = delta_values < 0

ties = delta_values == 0

print()
print(f"Median Pa beta chi²: " f"{np.median(pa_chi_values):.3f}")

print(f"Median Cs II chi²:   " f"{np.median(cs_chi_values):.3f}")

print(f"Median Δχ² " f"(Cs II - Pa beta): " f"{np.median(delta_values):.3f}")

print(f"Mean Δχ²: " f"{np.mean(delta_values):.3f}")

print(f"Minimum Δχ²: " f"{np.min(delta_values):.3f}")

print(f"Maximum Δχ²: " f"{np.max(delta_values):.3f}")

print()
print(f"Pixels favoring Pa beta: " f"{np.sum(pa_preferred)}")

print(f"Pixels favoring Cs II: " f"{np.sum(cs_preferred)}")

print(f"Ties: " f"{np.sum(ties)}")

print()
print(f"Median Pa beta S/N: " f"{np.median(pa_snr_values):.2f}")

print(f"Median Cs II amplitude S/N: " f"{np.median(cs_snr_values):.2f}")


# ============================================================
# Integrated propagated-error comparison
# ============================================================

print()
print("=" * 70)
print("INTEGRATED SPECTRAL COMPARISON")
print("=" * 70)

# Sum the spatial spectra over the accepted pixels.

accepted_coordinates = [(record["y"], record["x"]) for record in records]

integrated_flux = np.zeros(len(local_wavelength))

integrated_variance = np.zeros(len(local_wavelength))

for y, x in accepted_coordinates:

    flux = local_cube[:, y, x]
    uncertainty = local_error[:, y, x]

    valid = np.isfinite(flux) & np.isfinite(uncertainty) & (uncertainty > 0)

    integrated_flux[valid] += flux[valid]

    integrated_variance[valid] += uncertainty[valid] ** 2

integrated_error = np.sqrt(integrated_variance)

integrated_valid = (
    np.isfinite(integrated_flux)
    & np.isfinite(integrated_error)
    & (integrated_error > 0)
)

integrated_fit_pa = weighted_linear_amplitude(
    local_wavelength[integrated_valid],
    integrated_flux[integrated_valid],
    integrated_error[integrated_valid],
    pa_beta_center,
    instrument_sigma,
)

integrated_fit_cs = weighted_linear_amplitude(
    local_wavelength[integrated_valid],
    integrated_flux[integrated_valid],
    integrated_error[integrated_valid],
    csii_center,
    instrument_sigma,
)

if integrated_fit_pa is not None and integrated_fit_cs is not None:

    integrated_delta = integrated_fit_cs["chi2"] - integrated_fit_pa["chi2"]

    print()
    print("Pa beta:")
    print(f"  Amplitude: " f"{integrated_fit_pa['amplitude']:.8g}")
    print(f"  Amplitude error: " f"{integrated_fit_pa['amplitude_error']:.8g}")
    print(
        f"  S/N: "
        f"{integrated_fit_pa['amplitude'] / integrated_fit_pa['amplitude_error']:.2f}"
    )
    print(f"  Chi squared: " f"{integrated_fit_pa['chi2']:.3f}")

    print()
    print("Cs II:")
    print(f"  Amplitude: " f"{integrated_fit_cs['amplitude']:.8g}")
    print(f"  Amplitude error: " f"{integrated_fit_cs['amplitude_error']:.8g}")
    print(
        f"  S/N: "
        f"{integrated_fit_cs['amplitude'] / integrated_fit_cs['amplitude_error']:.2f}"
    )
    print(f"  Chi squared: " f"{integrated_fit_cs['chi2']:.3f}")

    print()
    print(f"Δχ² (Cs II - Pa beta): " f"{integrated_delta:.3f}")


# ============================================================
# Final interpretation
# ============================================================

print()
print("=" * 70)
print("FINAL INTERPRETATION")
print("=" * 70)

print()
print(
    "This is the final propagated-error comparison "
    "of the Pa beta and Cs II hypotheses."
)

print()
print("Both hypotheses use:")

print("  - the same local spectral window;")

print("  - the same local continuum model;")

print("  - the actual NIRSpec instrumental resolution;")

print("  - the independently measured " "+573.72 km/s M51 velocity;")

print("  - the propagated JWST S3D uncertainty cube;")

print("  - identical statistical treatment.")

print()

if np.median(delta_values) > 0:

    print(
        "RESULT: Pa beta is preferred over Cs II "
        "in the accepted high-S/N spatial pixels."
    )

else:

    print(
        "RESULT: The median spatial comparison " "does not prefer Pa beta over Cs II."
    )

print()

print(
    "The strongest evidence should be assessed from "
    "the combination of the integrated comparison, "
    "the spatial Δχ² distribution, and the independent "
    "velocity measurements."
)

print()
print(
    "The very large absolute chi-square values seen "
    "in earlier simple Gaussian fits should not be "
    "interpreted as a complete goodness-of-fit test. "
    "The primary purpose here is comparative model "
    "discrimination under identical assumptions."
)

print()
print("=" * 70)
print("FINAL PROPAGATED-ERROR TEST COMPLETE")
print("=" * 70)
