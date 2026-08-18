from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

from tools.m51_spectral_analysis import (
    load_x1d_spectrum,
    prepare_spectrum,
)


# ============================================================
# M51 JWST/NIRSpec X1D spectrum
# ============================================================

X1D_PATH = Path(
    "data/m51_jwst_level3/"
    "jw03435-o006_t010_nirspec_g140m-f100lp_x1d.fits"
)


# ============================================================
# Physical constants
# ============================================================

C_KMS = 299792.458


# ============================================================
# Adopted M51 systemic velocity
#
# This is only a provisional reference.
# We will NOT force the spectrum to agree with this value.
# ============================================================

M51_SYSTEMIC_VELOCITY = 460.0


# ============================================================
# Known near-IR nebular lines
#
# Wavelengths are vacuum rest wavelengths in nm.
#
# These are intended as velocity tracers, not as a
# comprehensive line-identification catalogue.
# ============================================================

NEBULAR_LINES = [
    {
        "name": "He I",
        "transition": "1.0833 um",
        "wavelength_nm": 1083.331,
        "notes": "strong He I",
    },
    {
        "name": "Pa gamma",
        "transition": "Pa gamma",
        "wavelength_nm": 1093.81,
        "notes": "H I recombination",
    },
    {
        "name": "[Fe II]",
        "transition": "1.257 um",
        "wavelength_nm": 1256.68,
        "notes": "forbidden iron",
    },
    {
        "name": "Pa beta",
        "transition": "Pa beta",
        "wavelength_nm": 1281.807,
        "notes": "H I recombination",
    },
    {
        "name": "O I",
        "transition": "1.3168 um",
        "wavelength_nm": 1316.8,
        "notes": "neutral oxygen",
    },
    {
        "name": "Brackett 17",
        "transition": "Br 17-4",
        "wavelength_nm": 1544.3,
        "notes": "H I recombination",
    },
    {
        "name": "Brackett 16",
        "transition": "Br 16-4",
        "wavelength_nm": 1556.1,
        "notes": "H I recombination",
    },
    {
        "name": "Brackett 15",
        "transition": "Br 15-4",
        "wavelength_nm": 1570.5,
        "notes": "H I recombination",
    },
    {
        "name": "Brackett 14",
        "transition": "Br 14-4",
        "wavelength_nm": 1588.5,
        "notes": "H I recombination",
    },
    {
        "name": "Brackett 13",
        "transition": "Br 13-4",
        "wavelength_nm": 1611.4,
        "notes": "H I recombination",
    },
    {
        "name": "Brackett 12",
        "transition": "Br 12-4",
        "wavelength_nm": 1641.2,
        "notes": "blended with [Fe II] 1.644 um",
    },
    {
        "name": "[Fe II]",
        "transition": "1.644 um",
        "wavelength_nm": 1643.9,
        "notes": "blended with Brackett 12",
    },
    {
        "name": "Brackett 11",
        "transition": "Br 11-4",
        "wavelength_nm": 1681.1,
        "notes": "H I recombination",
    },
    {
        "name": "He I",
        "transition": "1.7007 um",
        "wavelength_nm": 1700.7,
        "notes": "helium",
    },
    {
        "name": "Brackett 10",
        "transition": "Br 10-4",
        "wavelength_nm": 1736.7,
        "notes": "H I recombination",
    },
]


# ============================================================
# Gaussian + linear continuum model
# ============================================================


def gaussian_with_continuum(
    x,
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
        + slope * (x - center)
        + amplitude
        * np.exp(
            -0.5
            * (
                (x - center)
                / sigma
            ) ** 2
        )
    )


# ============================================================
# Relativistic Doppler velocity
# ============================================================


def wavelength_to_velocity(
    observed_nm,
    rest_nm,
):
    """
    Calculate radial velocity from observed/rest wavelength.

    Uses the relativistic Doppler relation.
    """

    ratio = observed_nm / rest_nm

    beta = (
        ratio**2 - 1.0
    ) / (
        ratio**2 + 1.0
    )

    return beta * C_KMS


# ============================================================
# Relativistic velocity relative to M51
# ============================================================


def velocity_relative_to_m51(
    observed_velocity,
    systemic_velocity,
):
    """
    Convert observed heliocentric-style velocity into a
    velocity relative to the adopted M51 systemic velocity.

    Relativistic velocity subtraction is used.
    """

    beta_obs = observed_velocity / C_KMS
    beta_sys = systemic_velocity / C_KMS

    beta_rel = (
        beta_obs - beta_sys
    ) / (
        1.0
        - beta_obs * beta_sys
    )

    return beta_rel * C_KMS


# ============================================================
# Fit one spectral line
# ============================================================


def fit_line(
    wavelength_nm,
    flux,
    uncertainty,
    rest_wavelength_nm,
    window_nm=4.0,
):
    """
    Fit a Gaussian emission line.

    The search window is deliberately broad enough to contain
    the expected M51 systemic velocity shift.
    """

    mask = (
        np.abs(
            wavelength_nm
            - rest_wavelength_nm
        )
        <= window_nm
    )

    if np.sum(mask) < 7:
        return None

    x = wavelength_nm[mask]
    y = flux[mask]
    err = uncertainty[mask]

    # --------------------------------------------------------
    # Initial continuum estimate
    # --------------------------------------------------------

    continuum_guess = np.median(y)

    amplitude_guess = (
        np.max(y)
        - continuum_guess
    )

    center_guess = x[
        np.argmax(y)
    ]

    sigma_guess = 0.55

    slope_guess = 0.0

    p0 = [
        amplitude_guess,
        center_guess,
        sigma_guess,
        continuum_guess,
        slope_guess,
    ]

    # --------------------------------------------------------
    # Reasonable parameter bounds
    # --------------------------------------------------------

    lower_bounds = [
        0.0,
        rest_wavelength_nm - window_nm,
        0.05,
        -np.inf,
        -np.inf,
    ]

    upper_bounds = [
        np.inf,
        rest_wavelength_nm + window_nm,
        3.0,
        np.inf,
        np.inf,
    ]

    try:

        popt, pcov = curve_fit(
            gaussian_with_continuum,
            x,
            y,
            p0=p0,
            sigma=err,
            absolute_sigma=True,
            bounds=(
                lower_bounds,
                upper_bounds,
            ),
            maxfev=20000,
        )

    except (
        RuntimeError,
        ValueError,
    ):

        return None

    (
        amplitude,
        center,
        sigma,
        continuum,
        slope,
    ) = popt

    # --------------------------------------------------------
    # Parameter uncertainties
    # --------------------------------------------------------

    try:

        perr = np.sqrt(
            np.diag(pcov)
        )

    except (
        ValueError,
        FloatingPointError,
    ):

        perr = np.full(
            len(popt),
            np.nan,
        )

    center_error = perr[1]

    sigma_error = perr[2]

    # --------------------------------------------------------
    # Fit quality
    # --------------------------------------------------------

    model = gaussian_with_continuum(
        x,
        *popt,
    )

    residuals = (
        y - model
    )

    chi_squared = np.sum(
        (
            residuals / err
        ) ** 2
    )

    degrees_of_freedom = (
        len(x) - len(popt)
    )

    if degrees_of_freedom > 0:

        reduced_chi_squared = (
            chi_squared
            / degrees_of_freedom
        )

    else:

        reduced_chi_squared = np.nan

    # --------------------------------------------------------
    # Approximate amplitude S/N
    # --------------------------------------------------------

    if (
        np.isfinite(perr[0])
        and perr[0] > 0
    ):

        amplitude_snr = (
            amplitude / perr[0]
        )

    else:

        amplitude_snr = np.nan

    # --------------------------------------------------------
    # Velocity
    # --------------------------------------------------------

    velocity = wavelength_to_velocity(
        center,
        rest_wavelength_nm,
    )

    # Propagate center uncertainty into velocity.
    #
    # For these small velocities the approximation
    # dv = c * d(lambda) / lambda is sufficient.

    velocity_error = (
        C_KMS
        * center_error
        / rest_wavelength_nm
    )

    velocity_relative = (
        velocity_relative_to_m51(
            velocity,
            M51_SYSTEMIC_VELOCITY,
        )
    )

    return {
        "rest_nm": rest_wavelength_nm,
        "center_nm": center,
        "center_error_nm": center_error,
        "velocity_kms": velocity,
        "velocity_error_kms": velocity_error,
        "velocity_relative_to_m51_kms": (
            velocity_relative
        ),
        "amplitude": amplitude,
        "amplitude_error": perr[0],
        "amplitude_snr": amplitude_snr,
        "sigma_nm": sigma,
        "sigma_error_nm": sigma_error,
        "fwhm_nm": 2.354820045 * sigma,
        "chi_squared": chi_squared,
        "degrees_of_freedom": (
            degrees_of_freedom
        ),
        "reduced_chi_squared": (
            reduced_chi_squared
        ),
    }


# ============================================================
# Main analysis
# ============================================================


print("=" * 70)
print("M51 JWST/NIRSpec NEBULAR VELOCITY INVESTIGATION")
print("=" * 70)

print()
print("Spectrum:")
print(X1D_PATH)

print()
print(
    "Adopted M51 systemic velocity:"
    f" {M51_SYSTEMIC_VELOCITY:.1f} km/s"
)

print()
print(
    "Important:"
)
print(
    "This systemic velocity is a provisional reference."
)
print(
    "The purpose of this test is to measure the"
)
print(
    "velocity actually present in the spectrum."
)


# ============================================================
# Load spectrum
# ============================================================


print()
print("Loading spectrum...")

spectrum = load_x1d_spectrum(
    X1D_PATH
)

clean = prepare_spectrum(
    spectrum
)

wavelength_um = clean[
    "wavelength"
]

flux = clean[
    "flux"
]

uncertainty = clean[
    "uncertainty"
]

wavelength_nm = (
    wavelength_um * 1000.0
)


print(
    f"Valid points: "
    f"{clean['valid_points']}"
)

print(
    f"Wavelength range: "
    f"{wavelength_nm.min():.3f}"
    f" - "
    f"{wavelength_nm.max():.3f} nm"
)


# ============================================================
# Analyze lines
# ============================================================


print()
print("=" * 70)
print("NEBULAR LINE VELOCITY MEASUREMENTS")
print("=" * 70)

results = []


for line in NEBULAR_LINES:

    rest_nm = line[
        "wavelength_nm"
    ]

    # --------------------------------------------------------
    # Skip lines outside the spectrum
    # --------------------------------------------------------

    if (
        rest_nm
        < wavelength_nm.min()
        or rest_nm
        > wavelength_nm.max()
    ):

        continue

    result = fit_line(
        wavelength_nm,
        flux,
        uncertainty,
        rest_nm,
    )

    if result is None:

        print()
        print(
            f"{line['name']:15s}"
            f" {rest_nm:10.3f} nm"
            f"  FIT FAILED"
        )

        continue

    result[
        "name"
    ] = line["name"]

    result[
        "transition"
    ] = line["transition"]

    result[
        "notes"
    ] = line["notes"]

    results.append(
        result
    )

    print()
    print(
        f"{line['name']} "
        f"({line['transition']})"
    )

    print(
        f"  Rest wavelength: "
        f"{rest_nm:.5f} nm"
    )

    print(
        f"  Fitted wavelength: "
        f"{result['center_nm']:.5f}"
        f" +/- "
        f"{result['center_error_nm']:.5f} nm"
    )

    print(
        f"  Velocity: "
        f"{result['velocity_kms']:+.2f}"
        f" +/- "
        f"{result['velocity_error_kms']:.2f}"
        f" km/s"
    )

    print(
        f"  Relative to M51 systemic: "
        f"{result['velocity_relative_to_m51_kms']:+.2f}"
        f" km/s"
    )

    print(
        f"  Amplitude S/N: "
        f"{result['amplitude_snr']:.1f}"
    )

    print(
        f"  FWHM: "
        f"{result['fwhm_nm']:.4f} nm"
    )

    print(
        f"  Reduced chi²: "
        f"{result['reduced_chi_squared']:.2f}"
    )

    print(
        f"  Notes: "
        f"{line['notes']}"
    )


# ============================================================
# Summary table
# ============================================================


print()
print("=" * 70)
print("VELOCITY SUMMARY")
print("=" * 70)

print()

print(
    f"{'Line':15s}"
    f"{'Rest':>11s}"
    f"{'Observed':>13s}"
    f"{'v_obs':>12s}"
    f"{'v_M51':>12s}"
)

print("-" * 70)

for result in results:

    print(
        f"{result['name']:15s}"
        f"{result['rest_nm']:11.3f}"
        f"{result['center_nm']:13.3f}"
        f"{result['velocity_kms']:12.1f}"
        f"{result['velocity_relative_to_m51_kms']:12.1f}"
    )


# ============================================================
# Compare Pa beta with the Cs II candidate
# ============================================================


print()
print("=" * 70)
print("SPECIAL CHECK: Pa BETA vs Cs II")
print("=" * 70)

csii_observed_nm = 1284.26130440

csii_rest_nm = 1284.61537587

csii_velocity = wavelength_to_velocity(
    csii_observed_nm,
    csii_rest_nm,
)

csii_m51_velocity = (
    velocity_relative_to_m51(
        csii_velocity,
        M51_SYSTEMIC_VELOCITY,
    )
)

print()
print(
    f"Cs II observed feature: "
    f"{csii_observed_nm:.6f} nm"
)

print(
    f"Cs II vacuum rest wavelength: "
    f"{csii_rest_nm:.6f} nm"
)

print(
    f"Cs II inferred velocity: "
    f"{csii_velocity:+.2f} km/s"
)

print(
    f"Cs II velocity relative to M51: "
    f"{csii_m51_velocity:+.2f} km/s"
)


# Find Pa beta result.

pa_beta = None

for result in results:

    if result["name"] == "Pa beta":

        pa_beta = result

        break


if pa_beta is not None:

    print()
    print(
        f"Pa beta fitted wavelength: "
        f"{pa_beta['center_nm']:.6f} nm"
    )

    print(
        f"Pa beta velocity relative to M51: "
        f"{pa_beta['velocity_relative_to_m51_kms']:+.2f}"
        f" km/s"
    )

    difference = (
        csii_m51_velocity
        - pa_beta[
            "velocity_relative_to_m51_kms"
        ]
    )

    print()
    print(
        "Difference between Cs II and Pa beta"
        " inferred M51-frame velocities:"
    )

    print(
        f"{difference:+.2f} km/s"
    )


# ============================================================
# Interpretation
# ============================================================


print()
print("=" * 70)
print("INTERPRETATION")
print("=" * 70)

print()
print(
    "The important result is the velocity pattern across"
)
print(
    "multiple independently identified nebular lines."
)

print()
print(
    "If Pa beta and the other strong nebular lines give"
)
print(
    "similar M51-frame velocities, they provide a local"
)
print(
    "kinematic reference for the 1284 nm feature."
)

print()
print(
    "If the 1284 nm feature has a substantially different"
)
print(
    "velocity from the nebular lines, the Cs II hypothesis"
)
print(
    "would require a distinct high-velocity component."
)

print()
print(
    "NOTE: Brackett 12 and [Fe II] 1.644 um are close enough"
)
print(
    "to be blended at this spectral resolution and should not"
)
print(
    "be treated as independent velocity measurements without"
)
print(
    "a multi-component fit."
)

print()
print("=" * 70)
print("TEST COMPLETE")
print("=" * 70)
