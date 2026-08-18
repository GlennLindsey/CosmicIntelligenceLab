from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# M51 VELOCITY-GRID TEST
# ============================================================
#
# Purpose:
#
# Determine whether the observed 1284.2613 nm feature is
# kinematically compatible with:
#
#     1. Pa beta
#     2. Cs II 1284.26406 nm
#
# using the actual M51 nebular velocity range measured from
# independent emission lines.
#
# This is a wavelength/velocity consistency test.
#
# It deliberately does NOT perform a chi-square fit at every
# velocity. The previous fixed-center experiment showed that
# the formal flux uncertainties produce extremely large
# reduced chi-square values.
#
# ============================================================


# ============================================================
# Observed M51 feature
# ============================================================

OBSERVED_FEATURE_NM = 1284.26130440
OBSERVED_FEATURE_ERROR_NM = 0.00134611


# ============================================================
# Laboratory wavelengths
# ============================================================
#
# Pa beta:
#
# The wavelength used in the previous M51 analysis was
# 1281.807 nm.
#
# Cs II:
#
# NIST reports:
#
#     Air:    1284.264060 nm
#     Vacuum: 1284.61537587 nm
#
# JWST/NIRSpec wavelengths are treated here as vacuum
# wavelengths, so the vacuum Cs II wavelength is used.
#
# ============================================================

PA_BETA_REST_NM = 1281.80700000

CSII_AIR_NM = 1284.26406000
CSII_VACUUM_NM = 1284.61537587


# ============================================================
# Empirical M51 nebular velocity reference
# ============================================================
#
# Measured previously from strong nebular lines:
#
# He I 1.0833          +505.00 km/s
# Pa gamma             +583.14 km/s
# [Fe II] 1.257        +573.72 km/s
# Brackett 17          +569.37 km/s
# Brackett 15          +539.69 km/s
# Brackett 13          +478.40 km/s
# [Fe II] 1.644        +510.82 km/s
# Brackett 10          +461.75 km/s
#
# Median = +525.25 km/s
# MAD    = 45.49 km/s
#
# ============================================================

NEBULAR_VELOCITIES_KMS = np.array(
    [
        505.00,
        583.14,
        573.72,
        569.37,
        539.69,
        478.40,
        510.82,
        461.75,
    ]
)

M51_MEDIAN_VELOCITY_KMS = np.median(
    NEBULAR_VELOCITIES_KMS
)

M51_MAD_KMS = np.median(
    np.abs(
        NEBULAR_VELOCITIES_KMS
        - M51_MEDIAN_VELOCITY_KMS
    )
)


# ============================================================
# Velocity grid
# ============================================================
#
# Broad enough to demonstrate where each transition would
# have to occur.
#
# The region around the measured M51 velocities is highlighted
# separately in the output.
#
# ============================================================

VELOCITY_MIN_KMS = -500.0
VELOCITY_MAX_KMS = 1000.0
VELOCITY_STEP_KMS = 1.0

velocities = np.arange(
    VELOCITY_MIN_KMS,
    VELOCITY_MAX_KMS + VELOCITY_STEP_KMS,
    VELOCITY_STEP_KMS,
)


# ============================================================
# Relativistic Doppler calculation
# ============================================================
#
# At several hundred km/s the classical expression is already
# quite accurate, but we use the relativistic Doppler relation
# for consistency:
#
#       lambda_obs
#       -----------
#       lambda_rest
#
#       = sqrt((1 + beta) / (1 - beta))
#
# where:
#
#       beta = v / c
#
# ============================================================

SPEED_OF_LIGHT_KMS = 299792.458


def observed_wavelength(
    rest_wavelength_nm,
    velocity_kms,
):
    """
    Calculate observed wavelength for a rest wavelength
    and radial velocity.

    Uses the relativistic Doppler relation.
    """

    beta = velocity_kms / SPEED_OF_LIGHT_KMS

    if np.any(np.abs(beta) >= 1.0):
        raise ValueError(
            "Velocity must be smaller than the speed of light."
        )

    doppler_factor = np.sqrt(
        (1.0 + beta)
        / (1.0 - beta)
    )

    return rest_wavelength_nm * doppler_factor


def velocity_for_wavelength(
    rest_wavelength_nm,
    observed_wavelength_nm,
):
    """
    Calculate the relativistic radial velocity required to
    shift a laboratory wavelength to an observed wavelength.
    """

    ratio = (
        observed_wavelength_nm
        / rest_wavelength_nm
    )

    beta = (
        ratio**2 - 1.0
    ) / (
        ratio**2 + 1.0
    )

    return beta * SPEED_OF_LIGHT_KMS


# ============================================================
# Main calculation
# ============================================================

print("=" * 70)
print("M51 VELOCITY-GRID TEST")
print("Pa beta versus Cs II")
print("=" * 70)

print()
print("Observed M51 feature:")
print(
    f"  Wavelength: "
    f"{OBSERVED_FEATURE_NM:.8f} nm"
)
print(
    f"  Uncertainty: "
    f"{OBSERVED_FEATURE_ERROR_NM:.8f} nm"
)

print()
print("Laboratory wavelengths:")
print(
    f"  Pa beta: "
    f"{PA_BETA_REST_NM:.8f} nm"
)
print(
    f"  Cs II air: "
    f"{CSII_AIR_NM:.8f} nm"
)
print(
    f"  Cs II vacuum: "
    f"{CSII_VACUUM_NM:.8f} nm"
)

print()
print("Empirical M51 nebular velocity reference:")
print(
    f"  Median velocity: "
    f"{M51_MEDIAN_VELOCITY_KMS:+.2f} km/s"
)
print(
    f"  MAD: "
    f"{M51_MAD_KMS:.2f} km/s"
)
print(
    f"  Minimum measured: "
    f"{NEBULAR_VELOCITIES_KMS.min():+.2f} km/s"
)
print(
    f"  Maximum measured: "
    f"{NEBULAR_VELOCITIES_KMS.max():+.2f} km/s"
)


# ============================================================
# Required velocities
# ============================================================

pa_beta_required_velocity = velocity_for_wavelength(
    PA_BETA_REST_NM,
    OBSERVED_FEATURE_NM,
)

csii_required_velocity = velocity_for_wavelength(
    CSII_VACUUM_NM,
    OBSERVED_FEATURE_NM,
)


print()
print("=" * 70)
print("VELOCITY REQUIRED TO PLACE EACH LINE AT THE OBSERVED FEATURE")
print("=" * 70)

print()
print("Pa beta:")
print(
    f"  Required velocity: "
    f"{pa_beta_required_velocity:+.3f} km/s"
)

print()
print("Cs II:")
print(
    f"  Required velocity: "
    f"{csii_required_velocity:+.3f} km/s"
)


# ============================================================
# Compare required velocities with M51 nebular reference
# ============================================================

pa_beta_velocity_difference = (
    pa_beta_required_velocity
    - M51_MEDIAN_VELOCITY_KMS
)

csii_velocity_difference = (
    csii_required_velocity
    - M51_MEDIAN_VELOCITY_KMS
)

print()
print("=" * 70)
print("VELOCITY COMPARISON WITH LOCAL M51 NEBULAR GAS")
print("=" * 70)

print()
print("Pa beta:")
print(
    f"  Required velocity: "
    f"{pa_beta_required_velocity:+.3f} km/s"
)
print(
    f"  Difference from M51 median: "
    f"{pa_beta_velocity_difference:+.3f} km/s"
)
print(
    f"  Difference in MAD units: "
    f"{pa_beta_velocity_difference / M51_MAD_KMS:+.2f}"
)

print()
print("Cs II:")
print(
    f"  Required velocity: "
    f"{csii_required_velocity:+.3f} km/s"
)
print(
    f"  Difference from M51 median: "
    f"{csii_velocity_difference:+.3f} km/s"
)
print(
    f"  Difference in MAD units: "
    f"{csii_velocity_difference / M51_MAD_KMS:+.2f}"
)


# ============================================================
# Predicted wavelengths across velocity grid
# ============================================================

pa_beta_predicted = observed_wavelength(
    PA_BETA_REST_NM,
    velocities,
)

csii_predicted = observed_wavelength(
    CSII_VACUUM_NM,
    velocities,
)


# ============================================================
# Predictions at the measured M51 median velocity
# ============================================================

pa_beta_at_m51 = observed_wavelength(
    PA_BETA_REST_NM,
    M51_MEDIAN_VELOCITY_KMS,
)

csii_at_m51 = observed_wavelength(
    CSII_VACUUM_NM,
    M51_MEDIAN_VELOCITY_KMS,
)


print()
print("=" * 70)
print("PREDICTED WAVELENGTHS AT M51 MEDIAN VELOCITY")
print("=" * 70)

print()
print("Pa beta:")
print(
    f"  Predicted wavelength: "
    f"{pa_beta_at_m51:.8f} nm"
)
print(
    f"  Observed feature: "
    f"{OBSERVED_FEATURE_NM:.8f} nm"
)
print(
    f"  Difference: "
    f"{OBSERVED_FEATURE_NM - pa_beta_at_m51:+.8f} nm"
)

print()
print("Cs II:")
print(
    f"  Predicted wavelength: "
    f"{csii_at_m51:.8f} nm"
)
print(
    f"  Observed feature: "
    f"{OBSERVED_FEATURE_NM:.8f} nm"
)
print(
    f"  Difference: "
    f"{OBSERVED_FEATURE_NM - csii_at_m51:+.8f} nm"
)


# ============================================================
# Find closest grid points
# ============================================================

pa_beta_distance = np.abs(
    pa_beta_predicted
    - OBSERVED_FEATURE_NM
)

csii_distance = np.abs(
    csii_predicted
    - OBSERVED_FEATURE_NM
)

pa_beta_best_index = np.argmin(
    pa_beta_distance
)

csii_best_index = np.argmin(
    csii_distance
)

print()
print("=" * 70)
print("BEST GRID MATCH")
print("=" * 70)

print()
print("Pa beta:")
print(
    f"  Velocity: "
    f"{velocities[pa_beta_best_index]:+.1f} km/s"
)
print(
    f"  Predicted wavelength: "
    f"{pa_beta_predicted[pa_beta_best_index]:.8f} nm"
)
print(
    f"  Wavelength difference: "
    f"{pa_beta_distance[pa_beta_best_index]:.8f} nm"
)

print()
print("Cs II:")
print(
    f"  Velocity: "
    f"{velocities[csii_best_index]:+.1f} km/s"
)
print(
    f"  Predicted wavelength: "
    f"{csii_predicted[csii_best_index]:.8f} nm"
)
print(
    f"  Wavelength difference: "
    f"{csii_distance[csii_best_index]:.8f} nm"
)


# ============================================================
# M51 velocity-window predictions
# ============================================================

m51_low_velocity = (
    M51_MEDIAN_VELOCITY_KMS
    - M51_MAD_KMS
)

m51_high_velocity = (
    M51_MEDIAN_VELOCITY_KMS
    + M51_MAD_KMS
)

pa_beta_low = observed_wavelength(
    PA_BETA_REST_NM,
    m51_low_velocity,
)

pa_beta_high = observed_wavelength(
    PA_BETA_REST_NM,
    m51_high_velocity,
)

csii_low = observed_wavelength(
    CSII_VACUUM_NM,
    m51_low_velocity,
)

csii_high = observed_wavelength(
    CSII_VACUUM_NM,
    m51_high_velocity,
)


print()
print("=" * 70)
print("M51 ± MAD VELOCITY WINDOW")
print("=" * 70)

print()
print(
    f"Velocity interval: "
    f"{m51_low_velocity:+.2f} "
    f"to "
    f"{m51_high_velocity:+.2f} km/s"
)

print()
print("Pa beta predicted wavelength range:")
print(
    f"  {pa_beta_low:.6f} - "
    f"{pa_beta_high:.6f} nm"
)

print()
print("Cs II predicted wavelength range:")
print(
    f"  {csii_low:.6f} - "
    f"{csii_high:.6f} nm"
)


# ============================================================
# Interpretation
# ============================================================

print()
print("=" * 70)
print("INTERPRETATION")
print("=" * 70)

if abs(
    pa_beta_required_velocity
    - M51_MEDIAN_VELOCITY_KMS
) < M51_MAD_KMS:

    print()
    print(
        "Pa beta is kinematically compatible with "
        "the local M51 nebular velocity."
    )

else:

    print()
    print(
        "Pa beta is not within one MAD of the "
        "local M51 nebular velocity."
    )


if abs(
    csii_required_velocity
    - M51_MEDIAN_VELOCITY_KMS
) < M51_MAD_KMS:

    print()
    print(
        "Cs II is kinematically compatible with "
        "the local M51 nebular velocity."
    )

else:

    print()
    print(
        "Cs II is NOT within one MAD of the "
        "local M51 nebular velocity."
    )


print()
print(
    "The velocity required for Cs II to reproduce "
    "the observed 1284.2613 nm feature should be "
    "compared directly with the velocities measured "
    "from independent nebular emission lines."
)

print()
print(
    "This test does not by itself establish the atomic "
    "identity of the feature. It tests only whether "
    "the proposed laboratory wavelengths are "
    "kinematically consistent with the M51 spectrum."
)


# ============================================================
# Plot
# ============================================================

print()
print("=" * 70)
print("GENERATING VELOCITY-GRID PLOT")
print("=" * 70)

fig, ax = plt.subplots(
    figsize=(11, 7)
)

ax.plot(
    velocities,
    pa_beta_predicted,
    label="Pa beta",
)

ax.plot(
    velocities,
    csii_predicted,
    label="Cs II",
)

ax.axhline(
    OBSERVED_FEATURE_NM,
    linestyle="--",
    label="Observed feature",
)

ax.axvline(
    M51_MEDIAN_VELOCITY_KMS,
    linestyle=":",
    label="M51 median velocity",
)

ax.axvspan(
    m51_low_velocity,
    m51_high_velocity,
    alpha=0.15,
    label="M51 ± MAD",
)

ax.scatter(
    [
        pa_beta_required_velocity,
        csii_required_velocity,
    ],
    [
        OBSERVED_FEATURE_NM,
        OBSERVED_FEATURE_NM,
    ],
    zorder=5,
)

ax.set_xlabel(
    "Radial velocity (km/s)"
)

ax.set_ylabel(
    "Predicted observed wavelength (nm)"
)

ax.set_title(
    "M51 1284.261 nm Feature: Pa beta vs Cs II"
)

ax.legend()

ax.grid(
    alpha=0.3
)

plt.tight_layout()

output_path = Path(
    "m51_velocity_grid_pa_beta_vs_csii.png"
)

plt.savefig(
    output_path,
    dpi=150,
)

plt.show()

print()
print(
    f"Plot saved to: {output_path}"
)


# ============================================================
# Final summary
# ============================================================

print()
print("=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print()
print(
    f"Observed feature: "
    f"{OBSERVED_FEATURE_NM:.8f} nm"
)

print(
    f"Pa beta required velocity: "
    f"{pa_beta_required_velocity:+.3f} km/s"
)

print(
    f"Cs II required velocity: "
    f"{csii_required_velocity:+.3f} km/s"
)

print(
    f"M51 median nebular velocity: "
    f"{M51_MEDIAN_VELOCITY_KMS:+.3f} km/s"
)

print(
    f"M51 MAD: "
    f"{M51_MAD_KMS:.3f} km/s"
)

print()
print(
    "Velocity-grid test complete."
)

print("=" * 70)
