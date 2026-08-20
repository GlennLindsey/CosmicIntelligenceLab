from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.io import fits
from scipy.optimize import nnls


# ============================================================
# M51 1284 NM
# Pa beta + C I PROFILE LIKELIHOOD TEST
# ============================================================

print("=" * 70)
print("M51 1284 NM PA BETA + C I PROFILE LIKELIHOOD")
print("NEXT-GENERATION SPECTROSCOPY AGENT TEST")
print("=" * 70)


# ============================================================
# Paths
# ============================================================

X1D_PATH = Path(
    "data/m51_jwst_level3/"
    "jw03435-o006_t010_nirspec_g140m-f100lp_x1d.fits"
)

CATALOGUE_PATH = Path(
    "data/atomic_lines/"
    "m51_atomic_line_catalogue.csv"
)

PROFILE_OUTPUT = Path(
    "data/atomic_lines/"
    "m51_1284_pabeta_ci_profile_likelihood.csv"
)

SURFACE_OUTPUT = Path(
    "data/atomic_lines/"
    "m51_1284_pabeta_ci_chi2_surface.csv"
)

PLOT_OUTPUT = Path(
    "m51_1284_pabeta_ci_profile_likelihood.png"
)

SURFACE_PLOT_OUTPUT = Path(
    "m51_1284_pabeta_ci_chi2_surface.png"
)


# ============================================================
# Constants
# ============================================================

C_KMS = 299792.458

REFERENCE_VELOCITY_KMS = 573.72

OBSERVED_FEATURE_NM = 1284.26130440

PA_BETA_REST_NM = 1281.8070

RESOLVING_POWER = 916.3

FIT_MIN_NM = 1278.0
FIT_MAX_NM = 1290.0

CI_FWHM_LIMIT = 1.0

VELOCITY_MIN_KMS = 250.0
VELOCITY_MAX_KMS = 900.0

VELOCITY_STEP_KMS = 5.0


# ============================================================
# Utility functions
# ============================================================

def velocity_to_wavelength(
    rest_nm,
    velocity_kms,
):
    beta = velocity_kms / C_KMS

    return (
        rest_nm
        * np.sqrt(
            (1.0 + beta)
            / (1.0 - beta)
        )
    )


def gaussian(
    wavelength_nm,
    center_nm,
    sigma_nm,
):
    return np.exp(
        -0.5
        * (
            (
                wavelength_nm
                - center_nm
            )
            / sigma_nm
        ) ** 2
    )


def reduced_chi2(
    residual,
    n_parameters,
):
    n = len(residual)

    dof = max(
        n - n_parameters,
        1,
    )

    return (
        np.sum(
            residual ** 2
        )
        / dof
    )


# ============================================================
# Load X1D
# ============================================================

print()
print("=" * 70)
print("LOADING M51 X1D SPECTRUM")
print("=" * 70)

with fits.open(
    X1D_PATH,
    memmap=False,
) as hdul:

    table = hdul[1].data

    wavelength_um = np.asarray(
        table["WAVELENGTH"],
        dtype=float,
    )

    flux = np.asarray(
        table["FLUX"],
        dtype=float,
    )

    flux_error = np.asarray(
        table["FLUX_ERROR"],
        dtype=float,
    )

wavelength_nm = (
    wavelength_um
    * 1000.0
)

print(
    f"Spectral points: "
    f"{len(wavelength_nm)}"
)

print(
    f"Wavelength range: "
    f"{wavelength_nm.min():.6f}"
    f" - "
    f"{wavelength_nm.max():.6f} nm"
)


# ============================================================
# Local spectral window
# ============================================================

mask = (
    np.isfinite(wavelength_nm)
    & np.isfinite(flux)
    & np.isfinite(flux_error)
    & (flux_error > 0)
    & (wavelength_nm >= FIT_MIN_NM)
    & (wavelength_nm <= FIT_MAX_NM)
)

wave = wavelength_nm[mask]
data = flux[mask]
error = flux_error[mask]

print()
print("=" * 70)
print("LOCAL SPECTRAL WINDOW")
print("=" * 70)

print(
    f"Window: "
    f"{FIT_MIN_NM:.3f} - "
    f"{FIT_MAX_NM:.3f} nm"
)

print(
    f"Usable points: "
    f"{len(wave)}"
)

if len(wave) < 10:
    raise RuntimeError(
        "Too few spectral points "
        "for profile likelihood."
    )


# ============================================================
# Instrument resolution
# ============================================================

instrument_fwhm_nm = (
    OBSERVED_FEATURE_NM
    / RESOLVING_POWER
)

instrument_sigma_nm = (
    instrument_fwhm_nm
    / 2.354820045
)

print()
print("=" * 70)
print("NIRSPEC INSTRUMENT RESOLUTION")
print("=" * 70)

print(
    f"R = {RESOLVING_POWER:.1f}"
)

print(
    f"FWHM = "
    f"{instrument_fwhm_nm:.6f} nm"
)

print(
    f"sigma = "
    f"{instrument_sigma_nm:.6f} nm"
)


# ============================================================
# Load atomic catalogue
# ============================================================

print()
print("=" * 70)
print("LOADING C I CATALOGUE")
print("=" * 70)

catalogue = pd.read_csv(
    CATALOGUE_PATH
)

ci = catalogue[
    catalogue["species"]
    .astype(str)
    .str.strip()
    .eq("C I")
].copy()

ci["rest_nm"] = pd.to_numeric(
    ci["ritz_wavelength_vacuum_nm"],
    errors="coerce",
)

ci["Aki"] = pd.to_numeric(
    ci["Aki_s-1"],
    errors="coerce",
)

ci = ci[
    np.isfinite(
        ci["rest_nm"]
    )
].copy()

ci["reference_predicted_nm"] = (
    velocity_to_wavelength(
        ci["rest_nm"].to_numpy(),
        REFERENCE_VELOCITY_KMS,
    )
)

ci["distance_nm"] = np.abs(
    ci["reference_predicted_nm"]
    - OBSERVED_FEATURE_NM
)

ci["distance_fwhm"] = (
    ci["distance_nm"]
    / instrument_fwhm_nm
)

ci_blend = ci[
    ci["distance_fwhm"]
    <= CI_FWHM_LIMIT
].copy()

ci_blend = ci_blend.reset_index(
    drop=True
)

if len(ci_blend) == 0:
    raise RuntimeError(
        "No C I transitions found "
        "within one instrumental FWHM."
    )

print(
    f"Total catalogue rows: "
    f"{len(catalogue)}"
)

print(
    f"C I catalogue rows: "
    f"{len(ci)}"
)

print()
print(
    "C I components included:"
)

for _, row in ci_blend.iterrows():

    print(
        f"  "
        f"{row['rest_nm']:.6f}"
        f" -> "
        f"{row['reference_predicted_nm']:.6f}"
        f" nm"
        f" | Aki = "
        f"{row['Aki']:.3g}"
    )


# ============================================================
# Velocity grids
# ============================================================

velocity_grid = np.arange(
    VELOCITY_MIN_KMS,
    VELOCITY_MAX_KMS
    + VELOCITY_STEP_KMS,
    VELOCITY_STEP_KMS,
)

print()
print("=" * 70)
print("VELOCITY GRID")
print("=" * 70)

print(
    f"Range: "
    f"{VELOCITY_MIN_KMS:.1f}"
    f" - "
    f"{VELOCITY_MAX_KMS:.1f} km/s"
)

print(
    f"Step: "
    f"{VELOCITY_STEP_KMS:.1f} km/s"
)

print(
    f"Grid points: "
    f"{len(velocity_grid)}"
)


# ============================================================
# Weighted linear model fitting
# ============================================================
#
# For any fixed Pa beta velocity and C I velocity:
#
#   flux =
#       continuum
#       + Pa beta amplitude
#       + C I component amplitudes
#
# The amplitudes are solved linearly.
#
# This avoids nonlinear optimizer degeneracy.
#
# ============================================================

def solve_model(
    pa_velocity,
    ci_velocity,
    include_ci=True,
):
    """
    Solve the linear amplitudes for fixed velocities.

    Parameters:
        pa_velocity:
            Fixed Pa beta velocity.

        ci_velocity:
            Fixed common C I velocity.

        include_ci:
            Whether C I components are included.

    Returns:
        chi2
        fitted_model
        coefficients
    """

    columns = []

    # --------------------------------------------------------
    # Continuum intercept
    # --------------------------------------------------------

    columns.append(
        np.ones_like(wave)
    )

    # --------------------------------------------------------
    # Continuum slope
    # --------------------------------------------------------

    columns.append(
        wave
        - OBSERVED_FEATURE_NM
    )

    # --------------------------------------------------------
    # Pa beta
    # --------------------------------------------------------

    pa_center = velocity_to_wavelength(
        PA_BETA_REST_NM,
        pa_velocity,
    )

    columns.append(
        gaussian(
            wave,
            pa_center,
            instrument_sigma_nm,
        )
    )

    # --------------------------------------------------------
    # C I components
    # --------------------------------------------------------

    if include_ci:

        for rest in ci_blend[
            "rest_nm"
        ].to_numpy():

            ci_center = (
                velocity_to_wavelength(
                    rest,
                    ci_velocity,
                )
            )

            columns.append(
                gaussian(
                    wave,
                    ci_center,
                    instrument_sigma_nm,
                )
            )

    design_matrix = np.column_stack(
        columns
    )

    weighted_matrix = (
        design_matrix
        / error[:, None]
    )

    weighted_data = (
        data
        / error
    )

    # --------------------------------------------------------
    # Non-negative least squares
    # --------------------------------------------------------
    #
    # Flux emission components are constrained
    # to non-negative amplitudes.
    #
    # Continuum coefficients are allowed to vary
    # freely.
    #
    # We therefore solve the continuum separately
    # if NNLS would incorrectly constrain it.
    #
    # --------------------------------------------------------

    n_columns = (
        design_matrix.shape[1]
    )

    if n_columns == 0:
        raise RuntimeError(
            "Empty design matrix."
        )

    # Continuum columns are unconstrained.
    continuum_matrix = (
        design_matrix[:, :2]
    )

    line_matrix = (
        design_matrix[:, 2:]
    )

    weighted_continuum = (
        continuum_matrix
        / error[:, None]
    )

    weighted_lines = (
        line_matrix
        / error[:, None]
    )

    # --------------------------------------------------------
    # Because the number of samples is small, use bounded
    # least squares for all coefficients.
    # --------------------------------------------------------

    from scipy.optimize import lsq_linear

    combined_matrix = np.column_stack(
        [
            weighted_continuum,
            weighted_lines,
        ]
    )

    lower_bounds = (
        [-np.inf, -np.inf]
        + [0.0]
        * (
            combined_matrix.shape[1]
            - 2
        )
    )

    upper_bounds = (
        [np.inf, np.inf]
        + [np.inf]
        * (
            combined_matrix.shape[1]
            - 2
        )
    )

    fit = lsq_linear(
        combined_matrix,
        weighted_data,
        bounds=(
            lower_bounds,
            upper_bounds,
        ),
        lsmr_tol="auto",
        max_iter=10000,
    )

    coefficients = fit.x

    fitted_model = (
        design_matrix
        @ coefficients
    )

    residual = (
        fitted_model
        - data
    ) / error

    chi2 = np.sum(
        residual ** 2
    )

    return (
        chi2,
        fitted_model,
        coefficients,
    )


# ============================================================
# Baseline Pa beta-only profile
# ============================================================

print()
print("=" * 70)
print("PA BETA-ONLY PROFILE")
print("=" * 70)

pa_profile_rows = []

for velocity in velocity_grid:

    chi2, model, coefficients = (
        solve_model(
            velocity,
            REFERENCE_VELOCITY_KMS,
            include_ci=False,
        )
    )

    pa_profile_rows.append(
        {
            "pa_beta_velocity_kms":
                velocity,
            "chi2":
                chi2,
            "pa_beta_amplitude":
                coefficients[2],
        }
    )

pa_profile = pd.DataFrame(
    pa_profile_rows
)

pa_minimum = pa_profile.loc[
    pa_profile["chi2"].idxmin()
]

pa_min_chi2 = (
    pa_minimum["chi2"]
)

best_pa_velocity = (
    pa_minimum[
        "pa_beta_velocity_kms"
    ]
)

print(
    f"Best Pa beta velocity: "
    f"{best_pa_velocity:.1f} km/s"
)

print(
    f"Minimum chi²: "
    f"{pa_min_chi2:.3f}"
)

print(
    f"Reference velocity: "
    f"{REFERENCE_VELOCITY_KMS:.2f} km/s"
)


# ============================================================
# Two-dimensional Pa beta + C I surface
# ============================================================

print()
print("=" * 70)
print("PA BETA + C I 2-D PROFILE LIKELIHOOD")
print("=" * 70)

print(
    "Scanning Pa beta velocity "
    "against C I velocity..."
)

surface_rows = []

chi2_surface = np.zeros(
    (
        len(velocity_grid),
        len(velocity_grid),
    )
)

for i, pa_velocity in enumerate(
    velocity_grid
):

    if (
        i % 20 == 0
    ):
        print(
            f"  Pa beta grid "
            f"{i + 1}/"
            f"{len(velocity_grid)}"
        )

    for j, ci_velocity in enumerate(
        velocity_grid
    ):

        chi2, model, coefficients = (
            solve_model(
                pa_velocity,
                ci_velocity,
                include_ci=True,
            )
        )

        chi2_surface[
            i,
            j
        ] = chi2

        surface_rows.append(
            {
                "pa_beta_velocity_kms":
                    pa_velocity,
                "ci_velocity_kms":
                    ci_velocity,
                "chi2":
                    chi2,
            }
        )


surface = pd.DataFrame(
    surface_rows
)

surface.to_csv(
    SURFACE_OUTPUT,
    index=False,
)


# ============================================================
# Global minimum
# ============================================================

minimum_index = np.unravel_index(
    np.argmin(
        chi2_surface
    ),
    chi2_surface.shape,
)

best_i = minimum_index[0]
best_j = minimum_index[1]

best_joint_pa_velocity = (
    velocity_grid[best_i]
)

best_joint_ci_velocity = (
    velocity_grid[best_j]
)

best_joint_chi2 = (
    chi2_surface[
        best_i,
        best_j
    ]
)


# ============================================================
# Profile over C I velocity
# ============================================================

ci_profile = np.min(
    chi2_surface,
    axis=0,
)

best_profile_ci_index = (
    np.argmin(
        ci_profile
    )
)

best_profile_ci_velocity = (
    velocity_grid[
        best_profile_ci_index
    ]
)

best_profile_ci_chi2 = (
    ci_profile[
        best_profile_ci_index
    ]
)


# ============================================================
# Profile over Pa beta velocity
# ============================================================

joint_pa_profile = np.min(
    chi2_surface,
    axis=1,
)

best_profile_pa_index = (
    np.argmin(
        joint_pa_profile
    )
)

best_profile_pa_velocity = (
    velocity_grid[
        best_profile_pa_index
    ]
)

best_profile_pa_chi2 = (
    joint_pa_profile[
        best_profile_pa_index
    ]
)


# ============================================================
# Δχ² profiles
# ============================================================

pa_delta_chi2 = (
    joint_pa_profile
    - best_joint_chi2
)

ci_delta_chi2 = (
    ci_profile
    - best_joint_chi2
)


# ============================================================
# Save profiles
# ============================================================

profile_rows = []

for i, velocity in enumerate(
    velocity_grid
):

    profile_rows.append(
        {
            "velocity_kms":
                velocity,
            "pa_beta_only_chi2":
                pa_profile.iloc[i][
                    "chi2"
                ],
            "joint_profile_pa_beta_chi2":
                joint_pa_profile[i],
            "joint_profile_ci_chi2":
                ci_profile[i],
            "joint_delta_chi2_pa_beta":
                pa_delta_chi2[i],
            "joint_delta_chi2_ci":
                ci_delta_chi2[i],
        }
    )

profile = pd.DataFrame(
    profile_rows
)

profile.to_csv(
    PROFILE_OUTPUT,
    index=False,
)


# ============================================================
# Print global results
# ============================================================

print()
print("=" * 70)
print("PROFILE-LIKELIHOOD RESULTS")
print("=" * 70)

print()
print(
    f"Pa beta-only minimum:"
)

print(
    f"  velocity = "
    f"{best_pa_velocity:.1f} km/s"
)

print(
    f"  chi² = "
    f"{pa_min_chi2:.3f}"
)

print()
print(
    f"Joint global minimum:"
)

print(
    f"  Pa beta velocity = "
    f"{best_joint_pa_velocity:.1f} km/s"
)

print(
    f"  C I velocity = "
    f"{best_joint_ci_velocity:.1f} km/s"
)

print(
    f"  chi² = "
    f"{best_joint_chi2:.3f}"
)

print()
print(
    f"C I profile minimum:"
)

print(
    f"  C I velocity = "
    f"{best_profile_ci_velocity:.1f} km/s"
)

print(
    f"  chi² = "
    f"{best_profile_ci_chi2:.3f}"
)

print()
print(
    f"Pa beta profile minimum "
    f"after allowing C I:"
)

print(
    f"  Pa beta velocity = "
    f"{best_profile_pa_velocity:.1f} km/s"
)

print(
    f"  chi² = "
    f"{best_profile_pa_chi2:.3f}"
)


# ============================================================
# Compare reference velocity
# ============================================================

reference_index = np.argmin(
    np.abs(
        velocity_grid
        - REFERENCE_VELOCITY_KMS
    )
)

reference_grid_velocity = (
    velocity_grid[
        reference_index
    ]
)

reference_ci_chi2 = (
    ci_profile[
        reference_index
    ]
)

reference_pa_chi2 = (
    joint_pa_profile[
        reference_index
    ]
)

print()
print("=" * 70)
print("REFERENCE VELOCITY COMPARISON")
print("=" * 70)

print(
    f"Independent M51 velocity:"
    f" {REFERENCE_VELOCITY_KMS:.2f} km/s"
)

print(
    f"Nearest grid velocity:"
    f" {reference_grid_velocity:.1f} km/s"
)

print(
    f"C I profile Δχ² at reference:"
    f" "
    f"{reference_ci_chi2 - best_joint_chi2:.3f}"
)

print(
    f"Pa beta profile Δχ² at reference:"
    f" "
    f"{reference_pa_chi2 - best_joint_chi2:.3f}"
)


# ============================================================
# Plot 1: profile likelihoods
# ============================================================

print()
print("=" * 70)
print("CREATING PROFILE LIKELIHOOD PLOT")
print("=" * 70)

fig, ax = plt.subplots(
    figsize=(12, 7)
)

ax.plot(
    velocity_grid,
    pa_delta_chi2,
    linewidth=2,
    label="Pa beta profile",
)

ax.plot(
    velocity_grid,
    ci_delta_chi2,
    linewidth=2,
    label="C I profile",
)

ax.axvline(
    REFERENCE_VELOCITY_KMS,
    linestyle="--",
    linewidth=1.5,
    label="M51 reference velocity",
)

ax.axhline(
    0.0,
    linestyle=":",
    linewidth=1,
)

ax.axhline(
    2.30,
    linestyle=":",
    linewidth=1,
    label="Δχ² = 2.30",
)

ax.axhline(
    6.17,
    linestyle=":",
    linewidth=1,
    label="Δχ² = 6.17",
)

ax.axhline(
    11.8,
    linestyle=":",
    linewidth=1,
    label="Δχ² = 11.8",
)

ax.set_xlabel(
    "Velocity (km/s)"
)

ax.set_ylabel(
    "Profile Δχ²"
)

ax.set_title(
    "M51 1284 nm Pa beta + C I "
    "Profile Likelihood"
)

ax.set_ylim(
    bottom=0
)

ax.grid(
    alpha=0.25
)

ax.legend()

fig.tight_layout()

fig.savefig(
    PLOT_OUTPUT,
    dpi=200,
)

plt.close(fig)


# ============================================================
# Plot 2: 2-D chi-square surface
# ============================================================

print(
    "Creating 2-D chi-square surface..."
)

fig, ax = plt.subplots(
    figsize=(10, 8)
)

delta_surface = (
    chi2_surface
    - best_joint_chi2
)

image = ax.imshow(
    delta_surface,
    origin="lower",
    aspect="auto",
    extent=[
        velocity_grid.min(),
        velocity_grid.max(),
        velocity_grid.min(),
        velocity_grid.max(),
    ],
)

contours = [
    2.30,
    6.17,
    11.8,
]

CS = ax.contour(
    velocity_grid,
    velocity_grid,
    delta_surface,
    levels=contours,
    linewidths=1.5,
)

ax.clabel(
    CS,
    inline=True,
    fontsize=9,
    fmt="Δχ² %.2f",
)

ax.scatter(
    [
        best_joint_ci_velocity
    ],
    [
        best_joint_pa_velocity
    ],
    marker="x",
    s=100,
    linewidths=2,
    label="Global minimum",
)

ax.axvline(
    REFERENCE_VELOCITY_KMS,
    linestyle="--",
    linewidth=1,
    label="M51 reference",
)

ax.axhline(
    REFERENCE_VELOCITY_KMS,
    linestyle="--",
    linewidth=1,
)

ax.set_xlabel(
    "C I velocity (km/s)"
)

ax.set_ylabel(
    "Pa beta velocity (km/s)"
)

ax.set_title(
    "M51 1284 nm Pa beta + C I "
    "Δχ² Surface"
)

fig.colorbar(
    image,
    ax=ax,
    label="Δχ²"
)

ax.legend()

fig.tight_layout()

fig.savefig(
    SURFACE_PLOT_OUTPUT,
    dpi=200,
)

plt.close(fig)


# ============================================================
# Final interpretation
# ============================================================

print()
print("=" * 70)
print("FINAL INTERPRETATION")
print("=" * 70)

print()
print(
    "This experiment profiles the likelihood surface "
    "rather than relying on a nonlinear optimizer."
)

print()
print(
    "For each fixed pair of velocities, the continuum "
    "and emission-line amplitudes are solved linearly."
)

print()
print(
    "The C I components share a common velocity but "
    "have independent amplitudes."
)

print()
print(
    "This avoids imposing Aki-based flux ratios."
)

print()
print(
    "The principal question is whether a genuine "
    "C I likelihood minimum exists near the independent "
    "M51 velocity of +573.72 km/s."
)

print()
print(
    "A C I solution should not be considered compelling "
    "merely because an optimizer can find one."
)

print()
print(
    "We will assess the C I hypothesis from the shape "
    "and depth of the profile likelihood and the "
    "two-dimensional Δχ² surface."
)

print()
print(
    "Outputs:"
)

print(
    f"  {PROFILE_OUTPUT}"
)

print(
    f"  {SURFACE_OUTPUT}"
)

print(
    f"  {PLOT_OUTPUT}"
)

print(
    f"  {SURFACE_PLOT_OUTPUT}"
)

print()
print("=" * 70)
print(
    "PROFILE LIKELIHOOD TEST COMPLETE"
)
print("=" * 70)
