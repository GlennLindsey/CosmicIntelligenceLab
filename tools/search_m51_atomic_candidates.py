#!/usr/bin/env python3

from pathlib import Path
import csv
import math


# ============================================================
# Configuration
# ============================================================

CATALOGUE_PATH = Path(
    "data/atomic_lines/"
    "m51_atomic_line_catalogue.csv"
)

OUTPUT_PATH = Path(
    "data/atomic_lines/"
    "m51_1284_candidate_lines.csv"
)

OBSERVED_WAVELENGTH_NM = 1284.261304400

REFERENCE_VELOCITY_KMS = 573.72

# Speed of light.
C_KMS = 299792.458

# NIRSpec G140M/F100LP resolution used in our M51 analysis.
RESOLVING_POWER = 916.3

INSTRUMENT_FWHM_NM = (
    OBSERVED_WAVELENGTH_NM / RESOLVING_POWER
)

HALF_FWHM_NM = INSTRUMENT_FWHM_NM / 2.0


# Candidate ranking limits.
#
# These are deliberately generous. We want the first catalogue
# search to show the atomic environment rather than prematurely
# eliminating possible blends.

MAX_VELOCITY_OFFSET_KMS = 1000.0

MAX_WAVELENGTH_DISTANCE_NM = 10.0


# ============================================================
# Doppler conversions
# ============================================================

def velocity_to_wavelength(
    rest_wavelength_nm,
    velocity_kms,
):
    """
    Relativistic Doppler conversion.

    Positive velocity = redshift.
    """

    beta = velocity_kms / C_KMS

    if abs(beta) >= 1.0:
        return math.nan

    return (
        rest_wavelength_nm
        * math.sqrt(
            (1.0 + beta)
            / (1.0 - beta)
        )
    )


def wavelength_to_velocity(
    observed_wavelength_nm,
    rest_wavelength_nm,
):
    """
    Relativistic velocity corresponding to an observed
    wavelength and laboratory rest wavelength.
    """

    if rest_wavelength_nm <= 0:
        return math.nan

    ratio = (
        observed_wavelength_nm
        / rest_wavelength_nm
    )

    beta = (
        ratio ** 2 - 1.0
    ) / (
        ratio ** 2 + 1.0
    )

    if abs(beta) >= 1.0:
        return math.nan

    return beta * C_KMS


# ============================================================
# Helpers
# ============================================================

def to_float(value):
    """Convert catalogue value to float or NaN."""

    if value is None:
        return math.nan

    value = str(value).strip()

    if not value:
        return math.nan

    try:
        number = float(value)

        if not math.isfinite(number):
            return math.nan

        return number

    except ValueError:
        return math.nan


def safe_text(value):
    """Return clean text for display/output."""

    if value is None:
        return ""

    return str(value).strip()


def rank_category(
    velocity_offset_kms,
    wavelength_distance_nm,
):
    """
    Simple transparent ranking category.

    This is NOT a formal Bayesian probability.
    """

    if not math.isfinite(velocity_offset_kms):
        return "unknown"

    if (
        abs(velocity_offset_kms) <= 25.0
        and wavelength_distance_nm <= HALF_FWHM_NM
    ):
        return "excellent"

    if (
        abs(velocity_offset_kms) <= 100.0
        and wavelength_distance_nm <= INSTRUMENT_FWHM_NM
    ):
        return "strong"

    if (
        abs(velocity_offset_kms) <= 250.0
        and wavelength_distance_nm <= 2.0 * INSTRUMENT_FWHM_NM
    ):
        return "plausible"

    if (
        abs(velocity_offset_kms) <= MAX_VELOCITY_OFFSET_KMS
        and wavelength_distance_nm <= MAX_WAVELENGTH_DISTANCE_NM
    ):
        return "distant"

    return "poor"


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("M51 1284 NM ATOMIC CANDIDATE SEARCH")
    print("LOCAL NIST ATOMIC-LINE CATALOGUE")
    print("=" * 70)

    print()
    print("Catalogue:")
    print(f"  {CATALOGUE_PATH}")

    print()
    print("Observed feature:")
    print(
        f"  {OBSERVED_WAVELENGTH_NM:.9f} nm"
    )

    print()
    print("M51 reference velocity:")
    print(
        f"  +{REFERENCE_VELOCITY_KMS:.2f} km/s"
    )

    print()
    print("NIRSpec resolution:")
    print(
        f"  R = {RESOLVING_POWER:.1f}"
    )
    print(
        f"  FWHM = {INSTRUMENT_FWHM_NM:.6f} nm"
    )

    if not CATALOGUE_PATH.exists():
        raise FileNotFoundError(
            f"Catalogue not found:\n{CATALOGUE_PATH}"
        )

    # --------------------------------------------------------
    # Read catalogue
    # --------------------------------------------------------

    with CATALOGUE_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        rows = list(
            csv.DictReader(f)
        )

    print()
    print(
        f"Catalogue transitions: {len(rows)}"
    )

    # --------------------------------------------------------
    # Evaluate transitions
    # --------------------------------------------------------

    candidates = []

    for row in rows:

        species = safe_text(
            row.get("species")
        )

        # Prefer Ritz vacuum wavelength.
        ritz_vacuum = to_float(
            row.get(
                "ritz_wavelength_vacuum_nm"
            )
        )

        observed_vacuum = to_float(
            row.get(
                "observed_wavelength_vacuum_nm"
            )
        )

        # If Ritz wavelength is unavailable,
        # fall back to observed laboratory wavelength.
        if math.isfinite(ritz_vacuum):

            rest_wavelength = ritz_vacuum
            wavelength_source = (
                "Ritz vacuum"
            )

        elif math.isfinite(observed_vacuum):

            rest_wavelength = observed_vacuum
            wavelength_source = (
                "Observed vacuum"
            )

        else:
            continue

        # ----------------------------------------------------
        # Predicted wavelength at M51 velocity
        # ----------------------------------------------------

        predicted_wavelength = (
            velocity_to_wavelength(
                rest_wavelength,
                REFERENCE_VELOCITY_KMS,
            )
        )

        if not math.isfinite(
            predicted_wavelength
        ):
            continue

        wavelength_difference = (
            predicted_wavelength
            - OBSERVED_WAVELENGTH_NM
        )

        wavelength_distance = abs(
            wavelength_difference
        )

        # ----------------------------------------------------
        # Velocity required to match observed feature
        # ----------------------------------------------------

        required_velocity = (
            wavelength_to_velocity(
                OBSERVED_WAVELENGTH_NM,
                rest_wavelength,
            )
        )

        if not math.isfinite(
            required_velocity
        ):
            continue

        velocity_offset = (
            required_velocity
            - REFERENCE_VELOCITY_KMS
        )

        # How many instrumental FWHM away?
        fwhm_distance = (
            wavelength_distance
            / INSTRUMENT_FWHM_NM
        )

        # ----------------------------------------------------
        # Atomic metadata
        # ----------------------------------------------------

        relative_intensity = to_float(
            row.get(
                "relative_intensity"
            )
        )

        aki = to_float(
            row.get(
                "Aki_s-1"
            )
        )

        lower_energy = to_float(
            row.get(
                "lower_energy_cm-1"
            )
        )

        upper_energy = to_float(
            row.get(
                "upper_energy_cm-1"
            )
        )

        # ----------------------------------------------------
        # Ranking
        # ----------------------------------------------------

        category = rank_category(
            velocity_offset,
            wavelength_distance,
        )

        candidates.append(
            {
                "species": species,

                "wavelength_source":
                    wavelength_source,

                "rest_wavelength_vacuum_nm":
                    rest_wavelength,

                "predicted_wavelength_at_M51_nm":
                    predicted_wavelength,

                "observed_feature_nm":
                    OBSERVED_WAVELENGTH_NM,

                "wavelength_difference_nm":
                    wavelength_difference,

                "absolute_wavelength_difference_nm":
                    wavelength_distance,

                "required_velocity_kms":
                    required_velocity,

                "reference_velocity_kms":
                    REFERENCE_VELOCITY_KMS,

                "velocity_offset_kms":
                    velocity_offset,

                "instrument_fwhm_nm":
                    INSTRUMENT_FWHM_NM,

                "distance_in_fwhm":
                    fwhm_distance,

                "relative_intensity":
                    relative_intensity,

                "Aki_s-1":
                    aki,

                "lower_energy_cm-1":
                    lower_energy,

                "upper_energy_cm-1":
                    upper_energy,

                "lower_configuration":
                    safe_text(
                        row.get(
                            "lower_configuration"
                        )
                    ),

                "lower_term":
                    safe_text(
                        row.get(
                            "lower_term"
                        )
                    ),

                "lower_J":
                    safe_text(
                        row.get(
                            "lower_J"
                        )
                    ),

                "upper_configuration":
                    safe_text(
                        row.get(
                            "upper_configuration"
                        )
                    ),

                "upper_term":
                    safe_text(
                        row.get(
                            "upper_term"
                        )
                    ),

                "upper_J":
                    safe_text(
                        row.get(
                            "upper_J"
                        )
                    ),

                "accuracy":
                    safe_text(
                        row.get(
                            "accuracy"
                        )
                    ),

                "line_reference":
                    safe_text(
                        row.get(
                            "line_reference"
                        )
                    ),

                "ranking":
                    category,
            }
        )

    # --------------------------------------------------------
    # Sort by physical velocity agreement
    # --------------------------------------------------------

    candidates.sort(
        key=lambda row: (
            abs(
                to_float(
                    row["velocity_offset_kms"]
                )
            ),
            abs(
                to_float(
                    row[
                        "absolute_wavelength_difference_nm"
                    ]
                )
            ),
        )
    )

    print()
    print(
        f"Usable catalogue transitions: "
        f"{len(candidates)}"
    )

    # --------------------------------------------------------
    # Write complete candidate catalogue
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(
        candidates[0].keys()
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(candidates)

    # --------------------------------------------------------
    # Display closest candidates
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CLOSEST ATOMIC TRANSITIONS")
    print("=" * 70)

    print()
    print(
        f"{'Species':10s} "
        f"{'Rest(nm)':>12s} "
        f"{'Pred(nm)':>12s} "
        f"{'Δλ(nm)':>10s} "
        f"{'Req v':>10s} "
        f"{'Δv':>10s} "
        f"{'FWHM':>8s} "
        f"{'Rank':>10s}"
    )

    print("-" * 90)

    for row in candidates[:30]:

        print(
            f"{row['species']:10s} "
            f"{to_float(row['rest_wavelength_vacuum_nm']):12.6f} "
            f"{to_float(row['predicted_wavelength_at_M51_nm']):12.6f} "
            f"{to_float(row['wavelength_difference_nm']):+10.6f} "
            f"{to_float(row['required_velocity_kms']):+10.2f} "
            f"{to_float(row['velocity_offset_kms']):+10.2f} "
            f"{to_float(row['distance_in_fwhm']):8.3f} "
            f"{row['ranking']:>10s}"
        )

    # --------------------------------------------------------
    # Species appearing near the feature
    # --------------------------------------------------------

    nearby = [
        row
        for row in candidates
        if (
            to_float(
                row[
                    "absolute_wavelength_difference_nm"
                ]
            )
            <= INSTRUMENT_FWHM_NM
        )
    ]

    species_nearby = sorted(
        {
            row["species"]
            for row in nearby
        }
    )

    print()
    print("=" * 70)
    print("TRANSITIONS WITHIN ONE INSTRUMENT FWHM")
    print("=" * 70)

    print()
    print(
        f"Number of transitions: {len(nearby)}"
    )

    print(
        f"Number of species: {len(species_nearby)}"
    )

    for species in species_nearby:
        count = sum(
            row["species"] == species
            for row in nearby
        )

        print(
            f"  {species:10s} {count:4d}"
        )

    # --------------------------------------------------------
    # Explicit Pa beta / Cs II check
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("EXPLICIT Pa BETA / Cs II CHECK")
    print("=" * 70)

    for target_species in (
        "H I",
        "Cs II",
    ):

        target_rows = [
            row
            for row in candidates
            if row["species"] == target_species
        ]

        print()
        print(target_species)

        if not target_rows:
            print("  No transition in catalogue.")
            continue

        for row in target_rows:

            print(
                f"  Rest vacuum: "
                f"{to_float(row['rest_wavelength_vacuum_nm']):.9f} nm"
            )

            print(
                f"  Predicted at M51: "
                f"{to_float(row['predicted_wavelength_at_M51_nm']):.9f} nm"
            )

            print(
                f"  Required velocity: "
                f"{to_float(row['required_velocity_kms']):+.3f} km/s"
            )

            print(
                f"  Velocity offset: "
                f"{to_float(row['velocity_offset_kms']):+.3f} km/s"
            )

            print(
                f"  Distance: "
                f"{to_float(row['distance_in_fwhm']):.3f} FWHM"
            )

            print(
                f"  Ranking: "
                f"{row['ranking']}"
            )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CANDIDATE SEARCH COMPLETE")
    print("=" * 70)

    print()
    print("Output:")
    print(f"  {OUTPUT_PATH}")

    print()
    print(
        "Important: ranking is based on wavelength and "
        "velocity agreement only."
    )

    print(
        "It is NOT a probability of atomic identification."
    )

    print(
        "Line intensity, Aki, excitation conditions, "
        "blending, and spatial morphology require "
        "separate analysis."
    )


if __name__ == "__main__":
    main()
