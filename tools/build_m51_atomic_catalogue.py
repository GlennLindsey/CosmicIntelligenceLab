#!/usr/bin/env python3

from pathlib import Path
import csv
import math


# ============================================================
# Paths
# ============================================================

RAW_PATH = Path(
    "data/atomic_lines/"
    "m51_nist_1270_1300_raw.csv"
)

OUTPUT_PATH = Path(
    "data/atomic_lines/"
    "m51_atomic_line_catalogue.csv"
)


# ============================================================
# NIST CSV cleaning
# ============================================================

def clean_nist_value(value):
    """
    Clean values from the NIST ASD CSV export.

    Handles NIST's Excel-compatible representation:

        =""1270.0492""

as well as ordinary quoted/unquoted values.
    """

    if value is None:
        return ""

    value = str(value).strip()

    if not value:
        return ""

    # Remove surrounding whitespace.
    value = value.strip()

    # NIST/Excel representation:
    # =""VALUE""
    if value.startswith('=""') and value.endswith('""'):
        value = value[3:-2]

    # Handle ordinary surrounding quotes.
    if len(value) >= 2:
        if value.startswith('"') and value.endswith('"'):
            value = value[1:-1]

    # Handle any remaining doubled quotes.
    value = value.replace('""', '"')

    # Remove a possible leading '='.
    if value.startswith("="):
        value = value[1:]

    # Remove quotes exposed by the previous operation.
    value = value.strip('"').strip()

    return value

def clean_header(value):
    """Normalize a NIST column heading."""

    if value is None:
        return ""

    return value.strip()


# ============================================================
# Numeric conversion
# ============================================================

def to_float(value):
    """Convert cleaned text to float, returning NaN if unavailable."""

    value = clean_nist_value(value)

    if not value:
        return math.nan

    try:
        return float(value)
    except ValueError:
        return math.nan


# ============================================================
# Species normalization
# ============================================================

def species_name(element, sp_num):
    """
    Convert NIST element + ion number into a readable species.

    NIST:
        H 1  -> H I
        Fe 2 -> Fe II
        Cs 2 -> Cs II
    """

    roman = {
        1: "I",
        2: "II",
        3: "III",
        4: "IV",
        5: "V",
        6: "VI",
        7: "VII",
        8: "VIII",
        9: "IX",
        10: "X",
    }

    element = clean_nist_value(element)
    sp_num_text = clean_nist_value(sp_num)

    if not element:
        return ""

    try:
        ion_number = int(sp_num_text)
    except ValueError:
        return element

    return f"{element} {roman.get(ion_number, sp_num_text)}"


# ============================================================
# Air -> vacuum wavelength
# ============================================================

def air_to_vacuum_nm(wavelength_air_nm):
    """
    Convert standard-air wavelength to vacuum wavelength.

    NIST reports our 1270-1300 nm catalogue wavelengths
    in air according to the selected wavelength convention.

    Uses the standard Edlén-type refractive-index relation
    appropriate for optical/near-IR wavelengths.

    Returns NaN for invalid input.
    """

    if not math.isfinite(wavelength_air_nm):
        return math.nan

    # Convert nm -> micrometres.
    wavelength_um = wavelength_air_nm / 1000.0

    # Wavenumber in inverse micrometres.
    sigma2 = (1.0 / wavelength_um) ** 2

    # Standard dry-air refractive index formulation.
    n_minus_1 = (
        6432.8
        + 2949810.0 / (146.0 - sigma2)
        + 25540.0 / (41.0 - sigma2)
    ) * 1.0e-8

    n = 1.0 + n_minus_1

    return wavelength_air_nm * n


# ============================================================
# Main parser
# ============================================================

def main():

    print("=" * 70)
    print("BUILD M51 LOCAL ATOMIC-LINE CATALOGUE")
    print("=" * 70)

    print()
    print("Raw NIST catalogue:")
    print(f"  {RAW_PATH}")

    print()
    print("Output catalogue:")
    print(f"  {OUTPUT_PATH}")

    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Raw NIST catalogue not found:\n{RAW_PATH}"
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Read raw NIST CSV
    # --------------------------------------------------------

    with RAW_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        raw_fieldnames = reader.fieldnames

        if raw_fieldnames is None:
            raise RuntimeError(
                "No CSV header found."
            )

        # Remove the empty final NIST column.
        fieldnames = [
            clean_header(name)
            for name in raw_fieldnames
            if clean_header(name)
        ]

        raw_rows = list(reader)

    print()
    print(f"Raw rows: {len(raw_rows)}")
    print(f"Raw columns: {len(fieldnames)}")

    # --------------------------------------------------------
    # Parse rows
    # --------------------------------------------------------

    output_rows = []

    for row_number, raw_row in enumerate(
        raw_rows,
        start=2,
    ):

        element = clean_nist_value(
            raw_row.get("element", "")
        )

        sp_num = clean_nist_value(
            raw_row.get("sp_num", "")
        )

        species = species_name(
            element,
            sp_num,
        )

        obs_air = to_float(
            raw_row.get("obs_wl_air(nm)", "")
        )

        obs_unc = to_float(
            raw_row.get("unc_obs_wl", "")
        )

        ritz_air = to_float(
            raw_row.get("ritz_wl_air(nm)", "")
        )

        ritz_unc = to_float(
            raw_row.get("unc_ritz_wl", "")
        )

        intensity = to_float(
            raw_row.get("intens", "")
        )

        aki = to_float(
            raw_row.get("Aki(s^-1)", "")
        )

        ei = to_float(
            raw_row.get("Ei(cm-1)", "")
        )

        ek = to_float(
            raw_row.get("Ek(cm-1)", "")
        )

        obs_vacuum = air_to_vacuum_nm(
            obs_air
        )

        ritz_vacuum = air_to_vacuum_nm(
            ritz_air
        )

        output_rows.append(
            {
                "species": species,
                "element": element,
                "ion_stage": sp_num,

                "observed_wavelength_air_nm": obs_air,
                "observed_wavelength_uncertainty_nm": obs_unc,

                "ritz_wavelength_air_nm": ritz_air,
                "ritz_wavelength_uncertainty_nm": ritz_unc,

                "observed_wavelength_vacuum_nm": obs_vacuum,
                "ritz_wavelength_vacuum_nm": ritz_vacuum,

                "relative_intensity": intensity,
                "Aki_s-1": aki,
                "accuracy": clean_nist_value(
                    raw_row.get("Acc", "")
                ),

                "lower_energy_cm-1": ei,
                "upper_energy_cm-1": ek,

                "lower_configuration": clean_nist_value(
                    raw_row.get("conf_i", "")
                ),
                "lower_term": clean_nist_value(
                    raw_row.get("term_i", "")
                ),
                "lower_J": clean_nist_value(
                    raw_row.get("J_i", "")
                ),

                "upper_configuration": clean_nist_value(
                    raw_row.get("conf_k", "")
                ),
                "upper_term": clean_nist_value(
                    raw_row.get("term_k", "")
                ),
                "upper_J": clean_nist_value(
                    raw_row.get("J_k", "")
                ),

                "transition_type": clean_nist_value(
                    raw_row.get("Type", "")
                ),

                "transition_probability_reference": clean_nist_value(
                    raw_row.get("tp_ref", "")
                ),
                "line_reference": clean_nist_value(
                    raw_row.get("line_ref", "")
                ),

                "source": "NIST ASD",
                "source_file": RAW_PATH.name,
            }
        )

    # --------------------------------------------------------
    # Write normalized catalogue
    # --------------------------------------------------------

    output_fieldnames = list(
        output_rows[0].keys()
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=output_fieldnames,
        )

        writer.writeheader()
        writer.writerows(output_rows)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    species_set = sorted(
        {
            row["species"]
            for row in output_rows
            if row["species"]
        }
    )

    print()
    print("=" * 70)
    print("CATALOGUE CREATED")
    print("=" * 70)

    print()
    print(f"Transitions: {len(output_rows)}")
    print(f"Species: {len(species_set)}")

    print()
    print("Species:")

    for species in species_set:
        count = sum(
            row["species"] == species
            for row in output_rows
        )

        print(
            f"  {species:8s} {count:4d}"
        )

    print()
    print("Output:")
    print(f"  {OUTPUT_PATH}")

    print()
    print("=" * 70)
    print("PARSER COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
