#!/usr/bin/env python3

"""
parse_storey_hummer.py

Small parser for the Storey & Hummer (1995) hydrogenic
recombination emissivity files.

Initial test:
    r1b0200.d

This is:
    Z = 1
    Case B
    Te = 20,000 K

Hydrogen transitions:
    Pa-beta  = 5 -> 3
    Pa-gamma = 6 -> 3

The parser extracts the emissivities for these transitions
at every electron density contained in the file.
"""

from pathlib import Path
import re


# ============================================================
# CONFIGURATION
# ============================================================

STOREY_DIR = Path(
    "~/Projects/cosmic_ai/data/atomic_lines/storey_hummer"
).expanduser()

INPUT_FILE = STOREY_DIR / "r1b0200.d"


# Hydrogen transitions of interest

PA_BETA = (5, 3)
PA_GAMMA = (6, 3)


# ============================================================
# PARSE STOREY-HUMMER FILE
# ============================================================

def parse_storey_file(filename):
    """
    Parse a Storey-Hummer primary output file.

    Returns a list of dictionaries containing:

        Z
        Te
        ne
        case
        upper
        lower
        emissivity
    """

    records = []

    current_Z = None
    current_Te = None
    current_ne = None
    current_case = None
    current_upper = None

    with open(filename, "r") as f:

        for line in f:

            line = line.rstrip()

            # ------------------------------------------------
            # Header line
            #
            # Example:
            #
            # E_NU=50 Z= 1 TE= 2.000E+04
            # NE= 1.000E+02 CASE=B
            # ------------------------------------------------

            match = re.search(
                r"E_NU=\s*(\d+).*?"
                r"Z=\s*(\d+).*?"
                r"TE=\s*([0-9.E+-]+).*?"
                r"NE=\s*([0-9.E+-]+).*?"
                r"CASE=\s*([AB])",
                line
            )

            if match:

                current_upper = int(match.group(1))
                current_Z = int(match.group(2))
                current_Te = float(match.group(3))
                current_ne = float(match.group(4))
                current_case = match.group(5)

                continue

            # ------------------------------------------------
            # Data lines
            #
            # Format:
            #
            # lower emissivity lower emissivity ...
            #
            # Example:
            #
            # 1 0.000E+00  2 3.263E-29 ...
            # ------------------------------------------------

            if current_upper is None:
                continue

            tokens = line.split()

            if len(tokens) < 2:
                continue

            # Data consist of pairs:
            #
            # lower-level-number
            # emissivity

            if len(tokens) % 2 != 0:
                continue

            for i in range(0, len(tokens), 2):

                try:

                    lower = int(tokens[i])
                    emissivity = float(tokens[i + 1])

                except ValueError:
                    continue

                records.append(
                    {
                        "Z": current_Z,
                        "Te": current_Te,
                        "ne": current_ne,
                        "case": current_case,
                        "upper": current_upper,
                        "lower": lower,
                        "emissivity": emissivity,
                    }
                )

    return records


# ============================================================
# CALCULATE PA-BETA / PA-GAMMA RATIOS
# ============================================================

def calculate_ratios(records):
    """
    Extract Pa-beta and Pa-gamma emissivities
    and calculate their ratio at each density.
    """

    grouped = {}

    for record in records:

        key = (
            record["Z"],
            record["case"],
            record["Te"],
            record["ne"],
        )

        grouped.setdefault(key, {})

        transition = (
            record["upper"],
            record["lower"],
        )

        grouped[key][transition] = record["emissivity"]

    ratios = []

    for key, transitions in grouped.items():

        pabeta = transitions.get(PA_BETA)
        pagamma = transitions.get(PA_GAMMA)

        if pabeta is None or pagamma is None:
            continue

        ratios.append(
            {
                "Z": key[0],
                "case": key[1],
                "Te": key[2],
                "ne": key[3],
                "Pa_beta": pabeta,
                "Pa_gamma": pagamma,
                "ratio": pabeta / pagamma,
            }
        )

    return ratios


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("STOREY-HUMMER HYDROGEN EMISSIVITY PARSER")
    print("=" * 70)

    print()
    print("Input file:")
    print(f"  {INPUT_FILE}")

    if not INPUT_FILE.exists():

        print()
        print("ERROR: Input file does not exist.")
        print()
        print("Expected:")
        print(f"  {INPUT_FILE}")
        return

    # --------------------------------------------------------
    # Parse file
    # --------------------------------------------------------

    records = parse_storey_file(INPUT_FILE)

    print()
    print("Parsed records:")
    print(f"  {len(records):,}")

    # --------------------------------------------------------
    # Basic file information
    # --------------------------------------------------------

    if records:

        first = records[0]

        print()
        print("File parameters:")
        print(f"  Z:     {first['Z']}")
        print(f"  Case:  {first['case']}")
        print(f"  Te:    {first['Te']:.0f} K")

    # --------------------------------------------------------
    # Calculate ratios
    # --------------------------------------------------------

    ratios = calculate_ratios(records)

    print()
    print("Pa-beta / Pa-gamma")
    print()
    print(
        f"{'ne':>12} "
        f"{'Pa-beta':>16} "
        f"{'Pa-gamma':>16} "
        f"{'ratio':>10}"
    )

    print("-" * 60)

    for row in ratios:

        print(
            f"{row['ne']:12.3e} "
            f"{row['Pa_beta']:16.6e} "
            f"{row['Pa_gamma']:16.6e} "
            f"{row['ratio']:10.4f}"
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print()
    print(f"Pa-beta transition:  {PA_BETA[0]} -> {PA_BETA[1]}")
    print(f"Pa-gamma transition: {PA_GAMMA[0]} -> {PA_GAMMA[1]}")

    print()
    print(f"Density points with both lines: {len(ratios)}")

    if ratios:

        print()
        print(
            "Electron-density range:"
        )

        densities = [row["ne"] for row in ratios]

        print(
            f"  {min(densities):.3e} "
            f"to "
            f"{max(densities):.3e} cm^-3"
        )

    print()
    print("Parser test complete.")


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
