from tools.spectral_line_lookup import (
    find_same_upper_level_transitions,
)


TARGET_ENERGY = 170973.7833


print("=" * 60)
print("Cs II SAME-UPPER-LEVEL SEARCH")
print("=" * 60)

print()
print("Species: Cs II")
print(
    f"Target upper energy: "
    f"{TARGET_ENERGY:.4f} cm-1"
)
print("Energy tolerance: 0.1 cm-1")
print()

result = find_same_upper_level_transitions(
    species="Cs II",
    upper_energy_cm1=TARGET_ENERGY,
    energy_tolerance_cm1=0.1,
)

print("Status:", result["status"])
print(
    "Candidate count:",
    result["candidate_count"],
)

print()

for i, candidate in enumerate(
    result["candidates"],
    start=1,
):

    print(f"Candidate {i}:")
    print(
        "  Species:",
        candidate.get("species"),
    )
    print(
        "  Observed wavelength:",
        candidate.get(
            "observed_wavelength_nm"
        ),
    )
    print(
        "  Ritz wavelength:",
        candidate.get(
            "ritz_wavelength_nm"
        ),
    )
    print(
        "  Upper energy:",
        candidate.get(
            "upper_energy_cm-1"
        ),
    )
    print(
        "  Upper configuration:",
        candidate.get(
            "upper_configuration"
        ),
    )
    print(
        "  Upper term:",
        candidate.get(
            "upper_term"
        ),
    )
    print(
        "  Upper J:",
        candidate.get(
            "upper_J"
        ),
    )
    print(
        "  Lower configuration:",
        candidate.get(
            "lower_configuration"
        ),
    )
    print(
        "  Lower term:",
        candidate.get(
            "lower_term"
        ),
    )
    print(
        "  Lower J:",
        candidate.get(
            "lower_J"
        ),
    )
    print(
        "  Relative intensity:",
        candidate.get(
            "relative_intensity"
        ),
    )
    print(
        "  Upper-energy difference:",
        candidate.get(
            "upper_energy_difference_cm-1"
        ),
    )
    print(
        "  Line reference:",
        candidate.get(
            "line_reference"
        ),
    )
    print()
