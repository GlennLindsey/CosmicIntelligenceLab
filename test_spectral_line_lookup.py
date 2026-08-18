from tools.spectral_line_lookup import spectral_line_lookup


print("=" * 60)
print("NIST SPECTRAL LINE LOOKUP TEST")
print("=" * 60)


# ------------------------------------------------------------
# Query
# ------------------------------------------------------------

query_wavelength_nm = 1284.263818
tolerance_nm = 0.5


print(
    f"\nQuery wavelength: {query_wavelength_nm:.6f} nm"
)

print(
    f"Tolerance: {tolerance_nm:.6f} nm"
)


# ------------------------------------------------------------
# Perform NIST lookup
# ------------------------------------------------------------

result = spectral_line_lookup(
    wavelength_nm=query_wavelength_nm,
    tolerance_nm=tolerance_nm,
)


print(
    f"Status: {result['status']}"
)

print(
    f"Candidate count: {result['candidate_count']}"
)


# ------------------------------------------------------------
# Display candidates
# ------------------------------------------------------------

print("\nCandidates:")


for index, candidate in enumerate(
    result["candidates"],
    start=1,
):

    print(f"\nCandidate {index}:")

    print(
        f"  Observed wavelength: "
        f"{candidate['observed_wavelength_nm']}"
    )

    print(
        f"  Observed uncertainty: "
        f"{candidate['observed_uncertainty_nm']}"
    )

    print(
        f"  Ritz wavelength: "
        f"{candidate['ritz_wavelength_nm']}"
    )

    print(
        f"  Ritz uncertainty: "
        f"{candidate['ritz_uncertainty_nm']}"
    )

    print(
        f"  Relative intensity: "
        f"{candidate['relative_intensity']}"
    )

    print(
        f"  Aki: "
        f"{candidate['transition_probability_s-1']}"
    )

    print(
        f"  Accuracy: "
        f"{candidate['accuracy']}"
    )

    print(
        f"  Lower energy: "
        f"{candidate['lower_energy_cm-1']}"
    )

    print(
        f"  Upper energy: "
        f"{candidate['upper_energy_cm-1']}"
    )

    print(
        f"  Lower configuration: "
        f"{candidate['lower_configuration']}"
    )

    print(
        f"  Lower term: "
        f"{candidate['lower_term']}"
    )

    print(
        f"  Upper configuration: "
        f"{candidate['upper_configuration']}"
    )

    print(
        f"  Upper term: "
        f"{candidate['upper_term']}"
    )

    print(
        f"  Transition type: "
        f"{candidate['transition_type']}"
    )

    print(
        f"  TP reference: "
        f"{candidate['transition_probability_reference']}"
    )

    print(
        f"  Line reference: "
        f"{candidate['line_reference']}"
    )

    print(
        f"  Species: "
        f"{candidate['species']}"
    )


print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
