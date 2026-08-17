from tools.m51_spectral_analysis import (
    load_x1d_spectrum,
    prepare_spectrum,
    characterize_candidate_feature,
    create_spectral_evidence,
)


X1D_PATH = (
    "data/m51_jwst_level3/"
    "jw03435-o006_t010_nirspec_g140m-f100lp_x1d.fits"
)


# ------------------------------------------------------------
# Load and prepare spectrum
# ------------------------------------------------------------

spectrum = load_x1d_spectrum(X1D_PATH)

prepared = prepare_spectrum(spectrum)


# ------------------------------------------------------------
# Characterize the M51 candidate
# ------------------------------------------------------------

characterization = characterize_candidate_feature(
    prepared,
    center_guess=1.284282,
    window=0.003,
)


# ------------------------------------------------------------
# Create Evidence Record
# ------------------------------------------------------------

evidence = create_spectral_evidence(
    characterization,
    source="JWST/NIRSpec",
    object_name="M51",
    dataset=(
        "jw03435-o006_t010_nirspec_g140m-f100lp_x1d.fits"
    ),
)


# ------------------------------------------------------------
# Display Evidence Record
# ------------------------------------------------------------

print("=" * 60)
print("M51 SPECTRAL EVIDENCE RECORD")
print("=" * 60)

print()

for key, value in evidence.items():

    print(f"{key}: {value}")

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
