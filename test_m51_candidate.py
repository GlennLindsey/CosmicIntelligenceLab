from tools.m51_spectral_analysis import (
    load_x1d_spectrum,
    prepare_spectrum,
    characterize_candidate_feature,
)


X1D_PATH = (
    "data/m51_jwst_level3/"
    "jw03435-o006_t010_nirspec_g140m-f100lp_x1d.fits"
)


spectrum = load_x1d_spectrum(X1D_PATH)

prepared = prepare_spectrum(spectrum)


# Pa-beta candidate from our M51 investigation.
candidate = characterize_candidate_feature(
    prepared,
    center_guess=1.284282,
    window=0.003,
)


print("=" * 60)
print("M51 CANDIDATE CHARACTERIZATION")
print("=" * 60)

print(
    f"\nCandidate wavelength: "
    f"{candidate['candidate_wavelength_um']:.9f} um"
)

print(
    f"Measured wavelength: "
    f"{candidate['measured_wavelength_um']:.9f} um"
)

print(
    f"Measured flux: "
    f"{candidate['measured_flux']:.8g}"
)

print(
    f"Local continuum: "
    f"{candidate['local_continuum']:.8g}"
)

print(
    f"Excess flux: "
    f"{candidate['excess_flux']:.8g}"
)


print("\nGaussian fit:")

for key, value in candidate["gaussian_fit"].items():
    print(f"  {key}: {value}")


print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
