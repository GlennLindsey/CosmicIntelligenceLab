"""
M51 JWST/NIRSpec Cs II Companion-Line Investigation
===================================================

Hypothesis
----------
The M51 feature near 1284.264 nm may be Cs II arising from the
upper level:

    5p^5 (2P° 1/2) 4f

with NIST upper-level energy:

    170973.7833 cm^-1

This script searches the M51 JWST/NIRSpec X1D spectrum for other
Cs II transitions arising from the same upper level.

Important
---------
NIST relative intensities are atomic-data information. They are
NOT predictions of the relative observed line strengths in M51.

This script therefore performs an exploratory wavelength/local-
spectrum comparison only. It does not establish a Cs II
identification.

The next stage will use Gaussian fitting and more rigorous
statistical tests.
"""


from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from tools.m51_spectral_analysis import (
    load_x1d_spectrum,
    prepare_spectrum,
)


# ============================================================
# M51 JWST/NIRSpec spectrum
# ============================================================

X1D_PATH = Path(
    "data/m51_jwst_level3/"
    "jw03435-o006_t010_nirspec_g140m-f100lp_x1d.fits"
)


# ============================================================
# Cs II transitions from the same upper level
# ============================================================
#
# Common NIST upper level:
#
#     5p^5 (2P° 1/2) 4f
#
# Upper energy:
#
#     170973.7833 cm^-1
#
# Wavelengths are NIST AIR wavelengths.
#
# The values below are the transitions identified in the
# same-upper-level NIST search.
#
# ============================================================

CSII_LINES = [
    {
        "wavelength_nm": 1284.26406,
        "relative_intensity": 4500.0,
        "label": "Cs II reference",
        "lower_level": "5p5 (2P° 1/2) 7s",
    },
    {
        "wavelength_nm": 1940.88700,
        "relative_intensity": 1700.0,
        "label": "Cs II companion",
        "lower_level": "5p5 (2P° 3/2) 7d",
    },
    {
        "wavelength_nm": 1970.12280,
        "relative_intensity": 1700.0,
        "label": "Cs II companion",
        "lower_level": "5p5 (2P° 3/2) 7d",
    },
    {
        "wavelength_nm": 2063.19010,
        "relative_intensity": 490.0,
        "label": "Cs II companion",
        "lower_level": "5p5 (2P° 3/2) 7d",
    },
    {
        "wavelength_nm": 2068.88860,
        "relative_intensity": 540.0,
        "label": "Cs II companion",
        "lower_level": "5p5 (2P° 1/2) 6d",
    },
    {
        "wavelength_nm": 2375.65350,
        "relative_intensity": 1100.0,
        "label": "Cs II companion",
        "lower_level": "5p5 (2P° 3/2) 7d",
    },
    {
        "wavelength_nm": 2498.07640,
        "relative_intensity": 1300.0,
        "label": "Cs II companion",
        "lower_level": "5p5 (2P° 1/2) 6d",
    },
]


# ============================================================
# Analysis settings
# ============================================================

# Half-width of the local spectral region inspected around
# each expected line.
WINDOW_NM = 3.0


# ============================================================
# Helper: inspect local spectrum
# ============================================================

def measure_local_region(
    wavelength_nm,
    flux,
    uncertainty,
    expected_nm,
    window_nm=3.0,
):
    """
    Measure the local spectrum around an expected wavelength.

    This is an exploratory measurement, not a formal line fit.
    """

    mask = (
        np.abs(
            wavelength_nm - expected_nm
        )
        <= window_nm
    )

    if np.sum(mask) < 5:
        return None

    local_wavelength = wavelength_nm[mask]
    local_flux = flux[mask]
    local_uncertainty = uncertainty[mask]

    # --------------------------------------------------------
    # Estimate local continuum.
    #
    # Median is deliberately used for this first-pass
    # investigation.
    # --------------------------------------------------------

    continuum = np.median(local_flux)

    # --------------------------------------------------------
    # Find strongest local flux point.
    #
    # This is NOT yet a Gaussian line detection.
    # --------------------------------------------------------

    peak_index = np.argmax(local_flux)

    peak_wavelength = (
        local_wavelength[peak_index]
    )

    peak_flux = (
        local_flux[peak_index]
    )

    # --------------------------------------------------------
    # Estimate local uncertainty.
    # --------------------------------------------------------

    finite_uncertainty = (
        np.isfinite(local_uncertainty)
        & (local_uncertainty > 0)
    )

    if np.any(finite_uncertainty):

        noise = np.median(
            local_uncertainty[
                finite_uncertainty
            ]
        )

    else:

        noise = np.nan

    # --------------------------------------------------------
    # Approximate peak S/N relative to local continuum.
    # --------------------------------------------------------

    if (
        np.isfinite(noise)
        and noise > 0
    ):

        peak_snr = (
            peak_flux - continuum
        ) / noise

    else:

        peak_snr = np.nan

    # --------------------------------------------------------
    # Wavelength displacement.
    # --------------------------------------------------------

    wavelength_offset = (
        peak_wavelength
        - expected_nm
    )

    return {
        "expected_nm": expected_nm,
        "peak_nm": peak_wavelength,
        "offset_nm": wavelength_offset,
        "continuum": continuum,
        "peak_flux": peak_flux,
        "noise": noise,
        "peak_snr": peak_snr,
        "point_count": int(np.sum(mask)),
        "local_wavelength": local_wavelength,
        "local_flux": local_flux,
        "local_uncertainty": local_uncertainty,
    }


# ============================================================
# Main
# ============================================================

print("=" * 60)
print("M51 / Cs II COMPANION-LINE INVESTIGATION")
print("=" * 60)


# ============================================================
# Confirm input file
# ============================================================

print()
print("X1D spectrum:")
print(X1D_PATH)

if not X1D_PATH.exists():

    raise FileNotFoundError(
        f"M51 X1D spectrum not found: {X1D_PATH}"
    )


# ============================================================
# Load spectrum
# ============================================================

print()
print("Loading M51 JWST/NIRSpec X1D spectrum...")

spectrum = load_x1d_spectrum(
    X1D_PATH
)


# ============================================================
# Prepare spectrum
# ============================================================

print("Preparing spectrum...")

clean = prepare_spectrum(
    spectrum
)


wavelength_um = clean["wavelength"]
flux = clean["flux"]
uncertainty = clean["uncertainty"]


# ============================================================
# Convert wavelength to nm
# ============================================================

wavelength_nm = (
    wavelength_um * 1000.0
)


# ============================================================
# Spectrum information
# ============================================================

print()
print("Spectrum loaded successfully.")

print(
    f"Valid points: "
    f"{clean['valid_points']}"
)

print(
    f"Rejected points: "
    f"{clean['rejected_points']}"
)

print(
    f"Wavelength range: "
    f"{wavelength_nm.min():.3f} - "
    f"{wavelength_nm.max():.3f} nm"
)


# ============================================================
# Determine Cs II wavelength coverage
# ============================================================

print()
print("=" * 60)
print("Cs II LINE COVERAGE")
print("=" * 60)

observable_lines = []


for line in CSII_LINES:

    expected_nm = line["wavelength_nm"]

    in_range = (
        wavelength_nm.min()
        <= expected_nm
        <= wavelength_nm.max()
    )

    if in_range:

        status = "IN RANGE"

        observable_lines.append(
            line
        )

    else:

        status = "OUT OF RANGE"

    print(
        f"{expected_nm:10.5f} nm   "
        f"{status:12s}   "
        f"{line['label']}"
    )


print()
print(
    f"Accessible Cs II transitions: "
    f"{len(observable_lines)}"
)


# ============================================================
# Measure local regions
# ============================================================

print()
print("=" * 60)
print("LOCAL Cs II SPECTRUM MEASUREMENTS")
print("=" * 60)


measurements = []


for line in observable_lines:

    expected_nm = line["wavelength_nm"]

    result = measure_local_region(
        wavelength_nm=wavelength_nm,
        flux=flux,
        uncertainty=uncertainty,
        expected_nm=expected_nm,
        window_nm=WINDOW_NM,
    )

    if result is None:

        print()
        print(
            f"{expected_nm:.5f} nm: "
            "insufficient spectral data"
        )

        continue

    result["relative_intensity"] = (
        line["relative_intensity"]
    )

    result["label"] = (
        line["label"]
    )

    result["lower_level"] = (
        line["lower_level"]
    )

    measurements.append(
        result
    )

    print()

    print(
        f"Expected wavelength: "
        f"{expected_nm:.5f} nm"
    )

    print(
        f"Local peak wavelength: "
        f"{result['peak_nm']:.5f} nm"
    )

    print(
        f"Wavelength offset: "
        f"{result['offset_nm']:+.5f} nm"
    )

    print(
        f"Continuum estimate: "
        f"{result['continuum']:.6g}"
    )

    print(
        f"Peak flux: "
        f"{result['peak_flux']:.6g}"
    )

    print(
        f"Noise estimate: "
        f"{result['noise']:.6g}"
    )

    print(
        f"Approximate peak S/N: "
        f"{result['peak_snr']:.3f}"
    )

    print(
        f"NIST relative intensity: "
        f"{line['relative_intensity']}"
    )

    print(
        f"Lower level: "
        f"{line['lower_level']}"
    )


# ============================================================
# Summary table
# ============================================================

print()
print("=" * 60)
print("Cs II COMPANION-LINE SUMMARY")
print("=" * 60)

print()

print(
    f"{'Expected':>12} "
    f"{'Peak':>12} "
    f"{'Offset':>12} "
    f"{'S/N':>10} "
    f"{'NIST Rel.':>12}"
)

print("-" * 60)


for result in measurements:

    print(
        f"{result['expected_nm']:12.5f} "
        f"{result['peak_nm']:12.5f} "
        f"{result['offset_nm']:+12.5f} "
        f"{result['peak_snr']:10.2f} "
        f"{str(result['relative_intensity']):>12}"
    )


# ============================================================
# Plot each accessible Cs II line
# ============================================================

print()
print(
    "Generating local spectral plots..."
)


for result in measurements:

    local_wavelength = (
        result["local_wavelength"]
    )

    local_flux = (
        result["local_flux"]
    )

    expected_nm = (
        result["expected_nm"]
    )

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        local_wavelength,
        local_flux,
        label="M51 NIRSpec",
    )

    plt.axvline(
        expected_nm,
        linestyle="--",
        label=(
            f"Cs II expected "
            f"{expected_nm:.5f} nm"
        ),
    )

    plt.xlabel(
        "Wavelength (nm)"
    )

    plt.ylabel(
        "Flux"
    )

    plt.title(
        "M51 JWST/NIRSpec — "
        f"Cs II at {expected_nm:.5f} nm"
    )

    plt.legend()

    plt.tight_layout()

    plt.show()


# ============================================================
# Final interpretation reminder
# ============================================================

print()
print("=" * 60)
print("INTERPRETATION")
print("=" * 60)

print()
print(
    "This is an exploratory companion-line test."
)

print(
    "A local peak near a predicted Cs II wavelength "
    "does NOT by itself establish a Cs II identification."
)

print(
    "The next analysis stage should use Gaussian fitting, "
    "local continuum modelling, wavelength calibration, "
    "instrument resolution, and statistical significance."
)

print()
print("=" * 60)
print("TEST COMPLETE")
print("=" * 60)
