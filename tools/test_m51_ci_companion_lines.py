#!/usr/bin/env python3

from pathlib import Path
import csv
import math

import numpy as np
from astropy.io import fits


# ============================================================
# Configuration
# ============================================================

X1D_PATH = Path(
    "data/m51_jwst_level3/"
    "jw03435-o006_t010_nirspec_g140m-f100lp_x1d.fits"
)

CATALOGUE_PATH = Path(
    "data/atomic_lines/"
    "m51_atomic_line_catalogue.csv"
)

OUTPUT_PATH = Path(
    "data/atomic_lines/"
    "m51_ci_companion_line_test.csv"
)

REFERENCE_VELOCITY_KMS = 573.72

OBSERVED_FEATURE_NM = 1284.261304400

C_KMS = 299792.458

RESOLVING_POWER = 916.3

INSTRUMENT_FWHM_NM = (
    OBSERVED_FEATURE_NM / RESOLVING_POWER
)

SEARCH_HALF_WIDTH_NM = (
    INSTRUMENT_FWHM_NM
)

DETECTION_SNR = 3.0

MARGINAL_SNR = 2.0


# ============================================================
# Utility functions
# ============================================================

def to_float(value):
    if value is None:
        return math.nan

    text = str(value).strip()

    if not text:
        return math.nan

    try:
        number = float(text)

        if not math.isfinite(number):
            return math.nan

        return number

    except (ValueError, TypeError):
        return math.nan


def velocity_to_wavelength(
    rest_wavelength_nm,
    velocity_kms,
):
    beta = velocity_kms / C_KMS

    if abs(beta) >= 1.0:
        return math.nan

    return (
        rest_wavelength_nm
        * np.sqrt(
            (1.0 + beta)
            / (1.0 - beta)
        )
    )


# ============================================================
# Header
# ============================================================

print("=" * 70)
print("M51 C I COMPANION-LINE CONSISTENCY TEST")
print("=" * 70)

print()
print("X1D:")
print(f"  {X1D_PATH}")

print()
print("Atomic catalogue:")
print(f"  {CATALOGUE_PATH}")

print()
print(
    "Reference M51 velocity:"
)
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

print()
print(
    "Companion search window:"
)
print(
    f"  ±{SEARCH_HALF_WIDTH_NM:.6f} nm"
)

# ============================================================
# Load X1D spectrum
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

    print()
    print("X1D columns:")

    for name in table.names:
        print(
            f"  {name}"
        )

    wavelength = np.asarray(
        table["WAVELENGTH"],
        dtype=float,
    )

    # JWST X1D wavelengths are stored in microns.
    # Convert to nm for comparison with the atomic catalogue.
    wavelength *= 1000.0

    flux = np.asarray(
        table["FLUX"],
        dtype=float,
    )

    flux_error = np.asarray(
        table["FLUX_ERROR"],
        dtype=float,
    )

print()
print(
    f"Spectral points: "
    f"{len(wavelength)}"
)

print(
    f"Wavelength range: "
    f"{np.nanmin(wavelength):.6f} - "
    f"{np.nanmax(wavelength):.6f} nm"
)


# ============================================================
# Validate X1D
# ============================================================

valid = (
    np.isfinite(wavelength)
    & np.isfinite(flux)
    & np.isfinite(flux_error)
    & (flux_error > 0)
)

print()
print(
    f"Valid flux/error points: "
    f"{np.sum(valid)}"
)

if np.sum(valid) < 10:
    raise RuntimeError(
        "Too few valid X1D flux/error points."
    )

wavelength = wavelength[valid]
flux = flux[valid]
flux_error = flux_error[valid]

order = np.argsort(wavelength)

wavelength = wavelength[order]
flux = flux[order]
flux_error = flux_error[order]


# ============================================================
# Load catalogue
# ============================================================

print()
print("=" * 70)
print("LOADING C I CATALOGUE")
print("=" * 70)

if not CATALOGUE_PATH.exists():
    raise FileNotFoundError(
        CATALOGUE_PATH
    )

with CATALOGUE_PATH.open(
    "r",
    encoding="utf-8",
    newline="",
) as f:

    catalogue_rows = list(
        csv.DictReader(f)
    )

ci_rows = [
    row
    for row in catalogue_rows
    if row.get("species", "").strip()
    == "C I"
]

print()
print(
    f"Total catalogue rows: "
    f"{len(catalogue_rows)}"
)

print(
    f"C I catalogue rows: "
    f"{len(ci_rows)}"
)

if not ci_rows:
    raise RuntimeError(
        "No C I transitions found."
    )


# ============================================================
# Test every C I transition
# ============================================================

results = []

for row_number, row in enumerate(
    ci_rows,
    start=1,
):

    # --------------------------------------------------------
    # Prefer Ritz vacuum wavelength.
    # --------------------------------------------------------

    rest_wavelength = to_float(
        row.get(
            "ritz_wavelength_vacuum_nm"
        )
    )

    wavelength_source = (
        "Ritz vacuum"
    )

    if not math.isfinite(
        rest_wavelength
    ):

        rest_wavelength = to_float(
            row.get(
                "observed_wavelength_vacuum_nm"
            )
        )

        wavelength_source = (
            "Observed vacuum"
        )

    if not math.isfinite(
        rest_wavelength
    ):
        continue

    # --------------------------------------------------------
    # Shift to M51 velocity.
    # --------------------------------------------------------

    predicted = (
        velocity_to_wavelength(
            rest_wavelength,
            REFERENCE_VELOCITY_KMS,
        )
    )

    if not math.isfinite(
        predicted
    ):
        continue

    # --------------------------------------------------------
    # Ignore transitions outside the actual X1D range.
    # --------------------------------------------------------

    if (
        predicted < wavelength.min()
        or predicted > wavelength.max()
    ):
        continue

    # --------------------------------------------------------
    # Local search window.
    # --------------------------------------------------------

    mask = (
        np.abs(
            wavelength - predicted
        )
        <= SEARCH_HALF_WIDTH_NM
    )

    if np.sum(mask) < 2:
        continue

    local_wavelength = (
        wavelength[mask]
    )

    local_flux = (
        flux[mask]
    )

    local_error = (
        flux_error[mask]
    )

    # --------------------------------------------------------
    # Local continuum.
    #
    # Use a linear continuum from the endpoints.
    # --------------------------------------------------------

    x = (
        local_wavelength
        - predicted
    )

    if len(local_wavelength) >= 3:

        edge_count = max(
            1,
            len(local_wavelength) // 4,
        )

        edge_indices = np.concatenate(
            [
                np.arange(
                    edge_count
                ),
                np.arange(
                    len(local_wavelength)
                    - edge_count,
                    len(local_wavelength),
                ),
            ]
        )

        try:

            continuum_coefficients = (
                np.polyfit(
                    x[edge_indices],
                    local_flux[
                        edge_indices
                    ],
                    1,
                    w=1.0
                    / local_error[
                        edge_indices
                    ],
                )
            )

            continuum = np.polyval(
                continuum_coefficients,
                x,
            )

        except (
            np.linalg.LinAlgError,
            ValueError,
        ):

            continuum = np.full(
                len(local_flux),
                np.median(
                    local_flux
                ),
            )

    else:

        continuum = np.full(
            len(local_flux),
            np.median(
                local_flux
            ),
        )

    residual = (
        local_flux - continuum
    )

    # --------------------------------------------------------
    # Individual-pixel S/N.
    # --------------------------------------------------------

    snr = (
        residual / local_error
    )

    # --------------------------------------------------------
    # Peak residual.
    # --------------------------------------------------------

    peak_index = int(
        np.argmax(
            residual / local_error
        )
    )

    peak_wavelength = float(
        local_wavelength[
            peak_index
        ]
    )

    peak_residual = float(
        residual[
            peak_index
        ]
    )

    peak_error = float(
        local_error[
            peak_index
        ]
    )

    peak_snr = (
        peak_residual
        / peak_error
    )

    # --------------------------------------------------------
    # Integrated weighted signal.
    #
    # Approximate line significance by summing the
    # continuum-subtracted flux over the local window.
    # --------------------------------------------------------

    positive_residual = np.maximum(
        residual,
        0.0,
    )

    if len(local_wavelength) >= 2:

        widths = np.gradient(
            local_wavelength
        )

        integrated_flux = float(
            np.sum(
                positive_residual
                * widths
            )
        )

        integrated_variance = float(
            np.sum(
                (
                    local_error
                    * widths
                ) ** 2
            )
        )

    else:

        integrated_flux = float(
            positive_residual[0]
        )

        integrated_variance = float(
            local_error[0] ** 2
        )

    if (
        integrated_variance > 0
        and math.isfinite(
            integrated_variance
        )
    ):

        integrated_error = math.sqrt(
            integrated_variance
        )

        integrated_snr = (
            integrated_flux
            / integrated_error
        )

    else:

        integrated_error = math.nan
        integrated_snr = math.nan

    # --------------------------------------------------------
    # Classification.
    # --------------------------------------------------------

    if peak_snr >= DETECTION_SNR:

        classification = (
            "detected"
        )

    elif peak_snr >= MARGINAL_SNR:

        classification = (
            "marginal"
        )

    else:

        classification = (
            "not_detected"
        )

    # --------------------------------------------------------
    # Catalogue metadata.
    # --------------------------------------------------------

    result = {

        "catalogue_row":
            row_number,

        "species":
            "C I",

        "wavelength_source":
            wavelength_source,

        "rest_wavelength_vacuum_nm":
            rest_wavelength,

        "predicted_wavelength_at_M51_nm":
            predicted,

        "reference_velocity_kms":
            REFERENCE_VELOCITY_KMS,

        "search_half_width_nm":
            SEARCH_HALF_WIDTH_NM,

        "peak_wavelength_nm":
            peak_wavelength,

        "peak_residual":
            peak_residual,

        "peak_error":
            peak_error,

        "peak_snr":
            peak_snr,

        "integrated_flux":
            integrated_flux,

        "integrated_error":
            integrated_error,

        "integrated_snr":
            integrated_snr,

        "relative_intensity":
            to_float(
                row.get(
                    "relative_intensity"
                )
            ),

        "Aki_s-1":
            to_float(
                row.get(
                    "Aki_s-1"
                )
            ),

        "lower_energy_cm-1":
            to_float(
                row.get(
                    "lower_energy_cm-1"
                )
            ),

        "upper_energy_cm-1":
            to_float(
                row.get(
                    "upper_energy_cm-1"
                )
            ),

        "lower_configuration":
            row.get(
                "lower_configuration",
                "",
            ),

        "lower_term":
            row.get(
                "lower_term",
                "",
            ),

        "lower_J":
            row.get(
                "lower_J",
                "",
            ),

        "upper_configuration":
            row.get(
                "upper_configuration",
                "",
            ),

        "upper_term":
            row.get(
                "upper_term",
                "",
            ),

        "upper_J":
            row.get(
                "upper_J",
                "",
            ),

        "accuracy":
            row.get(
                "accuracy",
                "",
            ),

        "line_reference":
            row.get(
                "line_reference",
                "",
            ),

        "classification":
            classification,
    }

    results.append(
        result
    )


# ============================================================
# Sort by predicted wavelength
# ============================================================

results.sort(
    key=lambda r:
        r[
            "predicted_wavelength_at_M51_nm"
        ]
)


# ============================================================
# Save CSV
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

if results:

    fieldnames = list(
        results[0].keys()
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

        writer.writerows(
            results
        )


# ============================================================
# Display results
# ============================================================

print()
print("=" * 70)
print("C I COMPANION-LINE RESULTS")
print("=" * 70)

print()

print(
    f"{'Rest':>12s} "
    f"{'Predicted':>12s} "
    f"{'Peak':>12s} "
    f"{'Peak S/N':>10s} "
    f"{'Int S/N':>10s} "
    f"{'Aki':>12s} "
    f"{'Result':>15s}"
)

print("-" * 100)

for result in results:

    aki = result[
        "Aki_s-1"
    ]

    if math.isfinite(aki):

        aki_text = (
            f"{aki:.3g}"
        )

    else:

        aki_text = "nan"

    print(
        f"{result['rest_wavelength_vacuum_nm']:12.6f} "
        f"{result['predicted_wavelength_at_M51_nm']:12.6f} "
        f"{result['peak_wavelength_nm']:12.6f} "
        f"{result['peak_snr']:10.2f} "
        f"{result['integrated_snr']:10.2f} "
        f"{aki_text:>12s} "
        f"{result['classification']:>15s}"
    )


# ============================================================
# Summary
# ============================================================

detected = [
    r for r in results
    if r["classification"]
    == "detected"
]

marginal = [
    r for r in results
    if r["classification"]
    == "marginal"
]

not_detected = [
    r for r in results
    if r["classification"]
    == "not_detected"
]

print()
print("=" * 70)
print("COMPANION-LINE SUMMARY")
print("=" * 70)

print()
print(
    f"C I catalogue rows: "
    f"{len(ci_rows)}"
)

print(
    f"Usable C I transitions: "
    f"{len(results)}"
)

print(
    f"Detected: "
    f"{len(detected)}"
)

print(
    f"Marginal: "
    f"{len(marginal)}"
)

print(
    f"Not detected: "
    f"{len(not_detected)}"
)


# ============================================================
# Strongest candidate companions
# ============================================================

if results:

    strongest = sorted(
        results,
        key=lambda r:
            r["peak_snr"],
        reverse=True,
    )

    print()
    print(
        "Strongest C I companion candidates:"
    )

    print()

    for result in strongest[:10]:

        print(
            f"  "
            f"{result['predicted_wavelength_at_M51_nm']:.6f} nm"
            f"  peak S/N = "
            f"{result['peak_snr']:.2f}"
            f"  "
            f"Aki = "
            f"{result['Aki_s-1']:.3g}"
            if math.isfinite(
                result["Aki_s-1"]
            )
            else
            f"  "
            f"{result['predicted_wavelength_at_M51_nm']:.6f} nm"
            f"  peak S/N = "
            f"{result['peak_snr']:.2f}"
        )


# ============================================================
# Explicit 1284-nm C I candidates
# ============================================================

near_1284 = [
    r for r in results
    if abs(
        r[
            "predicted_wavelength_at_M51_nm"
        ]
        - OBSERVED_FEATURE_NM
    )
    <= INSTRUMENT_FWHM_NM
]

print()
print("=" * 70)
print("C I TRANSITIONS WITHIN ONE FWHM OF 1284.2613 NM")
print("=" * 70)

print()

for result in near_1284:

    print(
        f"{result['rest_wavelength_vacuum_nm']:.6f} nm"
        f" -> "
        f"{result['predicted_wavelength_at_M51_nm']:.6f} nm"
        f" | peak S/N = "
        f"{result['peak_snr']:.2f}"
        f" | integrated S/N = "
        f"{result['integrated_snr']:.2f}"
        f" | Aki = "
        f"{result['Aki_s-1']:.3g}"
        if math.isfinite(
            result["Aki_s-1"]
        )
        else
        f"{result['rest_wavelength_vacuum_nm']:.6f} nm"
        f" -> "
        f"{result['predicted_wavelength_at_M51_nm']:.6f} nm"
        f" | peak S/N = "
        f"{result['peak_snr']:.2f}"
    )


# ============================================================
# Final interpretation
# ============================================================

print()
print("=" * 70)
print("INTERPRETATION")
print("=" * 70)

print()

if len(results) == 0:

    print(
        "No usable C I transitions were found."
    )

    print(
        "This indicates a catalogue/X1D selection "
        "problem and should NOT be interpreted as "
        "evidence against C I."
    )

else:

    print(
        "The C I catalogue was successfully matched "
        "to the M51 X1D spectral range."
    )

    print()
    print(
        "Companion-line detections are evaluated using "
        "the actual JWST X1D FLUX_ERROR values."
    )

    print()
    print(
        "A positive companion detection is evidence "
        "for C I being present in the spectrum, but "
        "does not by itself establish that C I causes "
        "the 1284.2613 nm feature."
    )

    print()
    print(
        "Conversely, absence of detectable companions "
        "does not prove C I is absent because excitation, "
        "line strength, blending, extinction, and "
        "instrumental sensitivity must be considered."
    )

print()
print(
    "Results saved to:"
)

print(
    f"  {OUTPUT_PATH}"
)

print()
print("=" * 70)
print("C I COMPANION-LINE TEST COMPLETE")
print("=" * 70)
