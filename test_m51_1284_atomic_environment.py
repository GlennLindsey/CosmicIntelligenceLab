from pathlib import Path
import html
import re
import time

import numpy as np
import pandas as pd
import requests
from astropy.io import fits


# ======================================================================
# M51 1284 NM ATOMIC-LINE ENVIRONMENT
# NIST ASD SPECIES-BY-SPECIES LINE SEARCH
# ======================================================================

print("=" * 70)
print("M51 1284 NM ATOMIC-LINE ENVIRONMENT")
print("NIST ASD SPECIES-BY-SPECIES LINE SEARCH")
print("=" * 70)


# ======================================================================
# CONFIGURATION
# ======================================================================

X1D_PATH = Path(
    "data/m51_jwst_level3/"
    "jw03435-o006_t010_nirspec_g140m-f100lp_x1d.fits"
)

OBSERVED_WAVELENGTH_NM = 1284.26130440

REFERENCE_VELOCITY_KMS = 573.72

C_KMS = 299792.458

SEARCH_LOW_NM = 1278.0
SEARCH_HIGH_NM = 1290.0

NIRSPEC_R = 916.3

INSTRUMENT_FWHM_NM = (
    OBSERVED_WAVELENGTH_NM
    / NIRSPEC_R
)


# ======================================================================
# ATOMIC SPECIES TO SEARCH
# ======================================================================

SPECTRA = [
    "H I",
    "He I",
    "He II",
    "C I",
    "C II",
    "C III",
    "C IV",
    "N I",
    "N II",
    "N III",
    "O I",
    "O II",
    "O III",
    "Ne I",
    "Ne II",
    "Ne III",
    "Na I",
    "Na II",
    "Mg I",
    "Mg II",
    "Al I",
    "Al II",
    "Al III",
    "Si I",
    "Si II",
    "Si III",
    "Si IV",
    "S I",
    "S II",
    "S III",
    "S IV",
    "Ar I",
    "Ar II",
    "Ar III",
    "Ca II",
    "Fe I",
    "Fe II",
    "Fe III",
    "Co II",
    "Ni I",
    "Ni II",
    "Cu II",
    "Zn II",
    "Cs II",
]


# ======================================================================
# UTILITY FUNCTIONS
# ======================================================================

def velocity_to_wavelength(
    rest_nm,
    velocity_kms,
):
    """
    Relativistic Doppler conversion.
    """

    beta = (
        velocity_kms
        /
        C_KMS
    )

    return (
        rest_nm
        *
        np.sqrt(
            (1.0 + beta)
            /
            (1.0 - beta)
        )
    )


def wavelength_to_velocity(
    observed_nm,
    rest_nm,
):
    """
    Relativistic wavelength -> velocity.
    """

    ratio = (
        observed_nm
        /
        rest_nm
    )

    beta = (
        ratio**2 - 1.0
    ) / (
        ratio**2 + 1.0
    )

    return (
        beta
        *
        C_KMS
    )


def air_to_vacuum_nm(
    wavelength_nm,
):
    """
    Convert air wavelength to vacuum wavelength.

    This is the standard Edlen-type refractive-index
    conversion appropriate to the optical/NIR range.
    """

    wavelength_nm = np.asarray(
        wavelength_nm,
        dtype=float,
    )

    wavelength_um = (
        wavelength_nm
        /
        1000.0
    )

    sigma2 = (
        1.0
        /
        wavelength_um
    ) ** 2

    n_minus_1 = (
        (
            5792105.0
            /
            (
                238.0185
                -
                sigma2
            )
        )
        +
        (
            167917.0
            /
            (
                57.362
                -
                sigma2
            )
        )
    ) * 1.0e-8

    return (
        wavelength_nm
        *
        (
            1.0
            +
            n_minus_1
        )
    )


def clean_species_name(
    species,
):
    """
    Convert a species name into a safe filename.
    """

    return re.sub(
        r"[^A-Za-z0-9]+",
        "_",
        species,
    ).strip("_")


# ======================================================================
# LOAD X1D SPECTRUM
# ======================================================================

print()
print("=" * 70)
print("M51 X1D SPECTRUM")
print("=" * 70)

if not X1D_PATH.exists():

    raise FileNotFoundError(
        f"X1D file not found:\n{X1D_PATH}"
    )


with fits.open(
    X1D_PATH,
    memmap=False,
) as hdul:

    print()
    print("HDU structure:")

    for i, hdu in enumerate(hdul):

        print(
            f"  HDU {i}: "
            f"{hdu.name:12s} "
            f"{getattr(hdu, 'shape', None)}"
        )

    table = hdul[1].data

    if table is None:

        raise RuntimeError(
            "X1D HDU 1 contains no table data."
        )

    columns = list(
        table.names
    )

    print()
    print("X1D columns:")

    for column in columns:

        print(
            f"  {column}"
        )

    required_columns = [
        "WAVELENGTH",
        "FLUX",
        "FLUX_ERROR",
    ]

    for column in required_columns:

        if column not in columns:

            raise RuntimeError(
                f"Required X1D column "
                f"{column} not found."
            )

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


# ======================================================================
# VALIDATE X1D
# ======================================================================

wavelength_nm = (
    wavelength_um
    *
    1000.0
)

valid = (
    np.isfinite(wavelength_nm)
    &
    np.isfinite(flux)
)

wavelength_nm = wavelength_nm[
    valid
]

flux = flux[
    valid
]

flux_error = flux_error[
    valid
]

if len(wavelength_nm) == 0:

    raise RuntimeError(
        "No valid X1D spectral points."
    )

if not np.all(
    np.diff(wavelength_nm) > 0
):

    raise RuntimeError(
        "X1D wavelength array is not "
        "strictly increasing."
    )


print()
print(
    f"Number of spectral points: "
    f"{len(wavelength_nm)}"
)

print(
    f"Wavelength range: "
    f"{wavelength_nm.min():.3f} - "
    f"{wavelength_nm.max():.3f} nm"
)

print(
    f"Observed feature: "
    f"{OBSERVED_WAVELENGTH_NM:.9f} nm"
)

print(
    "X1D wavelength-axis validation: PASSED"
)


# ======================================================================
# LOCAL SPECTRUM
# ======================================================================

local_mask = (
    (wavelength_nm >= SEARCH_LOW_NM)
    &
    (wavelength_nm <= SEARCH_HIGH_NM)
)

local_wavelength = (
    wavelength_nm[
        local_mask
    ]
)

local_flux = (
    flux[
        local_mask
    ]
)

local_error = (
    flux_error[
        local_mask
    ]
)

print()
print("=" * 70)
print("M51 SPECTRUM AROUND 1284 NM")
print("=" * 70)

print(
    f"Window: "
    f"{SEARCH_LOW_NM:.3f} - "
    f"{SEARCH_HIGH_NM:.3f} nm"
)

print(
    f"Spectral points: "
    f"{len(local_wavelength)}"
)

if len(local_wavelength) == 0:

    raise RuntimeError(
        "No X1D points in atomic-line search window."
    )

print(
    f"Actual X1D range: "
    f"{local_wavelength.min():.6f} - "
    f"{local_wavelength.max():.6f} nm"
)


# ======================================================================
# INSTRUMENT RESOLUTION
# ======================================================================

print()
print("=" * 70)
print("NIRSPEC INSTRUMENT RESOLUTION")
print("=" * 70)

print(
    f"Resolving power: "
    f"R = {NIRSPEC_R:.1f}"
)

print(
    f"Instrument FWHM: "
    f"{INSTRUMENT_FWHM_NM:.6f} nm"
)

print(
    f"Half FWHM: "
    f"{INSTRUMENT_FWHM_NM / 2.0:.6f} nm"
)


# ======================================================================
# NIST QUERY
# ======================================================================

NIST_URL = (
    "https://physics.nist.gov/"
    "cgi-bin/ASD/lines1.pl"
)


def query_nist_species(
    species,
):
    """
    Query one NIST ASD ion at a time.

    A failed species query does not terminate the experiment.
    """

    params = {
        "spectra": species,
        "low_w": f"{SEARCH_LOW_NM:.3f}",
        "upp_w": f"{SEARCH_HIGH_NM:.3f}",
        "unit": "1",
        "format": "1",
        "show_obs_wl": "1",
        "show_calc_wl": "1",
        "submit": "Retrieve Data",
    }

    try:

        response = requests.get(
            NIST_URL,
            params=params,
            timeout=60,
        )

        return (
            response.status_code,
            response.text,
        )

    except Exception as exc:

        return (
            None,
            str(exc),
        )


# ======================================================================
# NIST HTML TABLE PARSER
# ======================================================================

def parse_nist_html(
    text,
):
    """
    Extract HTML tables returned by NIST.

    NIST's line service returns HTML rather than a simple
    CSV response. We therefore use pandas.read_html().
    """

    try:

        tables = pd.read_html(
            text
        )

    except ValueError:

        return []

    return tables


def identify_wavelength_column(
    table,
):
    """
    Find the NIST observed wavelength column.

    Returns None if it cannot be identified.
    """

    candidates = []

    for column in table.columns:

        name = str(
            column
        ).upper()

        if (
            "OBS"
            in name
            and
            "WAVELENGTH"
            in name
        ):

            candidates.append(
                column
            )

        elif (
            "OBS"
            in name
            and
            "WAVE"
            in name
        ):

            candidates.append(
                column
            )

    if candidates:

        return candidates[0]

    return None


# ======================================================================
# OUTPUT DIRECTORIES
# ======================================================================

raw_directory = Path(
    "m51_nist_raw"
)

raw_directory.mkdir(
    exist_ok=True
)


# ======================================================================
# SEARCH ALL SPECIES
# ======================================================================

print()
print("=" * 70)
print("NIST ASD SPECIES SEARCH")
print("=" * 70)

print(
    f"Species requested: "
    f"{len(SPECTRA)}"
)

print(
    f"Wavelength interval: "
    f"{SEARCH_LOW_NM:.3f} - "
    f"{SEARCH_HIGH_NM:.3f} nm"
)

print()


successful_species = []

failed_species = []

candidate_tables = []


for index, species in enumerate(
    SPECTRA,
    start=1,
):

    print(
        f"[{index:02d}/{len(SPECTRA):02d}] "
        f"{species:8s}",
        end=" ",
        flush=True,
    )

    status, response_text = (
        query_nist_species(
            species
        )
    )

    safe_name = (
        clean_species_name(
            species
        )
    )

    raw_path = (
        raw_directory
        /
        f"{safe_name}.html"
    )

    raw_path.write_text(
        response_text
    )

    if status != 200:

        print(
            f"FAILED "
            f"(HTTP {status})"
        )

        failed_species.append(
            (
                species,
                status,
                "HTTP error",
            )
        )

        time.sleep(
            0.5
        )

        continue


    # --------------------------------------------------------------
    # Detect NIST error pages.
    # --------------------------------------------------------------

    lower_text = (
        response_text.lower()
    )

    if (
        "software error"
        in lower_text
        or
        "input error"
        in lower_text
        or
        "error message"
        in lower_text
    ):

        print(
            "FAILED "
            "(NIST error page)"
        )

        failed_species.append(
            (
                species,
                status,
                "NIST error page",
            )
        )

        time.sleep(
            0.5
        )

        continue


    # --------------------------------------------------------------
    # Parse returned HTML tables.
    # --------------------------------------------------------------

    tables = parse_nist_html(
        response_text
    )

    if not tables:

        print(
            "FAILED "
            "(no HTML table)"
        )

        failed_species.append(
            (
                species,
                status,
                "No HTML table",
            )
        )

        time.sleep(
            0.5
        )

        continue


    found_lines = 0

    for table in tables:

        wavelength_column = (
            identify_wavelength_column(
                table
            )
        )

        if wavelength_column is None:

            continue


        wavelength_values = pd.to_numeric(
            table[
                wavelength_column
            ],
            errors="coerce",
        )


        valid_rows = (
            wavelength_values.notna()
        )

        if not valid_rows.any():

            continue


        selected = table.loc[
            valid_rows
        ].copy()

        selected[
            "NIST_OBSERVED_AIR_NM"
        ] = wavelength_values.loc[
            valid_rows
        ].to_numpy()


        # Keep only requested wavelength range.

        selected = selected[
            (
                selected[
                    "NIST_OBSERVED_AIR_NM"
                ]
                >=
                SEARCH_LOW_NM
            )
            &
            (
                selected[
                    "NIST_OBSERVED_AIR_NM"
                ]
                <=
                SEARCH_HIGH_NM
            )
        ]

        if selected.empty:

            continue


        selected[
            "SPECIES"
        ] = species

        candidate_tables.append(
            selected
        )

        found_lines += (
            len(selected)
        )


    if found_lines > 0:

        print(
            f"OK "
            f"({found_lines} lines)"
        )

        successful_species.append(
            species
        )

    else:

        print(
            "OK "
            "(no lines in interval)"
        )

        successful_species.append(
            species
        )


    time.sleep(
        0.75
    )


# ======================================================================
# COMBINE NIST RESULTS
# ======================================================================

print()
print("=" * 70)
print("NIST SEARCH SUMMARY")
print("=" * 70)

print(
    f"Successful species queries: "
    f"{len(successful_species)}"
)

print(
    f"Failed species queries: "
    f"{len(failed_species)}"
)

if failed_species:

    print()
    print(
        "Failed species:"
    )

    for (
        species,
        status,
        reason,
    ) in failed_species:

        print(
            f"  {species:8s} "
            f"{reason}"
        )


if not candidate_tables:

    print()
    print(
        "No usable NIST line tables were returned."
    )

    print(
        "Raw responses are available in:"
    )

    print(
        f"  {raw_directory}/"
    )

    raise RuntimeError(
        "NIST search produced no candidate lines."
    )


nist = pd.concat(
    candidate_tables,
    ignore_index=True,
)


# ======================================================================
# REMOVE DUPLICATE LINES
# ======================================================================

if (
    "SPECIES"
    in nist.columns
    and
    "NIST_OBSERVED_AIR_NM"
    in nist.columns
):

    nist = (
        nist
        .drop_duplicates(
            subset=[
                "SPECIES",
                "NIST_OBSERVED_AIR_NM",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ======================================================================
# AIR -> VACUUM
# ======================================================================

nist[
    "NIST_VACUUM_NM"
] = air_to_vacuum_nm(
    nist[
        "NIST_OBSERVED_AIR_NM"
    ].to_numpy()
)


# ======================================================================
# PREDICTED M51 WAVELENGTH
# ======================================================================

nist[
    "PREDICTED_M51_NM"
] = velocity_to_wavelength(
    nist[
        "NIST_VACUUM_NM"
    ].to_numpy(),
    REFERENCE_VELOCITY_KMS,
)


# ======================================================================
# WAVELENGTH DIFFERENCE
# ======================================================================

nist[
    "DELTA_NM"
] = (
    nist[
        "PREDICTED_M51_NM"
    ]
    -
    OBSERVED_WAVELENGTH_NM
)


nist[
    "ABS_DELTA_NM"
] = np.abs(
    nist[
        "DELTA_NM"
    ]
)


# ======================================================================
# REQUIRED VELOCITY
# ======================================================================

nist[
    "REQUIRED_VELOCITY_KMS"
] = wavelength_to_velocity(
    OBSERVED_WAVELENGTH_NM,
    nist[
        "NIST_VACUUM_NM"
    ].to_numpy(),
)


nist[
    "VELOCITY_OFFSET_KMS"
] = (
    nist[
        "REQUIRED_VELOCITY_KMS"
    ]
    -
    REFERENCE_VELOCITY_KMS
)


# ======================================================================
# RESOLUTION COMPARISON
# ======================================================================

nist[
    "RESOLUTION_UNITS"
] = (
    nist[
        "ABS_DELTA_NM"
    ]
    /
    INSTRUMENT_FWHM_NM
)


nist[
    "WITHIN_HALF_FWHM"
] = (
    nist[
        "ABS_DELTA_NM"
    ]
    <=
    INSTRUMENT_FWHM_NM
    /
    2.0
)


nist[
    "WITHIN_ONE_FWHM"
] = (
    nist[
        "ABS_DELTA_NM"
    ]
    <=
    INSTRUMENT_FWHM_NM
)


nist[
    "WITHIN_TWO_FWHM"
] = (
    nist[
        "ABS_DELTA_NM"
    ]
    <=
    2.0
    *
    INSTRUMENT_FWHM_NM
)


# ======================================================================
# SORT BY WAVELENGTH AGREEMENT
# ======================================================================

nist = (
    nist
    .sort_values(
        "ABS_DELTA_NM"
    )
    .reset_index(
        drop=True
    )
)


# ======================================================================
# SAVE COMPLETE TABLE
# ======================================================================

output_path = Path(
    "m51_1284_atomic_candidates.csv"
)

nist.to_csv(
    output_path,
    index=False,
)


# ======================================================================
# DISPLAY RESULTS
# ======================================================================

display_columns = [
    "SPECIES",
    "NIST_OBSERVED_AIR_NM",
    "NIST_VACUUM_NM",
    "PREDICTED_M51_NM",
    "DELTA_NM",
    "REQUIRED_VELOCITY_KMS",
    "VELOCITY_OFFSET_KMS",
    "RESOLUTION_UNITS",
]


print()
print("=" * 70)
print("CLOSEST ATOMIC TRANSITIONS")
print("=" * 70)

print(
    nist[
        display_columns
    ]
    .head(50)
    .to_string(
        index=False,
        float_format=lambda x:
            f"{x:.6f}",
    )
)


# ======================================================================
# WITHIN ONE FWHM
# ======================================================================

one_fwhm = nist[
    nist[
        "WITHIN_ONE_FWHM"
    ]
]

print()
print("=" * 70)
print("CANDIDATES WITHIN ONE NIRSPEC FWHM")
print("=" * 70)

print(
    f"Number of candidates: "
    f"{len(one_fwhm)}"
)

if not one_fwhm.empty:

    print(
        one_fwhm[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

else:

    print(
        "None."
    )


# ======================================================================
# WITHIN HALF FWHM
# ======================================================================

half_fwhm = nist[
    nist[
        "WITHIN_HALF_FWHM"
    ]
]

print()
print("=" * 70)
print("CANDIDATES WITHIN HALF A NIRSPEC FWHM")
print("=" * 70)

print(
    f"Number of candidates: "
    f"{len(half_fwhm)}"
)

if not half_fwhm.empty:

    print(
        half_fwhm[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

else:

    print(
        "None."
    )


# ======================================================================
# CS II CHECK
# ======================================================================

print()
print("=" * 70)
print("Cs II CHECK")
print("=" * 70)

csii = nist[
    nist[
        "SPECIES"
    ]
    ==
    "Cs II"
]

if csii.empty:

    print(
        "No Cs II line was returned by NIST "
        "in the requested wavelength interval."
    )

    print(
        "This is a database-query result only and "
        "must not be interpreted as evidence that "
        "Cs II is physically absent."
    )

else:

    print(
        csii[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.9f}",
        )
    )


# ======================================================================
# FINAL SUMMARY
# ======================================================================

print()
print("=" * 70)
print("ATOMIC ENVIRONMENT SEARCH COMPLETE")
print("=" * 70)

print(
    f"Observed feature:"
)

print(
    f"  {OBSERVED_WAVELENGTH_NM:.9f} nm"
)

print(
    f"Reference M51 velocity:"
)

print(
    f"  {REFERENCE_VELOCITY_KMS:.2f} km/s"
)

print(
    f"NIRSpec resolving power:"
)

print(
    f"  R = {NIRSPEC_R:.1f}"
)

print(
    f"Instrument FWHM:"
)

print(
    f"  {INSTRUMENT_FWHM_NM:.6f} nm"
)

print(
    f"Total candidate transitions:"
)

print(
    f"  {len(nist)}"
)

print(
    f"Within 1 FWHM:"
)

print(
    f"  {len(one_fwhm)}"
)

print(
    f"Within 0.5 FWHM:"
)

print(
    f"  {len(half_fwhm)}"
)

print()
print(
    "Candidate table:"
)

print(
    f"  {output_path}"
)

print(
    "Raw NIST responses:"
)

print(
    f"  {raw_directory}/"
)

print()
print(
    "IMPORTANT:"
)

print(
    "This experiment identifies wavelength-compatible "
    "atomic transitions only."
)

print(
    "A wavelength match does not establish an atomic "
    "identification. Transition probabilities, excitation "
    "conditions, companion lines, spatial morphology, "
    "and kinematics must also be considered."
)

print("=" * 70)