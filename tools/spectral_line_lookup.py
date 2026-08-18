import re

import requests
from bs4 import BeautifulSoup

NIST_ASD_URL = "https://physics.nist.gov/cgi-bin/ASD/lines1.pl"


# ============================================================
# Basic helpers
# ============================================================


def _clean_text(value):
    """
    Normalize whitespace in a NIST table cell.

    Blank cells are returned as None.
    """

    if value is None:
        return None

    value = value.replace("\xa0", " ")
    value = re.sub(r"\s+", " ", value)
    value = value.strip()

    return value if value else None


def _to_float(value):
    """
    Convert a NIST numeric value to float.

    NIST sometimes inserts spaces as thousands separators,
    for example:

        1 283.78933
        15 490.077832

    Those spaces are removed before conversion.

    Returns None when the value is blank or cannot
    be interpreted as a number.
    """

    if value is None:
        return None

    value = value.strip()

    if not value:
        return None

    # Remove common NIST annotation characters.
    value = value.replace("*", "")
    value = value.replace(":", "")

    # NIST uses spaces as thousands separators.
    value = value.replace(" ", "")

    try:
        return float(value)

    except (ValueError, TypeError):
        return None


# ============================================================
# NIST species extraction
# ============================================================


def _roman_numeral(value):
    """
    Convert an integer ionization-stage value to a Roman numeral.

    NIST uses:

        spectr_charge=1  -> I
        spectr_charge=2  -> II
        spectr_charge=3  -> III
        ...

    Parameters
    ----------
    value : str or int
        Ionization stage supplied by NIST.

    Returns
    -------
    str or None
        Roman numeral representation, or None if invalid.
    """

    try:
        number = int(value)
    except (TypeError, ValueError):
        return None

    if number <= 0:
        return None

    roman_values = (
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    )

    result = []

    for arabic, roman in roman_values:
        count, number = divmod(number, arabic)
        result.append(roman * count)

    return "".join(result)


def _extract_species_from_row(row):
    """
    Extract the element and ionization stage from NIST HTML.

    NIST embeds the species information in the JavaScript
    associated with the line-reference link.

    Example:

        ...element=Th&spectr_charge=1...

    corresponds to:

        Th I

    Returns
    -------
    str or None
        Species/ion label such as "Th I".
    """

    # --------------------------------------------------------
    # Look through links in the row.
    # --------------------------------------------------------

    for link in row.find_all("a"):

        onclick = link.get("onclick", "")

        if not onclick:
            continue

        element_match = re.search(
            r"element=([^&'\"]+)",
            onclick,
        )

        charge_match = re.search(
            r"spectr_charge=(\d+)",
            onclick,
        )

        if not element_match or not charge_match:
            continue

        element = element_match.group(1)

        charge = _roman_numeral(charge_match.group(1))

        if charge is None:
            return element

        return f"{element} {charge}"

    # --------------------------------------------------------
    # Some NIST rows may expose the species through
    # mouse-over text instead of onclick.
    # --------------------------------------------------------

    for link in row.find_all("a"):

        status_text = link.get(
            "onmouseover",
            "",
        )

        if not status_text:
            continue

        match = re.search(
            r"for ([A-Z][a-z]?) ([IVX]+)",
            status_text,
        )

        if match:

            return f"{match.group(1)} " f"{match.group(2)}"

    return None


# ============================================================
# HTML extraction
# ============================================================


def _extract_row_cells(row):
    """
    Extract normalized text from a NIST result row.
    """

    cells = row.find_all(["td", "th"])

    return [
        _clean_text(
            cell.get_text(
                " ",
                strip=True,
            )
        )
        for cell in cells
    ]


# ============================================================
# NIST row parser
# ============================================================


def _parse_standard_row(
    values,
    species=None,
):
    """
    Parse a NIST ASD spectral-line data row.

    The actual NIST HTML data rows contain 17 cells:

        0   Observed wavelength
        1   Observed uncertainty
        2   Ritz wavelength
        3   Ritz uncertainty
        4   Relative intensity
        5   Aki
        6   Accuracy
        7   Lower energy
        8   separator
        9   Upper energy
        10  Lower configuration
        11  Lower term
        12  Upper configuration
        13  Upper term
        14  Type
        15  TP reference
        16  Line reference

    The species/ion is supplied separately from the HTML
    associated with the row's bibliographic reference.
    """

    values = list(values)

    raw_values = list(values)

    # --------------------------------------------------------
    # Guarantee enough positions.
    # --------------------------------------------------------

    while len(values) < 17:

        values.append(None)

    # --------------------------------------------------------
    # Parse wavelength/intensity fields.
    # --------------------------------------------------------

    observed_wavelength = _to_float(values[0])

    observed_uncertainty = _to_float(values[1])

    ritz_wavelength = _to_float(values[2])

    ritz_uncertainty = _to_float(values[3])

    relative_intensity = _to_float(values[4])

    transition_probability = _to_float(values[5])

    accuracy = values[6]

    lower_energy = _to_float(values[7])

    upper_energy = _to_float(values[9])

    # --------------------------------------------------------
    # Atomic level information.
    # --------------------------------------------------------

    lower_configuration = values[10]

    lower_term = values[11]

    upper_configuration = values[12]

    upper_term = values[13]

    # --------------------------------------------------------
    # Reference/type fields.
    # --------------------------------------------------------

    transition_type = values[14]

    transition_probability_reference = values[15]

    line_reference = values[16]

    return {
        "species": species,
        "observed_wavelength_nm": (observed_wavelength),
        "observed_uncertainty_nm": (observed_uncertainty),
        "ritz_wavelength_nm": (ritz_wavelength),
        "ritz_uncertainty_nm": (ritz_uncertainty),
        "relative_intensity": (relative_intensity),
        "transition_probability_s-1": (transition_probability),
        "accuracy": accuracy,
        "lower_energy_cm-1": (lower_energy),
        "upper_energy_cm-1": (upper_energy),
        "lower_configuration": (lower_configuration),
        "lower_term": lower_term,
        "upper_configuration": (upper_configuration),
        "upper_term": upper_term,
        "transition_type": (transition_type),
        "transition_probability_reference": (transition_probability_reference),
        "line_reference": line_reference,
        # Preserve everything NIST supplied.
        "raw_values": raw_values,
    }


# ============================================================
# Main NIST lookup
# ============================================================


def spectral_line_lookup(
    wavelength_nm,
    tolerance_nm=0.5,
):
    """
    Search the NIST Atomic Spectra Database for atomic
    transitions near a supplied wavelength.

    Parameters
    ----------
    wavelength_nm : float
        Reference wavelength in nanometers.

    tolerance_nm : float
        Search half-width in nanometers.

    Returns
    -------
    dict
        Structured NIST spectral-line evidence.

    Important
    ---------
    This function retrieves documented atomic transitions.

    It does NOT identify the physical origin of an
    astronomical spectral feature.

    NIST's default wavelength convention is:

        vacuum below 200 nm
        air from 200-2000 nm
        vacuum above 2000 nm

    Therefore a wavelength near 1284 nm is normally reported
    by NIST as an AIR wavelength.
    """

    # --------------------------------------------------------
    # Validate input.
    # --------------------------------------------------------

    if wavelength_nm <= 0:

        raise ValueError("wavelength_nm must be positive.")

    if tolerance_nm <= 0:

        raise ValueError("tolerance_nm must be positive.")

    # --------------------------------------------------------
    # Search interval.
    # --------------------------------------------------------

    lower_nm = wavelength_nm - tolerance_nm

    upper_nm = wavelength_nm + tolerance_nm

    # --------------------------------------------------------
    # NIST ASD query parameters.
    # --------------------------------------------------------

    params = {
        "low_w": f"{lower_nm:.6f}",
        "upp_w": f"{upper_nm:.6f}",
        # NIST unit code:
        # 1 = nanometers
        "unit": "1",
        "show_obs_wl": "1",
        "show_calc_wl": "1",
        "unc_out": "1",
        "intens_out": "1",
        "enrg_out": "1",
        "conf_out": "1",
        "term_out": "1",
        "bibrefs": "1",
        "allowed_out": "1",
        "forbid_out": "1",
        "format": "0",
        "page_size": "100",
        "show_av": "2",
    }

    # --------------------------------------------------------
    # Query NIST.
    # --------------------------------------------------------

    response = requests.get(
        NIST_ASD_URL,
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    # --------------------------------------------------------
    # Parse HTML.
    # --------------------------------------------------------

    soup = BeautifulSoup(
        response.text,
        "html.parser",
    )

    tables = soup.find_all("table")

    result_table = None

    for table in tables:

        text = table.get_text(
            " ",
            strip=True,
        )

        if "Observed" in text and "Wavelength" in text:

            result_table = table
            break

    # --------------------------------------------------------
    # No result table.
    # --------------------------------------------------------

    if result_table is None:

        return {
            "status": "no_results",
            "source": "NIST ASD",
            "query_wavelength_nm": (wavelength_nm),
            "tolerance_nm": (tolerance_nm),
            "wavelength_unit": "nm",
            "nist_wavelength_convention": ("air wavelength for 200-2000 nm"),
            "candidate_count": 0,
            "candidates": [],
        }

    # --------------------------------------------------------
    # Parse result rows.
    # --------------------------------------------------------

    rows = result_table.find_all("tr")

    candidates = []

    for row in rows:

        values = _extract_row_cells(row)

        # ----------------------------------------------------
        # Skip empty rows.
        # ----------------------------------------------------

        if not values:
            continue

        if not any(values):
            continue

        # ----------------------------------------------------
        # Skip the column heading row.
        # ----------------------------------------------------

        joined = " ".join(value or "" for value in values)

        if "Observed" in joined and "Wavelength" in joined:

            continue

        # ----------------------------------------------------
        # Skip separator rows.
        # ----------------------------------------------------

        nonempty_values = [value for value in values if value is not None]

        if nonempty_values and all(value in {"---", "-"} for value in nonempty_values):

            continue

        # ----------------------------------------------------
        # A legitimate spectral-line row must have either
        # an observed or Ritz wavelength.
        # ----------------------------------------------------

        numeric_values = [_to_float(value) for value in values[:4]]

        if not any(value is not None for value in numeric_values):

            continue

        # ----------------------------------------------------
        # Extract species directly from the NIST HTML.
        # ----------------------------------------------------

        species = _extract_species_from_row(row)

        # ----------------------------------------------------
        # Parse the actual 17-cell NIST row.
        # ----------------------------------------------------

        candidate = _parse_standard_row(
            values,
            species=species,
        )

        candidates.append(candidate)

    # --------------------------------------------------------
    # Return structured evidence.
    # --------------------------------------------------------

    return {
        "status": ("results_found" if candidates else "no_results"),
        "source": "NIST ASD",
        "query_wavelength_nm": (wavelength_nm),
        "tolerance_nm": (tolerance_nm),
        "search_range_nm": {
            "lower": lower_nm,
            "upper": upper_nm,
        },
        "wavelength_unit": "nm",
        "nist_wavelength_convention": ("air wavelength for 200-2000 nm"),
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def find_same_upper_level_transitions(
    species,
    upper_energy_cm1,
    energy_tolerance_cm1=0.1,
):
    """
    Find spectral transitions belonging to a specified upper
    energy level.

    This is intended for independent spectral-line validation.

    Example
    -------
    find_same_upper_level_transitions(
        species="Cs II",
        upper_energy_cm1=170973.7833,
        energy_tolerance_cm1=0.1,
    )

    Parameters
    ----------
    species : str
        NIST species designation, e.g. "Cs II".

    upper_energy_cm1 : float
        Target upper-level energy in inverse centimetres.

    energy_tolerance_cm1 : float
        Allowed difference between the NIST upper-level energy
        and the requested energy.

    Returns
    -------
    dict
        Structured collection of transitions sharing the
        requested upper level.
    """

    if not species:
        raise ValueError("species must be provided.")

    if upper_energy_cm1 <= 0:
        raise ValueError("upper_energy_cm1 must be positive.")

    if energy_tolerance_cm1 <= 0:
        raise ValueError("energy_tolerance_cm1 must be positive.")

    # --------------------------------------------------------
    # Query a broad wavelength interval for the species.
    #
    # We deliberately retrieve the species rather than
    # searching only around 1284 nm. Companion transitions
    # may occur elsewhere in the spectrum.
    # --------------------------------------------------------

    params = {
        "spectra": species,
        "low_w": "100",
        "upp_w": "5000",
        "unit": "1",
        "show_obs_wl": "1",
        "show_calc_wl": "1",
        "unc_out": "1",
        "intens_out": "1",
        "enrg_out": "1",
        "conf_out": "1",
        "term_out": "1",
        "bibrefs": "1",
        "allowed_out": "1",
        "forbid_out": "1",
        "format": "0",
        "page_size": "100",
        "show_av": "2",
        "J_out": "on",
    }

    response = requests.get(
        NIST_ASD_URL,
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    soup = BeautifulSoup(
        response.text,
        "html.parser",
    )

    # --------------------------------------------------------
    # Find the NIST result table.
    # --------------------------------------------------------

    result_table = None

    for table in soup.find_all("table"):

        text = table.get_text(
            " ",
            strip=True,
        )

        if (
            "Observed" in text
            and "Wavelength" in text
            and "Lower Level" in text
            and "Upper Level" in text
        ):
            result_table = table
            break

    if result_table is None:

        return {
            "status": "no_results",
            "source": "NIST ASD",
            "species": species,
            "target_upper_energy_cm-1": upper_energy_cm1,
            "energy_tolerance_cm-1": energy_tolerance_cm1,
            "candidate_count": 0,
            "candidates": [],
        }

    # --------------------------------------------------------
    # Parse rows.
    # --------------------------------------------------------

    matching_candidates = []

    for row in result_table.find_all("tr"):

        values = _extract_row_cells(row)

        if not values:
            continue

        # Skip header rows.
        joined = " ".join(value or "" for value in values)

        if "Observed" in joined and "Wavelength" in joined:
            continue

        # Need at least enough fields to contain upper energy.
        if len(values) < 10:
            continue

        # ----------------------------------------------------
        # NIST's actual result table has this structure:
        #
        # 0 observed wavelength
        # 1 observed uncertainty
        # 2 Ritz wavelength
        # 3 Ritz uncertainty
        # 4 relative intensity
        # 5 Aki
        # 6 accuracy
        # 7 lower energy
        # 8 separator
        # 9 upper energy
        # ----------------------------------------------------

        upper_energy = _to_float(values[9])

        if upper_energy is None:
            continue

        difference = abs(upper_energy - upper_energy_cm1)

        if difference > energy_tolerance_cm1:
            continue

        # ----------------------------------------------------
        # Extract species from the actual HTML row.
        # ----------------------------------------------------

        row_species = _extract_species_from_row(row)

        # ----------------------------------------------------
        # Parse the standard NIST row.
        # ----------------------------------------------------

        candidate = _parse_standard_row(
            values,
            species=row_species or species,
        )

        candidate["upper_energy_difference_cm-1"] = difference

        matching_candidates.append(candidate)

    return {
        "status": (
            "results_found" if matching_candidates else "no_matching_upper_level"
        ),
        "source": "NIST ASD",
        "species": species,
        "target_upper_energy_cm-1": upper_energy_cm1,
        "energy_tolerance_cm-1": energy_tolerance_cm1,
        "candidate_count": len(matching_candidates),
        "candidates": matching_candidates,
    }
