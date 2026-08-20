import requests

URL = "https://physics.nist.gov/cgi-bin/ASD/lines1.pl"

params = {
    # --------------------------------------------------------------
    # Spectrum
    # --------------------------------------------------------------
    "spectra": "Fe II",

    # --------------------------------------------------------------
    # Wavelength limits
    # IMPORTANT: NIST currently uses low_wl / upp_wl
    # --------------------------------------------------------------
    "low_wl": "1278",
    "upp_wl": "1290",

    # --------------------------------------------------------------
    # Wavelength units
    # 1 = nm
    # --------------------------------------------------------------
    "unit": "1",

    # --------------------------------------------------------------
    # Output
    # --------------------------------------------------------------
    "format": "0",
    "output": "0",
    "page_size": "15",

    # --------------------------------------------------------------
    # Wavelength information
    # --------------------------------------------------------------
    "show_obs_wl": "1",
    "show_calc_wl": "1",

    # --------------------------------------------------------------
    # Line information
    # --------------------------------------------------------------
    "intens_out": "on",
    "enrg_out": "on",
    "conf_out": "on",
    "term_out": "on",
    "J_out": "on",

    # --------------------------------------------------------------
    # Transition information
    # --------------------------------------------------------------
    "allowed_out": "1",
    "forbid_out": "1",

    # --------------------------------------------------------------
    # Uncertainties
    # --------------------------------------------------------------
    "unc_out": "1",

    # --------------------------------------------------------------
    # Submit
    # --------------------------------------------------------------
    "submit": "Retrieve Data",
}


print("=" * 70)
print("NIST ASD CONNECTION TEST")
print("=" * 70)

print()
print("Query:")
print("  Spectrum: Fe II")
print("  Wavelength: 1278 - 1290 nm")

print()
print("Sending request...")

response = requests.get(
    URL,
    params=params,
    timeout=60,
)

print()
print("HTTP status:")
print(
    response.status_code
)

print()
print("Response length:")
print(
    len(response.text)
)

print()
print("Final URL:")
print(
    response.url
)


# ======================================================================
# ERROR CHECK
# ======================================================================

if response.status_code != 200:

    print()
    print("NIST returned an HTTP error.")

    print()
    print("Response:")
    print(
        response.text[:3000]
    )

    raise RuntimeError(
        f"NIST returned HTTP "
        f"{response.status_code}"
    )


text = response.text

lower = text.lower()


if (
    "software error"
    in lower
):

    print()
    print(
        "NIST returned a server-side "
        "software error."
    )

    print(
        text[:3000]
    )

    raise RuntimeError(
        "NIST server-side error."
    )


if (
    "input error"
    in lower
):

    print()
    print(
        "NIST rejected the query."
    )

    print(
        text[:3000]
    )

    raise RuntimeError(
        "NIST input error."
    )


# ======================================================================
# SUCCESS
# ======================================================================

print()
print("=" * 70)
print("NIST RESPONSE SUCCESSFUL")
print("=" * 70)

print()

# Look for recognizable NIST output.

for phrase in [
    "Atomic Spectra Database",
    "Lines of Data Found",
    "Observed",
    "Ritz",
    "Fe II",
]:

    if phrase in text:

        print(
            f"FOUND: {phrase}"
        )

    else:

        print(
            f"NOT FOUND: {phrase}"
        )


print()
print("=" * 70)
print("RESPONSE PREVIEW")
print("=" * 70)

print()

print(
    text[:5000]
)

print()
print("=" * 70)
print("TEST COMPLETE")
print("=" * 70)
