from pathlib import Path

import numpy as np
from astropy.io import fits


# ============================================================
# M51 NIRSpec Level-3 S3D diagnostic
# ============================================================

S3D_PATH = Path(
    "data/m51_jwst_level3/"
    "jw03435-o006_t010_nirspec_g140m-f100lp_s3d.fits"
)


print("=" * 70)
print("M51 JWST/NIRSpec S3D CUBE DIAGNOSTIC")
print("=" * 70)

print()
print("FILE:")
print(S3D_PATH)


# ============================================================
# Open cube
# ============================================================

with fits.open(S3D_PATH) as hdul:

    print()
    print("=" * 70)
    print("HDU STRUCTURE")
    print("=" * 70)

    hdul.info()

    for index, hdu in enumerate(hdul):

        print()
        print("-" * 70)
        print(f"HDU {index}")
        print("-" * 70)

        print(
            "Type:",
            type(hdu).__name__
        )

        print(
            "Shape:",
            getattr(
                hdu.data,
                "shape",
                None,
            )
        )

        print(
            "Data type:",
            getattr(
                getattr(hdu, "data", None),
                "dtype",
                None,
            )
        )

        print(
            "EXTNAME:",
            hdu.header.get(
                "EXTNAME",
                None,
            )
        )

        print(
            "BUNIT:",
            hdu.header.get(
                "BUNIT",
                None,
            )
        )


    # ========================================================
    # Identify science cube
    # ========================================================

    science_hdu = None

    for index, hdu in enumerate(hdul):

        if hdu.data is None:
            continue

        if not hasattr(hdu.data, "ndim"):
            continue

        if hdu.data.ndim == 3:

            science_hdu = hdu

            print()
            print("=" * 70)
            print("3-D SCIENCE CUBE FOUND")
            print("=" * 70)

            print(
                f"HDU: {index}"
            )

            print(
                f"EXTNAME: "
                f"{hdu.header.get('EXTNAME', '')}"
            )

            print(
                f"Shape: "
                f"{hdu.data.shape}"
            )

            print(
                f"BUNIT: "
                f"{hdu.header.get('BUNIT', '')}"
            )

            break


    if science_hdu is None:

        raise RuntimeError(
            "No 3-D science cube was found."
        )


    # ========================================================
    # Cube statistics
    # ========================================================

    cube = np.asarray(
        science_hdu.data,
        dtype=float,
    )

    print()
    print("=" * 70)
    print("CUBE STATISTICS")
    print("=" * 70)

    print(
        "Shape:",
        cube.shape,
    )

    print(
        "Number of elements:",
        cube.size,
    )

    finite = np.isfinite(cube)

    print(
        "Finite elements:",
        np.sum(finite),
    )

    print(
        "Non-finite elements:",
        np.sum(~finite),
    )

    if np.any(finite):

        print(
            "Minimum:",
            np.nanmin(cube),
        )

        print(
            "Maximum:",
            np.nanmax(cube),
        )

        print(
            "Median:",
            np.nanmedian(cube),
        )


    # ========================================================
    # Wavelength / spectral-axis keywords
    # ========================================================

    header = science_hdu.header

    print()
    print("=" * 70)
    print("SPECTRAL WCS KEYWORDS")
    print("=" * 70)

    wavelength_keywords = [
        "CTYPE1",
        "CTYPE2",
        "CTYPE3",
        "CUNIT1",
        "CUNIT2",
        "CUNIT3",
        "CRVAL1",
        "CRVAL2",
        "CRVAL3",
        "CRPIX1",
        "CRPIX2",
        "CRPIX3",
        "CDELT1",
        "CDELT2",
        "CDELT3",
        "CD1_1",
        "CD2_2",
        "CD3_3",
        "PC1_1",
        "PC2_2",
        "PC3_3",
    ]

    for key in wavelength_keywords:

        if key in header:

            print(
                f"{key:8s}: "
                f"{header[key]!r}"
            )


    # ========================================================
    # All keywords mentioning spectral information
    # ========================================================

    print()
    print("=" * 70)
    print("ALL SPECTRAL / WAVELENGTH KEYWORDS")
    print("=" * 70)

    for key in header:

        upper = key.upper()

        if any(
            term in upper
            for term in [
                "WAVE",
                "SPEC",
                "CTYPE",
                "CUNIT",
                "CRVAL",
                "CRPIX",
                "CDELT",
                "CD",
                "PC",
            ]
        ):

            print(
                f"{key:20s}: "
                f"{header[key]!r}"
            )


    # ========================================================
    # Target / observation metadata
    # ========================================================

    print()
    print("=" * 70)
    print("TARGET / OBSERVATION METADATA")
    print("=" * 70)

    metadata_keys = [
        "TELESCOP",
        "INSTRUME",
        "DETECTOR",
        "FILTER",
        "GRATING",
        "EXP_TYPE",
        "PROGRAM",
        "DATE-OBS",
        "TIME-OBS",
        "TARGNAME",
        "TARG_RA",
        "TARG_DEC",
        "SRCTYPE",
        "BUNIT",
    ]

    for key in metadata_keys:

        if key in header:

            print(
                f"{key:12s}: "
                f"{header[key]!r}"
            )


    # ========================================================
    # Determine likely spectral axis
    # ========================================================

    print()
    print("=" * 70)
    print("LIKELY SPECTRAL AXIS")
    print("=" * 70)

    for axis in range(1, 4):

        ctype = header.get(
            f"CTYPE{axis}",
            "",
        )

        cunit = header.get(
            f"CUNIT{axis}",
            "",
        )

        crval = header.get(
            f"CRVAL{axis}",
            None,
        )

        cdelt = header.get(
            f"CDELT{axis}",
            None,
        )

        crpix = header.get(
            f"CRPIX{axis}",
            None,
        )

        print()
        print(
            f"Axis {axis}:"
        )

        print(
            f"  CTYPE = {ctype!r}"
        )

        print(
            f"  CUNIT = {cunit!r}"
        )

        print(
            f"  CRVAL = {crval!r}"
        )

        print(
            f"  CDELT = {cdelt!r}"
        )

        print(
            f"  CRPIX = {crpix!r}"
        )


    # ========================================================
    # Look for wavelength-like axis
    # ========================================================

    print()
    print("=" * 70)
    print("WAVELENGTH AXIS TEST")
    print("=" * 70)

    spectral_axis = None

    for axis in range(1, 4):

        ctype = str(
            header.get(
                f"CTYPE{axis}",
                "",
            )
        ).upper()

        cunit = str(
            header.get(
                f"CUNIT{axis}",
                "",
            )
        ).lower()

        if (
            "WAVE" in ctype
            or "FREQ" in ctype
            or "WAVE" in cunit
        ):

            spectral_axis = axis

            print(
                f"Candidate spectral axis: "
                f"{axis}"
            )

            print(
                f"CTYPE{axis}: "
                f"{header.get(f'CTYPE{axis}')}"
            )

            print(
                f"CUNIT{axis}: "
                f"{header.get(f'CUNIT{axis}')}"
            )

            break


    if spectral_axis is None:

        print(
            "No obvious wavelength axis found "
            "from the simple keyword test."
        )

    else:

        n_axis = cube.shape[
            3 - spectral_axis
        ]

        crval = header.get(
            f"CRVAL{spectral_axis}"
        )

        crpix = header.get(
            f"CRPIX{spectral_axis}"
        )

        cdelt = header.get(
            f"CDELT{spectral_axis}"
        )

        print()
        print(
            f"Spectral axis length: "
            f"{n_axis}"
        )

        print(
            f"CRVAL: {crval}"
        )

        print(
            f"CRPIX: {crpix}"
        )

        print(
            f"CDELT: {cdelt}"
        )

        if (
            crval is not None
            and crpix is not None
            and cdelt is not None
        ):

            pixels = np.arange(
                1,
                n_axis + 1,
                dtype=float,
            )

            wavelength = (
                crval
                + (
                    pixels
                    - crpix
                )
                * cdelt
            )

            print()
            print(
                "First wavelength coordinate:"
            )

            print(
                wavelength[0]
            )

            print(
                "Last wavelength coordinate:"
            )

            print(
                wavelength[-1]
            )

            print()
            print(
                "Approximate wavelength range:"
            )

            print(
                f"{np.min(wavelength):.9f}"
                " to "
                f"{np.max(wavelength):.9f}"
            )


    # ========================================================
    # Search spectral WCS for 1284 nm
    # ========================================================

    print()
    print("=" * 70)
    print("1284 nm WAVELENGTH CHECK")
    print("=" * 70)

    if spectral_axis is not None:

        crval = header.get(
            f"CRVAL{spectral_axis}"
        )

        crpix = header.get(
            f"CRPIX{spectral_axis}"
        )

        cdelt = header.get(
            f"CDELT{spectral_axis}"
        )

        cunit = str(
            header.get(
                f"CUNIT{spectral_axis}",
                "",
            )
        ).lower()

        if all(
            value is not None
            for value in [
                crval,
                crpix,
                cdelt,
            ]
        ):

            n_axis = cube.shape[
                3 - spectral_axis
            ]

            pixels = np.arange(
                1,
                n_axis + 1,
                dtype=float,
            )

            coordinate = (
                crval
                + (
                    pixels
                    - crpix
                )
                * cdelt
            )

            # Convert to nm for the comparison.
            if "um" in cunit:
                wavelength_nm = (
                    coordinate * 1000.0
                )

            elif "micron" in cunit:
                wavelength_nm = (
                    coordinate * 1000.0
                )

            elif "m" == cunit:
                wavelength_nm = (
                    coordinate * 1e9
                )

            else:
                wavelength_nm = coordinate

            index = np.argmin(
                np.abs(
                    wavelength_nm
                    - 1284.2613
                )
            )

            print(
                f"Closest spectral index: "
                f"{index}"
            )

            print(
                f"Cube wavelength: "
                f"{wavelength_nm[index]:.9f} nm"
            )

            print(
                f"Difference from feature: "
                f"{wavelength_nm[index] - 1284.2613:+.9f} nm"
            )


print()
print("=" * 70)
print("S3D DIAGNOSTIC COMPLETE")
print("=" * 70)
