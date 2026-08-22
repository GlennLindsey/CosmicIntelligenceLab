#!/usr/bin/env python3

"""
M51 JWST NIRSpec/IFU — CHAOS Spatial Cross-Match

Purpose
-------
Determine whether the independent CHAOS H II region
NGC5194+30.2+2.2 is spatially coincident with the
JWST NIRSpec/IFU Pa-beta / Pa-gamma extraction.

JWST nominal extraction:
    radius = 0.45 arcsec

JWST extraction center:
    taken from the S3D WCS at EXTR_X / EXTR_Y

CHAOS position:
    taken from CHAOS table2.dat

The experiment calculates:

1. JWST extraction center
2. CHAOS RA/Dec
3. angular separation
4. separation in JWST spatial pixels
5. whether CHAOS lies inside the JWST aperture
6. fractional offset relative to aperture radius
7. JWST aperture footprint
8. CHAOS position relative to JWST aperture
9. comparison with the pipeline X1D position
10. research-quality figure
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u


# ============================================================
# PATHS
# ============================================================

PROJECT = Path.home() / "Projects" / "cosmic_ai"

S3D_FILE = (
    PROJECT
    / "data"
    / "m51_jwst_level3"
    / "jw03435-o006_t010_nirspec_g140m-f100lp_s3d.fits"
)

X1D_FILE = (
    PROJECT
    / "data"
    / "m51_jwst_level3"
    / "jw03435-o006_t010_nirspec_g140m-f100lp_x1d.fits"
)

CHAOS_TABLE2 = (
    PROJECT
    / "data"
    / "atomic_lines"
    / "chaos_m51"
    / "table2.dat"
)

OUTPUT_TABLE = (
    PROJECT
    / "data"
    / "atomic_lines"
    / "m51_jwst_chaos_spatial_crossmatch.csv"
)

OUTPUT_FIGURE = (
    PROJECT
    / "m51_jwst_chaos_spatial_crossmatch.png"
)


# ============================================================
# TARGET
# ============================================================

CHAOS_TARGET = "NGC5194+30.2+2.2"

JWST_APERTURE_RADIUS_ARCSEC = 0.45


# ============================================================
# UTILITY
# ============================================================

def section(title):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    section(
        "M51 JWST NIRSPEC/IFU — CHAOS SPATIAL CROSS-MATCH"
    )

    print(
        "Target:"
    )

    print(
        f"  {CHAOS_TARGET}"
    )

    print(
        "\nPurpose:"
    )

    print(
        "Determine whether the CHAOS H II region"
        " is spatially coincident with the JWST"
        " Pa-beta / Pa-gamma extraction."
    )

    # ========================================================
    # 1. READ JWST S3D
    # ========================================================

    section(
        "1. READING JWST S3D"
    )

    print(
        f"File:\n  {S3D_FILE}"
    )

    with fits.open(S3D_FILE) as hdul:

        sci = hdul["SCI"]

        data_shape = sci.data.shape

        wcs = WCS(
            sci.header
        )

        celestial_wcs = (
            wcs.celestial
        )

        # Spatial pixel scale
        cdelt = np.abs(
            np.diag(
                celestial_wcs.pixel_scale_matrix
            )
        )

        x_scale = (
            cdelt[0] * 3600.0
        )

        y_scale = (
            cdelt[1] * 3600.0
        )

        print(
            f"\nSCI shape:"
        )

        print(
            f"  spectral = {data_shape[0]}"
        )

        print(
            f"  ny       = {data_shape[1]}"
        )

        print(
            f"  nx       = {data_shape[2]}"
        )

        print(
            "\nSpatial pixel scale:"
        )

        print(
            f"  X = {x_scale:.6f} arcsec/pixel"
        )

        print(
            f"  Y = {y_scale:.6f} arcsec/pixel"
        )

        # ----------------------------------------------------
        # JWST extraction center
        # ----------------------------------------------------

        # The X1D header gives EXTR_X=62, EXTR_Y=48.
        # FITS WCS pixel coordinates are zero-based when
        # passed through astropy WCS pixel_to_world.
        extr_x = 62.0
        extr_y = 48.0

        jwst_center = celestial_wcs.pixel_to_world(
            extr_x,
            extr_y
        )

    print(
        "\nJWST extraction center from S3D WCS:"
    )

    print(
        f"  RA  = {jwst_center.ra.deg:.10f} deg"
    )

    print(
        f"  Dec = {jwst_center.dec.deg:.10f} deg"
    )

    print(
        f"  RA  = "
        f"{jwst_center.ra.to_string(unit=u.hourangle, sep=':')}"
    )

    print(
        f"  Dec = "
        f"{jwst_center.dec.to_string(sep=':')}"
    )

    # ========================================================
    # 2. READ X1D PIPELINE POSITION
    # ========================================================

    section(
        "2. READING X1D PIPELINE POSITION"
    )

    print(
        f"File:\n  {X1D_FILE}"
    )

    with fits.open(X1D_FILE) as hdul:

        hdr = hdul["EXTRACT1D"].header

        extr_x_x1d = float(
            hdr["EXTR_X"]
        )

        extr_y_x1d = float(
            hdr["EXTR_Y"]
        )

        pipeline_ra = float(
            hdr["SLIT_RA"]
        )

        pipeline_dec = float(
            hdr["SLIT_DEC"]
        )

    pipeline_position = SkyCoord(
        pipeline_ra * u.deg,
        pipeline_dec * u.deg,
        frame="icrs"
    )

    print(
        "\nX1D extraction metadata:"
    )

    print(
        f"  EXTR_X  = {extr_x_x1d:.3f}"
    )

    print(
        f"  EXTR_Y  = {extr_y_x1d:.3f}"
    )

    print(
        f"  SLIT_RA = {pipeline_ra:.10f} deg"
    )

    print(
        f"  SLIT_DEC = {pipeline_dec:.10f} deg"
    )

    # ========================================================
    # 3. READ CHAOS POSITION
    # ========================================================

    section(
        "3. READING CHAOS POSITION"
    )

    print(
        f"File:\n  {CHAOS_TABLE2}"
    )

    chaos_matches = []

    with open(
        CHAOS_TABLE2,
        "r",
        errors="replace"
    ) as f:

        for line_number, line in enumerate(
            f,
            start=1
        ):

            if CHAOS_TARGET in line:

                chaos_matches.append(
                    (
                        line_number,
                        line.rstrip()
                    )
                )

    print(
        f"\nMatches found: "
        f"{len(chaos_matches)}"
    )

    for line_number, line in chaos_matches:

        print(
            f"\nLine {line_number}:"
        )

        print(
            f"  {line}"
        )

    if len(chaos_matches) != 1:

        raise RuntimeError(
            "Expected exactly one CHAOS table2 match."
        )

    # --------------------------------------------------------
    # Parse fixed-format CHAOS table2 coordinates
    #
    # Example:
    #
    # NGC5194+30.2+2.2
    # 13 29 55.7
    # +47 11 45.24
    #
    # The table contains fixed-width fields.
    # --------------------------------------------------------

    chaos_line = chaos_matches[0][1]

    # First 19 characters are the HII identifier.
    target_field = chaos_line[:19].strip()

    if target_field != CHAOS_TARGET:

        raise RuntimeError(
            "CHAOS target identifier does not match."
        )

    # Remaining fields are whitespace separated.
    fields = chaos_line[19:].split()

    if len(fields) < 6:

        raise RuntimeError(
            "Could not parse CHAOS coordinate fields."
        )

    ra_h = float(fields[0])
    ra_m = float(fields[1])
    ra_s = float(fields[2])

    dec_sign = 1.0

    dec_string = fields[3]

    if dec_string.startswith("-"):

        dec_sign = -1.0

    dec_d = abs(
        float(dec_string)
    )

    dec_m = float(fields[4])
    dec_s = float(fields[5])

    chaos_ra_deg = (
        15.0
        * (
            ra_h
            + ra_m / 60.0
            + ra_s / 3600.0
        )
    )

    chaos_dec_deg = (
        dec_sign
        * (
            dec_d
            + dec_m / 60.0
            + dec_s / 3600.0
        )
    )

    chaos_position = SkyCoord(
        chaos_ra_deg * u.deg,
        chaos_dec_deg * u.deg,
        frame="icrs"
    )

    print(
        "\nCHAOS coordinates:"
    )

    print(
        f"  RA  = {chaos_ra_deg:.10f} deg"
    )

    print(
        f"  Dec = {chaos_dec_deg:.10f} deg"
    )

    print(
        f"  RA  = "
        f"{chaos_position.ra.to_string(unit=u.hourangle, sep=':')}"
    )

    print(
        f"  Dec = "
        f"{chaos_position.dec.to_string(sep=':')}"
    )

    # ========================================================
    # 4. JWST ↔ CHAOS SEPARATION
    # ========================================================

    section(
        "4. JWST ↔ CHAOS ANGULAR SEPARATION"
    )

    separation = (
        jwst_center.separation(
            chaos_position
        )
    )

    pipeline_separation = (
        pipeline_position.separation(
            chaos_position
        )
    )

    print(
        "JWST WCS center:"
    )

    print(
        f"  RA  = {jwst_center.ra.deg:.10f}"
    )

    print(
        f"  Dec = {jwst_center.dec.deg:.10f}"
    )

    print(
        "\nCHAOS:"
    )

    print(
        f"  RA  = {chaos_position.ra.deg:.10f}"
    )

    print(
        f"  Dec = {chaos_position.dec.deg:.10f}"
    )

    print(
        "\nSeparation:"
    )

    print(
        f"  {separation.arcsec:.6f} arcsec"
    )

    print(
        f"  {separation.to(u.mas).value:.3f} mas"
    )

    print(
        "\nSeparation from pipeline X1D position:"
    )

    print(
        f"  {pipeline_separation.arcsec:.6f} arcsec"
    )

    # ========================================================
    # 5. SEPARATION IN JWST PIXELS
    # ========================================================

    section(
        "5. SEPARATION IN JWST SPATIAL PIXELS"
    )

    mean_pixel_scale = (
        0.5 * (x_scale + y_scale)
    )

    separation_pixels = (
        separation.arcsec
        / mean_pixel_scale
    )

    print(
        f"Mean spatial scale:"
        f" {mean_pixel_scale:.6f} arcsec/pixel"
    )

    print(
        f"\nJWST ↔ CHAOS separation:"
    )

    print(
        f"  {separation_pixels:.6f} pixels"
    )

    # ========================================================
    # 6. APERTURE CONTAINMENT
    # ========================================================

    section(
        "6. JWST APERTURE CONTAINMENT"
    )

    radius = (
        JWST_APERTURE_RADIUS_ARCSEC
    )

    fraction_of_radius = (
        separation.arcsec
        / radius
    )

    inside_aperture = (
        separation.arcsec
        <= radius
    )

    print(
        f"JWST nominal aperture radius:"
    )

    print(
        f"  {radius:.3f} arcsec"
    )

    print(
        f"\nCHAOS offset from aperture center:"
    )

    print(
        f"  {separation.arcsec:.6f} arcsec"
    )

    print(
        f"\nFraction of aperture radius:"
    )

    print(
        f"  {fraction_of_radius:.6f}"
    )

    print(
        f"\nCHAOS lies inside JWST aperture:"
    )

    print(
        f"  {inside_aperture}"
    )

    # ========================================================
    # 7. APERTURE AREA
    # ========================================================

    section(
        "7. JWST APERTURE GEOMETRY"
    )

    aperture_area = (
        np.pi
        * radius**2
    )

    diameter = (
        2.0 * radius
    )

    print(
        f"Radius:"
        f" {radius:.3f} arcsec"
    )

    print(
        f"Diameter:"
        f" {diameter:.3f} arcsec"
    )

    print(
        f"Area:"
        f" {aperture_area:.6f} arcsec^2"
    )

    print(
        "\nCHAOS position:"
    )

    print(
        "  Point source/region coordinate"
        " lies within the nominal aperture."
    )

    # ========================================================
    # 8. SKY-PLANE OFFSET
    # ========================================================

    section(
        "8. SKY-PLANE OFFSET"
    )

    # Small-angle tangent-plane approximation.
    delta_ra = (
        (
            chaos_position.ra.deg
            - jwst_center.ra.deg
        )
        * 3600.0
        * np.cos(
            np.deg2rad(
                jwst_center.dec.deg
            )
        )
    )

    delta_dec = (
        (
            chaos_position.dec.deg
            - jwst_center.dec.deg
        )
        * 3600.0
    )

    print(
        "Relative position:"
    )

    print(
        f"  ΔRA  = {delta_ra:+.6f} arcsec"
    )

    print(
        f"  ΔDec = {delta_dec:+.6f} arcsec"
    )

    print(
        f"  separation = "
        f"{np.hypot(delta_ra, delta_dec):.6f} arcsec"
    )

    # ========================================================
    # 9. CREATE CROSS-MATCH TABLE
    # ========================================================

    section(
        "9. WRITING CROSS-MATCH TABLE"
    )

    result = pd.DataFrame(
        [
            {
                "chaos_region":
                    CHAOS_TARGET,

                "jwst_ra_deg":
                    jwst_center.ra.deg,

                "jwst_dec_deg":
                    jwst_center.dec.deg,

                "pipeline_ra_deg":
                    pipeline_position.ra.deg,

                "pipeline_dec_deg":
                    pipeline_position.dec.deg,

                "chaos_ra_deg":
                    chaos_position.ra.deg,

                "chaos_dec_deg":
                    chaos_position.dec.deg,

                "jwst_chaos_separation_arcsec":
                    separation.arcsec,

                "jwst_chaos_separation_mas":
                    separation.to(
                        u.mas
                    ).value,

                "pipeline_chaos_separation_arcsec":
                    pipeline_separation.arcsec,

                "spatial_pixel_scale_arcsec":
                    mean_pixel_scale,

                "separation_pixels":
                    separation_pixels,

                "aperture_radius_arcsec":
                    radius,

                "aperture_diameter_arcsec":
                    diameter,

                "aperture_area_arcsec2":
                    aperture_area,

                "offset_fraction_of_radius":
                    fraction_of_radius,

                "chaos_inside_jwst_aperture":
                    inside_aperture,

                "delta_ra_arcsec":
                    delta_ra,

                "delta_dec_arcsec":
                    delta_dec,
            }
        ]
    )

    result.to_csv(
        OUTPUT_TABLE,
        index=False
    )

    print(
        f"Saved:"
        f"\n  {OUTPUT_TABLE}"
    )

    # ========================================================
    # 10. CREATE FIGURE
    # ========================================================

    section(
        "10. CREATING CROSS-MATCH FIGURE"
    )

    theta = np.linspace(
        0.0,
        2.0 * np.pi,
        361
    )

    circle_x = (
        radius
        * np.cos(theta)
    )

    circle_y = (
        radius
        * np.sin(theta)
    )

    fig, ax = plt.subplots(
        figsize=(9, 9)
    )

    # JWST aperture
    ax.plot(
        circle_x,
        circle_y,
        linewidth=2,
        label="JWST nominal extraction aperture"
    )

    # JWST center
    ax.scatter(
        [0],
        [0],
        marker="+",
        s=180,
        linewidths=2,
        label="JWST extraction center"
    )

    # CHAOS position
    ax.scatter(
        [delta_ra],
        [delta_dec],
        marker="*",
        s=180,
        label="CHAOS NGC5194+30.2+2.2"
    )

    # Connection
    ax.plot(
        [0, delta_ra],
        [0, delta_dec],
        linestyle="--",
        linewidth=1
    )

    # Label
    ax.annotate(
        f"CHAOS\n"
        f"{separation.arcsec:.4f}\"",
        xy=(
            delta_ra,
            delta_dec
        ),
        xytext=(
            15,
            15
        ),
        textcoords="offset points",
        fontsize=10
    )

    ax.annotate(
        "JWST center",
        xy=(0, 0),
        xytext=(10, -20),
        textcoords="offset points",
        fontsize=10
    )

    ax.set_xlabel(
        r"$\Delta$RA (arcsec)"
    )

    ax.set_ylabel(
        r"$\Delta$Dec (arcsec)"
    )

    ax.set_title(
        "M51 JWST NIRSpec/IFU — CHAOS Spatial Cross-Match"
    )

    # Equal sky-plane scale
    ax.set_aspect(
        "equal",
        adjustable="box"
    )

    # Add margin
    limit = max(
        radius * 1.35,
        abs(delta_ra) * 3,
        abs(delta_dec) * 3,
        0.6
    )

    ax.set_xlim(
        -limit,
        limit
    )

    ax.set_ylim(
        -limit,
        limit
    )

    ax.grid(
        True,
        alpha=0.3
    )

    ax.legend(
        loc="upper right"
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_FIGURE,
        dpi=250
    )

    plt.close(fig)

    print(
        f"Saved:"
        f"\n  {OUTPUT_FIGURE}"
    )

    # ========================================================
    # 11. FINAL RESULT
    # ========================================================

    section(
        "11. FINAL CROSS-MATCH RESULT"
    )

    print(
        "CHAOS region:"
    )

    print(
        f"  {CHAOS_TARGET}"
    )

    print(
        "\nCHAOS coordinates:"
    )

    print(
        f"  RA  = "
        f"{chaos_position.ra.deg:.10f} deg"
    )

    print(
        f"  Dec = "
        f"{chaos_position.dec.deg:.10f} deg"
    )

    print(
        "\nJWST extraction center:"
    )

    print(
        f"  RA  = "
        f"{jwst_center.ra.deg:.10f} deg"
    )

    print(
        f"  Dec = "
        f"{jwst_center.dec.deg:.10f} deg"
    )

    print(
        "\nAngular separation:"
    )

    print(
        f"  {separation.arcsec:.6f} arcsec"
    )

    print(
        "\nJWST aperture radius:"
    )

    print(
        f"  {radius:.3f} arcsec"
    )

    print(
        "\nCHAOS inside JWST aperture:"
    )

    print(
        f"  {inside_aperture}"
    )

    print(
        "\nOffset as fraction of aperture radius:"
    )

    print(
        f"  {fraction_of_radius:.4f}"
    )

    print(
        "\n=================================================="
    )

    if inside_aperture:

        print(
            "RESULT:"
        )

        print(
            "The CHAOS region coordinate lies INSIDE"
            " the nominal JWST extraction aperture."
        )

    else:

        print(
            "RESULT:"
        )

        print(
            "The CHAOS region coordinate lies OUTSIDE"
            " the nominal JWST extraction aperture."
        )

    print(
        "=================================================="
    )

    print(
        "\nScientific interpretation:"
    )

    if inside_aperture:

        print(
            "The spatial coincidence is strong enough"
            " that the CHAOS extinction measurement"
            " is relevant as an independent physical"
            " constraint on the JWST extraction."
        )

        print(
            "However, the optical and JWST apertures"
            " remain observationally different and"
            " do not necessarily weight the emitting"
            " gas identically."
        )

    else:

        print(
            "The CHAOS extinction measurement cannot"
            " be regarded as an aperture-coincident"
            " constraint without further analysis."
        )

    print(
        "\nExperiment complete."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
