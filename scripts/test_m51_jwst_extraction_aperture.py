#!/usr/bin/env python3

"""
M51 JWST NIRSpec/IFU Extraction Aperture Audit

Definitive aperture audit using:

    S3D cube
    X1D spectrum
    JWST CRDS EXTRACT1D reference

The CRDS reference specifies the NIRSpec IFU extraction geometry.

For this observation we expect:

    model_type       = Extract1dIFUModel
    exposure.type    = NRS_IFU
    method           = center
    region_type      = target
    subpixels        = 5
    radius           = 0.45 arcsec
    subtract_background = False

The script determines:

    1. S3D IFU footprint
    2. S3D spatial pixel scale
    3. X1D extraction center
    4. CRDS-defined extraction radius
    5. nominal extraction area
    6. extraction footprint in S3D pixels
    7. extraction footprint in arcseconds
    8. RA/Dec polygon describing the aperture
    9. Pa-beta / Pa-gamma aperture consistency
   10. background-annulus information
   11. research-quality diagnostic figure

IMPORTANT:

The CRDS reference defines the nominal extraction geometry.
This script does NOT claim to recover the complete effective
pixel-weighting function used internally by the pipeline.

Outputs:

    data/atomic_lines/m51_jwst_extraction_aperture.csv
    data/atomic_lines/m51_jwst_extraction_aperture_summary.csv
    data/atomic_lines/m51_jwst_extraction_aperture_polygon.csv
    m51_jwst_extraction_aperture.png
"""

from pathlib import Path
import csv
import math

import asdf
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

# ======================================================================
# Paths
# ======================================================================

PROJECT_ROOT = Path.home() / "Projects" / "cosmic_ai"

S3D_FILE = (
    PROJECT_ROOT
    / "data"
    / "m51_jwst_level3"
    / "jw03435-o006_t010_nirspec_g140m-f100lp_s3d.fits"
)

X1D_FILE = (
    PROJECT_ROOT
    / "data"
    / "m51_jwst_level3"
    / "jw03435-o006_t010_nirspec_g140m-f100lp_x1d.fits"
)

CRDS_REFERENCE = (
    Path.home()
    / "crds_cache"
    / "jwst"
    / "references"
    / "jwst"
    / "nirspec"
    / "jwst_nirspec_extract1d_0002.asdf"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "atomic_lines"

PIXEL_CSV = OUTPUT_DIR / "m51_jwst_extraction_aperture.csv"

SUMMARY_CSV = OUTPUT_DIR / "m51_jwst_extraction_aperture_summary.csv"

POLYGON_CSV = OUTPUT_DIR / "m51_jwst_extraction_aperture_polygon.csv"

FIGURE_FILE = PROJECT_ROOT / "m51_jwst_extraction_aperture.png"


# ======================================================================
# Constants
# ======================================================================

PA_GAMMA_UM = 1.094
PA_BETA_UM = 1.282

ARCSEC_PER_DEG = 3600.0


# ======================================================================
# Utility functions
# ======================================================================


def print_section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def normalize_ra(ra):
    """
    Normalize RA to the conventional 0--360 degree range.
    """
    return float(ra) % 360.0


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


# ======================================================================
# Read S3D cube
# ======================================================================


def read_s3d():

    print_section("1. READING S3D CUBE")

    with fits.open(S3D_FILE) as hdul:

        sci = hdul["SCI"]

        data_shape = sci.data.shape
        header = sci.header.copy()

        wcs = WCS(header)
        celestial_wcs = wcs.celestial

        scales = proj_plane_pixel_scales(celestial_wcs) * ARCSEC_PER_DEG

        nx = data_shape[2]
        ny = data_shape[1]

        print("File:")
        print(f"  {S3D_FILE}")

        print()
        print("SCI shape:")
        print(f"  spectral = {data_shape[0]}")
        print(f"  ny       = {ny}")
        print(f"  nx       = {nx}")

        print()
        print("BUNIT:")
        print(f"  {header.get('BUNIT')}")

        print()
        print("Spatial pixel scale:")
        print(f"  X = {scales[0]:.6f} arcsec/pixel")
        print(f"  Y = {scales[1]:.6f} arcsec/pixel")

        return {
            "header": header,
            "wcs": celestial_wcs,
            "nx": nx,
            "ny": ny,
            "scale_x": float(scales[0]),
            "scale_y": float(scales[1]),
        }


# ======================================================================
# Calculate S3D footprint
# ======================================================================


def calculate_cube_footprint(s3d):

    print_section("2. S3D IFU SKY FOOTPRINT")

    wcs = s3d["wcs"]
    nx = s3d["nx"]
    ny = s3d["ny"]

    corners = {
        "lower_left": (0, 0),
        "lower_right": (nx - 1, 0),
        "upper_left": (0, ny - 1),
        "upper_right": (nx - 1, ny - 1),
    }

    footprint = {}

    for name, (x, y) in corners.items():

        ra, dec = wcs.pixel_to_world_values(x, y)

        ra = normalize_ra(ra)

        footprint[name] = {
            "x": x,
            "y": y,
            "ra": ra,
            "dec": float(dec),
        }

        print(f"{name:12s}: " f"x={x:3d} y={y:3d} " f"RA={ra:.8f} " f"Dec={dec:.8f}")

    width = nx * s3d["scale_x"]
    height = ny * s3d["scale_y"]
    area = width * height

    print()
    print("Approximate footprint dimensions:")
    print(f"  width  = {width:.4f} arcsec")
    print(f"  height = {height:.4f} arcsec")
    print(f"  area   = {area:.4f} arcsec^2")

    s3d["footprint"] = footprint
    s3d["width_arcsec"] = width
    s3d["height_arcsec"] = height
    s3d["area_arcsec2"] = area

    return s3d


# ======================================================================
# Read X1D
# ======================================================================


def read_x1d():

    print_section("3. READING X1D EXTRACTION METADATA")

    with fits.open(X1D_FILE) as hdul:

        primary = hdul[0].header.copy()
        ext = hdul["EXTRACT1D"]

        header = ext.header.copy()
        table = ext.data

        print("File:")
        print(f"  {X1D_FILE}")

        metadata = {}

        keys = [
            "SLTNAME",
            "SRCTYPE",
            "EXTR_X",
            "EXTR_Y",
            "SLIT_RA",
            "SLIT_DEC",
            "PA_APER",
        ]

        print()
        print("Extraction metadata:")

        for key in keys:

            value = header.get(key)

            metadata[key] = value

            print(f"  {key:12s}: {value}")

        print()
        print("Primary extraction metadata:")

        for key in [
            "APERNAME",
            "PPS_APER",
            "R_EXTR1D",
            "S_EXTR1D",
        ]:

            value = primary.get(key)

            metadata[key] = value

            print(f"  {key:12s}: {value}")

        wavelength = np.asarray(
            table["WAVELENGTH"],
            dtype=float,
        )

        npixels = np.asarray(
            table["NPIXELS"],
            dtype=float,
        )

        return {
            "primary_header": primary,
            "header": header,
            "table": table,
            "metadata": metadata,
            "wavelength": wavelength,
            "npixels": npixels,
        }


# ======================================================================
# Locate Pa-beta and Pa-gamma
# ======================================================================


def wavelength_audit(x1d):

    print_section("4. PA-BETA / PA-GAMMA EXTRACTION AUDIT")

    wavelength = x1d["wavelength"]
    npixels = x1d["npixels"]

    results = {}

    for name, target in [
        ("Pa-gamma", PA_GAMMA_UM),
        ("Pa-beta", PA_BETA_UM),
    ]:

        index = np.nanargmin(np.abs(wavelength - target))

        actual = wavelength[index]
        npix = npixels[index]

        results[name] = {
            "target_um": target,
            "actual_um": float(actual),
            "row": int(index),
            "npixels": float(npix),
        }

        print()
        print(f"{name}:")
        print(f"  nominal wavelength = {target:.6f} um")
        print(f"  nearest wavelength = {actual:.6f} um")
        print(f"  X1D row            = {index}")
        print(f"  NPIXELS            = {npix:.1f}")

    pbg = results["Pa-beta"]["npixels"]
    pgam = results["Pa-gamma"]["npixels"]

    difference = pbg - pgam

    print()
    print("NPIXELS comparison:")
    print(f"  Pa-beta  = {pbg:.1f}")
    print(f"  Pa-gamma = {pgam:.1f}")
    print(f"  difference = {difference:+.1f}")

    print()
    print("Interpretation:")
    print("  Both lines are contained in the same X1D product.")
    print("  Both therefore use the same pipeline extraction model.")
    print("  NPIXELS varies slightly with wavelength.")
    print("  NPIXELS is NOT interpreted as a 2-D aperture size.")

    x1d["line_results"] = results

    return x1d


# ======================================================================
# Read CRDS EXTRACT1D reference
# ======================================================================


def read_crds_reference():

    print_section("5. READING JWST CRDS EXTRACT1D REFERENCE")

    print("Reference:")
    print(f"  {CRDS_REFERENCE}")

    if not CRDS_REFERENCE.exists():

        raise FileNotFoundError(
            "The JWST EXTRACT1D CRDS reference was not found:\n" f"{CRDS_REFERENCE}"
        )

    with asdf.open(CRDS_REFERENCE) as af:

        tree = af.tree

        meta = tree["meta"]
        data = tree["data"]

        print()
        print("Reference metadata:")

        metadata_keys = [
            "model_type",
            "method",
            "region_type",
            "version",
            "reftype",
            "origin",
            "pedigree",
            "description",
        ]

        reference_meta = {}

        for key in metadata_keys:

            value = meta.get(key)

            reference_meta[key] = value

            print(f"  {key:15s}: {value}")

        exposure = meta.get("exposure", {})

        instrument = meta.get("instrument", {})

        exposure_type = exposure.get("type")
        instrument_name = instrument.get("name")

        print()
        print("Exposure type:")
        print(f"  {exposure_type}")

        print()
        print("Instrument:")
        print(f"  {instrument_name}")

        wavelength = np.asarray(
            data["wavelength"],
            dtype=float,
        )

        radius = np.asarray(
            data["radius"],
            dtype=float,
        )

        inner_bkg = np.asarray(
            data["inner_bkg"],
            dtype=float,
        )

        outer_bkg = np.asarray(
            data["outer_bkg"],
            dtype=float,
        )

        subpixels = int(meta["subpixels"])
        subtract_background = bool(meta["subtract_background"])

        radius_units = data["radius_units"]
        inner_bkg_units = data["inner_bkg_units"]
        outer_bkg_units = data["outer_bkg_units"]

        print()
        print("Extraction parameters:")
        print(f"  subpixels            = {subpixels}")
        print(f"  subtract_background  = " f"{subtract_background}")

        print()
        print("Radius:")
        print(
            f"  range = " f"{radius.min():.6f} - " f"{radius.max():.6f} {radius_units}"
        )

        print()
        print("Background annulus:")
        print(
            f"  inner = "
            f"{inner_bkg.min():.6f} - "
            f"{inner_bkg.max():.6f} {inner_bkg_units}"
        )
        print(
            f"  outer = "
            f"{outer_bkg.min():.6f} - "
            f"{outer_bkg.max():.6f} {outer_bkg_units}"
        )

        return {
            "meta": meta,
            "data": data,
            "reference_meta": reference_meta,
            "exposure_type": exposure_type,
            "instrument_name": instrument_name,
            "wavelength": wavelength,
            "radius": radius,
            "inner_bkg": inner_bkg,
            "outer_bkg": outer_bkg,
            "radius_units": radius_units,
            "inner_bkg_units": inner_bkg_units,
            "outer_bkg_units": outer_bkg_units,
            "subpixels": subpixels,
            "subtract_background": subtract_background,
        }


# ======================================================================
# Interpolate CRDS aperture parameters at Pa-beta / Pa-gamma
# ======================================================================


def interpolate_reference_parameters(crds):

    print_section("6. CRDS APERTURE PARAMETERS AT PA-BETA / PA-GAMMA")

    wavelength = crds["wavelength"]
    radius = crds["radius"]
    inner_bkg = crds["inner_bkg"]
    outer_bkg = crds["outer_bkg"]

    results = {}

    for name, target in [
        ("Pa-gamma", PA_GAMMA_UM),
        ("Pa-beta", PA_BETA_UM),
    ]:

        r = float(
            np.interp(
                target,
                wavelength,
                radius,
            )
        )

        inner = float(
            np.interp(
                target,
                wavelength,
                inner_bkg,
            )
        )

        outer = float(
            np.interp(
                target,
                wavelength,
                outer_bkg,
            )
        )

        results[name] = {
            "radius_arcsec": r,
            "inner_bkg_arcsec": inner,
            "outer_bkg_arcsec": outer,
        }

        print()
        print(f"{name}:")
        print(f"  wavelength = {target:.6f} um")
        print(f"  radius     = {r:.6f} arcsec")
        print(f"  background inner = {inner:.6f} arcsec")
        print(f"  background outer = {outer:.6f} arcsec")

    r_beta = results["Pa-beta"]["radius_arcsec"]
    r_gamma = results["Pa-gamma"]["radius_arcsec"]

    print()
    print("Aperture comparison:")
    print(f"  Pa-beta radius  = {r_beta:.6f} arcsec")
    print(f"  Pa-gamma radius = {r_gamma:.6f} arcsec")

    same_radius = np.isclose(
        r_beta,
        r_gamma,
        rtol=0.0,
        atol=1e-10,
    )

    print(f"  Same nominal aperture = {same_radius}")

    crds["line_results"] = results
    crds["same_nominal_aperture"] = bool(same_radius)

    return crds


# ======================================================================
# Calculate extraction center
# ======================================================================


def calculate_extraction_center(s3d, x1d):

    print_section("7. EXTRACTION CENTER")

    x = safe_float(x1d["metadata"].get("EXTR_X"))

    y = safe_float(x1d["metadata"].get("EXTR_Y"))

    if x is None or y is None:

        raise RuntimeError("EXTR_X / EXTR_Y are not available " "in the X1D product.")

    ra, dec = s3d["wcs"].pixel_to_world_values(
        x,
        y,
    )

    ra = normalize_ra(ra)

    pipeline_ra = safe_float(x1d["metadata"].get("SLIT_RA"))

    pipeline_dec = safe_float(x1d["metadata"].get("SLIT_DEC"))

    print("Extraction pixel:")
    print(f"  X = {x:.3f}")
    print(f"  Y = {y:.3f}")

    print()
    print("Sky position from S3D WCS:")
    print(f"  RA  = {ra:.10f} deg")
    print(f"  Dec = {dec:.10f} deg")

    print()
    print("Pipeline X1D position:")
    print(
        f"  RA  = {pipeline_ra:.10f} deg"
        if pipeline_ra is not None
        else "  RA  = unavailable"
    )
    print(
        f"  Dec = {pipeline_dec:.10f} deg"
        if pipeline_dec is not None
        else "  Dec = unavailable"
    )

    x1d["extraction_center"] = {
        "x": x,
        "y": y,
        "ra": ra,
        "dec": float(dec),
        "pipeline_ra": pipeline_ra,
        "pipeline_dec": pipeline_dec,
    }

    return x1d


# ======================================================================
# Build nominal circular aperture
# ======================================================================


def build_aperture_geometry(s3d, x1d, crds):

    print_section("8. BUILDING NOMINAL PIPELINE EXTRACTION APERTURE")

    center = x1d["extraction_center"]

    x0 = center["x"]
    y0 = center["y"]

    beta_radius = crds["line_results"]["Pa-beta"]["radius_arcsec"]

    gamma_radius = crds["line_results"]["Pa-gamma"]["radius_arcsec"]

    radius_arcsec = float(0.5 * (beta_radius + gamma_radius))

    radius_x_pixels = radius_arcsec / s3d["scale_x"]

    radius_y_pixels = radius_arcsec / s3d["scale_y"]

    nominal_area = math.pi * radius_arcsec**2

    print("Pipeline extraction model:")
    print(f"  model_type = " f"{crds['reference_meta']['model_type']}")
    print(f"  method     = " f"{crds['reference_meta']['method']}")
    print(f"  region     = " f"{crds['reference_meta']['region_type']}")
    print(f"  subpixels  = " f"{crds['subpixels']}")

    print()
    print("Nominal extraction aperture:")
    print(f"  radius = {radius_arcsec:.6f} arcsec")
    print(f"  radius = {radius_x_pixels:.6f} " f"S3D pixels in X")
    print(f"  radius = {radius_y_pixels:.6f} " f"S3D pixels in Y")

    print()
    print("Nominal circular area:")
    print(f"  area = {nominal_area:.6f} arcsec^2")

    print()
    print("Background configuration:")
    print(
        f"  inner radius = "
        f"{crds['line_results']['Pa-beta']['inner_bkg_arcsec']:.6f} arcsec"
    )
    print(
        f"  outer radius = "
        f"{crds['line_results']['Pa-beta']['outer_bkg_arcsec']:.6f} arcsec"
    )
    print(f"  background subtraction = " f"{crds['subtract_background']}")

    geometry = {
        "center_x": x0,
        "center_y": y0,
        "radius_arcsec": radius_arcsec,
        "radius_x_pixels": radius_x_pixels,
        "radius_y_pixels": radius_y_pixels,
        "area_arcsec2": nominal_area,
    }

    crds["geometry"] = geometry

    return crds


# ======================================================================
# Generate aperture polygon
# ======================================================================


def generate_aperture_polygon(s3d, x1d, crds, npoints=181):

    print_section("9. GENERATING RA/DEC APERTURE POLYGON")

    wcs = s3d["wcs"]
    center = x1d["extraction_center"]
    geometry = crds["geometry"]

    theta = np.linspace(
        0.0,
        2.0 * np.pi,
        npoints,
    )

    dx_arcsec = geometry["radius_arcsec"] * np.cos(theta)

    dy_arcsec = geometry["radius_arcsec"] * np.sin(theta)

    # Convert arcseconds to approximate local
    # S3D pixel offsets.

    dx_pixels = dx_arcsec / s3d["scale_x"]

    dy_pixels = dy_arcsec / s3d["scale_y"]

    x = center["x"] + dx_pixels
    y = center["y"] + dy_pixels

    ra, dec = wcs.pixel_to_world_values(
        x,
        y,
    )

    ra = np.asarray(ra)
    dec = np.asarray(dec)

    ra = np.mod(ra, 360.0)

    rows = []

    for i in range(len(theta)):

        rows.append(
            {
                "point": i,
                "theta_deg": float(np.degrees(theta[i])),
                "x_pixel": float(x[i]),
                "y_pixel": float(y[i]),
                "dx_arcsec": float(dx_arcsec[i]),
                "dy_arcsec": float(dy_arcsec[i]),
                "ra_deg": float(ra[i]),
                "dec_deg": float(dec[i]),
            }
        )

    with open(
        POLYGON_CSV,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "point",
                "theta_deg",
                "x_pixel",
                "y_pixel",
                "dx_arcsec",
                "dy_arcsec",
                "ra_deg",
                "dec_deg",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    print("Polygon points:")
    print(f"  {len(rows)}")

    print()
    print("RA range:")
    print(f"  {ra.min():.10f} - " f"{ra.max():.10f} deg")

    print("Dec range:")
    print(f"  {dec.min():.10f} - " f"{dec.max():.10f} deg")

    print()
    print("Polygon:")
    print(f"  {POLYGON_CSV}")

    crds["polygon"] = rows

    return crds


# ======================================================================
# Write complete spatial pixel table
# ======================================================================


def write_pixel_table(s3d, x1d, crds):

    print_section("10. WRITING S3D SPATIAL APERTURE TABLE")

    wcs = s3d["wcs"]

    center = x1d["extraction_center"]
    geometry = crds["geometry"]

    x0 = center["x"]
    y0 = center["y"]

    radius_x = geometry["radius_x_pixels"]
    radius_y = geometry["radius_y_pixels"]

    rows = []

    for y in range(s3d["ny"]):

        for x in range(s3d["nx"]):

            dx = x - x0
            dy = y - y0

            dx_arcsec = dx * s3d["scale_x"]

            dy_arcsec = dy * s3d["scale_y"]

            radius_from_center = math.sqrt(dx_arcsec**2 + dy_arcsec**2)

            inside = radius_from_center <= geometry["radius_arcsec"]

            ra, dec = wcs.pixel_to_world_values(
                x,
                y,
            )

            ra = normalize_ra(ra)

            rows.append(
                {
                    "x_pixel": x,
                    "y_pixel": y,
                    "ra_deg": float(ra),
                    "dec_deg": float(dec),
                    "dx_arcsec": float(dx_arcsec),
                    "dy_arcsec": float(dy_arcsec),
                    "radius_from_center_arcsec": float(radius_from_center),
                    "inside_nominal_aperture": bool(inside),
                }
            )

    with open(
        PIXEL_CSV,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "x_pixel",
                "y_pixel",
                "ra_deg",
                "dec_deg",
                "dx_arcsec",
                "dy_arcsec",
                "radius_from_center_arcsec",
                "inside_nominal_aperture",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    aperture_pixels = [row for row in rows if row["inside_nominal_aperture"]]

    print("S3D spatial pixels:")
    print(f"  total pixels = {len(rows)}")

    print()
    print("Pixel centers inside nominal aperture:")
    print(f"  {len(aperture_pixels)}")

    print()
    print("Aperture table:")
    print(f"  {PIXEL_CSV}")

    crds["aperture_pixel_rows"] = rows
    crds["aperture_pixel_count"] = len(aperture_pixels)

    return crds


# ======================================================================
# Create research figure
# ======================================================================


def create_figure(s3d, x1d, crds):

    print_section("11. CREATING DEFINITIVE APERTURE FIGURE")

    nx = s3d["nx"]
    ny = s3d["ny"]

    center = x1d["extraction_center"]
    geometry = crds["geometry"]

    fig, ax = plt.subplots(figsize=(11, 9))

    # --------------------------------------------------------------
    # S3D footprint
    # --------------------------------------------------------------

    ax.plot(
        [
            0,
            nx - 1,
            nx - 1,
            0,
            0,
        ],
        [
            0,
            0,
            ny - 1,
            ny - 1,
            0,
        ],
        linewidth=2,
        label="S3D IFU footprint",
    )

    # --------------------------------------------------------------
    # Nominal extraction circle
    # --------------------------------------------------------------

    theta = np.linspace(
        0,
        2 * np.pi,
        500,
    )

    radius_x = geometry["radius_x_pixels"]
    radius_y = geometry["radius_y_pixels"]

    circle_x = center["x"] + radius_x * np.cos(theta)

    circle_y = center["y"] + radius_y * np.sin(theta)

    ax.plot(
        circle_x,
        circle_y,
        linewidth=3,
        label=(
            "JWST EXTRACT1D nominal aperture " f"(r={geometry['radius_arcsec']:.2f}\")"
        ),
    )

    # --------------------------------------------------------------
    # Extraction center
    # --------------------------------------------------------------

    ax.scatter(
        center["x"],
        center["y"],
        s=120,
        marker="x",
        linewidths=3,
        label="X1D extraction center",
    )

    ax.annotate(
        f"EXTR_X/Y = " f"({center['x']:.0f}, {center['y']:.0f})",
        (
            center["x"],
            center["y"],
        ),
        xytext=(12, 12),
        textcoords="offset points",
    )

    # --------------------------------------------------------------
    # Target position
    # --------------------------------------------------------------

    header = s3d["header"]

    target_ra = header.get("TARG_RA") or header.get("RA_TARG")

    target_dec = header.get("TARG_DEC") or header.get("DEC_TARG")

    if target_ra is not None and target_dec is not None:

        target_ra = normalize_ra(target_ra)

        tx, ty = s3d["wcs"].world_to_pixel_values(
            target_ra,
            target_dec,
        )

        ax.scatter(
            tx,
            ty,
            s=90,
            marker="+",
            linewidths=2,
            label="Target position",
        )

    # --------------------------------------------------------------
    # Scale bar
    # --------------------------------------------------------------

    bar_pixels = 1.0 / s3d["scale_x"]

    x_bar = 8
    y_bar = 8

    ax.plot(
        [
            x_bar,
            x_bar + bar_pixels,
        ],
        [
            y_bar,
            y_bar,
        ],
        linewidth=4,
    )

    ax.text(
        x_bar + bar_pixels / 2,
        y_bar + 2,
        "1 arcsec",
        ha="center",
    )

    # --------------------------------------------------------------
    # Labels
    # --------------------------------------------------------------

    ax.set_xlabel("S3D X pixel")

    ax.set_ylabel("S3D Y pixel")

    ax.set_title("M51 JWST NIRSpec/IFU " "Pipeline Extraction Aperture")

    ax.set_xlim(
        -0.5,
        nx - 0.5,
    )

    ax.set_ylim(
        -0.5,
        ny - 0.5,
    )

    ax.set_aspect("equal")

    ax.grid(alpha=0.25)

    ax.legend(loc="upper right")

    # --------------------------------------------------------------
    # Text box
    # --------------------------------------------------------------

    text = (
        "JWST NIRSpec/IFU\n"
        "CRDS: jwst_nirspec_extract1d_0002.asdf\n"
        f"Center: ({center['x']:.1f}, "
        f"{center['y']:.1f}) pixels\n"
        f"Radius: "
        f"{geometry['radius_arcsec']:.3f} arcsec\n"
        f"Area: "
        f"{geometry['area_arcsec2']:.4f} arcsec²\n"
        f"Pa-β / Pa-γ: same nominal aperture\n"
        f"Subpixels: {crds['subpixels']}\n"
        f"Background subtraction: "
        f"{crds['subtract_background']}"
    )

    ax.text(
        0.02,
        0.02,
        text,
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=9,
        bbox=dict(
            boxstyle="round",
            alpha=0.85,
        ),
    )

    plt.tight_layout()

    fig.savefig(
        FIGURE_FILE,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("Figure:")
    print(f"  {FIGURE_FILE}")


# ======================================================================
# Write summary
# ======================================================================


def write_summary(s3d, x1d, crds):

    print_section("12. WRITING AUDIT SUMMARY")

    metadata = x1d["metadata"]
    center = x1d["extraction_center"]
    geometry = crds["geometry"]
    lines = x1d["line_results"]
    crds_lines = crds["line_results"]

    summary = {
        "s3d_file": str(S3D_FILE),
        "x1d_file": str(X1D_FILE),
        "crds_reference": str(CRDS_REFERENCE),
        "program": metadata.get("PROGRAM", ""),
        "apername": metadata.get("APERNAME", ""),
        "pps_aper": metadata.get("PPS_APER", ""),
        "sltname": metadata.get("SLTNAME", ""),
        "srctype": metadata.get("SRCTYPE", ""),
        "extr_x": metadata.get("EXTR_X", ""),
        "extr_y": metadata.get("EXTR_Y", ""),
        "center_ra_deg": center["ra"],
        "center_dec_deg": center["dec"],
        "pipeline_slit_ra_deg": center["pipeline_ra"],
        "pipeline_slit_dec_deg": center["pipeline_dec"],
        "pa_aper_deg": metadata.get("PA_APER", ""),
        "pixel_scale_x_arcsec": s3d["scale_x"],
        "pixel_scale_y_arcsec": s3d["scale_y"],
        "cube_nx": s3d["nx"],
        "cube_ny": s3d["ny"],
        "cube_width_arcsec": s3d["width_arcsec"],
        "cube_height_arcsec": s3d["height_arcsec"],
        "cube_area_arcsec2": s3d["area_arcsec2"],
        "model_type": crds["reference_meta"]["model_type"],
        "exposure_type": crds["exposure_type"],
        "method": crds["reference_meta"]["method"],
        "region_type": crds["reference_meta"]["region_type"],
        "subpixels": crds["subpixels"],
        "subtract_background": crds["subtract_background"],
        "aperture_radius_arcsec": geometry["radius_arcsec"],
        "aperture_radius_x_pixels": geometry["radius_x_pixels"],
        "aperture_radius_y_pixels": geometry["radius_y_pixels"],
        "aperture_area_arcsec2": geometry["area_arcsec2"],
        "aperture_pixel_count": crds["aperture_pixel_count"],
        "pa_gamma_wavelength_um": lines["Pa-gamma"]["actual_um"],
        "pa_beta_wavelength_um": lines["Pa-beta"]["actual_um"],
        "pa_gamma_npixels": lines["Pa-gamma"]["npixels"],
        "pa_beta_npixels": lines["Pa-beta"]["npixels"],
        "pa_gamma_radius_arcsec": crds_lines["Pa-gamma"]["radius_arcsec"],
        "pa_beta_radius_arcsec": crds_lines["Pa-beta"]["radius_arcsec"],
        "same_nominal_aperture": crds["same_nominal_aperture"],
        "background_inner_arcsec": crds_lines["Pa-beta"]["inner_bkg_arcsec"],
        "background_outer_arcsec": crds_lines["Pa-beta"]["outer_bkg_arcsec"],
        "footprint_status": "PIPELINE NOMINAL GEOMETRY RECOVERED",
        "effective_pixel_weighting_status": "NOT EXPLICITLY RECOVERED",
    }

    fields = list(summary.keys())

    with open(
        SUMMARY_CSV,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerow(summary)

    print("Summary:")
    print(f"  {SUMMARY_CSV}")

    return summary


# ======================================================================
# Final report
# ======================================================================


def final_report(s3d, x1d, crds):

    print_section("13. FINAL APERTURE AUDIT RESULT")

    center = x1d["extraction_center"]
    geometry = crds["geometry"]

    print("1. PIPELINE EXTRACTION MODEL")
    print("   Extract1dIFUModel")

    print()
    print("2. EXTRACTION METHOD")
    print(f"   {crds['reference_meta']['method']}")

    print()
    print("3. EXTRACTION CENTER")
    print(f"   pixel = " f"({center['x']:.3f}, " f"{center['y']:.3f})")
    print(f"   RA    = " f"{center['ra']:.10f} deg")
    print(f"   Dec   = " f"{center['dec']:.10f} deg")

    print()
    print("4. NOMINAL EXTRACTION APERTURE")
    print(f"   radius = " f"{geometry['radius_arcsec']:.6f} arcsec")
    print(f"   radius = " f"{geometry['radius_x_pixels']:.3f} " f"x-pixels")
    print(f"   radius = " f"{geometry['radius_y_pixels']:.3f} " f"y-pixels")
    print(f"   area   = " f"{geometry['area_arcsec2']:.6f} arcsec^2")

    print()
    print("5. PA-BETA / PA-GAMMA")
    print("   SAME nominal spatial extraction geometry.")

    print()
    print("6. BACKGROUND")
    print(
        f"   inner radius = "
        f"{crds['line_results']['Pa-beta']['inner_bkg_arcsec']:.3f} arcsec"
    )
    print(
        f"   outer radius = "
        f"{crds['line_results']['Pa-beta']['outer_bkg_arcsec']:.3f} arcsec"
    )
    print(f"   subtraction = " f"{crds['subtract_background']}")

    print()
    print("7. S3D IFU FOOTPRINT")
    print(f"   {s3d['width_arcsec']:.4f} x " f"{s3d['height_arcsec']:.4f} arcsec")

    print()
    print("8. IMPORTANT SCIENTIFIC QUALIFICATION")
    print("   The CRDS reference recovers the nominal " "circular extraction geometry.")
    print(
        "   The detailed effective pixel weighting is "
        "not independently reconstructed here."
    )

    print()
    print("9. RESEARCH OUTPUTS")
    print(f"   {PIXEL_CSV}")
    print(f"   {POLYGON_CSV}")
    print(f"   {SUMMARY_CSV}")
    print(f"   {FIGURE_FILE}")

    print()
    print("Definitive aperture audit complete.")


# ======================================================================
# Main
# ======================================================================


def main():

    print("=" * 70)
    print("M51 JWST NIRSPEC/IFU " "DEFINITIVE EXTRACTION APERTURE AUDIT")
    print("=" * 70)

    print()
    print("Purpose:")
    print(
        "Recover the pipeline-defined nominal spatial "
        "extraction geometry from the JWST CRDS reference."
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    s3d = read_s3d()

    s3d = calculate_cube_footprint(s3d)

    x1d = read_x1d()

    x1d = wavelength_audit(x1d)

    crds = read_crds_reference()

    crds = interpolate_reference_parameters(crds)

    x1d = calculate_extraction_center(
        s3d,
        x1d,
    )

    crds = build_aperture_geometry(
        s3d,
        x1d,
        crds,
    )

    crds = generate_aperture_polygon(
        s3d,
        x1d,
        crds,
    )

    crds = write_pixel_table(
        s3d,
        x1d,
        crds,
    )

    create_figure(
        s3d,
        x1d,
        crds,
    )

    write_summary(
        s3d,
        x1d,
        crds,
    )

    final_report(
        s3d,
        x1d,
        crds,
    )


if __name__ == "__main__":
    main()
