"""
Catalog utilities for the Cosmic Intelligence Lab.

Reusable functions for loading and preparing astronomical catalogs.
"""

from pathlib import Path

from astropy.table import Table, join

# ==========================================================
# JADES Catalog Location
# ==========================================================

JADES_FILENAME = "hlsp_jades_jwst_nircam_goods-s_photometry_v5.0_catalog.fits"


# ==========================================================
# Load One JADES Table
# ==========================================================


def load_jades_table(project_root, table_name):
    """
    Load one table from the official JADES DR5 catalog.

    Parameters
    ----------
    project_root : pathlib.Path
        Root directory of the Cosmic Intelligence Lab project.

    table_name : str
        FITS extension name (e.g. KRON, PHOTOZ, SIZE, FLAG).

    Returns
    -------
    astropy.table.Table
        Requested JADES table.
    """

    filename = (
        Path(project_root) / "data" / "catalogs" / "jades" / "dr5" / JADES_FILENAME
    )

    print("=" * 60)
    print("Cosmic Intelligence Lab")
    print(f"Loading JADES table: {table_name}")
    print("=" * 60)

    table = Table.read(
        filename,
        hdu=table_name,
        memmap=True,
    )

    print(f"Rows    : {len(table):,}")
    print(f"Columns : {len(table.colnames)}")
    print()

    return table


# ==========================================================
# Load Working JADES Catalog
# ==========================================================


def load_working_jades_catalog(project_root):
    """
    Load and join the KRON and PHOTOZ tables.

    Returns
    -------
    astropy.table.Table
        Working JADES catalog.
    """

    kron = load_jades_table(project_root, "KRON")

    photoz = load_jades_table(project_root, "PHOTOZ")

    jades = join(
        kron,
        photoz,
        keys="ID",
        join_type="inner",
    )

    print("=" * 60)
    print("Working JADES Catalog")
    print("=" * 60)
    print(f"Rows    : {len(jades):,}")
    print(f"Columns : {len(jades.colnames)}")
    print()

    return jades


# ==========================================================
# Select Core Columns
# ==========================================================


def select_jades_core_columns(jades):
    """
    Select the columns most useful for local environment studies.

    Parameters
    ----------
    jades : astropy.table.Table

    Returns
    -------
    astropy.table.Table
        Reduced working catalog.
    """

    columns = [
        "ID",
        "RA",
        "DEC",
        "A_KRON",
        "B_KRON",
        "THETA_KRON",
        "z_spec",
        "z_peak",
        "z_ml",
    ]

    return jades[columns]


# ==========================================================
# Select Core JADES Columns
# ==========================================================


def select_jades_core_columns(jades):
    """
    Select the columns most useful for local environment studies.

    Parameters
    ----------
    jades : astropy.table.Table
        Working JADES catalog.

    Returns
    -------
    astropy.table.Table
        Reduced catalog containing only the core columns.
    """

    columns = [
        "ID",
        "RA",
        "DEC",
        "A_KRON",
        "B_KRON",
        "THETA_KRON",
        "z_spec",
        "z_peak",
        "z_ml",
    ]

    return jades[columns]


# ==========================================================
# Build JADES Sky Coordinates
# ==========================================================

from astropy.coordinates import SkyCoord


def build_jades_coordinates(jades_core):
    """
    Build SkyCoord objects from the JADES core catalog.

    Parameters
    ----------
    jades_core : astropy.table.Table

    Returns
    -------
    astropy.coordinates.SkyCoord
    """

    return SkyCoord(
        ra=jades_core["RA"],
        dec=jades_core["DEC"],
        frame="icrs",
    )


# ==========================================================
# Cone Search JADES Catalog
# ==========================================================

import astropy.units as u


def cone_search_jades(
    jades_core,
    jades_coords,
    center,
    radius=30 * u.arcsec,
):
    """
    Perform a cone search on the JADES core catalog.

    Parameters
    ----------
    jades_core : astropy.table.Table
        Core JADES catalog.

    jades_coords : astropy.coordinates.SkyCoord
        Sky coordinates corresponding to jades_core.

    center : astropy.coordinates.SkyCoord
        Target position.

    radius : astropy.units.Quantity
        Search radius (e.g. 30 * u.arcsec).

    Returns
    -------
    astropy.table.Table
        Nearby galaxies with an added 'separation_arcsec' column.
    """

    # Compute angular separations
    separations = center.separation(jades_coords)

    # Select galaxies inside the cone
    inside = separations <= radius

    # Create a copy so the original catalog is untouched
    neighbors = jades_core[inside].copy()

    # Add separation column (in arcseconds)
    neighbors["separation_arcsec"] = separations[inside].to(u.arcsec).value

    # Sort by increasing separation
    neighbors.sort("separation_arcsec")

    return neighbors
