"""
Catalog utilities for the Cosmic Intelligence Lab.

This module contains reusable functions for loading
astronomical catalogs used throughout the project.
"""

from pathlib import Path

from astropy.table import Table


def load_jades_catalog(project_root):
    """
    Load the official JADES DR5 GOODS-S photometric catalog.

    Parameters
    ----------
    project_root : pathlib.Path
        Root directory of the Cosmic Intelligence Lab project.

    Returns
    -------
    astropy.table.Table
        JADES DR5 photometric catalog.
    """

    filename = (
        Path(project_root)
        / "data"
        / "catalogs"
        / "jades"
        / "dr5"
        / "hlsp_jades_jwst_nircam_goods-s_photometry_v5.0_catalog.fits"
    )

    print("=" * 60)
    print("Cosmic Intelligence Lab")
    print("Loading JADES DR5 Photometric Catalog")
    print("=" * 60)
    print(filename)
    print()

    catalog = Table.read(filename, memmap=True)

    print(f"Loaded {len(catalog):,} catalog entries.")

    return catalog
