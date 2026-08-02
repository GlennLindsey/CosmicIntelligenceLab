# Notebook 22 – Identify ALMA Observations

## Purpose

This notebook searches the ALMA Science Archive for observations associated with
Einstein Object 39 (EIS J033238.61−274631.6).

Rather than querying individual published surveys, the notebook performs a
direct search of the official ALMA Science Archive using the International
Virtual Observatory Alliance (IVOA) Table Access Protocol (TAP). This approach
provides a comprehensive inventory of archival ALMA observations surrounding
the target galaxy.

The notebook identifies nearby observing programs, computes their angular
separation from the target, ranks the observations by proximity, and summarizes
the available observing programs for future scientific analysis.

---

## Scientific Goals

- Retrieve the coordinates of Einstein Object 39.
- Connect to the ALMA Science Archive TAP service.
- Search for ALMA observations within a configurable search radius.
- Compute angular separations between the target galaxy and nearby ALMA
  pointing centres.
- Group multiple spectral windows into unique observing programs.
- Rank observing programs by proximity.
- Produce an AI-generated scientific assessment of the available ALMA archive
  coverage.

---

## Inputs

- Einstein Master Catalog
  (`einstein_master_catalog.ecsv`)

- Target Object
  - Einstein Label: 39
  - Name: EIS J033238.61−274631.6
  - Type: Galaxy
  - Redshift: 0.621743

---

## Outputs

The notebook produces:

- Complete list of ALMA archive observations near the target
- Summary of unique observing programs
- Ranked observing-program table
- Einstein scientific assessment
- Archive metadata for future notebooks

---

## Scientific Value

This notebook establishes the observational history of Object 39 within the
ALMA archive.

Rather than relying on published catalogs, it directly interrogates the archive,
ensuring that recently acquired observations and newly released programs are
included.

The resulting observing-program inventory serves as the foundation for later
analysis of millimetre continuum emission, molecular gas observations, and
multi-wavelength comparisons.

---

## Relationship to Subsequent Notebooks

Notebook 23 builds upon this work by integrating these ALMA results with
information from astronomical databases and imaging surveys to create a
comprehensive Object Knowledge Report.

Future notebooks may retrieve calibrated ALMA image products and spectral data
for detailed scientific analysis.

---

## Key Python Packages

- astroquery
- pyvo
- astropy
- numpy

---

## Data Sources

- ALMA Science Archive (ESO TAP Service)
- Einstein Master Catalog

---

## Author

Glenn Lindsey

Project: Cosmic Intelligence Lab

AI Research Assistant: Einstein
