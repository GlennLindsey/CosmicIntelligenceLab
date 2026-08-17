# Notebook 21 – Object 39 Knowledge Report

**Project:** Cosmic Intelligence Lab  
**Research Assistant:** Einstein  
**Notebook:** 21 – Comparative Galaxy Analysis  
**Target Object:** Einstein Object 39 (EIS J033238.61-274631.6)

---

# Objective

The objective of Notebook 21 was to investigate the local environment of
Einstein Object 39 by comparing it with nearby galaxies in the GOODS-MUSIC
catalog. This notebook extends the object-level knowledge developed in
Notebook 20 by examining the galaxy's surroundings, photometric properties,
and relationship to neighbouring galaxies of similar redshift.

Specifically, this notebook aimed to:

- Cross-match Object 39 with the GOODS-MUSIC catalog.
- Identify nearby galaxies within the GOODS-South field.
- Select galaxies having similar redshifts.
- Compare infrared photometric properties.
- Produce visualizations of the local galaxy environment.
- Generate an automated scientific interpretation.

---

# Data Sources

The analysis used the following resources:

- Einstein Master Catalog
- GOODS-MUSIC Catalog
- CDS TAP Service (Centre de Données astronomiques de Strasbourg)
- Astropy
- Astroquery
- NumPy
- Matplotlib
- Python / Jupyter Notebook

---

# Target Object

| Property | Value |
|----------|------|
| Einstein Label | 39 |
| Object Name | EIS J033238.61-274631.6 |
| Object Type | Galaxy |
| Right Ascension | 53.160803° |
| Declination | -27.775437° |
| Einstein Redshift | 0.621743 |

---

# GOODS-MUSIC Cross-Match

A cone search was performed around the target coordinates using the CDS TAP
service.

Search parameters:

- Search radius: 0.01 degrees (36 arcseconds)
- Initial GOODS-MUSIC objects returned: 120

Each source was assigned a "best redshift," using spectroscopic redshift when
available and photometric redshift otherwise.

Galaxies were then filtered to satisfy:

```
| best_z − target_z | ≤ 0.05
```

Stars were removed from the comparison sample.

Final comparison sample:

- **8 nearby galaxies**

---

# Identification of the GOODS-MUSIC Counterpart

The nearest GOODS-MUSIC source was identified by calculating angular
separations between Object 39 and every galaxy in the comparison sample.

The closest source is:

| Property | Value |
|----------|------|
| GOODS-MUSIC Sequence | 10989 |
| Angular Separation | 0.215 arcseconds |
| Photometric Redshift | 0.640 |

The extremely small positional offset provides strong confidence that this is
the correct GOODS-MUSIC counterpart.

---

# Photometric Properties

The following infrared magnitudes were extracted for the identified
counterpart.

| Band | Magnitude |
|------|----------:|
| J | 21.075 |
| H | 20.730 |
| Ks | 20.489 |
| IR3.6 | 20.838 |
| IR4.5 | 21.232 |

These values were subsequently used to compare Object 39 with neighbouring
galaxies.

---

# Local Galaxy Environment

The comparison sample represents galaxies within:

- 36 arcseconds of Object 39
- ±0.05 in redshift

This provides a reasonable approximation of the local galaxy environment.

Analysis of the comparison sample indicates:

- Object 39 is neither unusually bright nor unusually faint in the Ks band.
- Its infrared brightness lies near the middle of the local distribution.
- No galaxies in the comparison sample are flagged as active galactic nuclei
  (AGN).
- The surrounding galaxies appear representative of a normal galaxy
  population.

---

# Figures Produced

Notebook 21 generated the following figures:

1. Sky map showing Object 39 and nearby comparison galaxies.
2. Redshift distribution of nearby galaxies.
3. Ks Magnitude versus Redshift comparison plot.

These visualizations provide complementary views of the spatial and
photometric properties of the local galaxy environment.

---

# Scientific Findings

Notebook 21 successfully established a reliable cross-identification between
Einstein Object 39 and its GOODS-MUSIC counterpart.

The positional agreement of only 0.215 arcseconds strongly supports the
association between the two catalogs.

Comparison with neighbouring galaxies indicates that Object 39 occupies a
typical position within its local environment. Its Ks-band luminosity is
representative of nearby galaxies having similar redshifts, and no evidence
was found that it resides within an AGN-dominated population.

The analysis demonstrates that Object 39 appears to be a normal galaxy within
its immediate environment rather than an obvious outlier.

---

# Conclusions

Notebook 21 extends the work of Notebook 20 by moving beyond individual object
identification toward environmental analysis.

The notebook successfully:

- identified the GOODS-MUSIC counterpart,
- characterized the local galaxy population,
- compared infrared photometry,
- visualized the surrounding environment,
- and produced an automated scientific assessment.

This notebook establishes a reusable workflow for analysing additional
Einstein catalog objects using the same methodology.

---

# Future Work

Potential extensions include:

- Cross-matching with ALMA observations.
- Investigating millimetre continuum detections.
- Estimating star-formation activity.
- Measuring projected physical separations.
- Studying local galaxy density.
- Constructing colour–magnitude diagrams.
- Applying the workflow to additional Einstein catalog galaxies.

Notebook 22 will expand the analysis by incorporating observations at
millimetre wavelengths, providing a multi-wavelength view of Object 39 and
its surrounding environment.

---

# Summary

Notebook 21 demonstrates a complete comparative galaxy analysis workflow.

Beginning with an Einstein catalog object, the notebook identified a reliable
counterpart in GOODS-MUSIC, constructed a carefully filtered comparison
sample, compared photometric properties, visualized the local environment, and
generated a reproducible scientific interpretation.

The workflow provides a strong foundation for future multi-wavelength studies
within the Cosmic Intelligence Lab project.
