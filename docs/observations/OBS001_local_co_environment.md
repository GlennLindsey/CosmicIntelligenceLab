# OBS001 — Local CO Environment of ASPECS C19

**Project:** Cosmic Intelligence Lab

**Observation ID:** OBS001

**Status:** Completed (Phase 1)

---

# Scientific Objective

Determine whether ASPECS C19 is isolated or resides within a physically associated environment by examining nearby ASPECS CO detections.

---

# Target

**ASPECS C19**

RA (J2000): 03:32:36.19

Dec (J2000): −27:46:28.0

Redshift:

z = 2.574

---

# Data Sources

- ASPECS CO Line Catalog (Decarli et al.)
- Einstein Master Catalog
- Astropy SkyCoord coordinate calculations
- Planck18 cosmology

---

# Methods

The ASPECS CO catalog was imported into Python and converted into celestial coordinates using Astropy.

Angular separations between every CO detection and C19 were calculated.

For each source we computed:

- Angular separation
- Redshift difference (Δz)
- Approximate velocity difference
- Projected physical separation

---

# Results

## CO detections within 10"

| ID | z | Separation |
|----|------|-----------|
| C19 | 2.574 | 0.34" |
| 15 | 1.096 | 5.49" |
| 8 | 1.382 | 8.30" |
| MP02 | 1.087 | 9.54" |

Only C19 is at the target redshift.

---

## CO detections within 20"

No additional galaxies consistent with the redshift of C19 were identified.

---

## CO detections within 30"

One additional galaxy appears:

| ID | z | Separation |
|----|------|-----------|
| 14 | 1.098 | 21.98" |

Its redshift differs substantially from C19.

---

# Same-Redshift Candidates

Only one ASPECS CO source satisfies

Δz < 0.05

| ID | z | Δz | Separation |
|----|------|------|-----------|
| 1 | 2.543 | 0.031 | 31.88" |

---

# Projected Physical Separation

Assuming the cosmology adopted in this project,

CO ID 1 lies approximately

**262 kpc**

from C19 in projected distance.

---

# Interpretation

Within approximately 30 arcseconds, C19 does **not** appear to belong to a compact group of CO emitters.

However,

CO ID 1 is remarkable because

- its redshift differs by only Δz = 0.031,
- it lies only ~262 kpc away in projection.

This separation is consistent with galaxies inhabiting the same larger-scale environment or forming part of a common overdensity.

---

# Conclusions

The immediate surroundings of C19 contain no nearby CO emitters at comparable redshift.

Nevertheless, CO ID 1 is identified as an excellent candidate for further investigation owing to its small redshift difference and moderate projected separation.

Whether these galaxies are physically associated requires additional observational evidence.

---

# Follow-up Observations

OBS003 — ALMA continuum analysis

Determine whether CO ID 1 also exhibits detectable 1.2 mm continuum emission.

Future observations should include

- continuum comparison,
- JWST morphology,
- HST morphology,
- NED and SIMBAD cross-identifications,
- stellar mass comparison,
- star-formation rate comparison.

---

# Scientific Outcome

OBS001 establishes that C19 is not embedded within a compact concentration of nearby CO emitters, but identifies CO ID 1 as a strong candidate member of the same larger-scale environment.

This observation motivates subsequent ALMA and JWST analyses.