# M51 JWST Research Notes

**Project:** Cosmic AI  
**Investigation:** M51 JWST archival observations  
**Notebook:** `notebooks/01_m51_jwst_exploration.ipynb`

---

## 1. Research Objective

Investigate the actual JWST observations returned by the MAST archive for M51.

The purpose of this investigation is to understand the structure of JWST archival
data before designing the next Cosmic AI astronomical research tool.

Particular attention is being given to:

- JWST observation records
- observing proposals
- instruments and observing modes
- filters and spectroscopic configurations
- target names
- calibration levels
- MAST Product Group IDs (`obsid`)
- associated science products
- FITS data products

The eventual goal is to develop a robust workflow that can search archival
astronomical data in support of transient and supernova research.

---

## 2. Method

The investigation is being conducted in JupyterLab using:

- Python
- `astroquery`
- `astropy`
- MAST `Observations`

The initial search was:

```python
Observations.query_object(
    "M51",
    radius="0.02 deg"
)

## 3. Initial MAST Results

The MAST query returned observations associated with M51. The results
included observations from multiple missions.

The Cosmic AI MAST tool reported:

- Total observations: 2,613
- JWST observations: 214
- HST observations: 1,181

The JWST observations were then examined in greater detail in JupyterLab.

## 4. JWST Observation Investigation

A JWST observation was selected for detailed examination:

- Proposal: 3435
- Observation: `jw03435-o006_t010_nirspec_g140m-f100lp`
- Target: `M51-N-AURORAL`
- Instrument: NIRSpec/IFU
- Disperser: G140M
- Filter: F100LP
- Calibration level: Level 3

## 5. JWST Data Products

The selected observation has associated Level-3 products including an
S3D data cube and an X1D extracted spectrum.

Local copies were obtained for analysis:

- `jw03435-o006_t010_nirspec_g140m-f100lp_s3d.fits`
- `jw03435-o006_t010_nirspec_g140m-f100lp_x1d.fits`

The S3D and X1D products were initially visualized in JupyterLab.
Further scientific interpretation remains to be completed.

---

## 6. NIRSpec/IFU Spectral Feature Investigation - 2026-08-13

The Level-3 NIRSpec/IFU S3D science cube was examined to identify
potentially significant spectral features.

The SCI array has:

- Shape: `(1447, 97, 125)`
- Unit: `MJy/sr`
- Wavelength range: approximately 0.9703–1.8900 µm

The maximum finite value in the cube was:

- **Maximum surface brightness:** 23054.594 MJy/sr
- **Spectral index:** 1104
- **Spatial position:** `(x=74, y=62)`
- **Wavelength:** 1.6724620414 µm

The spectrum extracted from spatial pixel `(74, 62)` shows a very
strong feature centered near 1.67246 µm.

The spectral samples surrounding the feature were:

| Wavelength (µm) | Surface brightness (MJy/sr) |
|---:|---:|
| 1.671190041 | 35.591 |
| 1.671826041 | 1962.812 |
| 1.672462041 | 23054.594 |
| 1.673098041 | 2232.636 |
| 1.673734041 | 116.984 |
| 1.674370041 | 20.399 |

The feature therefore extends across multiple adjacent spectral
samples rather than being confined to a single spectral sample.

A spatial comparison at the peak wavelength also showed substantial
signal in neighboring spatial pixels:

| Spatial position | Surface brightness (MJy/sr) |
|---|---:|
| `(74, 62)` | 23054.594 |
| `(75, 62)` | 8544.437 |
| `(73, 62)` | 494.777 |
| `(74, 63)` | -0.599 |
| `(74, 61)` | 8657.464 |

These measurements provide preliminary evidence that the feature has
both spectral and spatial structure.

At this stage, the physical origin of the feature has **not** been
established. It has not yet been identified as a particular emission
line, astrophysical phenomenon, or transient.

The next step is validation using the associated Level-3 X1D product:

`jw03435-o006_t010_nirspec_g140m-f100lp_x1d.fits`

The purpose of this comparison is to determine whether the feature
identified in the S3D cube is independently reproduced in the
extracted X1D spectrum.

### X1D Product Validation

The associated Level-3 X1D product was examined to determine whether
the spectral feature identified in the S3D cube is reproduced in the
extracted spectrum.

The X1D `EXTRACT1D` table contains 1,447 spectral samples and includes
wavelength, flux, surface-brightness, uncertainty, data-quality,
background, variance, and extraction-pixel information.

The X1D wavelength axis has the following properties:

- **Unit:** µm
- **First wavelength:** 0.9703180286 µm
- **Last wavelength:** 1.8899740454 µm
- **Number of spectral samples:** 1,447

The S3D feature occurs at:

- **Wavelength:** 1.6724620414 µm
- **S3D maximum:** 23054.594 MJy/sr
- **S3D spatial position:** `(x=74, y=62)`

The X1D spectrum contains a corresponding feature at:

- **X1D index:** 1104
- **X1D wavelength:** 1.6724620414 µm
- **SURF_BRIGHT:** 16.119589 MJy/sr
- **SB_ERROR:** 0.199882 MJy/sr
- **DQ:** 0
- **NPIXELS:** 4698

The agreement in wavelength between the S3D and X1D products provides
strong evidence that the feature identified in the S3D cube is also
present in the associated extracted X1D spectrum.

The X1D product is derived from the same JWST observation and therefore
does not constitute an independent observational confirmation.

The physical origin of the feature has not yet been established. No
specific emission line or astrophysical process has yet been assigned
to it.

The next step is to examine the X1D uncertainty and data-quality
information across the feature.

### Background-Subtracted X1D Spectrum

The X1D surface-brightness spectrum was examined after subtracting the
pipeline-provided `BACKGROUND` column.

At the feature wavelength:

- **Wavelength:** 1.672462041 µm
- **SURF_BRIGHT:** 16.1196 MJy/sr
- **BACKGROUND:** 7.2064 MJy/sr
- **Background-subtracted excess:** 8.9132 MJy/sr
- **SB_ERROR:** 0.1999 MJy/sr
- **DQ:** 0

The background remains smooth through the feature and does not show a
corresponding peak. The feature therefore persists after subtraction
of the reported background.

The neighboring background-subtracted values are:

| Wavelength (µm) | Excess (MJy/sr) |
|---:|---:|
| 1.671826041 | 0.9639 |
| 1.672462041 | 8.9132 |
| 1.673098041 | 1.1931 |
| 1.673734041 | 0.4011 |

The result provides additional evidence that the feature is present in
the extracted spectrum and is not explained by a corresponding
background enhancement.

The physical identity of the feature has not been established.
In particular, it has not yet been identified as a specific emission
line or associated with a supernova or other transient.

The statistical significance of the feature also requires further
careful treatment because the X1D surface-brightness variance-component
columns examined previously are zero. The pipeline-provided
`SB_ERROR` should therefore not be treated as a complete independent
line-significance analysis.

The next stage is to investigate possible physical and spectroscopic
identifications of the 1.672462 µm feature using authoritative
astronomical/atomic data.

---

## 7. Initial Spectroscopic Identification Investigation - 2026-08-13

The observed feature at 1.672462041 µm was compared with external
astronomical and atomic spectroscopy information to investigate
possible physical identifications.

The purpose of this stage is to generate and test candidate
identifications. A wavelength match alone is not considered sufficient
to identify the feature.

### 7.1 Candidate: Al I

Published near-infrared stellar spectroscopy identifies a group of
neutral aluminum (Al I) lines near this wavelength. APOGEE uses three
near-infrared Al I lines at approximately:

- 1.67235 µm (16723.5 Å, vacuum)
- 1.6755 µm (16755 Å, vacuum)
- 1.6768 µm (16768 Å, vacuum)

The observed JWST feature at:

- **1.672462041 µm**

is therefore very close in wavelength to the Al I feature near
1.67235 µm.

However, the astronomical literature identifies these Al I features
primarily as stellar photospheric absorption features. The present
M51 X1D spectrum shows a positive, background-subtracted spectral
feature.

Therefore, the wavelength agreement alone does not establish an Al I
identification.

The Al I hypothesis remains a candidate requiring further testing.

### 7.2 Candidate: [Fe II]

Near-infrared [Fe II] lines are important diagnostics of ionized and
shocked gas. Published infrared spectroscopy identifies [Fe II]
features near 1.644 µm, 1.664 µm, and 1.677 µm, and their relative
strengths can be used as diagnostics of physical conditions such as
electron density.

The observed feature at 1.672462041 µm is not an obvious direct match
to the commonly used [Fe II] wavelengths.

A velocity-shifted identification cannot be ruled out at this stage,
but it requires further quantitative analysis.

The [Fe II] hypothesis therefore remains unestablished.

### 7.3 Hydrogen recombination lines

The STScI infrared line list gives the following Brackett-series
vacuum wavelengths:

- Brackett 12-4: 1.6412 µm
- Brackett 11-4: 1.6811 µm
- Brackett 10-4: 1.7367 µm

The observed feature at 1.672462041 µm does not directly coincide with
these listed hydrogen recombination lines.

Hydrogen therefore does not currently provide an obvious
identification of the feature.

### 7.4 Current scientific assessment

The external comparison has produced candidate hypotheses but has not
established the physical identity of the feature.

Current evidence:

- The feature is observed in the Level-3 S3D cube.
- It has spatial structure.
- It has spectral structure.
- It is reproduced in the associated X1D product.
- The X1D data-quality value at the feature is `DQ = 0`.
- The reported background remains smooth through the feature.
- The feature remains prominent after subtraction of the reported
  background.
- The observed wavelength is close to a known Al I near-infrared
  feature.
- The observed wavelength is not an obvious direct match to the
  hydrogen Brackett lines examined.
- [Fe II] remains a possible but presently unestablished alternative.

The physical origin of the feature is therefore **NOT ESTABLISHED**.

### 7.5 Next investigation

The next step is to test candidate identifications using their expected
companion spectral features rather than relying on the wavelength of a
single feature.

In particular:

1. Search the complete X1D spectrum for other Al I features near
   1.6755 and 1.6768 µm.
2. Search for the expected near-infrared [Fe II] features, including
   the 1.644 µm and 1.677 µm regions.
3. Examine the relative strengths and profiles of candidate lines.
4. Consider the wavelength calibration and possible velocity shifts.
5. Determine whether the spatial distribution of candidate features
   is consistent with the feature identified in the S3D cube.

No candidate should be identified as the physical origin of the M51
feature until the broader spectral evidence supports the
identification.
