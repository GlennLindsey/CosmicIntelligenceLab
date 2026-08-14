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
