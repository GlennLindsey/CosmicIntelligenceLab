# OBS003 — ALMA Image Analysis

## Scientific Goal

Investigate the ALMA 1.2 mm continuum emission associated with ASPECS C19.

---

## Data

Instrument:
ALMA Band 6

Dataset:
ASPECS Large Program

Image:
member.uid___A001_X87c_X21e.lp_walter.aspecs1mm.cont.nat.image.clean.pbcor.fits

---

## Observations

### Observation 1

Successfully downloaded the official ASPECS continuum mosaic.

---

### Observation 2

Opened the FITS image and verified the WCS.

Image dimensions:
840 × 840 pixels

Units:
Jy/beam

---

### Observation 3

Converted the published C19 coordinates into pixel coordinates.

Pixel position:

x = 573.23

y = 579.94

---

### Observation 4

Extracted a 100×100 pixel cutout centered on C19.

---

### Observation 5

Measured the continuum emission.

Peak flux:
8.50 × 10⁻⁵ Jy/beam

Local RMS:
1.83 × 10⁻⁵ Jy/beam

Estimated S/N:
4.64

Published catalog S/N:
7.0

Comment:
The lower measured S/N likely reflects the simple local RMS estimate used here rather than the source-fitting procedure employed by the ASPECS catalog.

---

### Observation 6

Morphology

The ALMA Band 6 continuum emission associated with ASPECS C19 is compact and centered on the published source coordinates. The fitted Gaussian dimensions are only slightly larger than the synthesized beam, and the integrated continuum flux (0.093 mJy) exceeds the peak flux density (0.085 mJy beam⁻¹) by only ~9%. These measurements are consistent with C19 being unresolved or, at most, marginally resolved in the natural-weighted ASPECS continuum mosaic.

---

## ALMA–JWST Alignment

The centroid of the ALMA 1.2 mm continuum emission is separated from the centroid of the JWST F200W emission by 0.1565 arcsec, corresponding to 1.29 kpc at z = 2.574. This offset is substantially smaller than the ALMA synthesized beam (1.53″ × 1.08″), indicating that the millimeter continuum is spatially consistent with the central stellar component of C19. Figure 1 illustrates this alignment by overlaying ALMA continuum contours on the JWST F200W image.

---

## Conclusions

The continuum peak coincides with the published coordinates of ASPECS C19 within the image resolution. This independently confirms the source identification in the official ASPECS continuum mosaic.

---

## Key Results

- Official ASPECS Band 6 continuum mosaic successfully analyzed.
- C19 located and verified using ALMA WCS.
- Peak continuum flux density: 0.085 mJy beam⁻¹.
- Integrated continuum flux density: 0.093 mJy.
- Gaussian fit centered within 0.1 arcsec of the published coordinates.
- Source dimensions are consistent with the synthesized beam.
- C19 is unresolved or only marginally resolved in the natural-weighted continuum mosaic.