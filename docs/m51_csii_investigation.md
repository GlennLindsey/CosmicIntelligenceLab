# M51 / Cs II 1284.264 nm Investigation

## Purpose

Investigate whether the strong emission feature near **1284.261 nm** in
the JWST/NIRSpec spectrum of M51 could plausibly be identified as a
highly excited Cs II transition.

The investigation began because a NIST Atomic Spectra Database query
returned a Cs II transition with an air wavelength extremely close to
the observed feature.

The primary alternative is **hydrogen Pa beta (Pa β)**.

The goal is not to assume either identification, but to test the two
hypotheses using independent wavelength, velocity, spectral-fitting,
spatial, and kinematic evidence.

---

# 1. Initial Observed Feature

The M51 JWST/NIRSpec X1D spectrum contains a strong emission feature at
approximately:

    1284.2613 nm

Initial Gaussian fitting gave:

    Gaussian center:
    1284.26130440 ± 0.00134611 nm

    Amplitude:
    0.00841121 ± 0.00001887

    Sigma:
    0.545346 ± 0.001538 nm

    FWHM:
    1.284192 ± 0.003621 nm

The feature has very high formal amplitude S/N:

    S/N ≈ 446

The simple Gaussian model, however, has a very large reduced chi-square,
so the absolute goodness of fit should not be interpreted literally.

---

# 2. NIST Cs II Candidate

A NIST Atomic Spectra Database search identified the following Cs II
transition:

    Observed wavelength:
    1284.26406 nm

    Ritz wavelength:
    1284.26391 nm

    Relative intensity:
    4500

    Lower energy:
    163189.3511 cm^-1

    Upper energy:
    170973.7833 cm^-1

    Lower configuration:
    5p5 (2P° 1/2) 7s

    Lower term:
    2 [1/2]°

    Upper configuration:
    5p5 (2P° 1/2) 4f

    Upper term:
    2 [5/2]

    Species:
    Cs II

    Line reference:
    L12126

The NIST wavelength is remarkably close to the measured feature when
expressed as an air wavelength.

---

# 3. Air-to-Vacuum Correction

The NIST wavelength is an air wavelength.

Using the adopted air-to-vacuum conversion:

    Cs II air wavelength:
    1284.26406000 nm

    Cs II vacuum wavelength:
    1284.61537587 nm

This distinction is critical because JWST/NIRSpec wavelengths are treated
as vacuum wavelengths.

The observed feature therefore cannot be compared directly with the
NIST air wavelength without conversion.

---

# 4. Same-Upper-Level Cs II Search

The upper energy of the candidate Cs II transition is:

    170973.7833 cm^-1

A NIST search for Cs II transitions sharing this upper energy produced
22 candidate transitions.

The same upper configuration is:

    5p5 (2P° 1/2) 4f

Several of these transitions are substantially stronger in the NIST
relative-intensity scale than the 1284 nm transition.

Examples include:

    315.11865 nm   relative intensity 29000
    686.37174 nm   relative intensity 4700
    211.22364 nm   relative intensity 1400
    234.37456 nm   relative intensity 740
    2375.6535 nm   relative intensity 1100
    2498.0764 nm   relative intensity 1300

The 1284.26406 nm transition has:

    relative intensity = 4500

This raised an important physical question:

> If Cs II were responsible for the 1284 nm feature, should other
> transitions arising from the same highly excited upper level also be
> observable?

The available M51 NIRSpec X1D spectrum covers approximately:

    970.318 - 1889.974 nm

Therefore only the 1284 nm transition among the strongest accessible
same-upper-level candidates falls inside the current spectrum.

The other strong companion transitions identified above lie outside the
available wavelength range.

---

# 5. Initial Companion-Line Search

A search was performed for the Cs II transitions arising from the same
upper level.

Accessible wavelength range:

    970.318 - 1889.974 nm

The only same-upper-level transition accessible in the current X1D
spectrum was:

    1284.26406 nm

Therefore the current dataset cannot provide a decisive companion-line
test.

This remains an important limitation.

---

# 6. Initial Local Spectral Measurement

A local ±3 nm window around the candidate wavelength was examined.

The nearest spectral sample occurred at approximately:

    1284.502 nm

The local peak was:

    1284.50203 nm

The original local measurement therefore showed a large apparent
offset from the NIST air wavelength.

This prompted a more rigorous Gaussian fitting and wavelength-frame
investigation.

---

# 7. NIRSpec Instrument and X1D Metadata

The relevant X1D product is:

    data/m51_jwst_level3/
    jw03435-o006_t010_nirspec_g140m-f100lp_x1d.fits

Instrument:

    JWST / NIRSpec

Detector:

    NRS1

Grating:

    G140M

Filter:

    F100LP

Observation:

    2024-04-06

Exposure type:

    NRS_IFU

Target description:

    M51-N-AURORAL

Target coordinates:

    RA  = 202.4820833 deg
    Dec = 47.1959000 deg

The X1D spectrum contains:

    1447 wavelength points

with wavelength range:

    970.318 - 1889.974 nm

Spectral sampling:

    approximately 0.636 nm

The X1D wavelength units are:

    microns

The flux units are:

    Jy

The spectrum is marked:

    SRCTYPE = EXTENDED

---

# 8. Initial Gaussian Fit and Cs II Vacuum Wavelength

The integrated feature was fitted with a Gaussian.

Measured center:

    1284.26130440 ± 0.00134611 nm

Cs II vacuum wavelength:

    1284.61537587 nm

Difference:

    -0.35407147 nm

Corresponding velocity difference:

    approximately -82.6 km/s

This result suggested that the feature would require a distinct velocity
if interpreted as Cs II.

---

# 9. Independent M51 Nebular Velocity Reference

Known strong nebular emission lines were fitted in the X1D spectrum to
measure the velocity actually present in the M51 spectrum.

The measured velocities included approximately:

    He I 1.0833 μm       +505.0 km/s
    Pa gamma             +583.1 km/s
    [Fe II] 1.257 μm     +573.7 km/s
    Pa beta              +573.4 km/s
    Brackett 17          +569.4 km/s
    Brackett 15          +539.7 km/s
    Brackett 13          +478.4 km/s
    [Fe II] 1.644 μm     +510.8 km/s
    Brackett 10          +461.7 km/s

Some weak or blended lines produced unreliable velocity fits and were
not treated as independent references.

The provisional empirical M51 nebular velocity reference was:

    Median = +525.25 km/s
    MAD    = 45.49 km/s

This reference is empirical and is used to test whether candidate
laboratory wavelengths are kinematically consistent with the observed
nebular gas.

---

# 10. Pa Beta versus Cs II at the Local M51 Velocity

Using the empirical median velocity:

    +525.25 km/s

the predicted wavelengths are:

    Pa beta:
    1284.05477630 nm

    Cs II:
    1286.86807693 nm

Observed:

    1284.26130440 nm

Thus Pa beta is much closer to the observed feature.

The velocity required to place each transition directly at the observed
wavelength is:

    Pa beta:
    +573.470 km/s

    Cs II:
    -82.642 km/s

Compared with the local M51 velocity reference:

    Pa beta difference:
    +48.215 km/s

    Cs II difference:
    -607.897 km/s

In MAD units:

    Pa beta:
    +1.06 MAD

    Cs II:
    -13.36 MAD

This strongly disfavors Cs II as the explanation for the feature under
the assumption that it originates in the same general nebular velocity
field.

---

# 11. Initial Fixed-Velocity Statistical Comparison

An initial fixed-center comparison was performed using the same local
continuum treatment and NIRSpec instrumental resolution.

The approximate instrument resolving power near the feature was:

    R ≈ 916

giving:

    Instrument sigma ≈ 0.595 nm
    Instrument FWHM ≈ 1.402 nm

The initial comparison gave:

    Pa beta χ² = 23671.322

    Cs II χ² = 128293.371

    Δχ² = 104622.050

in favor of Pa beta.

The absolute reduced χ² values were extremely large, so the formal
absolute χ² probabilities were not interpreted literally.

The comparison was treated as a model-discrimination experiment.

---

# 12. Constrained Pa Beta versus Cs II Fit

The comparison was subsequently refined using the independently measured
local [Fe II] velocity:

    +573.72 km/s

Predicted centers:

    Pa beta:
    1284.26002473 nm

    Cs II:
    1287.07377505 nm

### Pa beta

    Amplitude:
    0.0080310548

    Amplitude error:
    0.0000152400

    Amplitude S/N:
    526.97

    χ²:
    1839.627

    Degrees of freedom:
    16

    Reduced χ²:
    114.977

    AIC:
    1845.627

    BIC:
    1848.460

### Cs II

    Amplitude:
    approximately 1.7 × 10^-23

    Amplitude S/N:
    approximately 0

    χ²:
    279538.895

    Degrees of freedom:
    16

    Reduced χ²:
    17471.181

    AIC:
    279544.895

    BIC:
    279547.729

Therefore:

    Δχ² = +277699.268
    ΔAIC = +277699.268
    ΔBIC = +277699.268

in favor of Pa beta.

The Cs II constrained model produces effectively no positive emission
amplitude at its predicted wavelength.

---

# 13. Free-Centroid Pa Beta Fit

A free-centroid Gaussian fit gave:

    Center:
    1284.26127779 ± 0.00142234 nm

Assuming Pa beta:

    Velocity:
    +574.013 ± 0.333 km/s

This agrees extremely closely with the independently measured local
nebular velocity.

---

# 14. Preliminary S3D Spatial Extraction

The relevant Level-3 S3D cube is:

    data/m51_jwst_level3/
    jw03435-o006_t010_nirspec_g140m-f100lp_s3d.fits

Cube shape:

    1447 × 97 × 125

Wavelength range:

    970.318 - 1889.974 nm

Spectral sampling:

    0.636 nm

Spatial dimensions:

    125 × 97

BUNIT:

    MJy/sr

Three preliminary spatial maps were examined:

1. The 1284 nm feature
2. Pa beta
3. [Fe II] 1.257 μm

The 1284 nm feature and Pa beta use the same four S3D spectral planes:

    1283.2300 nm
    1283.8660 nm
    1284.5020 nm
    1285.1380 nm

Because the wavelength difference between the two hypotheses is much
smaller than the S3D spectral sampling, the preliminary 1284 nm and
Pa beta maps are identical.

Therefore:

    Pearson r = 1.00000

This is NOT independent evidence for Pa beta.

The [Fe II] map uses five nearby spectral planes centered around
1259.0872 nm.

Peak locations:

    1284 nm feature:
    x = 53
    y = 47

    [Fe II] 1.257 μm:
    x = 53
    y = 47

Spatial correlation:

    Pearson r = 0.76794

This indicates substantial similarity in spatial morphology and a common
bright emission location.

However, these preliminary maps were simple spectral integrations and
did not yet provide spatially resolved velocity information.

---

# 15. Spatially Resolved Fixed-Velocity Pa Beta versus Cs II

A spatially resolved comparison was performed using the same local
spectral window, the same instrumental Gaussian width, local noise
estimation, and fixed line centers based on:

    +573.72 km/s

Predicted centers:

    Pa beta:
    1284.26002473 nm

    Cs II:
    1287.07377505 nm

Total spatial pixels attempted:

    12125

Accepted pixels:

    1379

Acceptance fraction:

    0.1137

Model preference:

    Median Δχ² (Cs II - Pa beta):
    +45.242

    Minimum:
    -34.817

    Maximum:
    +17013.250

Pixels favoring Pa beta:

    1376

Pixels favoring Cs II:

    3

This strongly favors Pa beta over Cs II throughout the accepted bright
emission region, although a small number of pixels favored Cs II and
require caution in interpreting the result.

---

# 16. Free-Centroid 1284 nm Velocity Map

A free-centroid S3D analysis was then performed.

The line centroid was allowed to vary independently in sufficiently
strong spatial pixels.

Accepted pixels:

    1882

Acceptance fraction:

    0.4004

Assuming Pa beta:

    Median velocity:
    574.97 km/s

    Mean velocity:
    572.70 km/s

    Standard deviation:
    54.88 km/s

    16th percentile:
    546.45 km/s

    84th percentile:
    601.61 km/s

    Median velocity uncertainty:
    13.93 km/s

Independent [Fe II] reference:

    +573.72 km/s

Difference:

    +1.25 km/s

If interpreted as Cs II instead, the same measured centroid corresponds
to approximately:

    -82.6 km/s

The resulting velocity map is therefore highly consistent with the
Pa beta interpretation.

---

# 17. Direct Pa Beta versus [Fe II] Spatial Velocity Comparison

Pa beta and [Fe II] 1.257 μm were fitted independently in the same
spatial pixels.

Initial paired sample:

    696 pixels

Results:

    Pa beta median:
    574.360 km/s

    [Fe II] median:
    575.824 km/s

    Median difference:
    -0.853 km/s

However, the detailed pixel-by-pixel velocity correlation was weak:

    Pearson r = 0.041

The analysis was therefore repeated with a strict high-S/N selection.

Selection:

    Minimum S/N = 10
    Maximum velocity error = 15 km/s
    Velocity range = 400–700 km/s

High-quality paired pixels:

    143

Results:

    Pa beta median:
    573.414 km/s

    [Fe II] median:
    577.321 km/s

    Median difference:
    -2.107 km/s

Correlation:

    Pearson r = 0.162695
    Spearman rho = 0.210820

The Pa beta and [Fe II] lines therefore show similar bulk velocities
but weak detailed pixel-by-pixel velocity correlation.

This is important: the evidence does not require the two species to
trace exactly the same detailed velocity structure.

---

# 18. Pa Beta versus Pa Gamma Spatial Velocity Correlation

A more direct physical comparison was made between Pa beta and Pa gamma,
because both transitions are hydrogen recombination lines.

High-S/N selection:

    Minimum S/N = 10
    Maximum centroid velocity error = 15 km/s
    Velocity range = 400–700 km/s

High-quality paired pixels:

    233

Results:

### Pa beta

    Median:
    574.910 km/s

    Mean:
    574.990 km/s

    Standard deviation:
    13.955 km/s

    Median S/N:
    65.30

### Pa gamma

    Median:
    582.123 km/s

    Mean:
    581.870 km/s

    Standard deviation:
    21.057 km/s

    Median S/N:
    25.53

Velocity difference:

    Median:
    -6.872 km/s

    Mean:
    -6.880 km/s

    Standard deviation:
    13.874 km/s

    Median combined uncertainty:
    7.057 km/s

Velocity correlation:

    Pearson r:
    0.758285

    Pearson p-value:
    8.28 × 10^-45

    Spearman rho:
    0.783981

    Spearman p-value:
    9.85 × 10^-50

The Pa beta and Pa gamma velocity fields therefore show a strong
positive spatial correlation.

---

# 19. Propagated JWST S3D Error-Cube Test

The Pa beta / Pa gamma comparison was repeated using the actual
propagated JWST/NIRSpec S3D uncertainty cube.

The S3D FITS file contains:

    SCI:
    1447 × 97 × 125

    ERR:
    1447 × 97 × 125

The propagated-error analysis avoids relying solely on the earlier
local spectral-noise estimate.

High-quality paired pixels:

    262

### Pa beta

    Median velocity:
    575.825 km/s

    Mean velocity:
    576.891 km/s

    Standard deviation:
    13.910 km/s

    Median velocity uncertainty:
    2.305 km/s

    Median S/N:
    51.52

### Pa gamma

    Median velocity:
    584.203 km/s

    Mean velocity:
    584.423 km/s

    Standard deviation:
    21.978 km/s

    Median velocity uncertainty:
    7.008 km/s

    Median S/N:
    23.10

### Velocity difference

    Median:
    -7.243 km/s

    Mean:
    -7.532 km/s

    Standard deviation:
    14.106 km/s

    Median combined uncertainty:
    7.353 km/s

### Correlation

    Pearson r:
    0.781020

    Pearson p-value:
    4.36 × 10^-55

    Spearman rho:
    0.782055

    Spearman p-value:
    2.54 × 10^-55

### Agreement within propagated uncertainties

    Within 1 sigma:
    104 / 262 = 39.7%

    Within 2 sigma:
    184 / 262 = 70.2%

    Within 3 sigma:
    224 / 262 = 85.5%

The strong Pa beta / Pa gamma spatial velocity correlation therefore
survives use of the propagated JWST error cube.

This is currently one of the strongest independent pieces of evidence
supporting the Pa beta interpretation.

---

# 20. Major Milestone

## Cs II Hypothesis Strongly Disfavored

**Date: 2026-08-18**

The investigation has now accumulated convergent evidence from:

1. Laboratory wavelength comparison
2. Air-to-vacuum wavelength conversion
3. NIST same-upper-level transition search
4. Independent M51 nebular velocity measurements
5. Velocity-grid analysis
6. Constrained Pa beta versus Cs II fitting
7. Free-centroid Gaussian fitting
8. S3D spatial morphology
9. Spatially resolved velocity analysis
10. Pa beta / [Fe II] comparison
11. Pa beta / Pa gamma comparison
12. Propagated JWST uncertainty analysis

The combined evidence strongly favors **Pa beta** as the identification
of the 1284.261 nm emission feature.

The proposed highly excited **Cs II** transition is currently
**strongly disfavored**.

The evidence is convergent:

- The observed wavelength corresponds naturally to Pa beta at the local
  M51 nebular velocity.
- Cs II requires a velocity near -82.6 km/s.
- The constrained Pa beta model overwhelmingly outperforms the Cs II
  model.
- The constrained Cs II model produces essentially no positive emission
  amplitude.
- The 1284 nm feature is spatially associated with the bright nebular
  emission region.
- Its free-centroid velocity field has a median velocity of approximately
  575 km/s when interpreted as Pa beta.
- The Pa beta velocity field has a strong spatial correlation with Pa
  gamma.
- The Pa beta / Pa gamma result remains strong when the propagated JWST
  S3D error cube is used.

### Current scientific status

**Pa beta: strongly favored**

**Cs II: strongly disfavored**

This is not considered an absolute atomic-identification proof.

The simple Gaussian models have large reduced chi-square values, showing
that the detailed spectral model remains incomplete. The S3D analyses
also remain subject to limitations in extraction, spatial sampling,
continuum treatment, and line fitting.

Nevertheless, the independent wavelength, velocity, spatial, and
hydrogen-line evidence now converges strongly on Pa beta.

The Cs II hypothesis should remain documented as an investigated
alternative rather than being silently discarded.

---

# 21. Remaining Final Comparison

The next experiment is a final internally consistent comparison of the
two competing atomic hypotheses using the propagated JWST S3D error
cube.

The comparison should use:

- propagated JWST S3D uncertainties;
- identical local continuum treatment;
- actual NIRSpec instrumental resolution;
- independently measured nebular velocity;
- identical fitting windows;
- identical statistical treatment;
- Δχ²;
- AIC;
- BIC.

The purpose is to establish the final quantitative Pa beta versus Cs II
comparison using the best available uncertainty information.

Following this comparison, the investigation can move toward the
physical interpretation of the 1284 nm Pa beta emission in M51.

---

# 22. Scientific Caution

Several limitations remain important.

### S3D spectral sampling

The S3D cube samples the spectrum at approximately:

    0.636 nm per spectral plane

The 1284 nm feature and the predicted Pa beta wavelength differ by only
approximately 0.0013 nm in the relevant comparison.

Therefore the preliminary integrated 1284 nm and Pa beta maps were
identical because they selected the same spectral planes.

This is not independent spatial confirmation of the Pa beta identity.

### Integrated X1D spectrum

The X1D spectrum is an integrated spectrum from an extended NIRSpec IFU
observation.

Spatially resolved analysis is therefore essential when interpreting
possible velocity components.

### Model goodness of fit

The simple Gaussian fits have reduced chi-square values substantially
larger than unity.

The large absolute chi-square values therefore should not be treated
as evidence that the spectral model is a complete physical description
of the emission.

The strongest use of the statistical comparisons is differential:
comparing the competing hypotheses under the same assumptions.

### Propagated uncertainties

The latest Pa beta / Pa gamma experiment uses the propagated JWST S3D
error cube and is therefore preferred over the earlier local-noise
experiment for uncertainty-based conclusions.

The propagated-error analysis should likewise be used for the final
Pa beta versus Cs II comparison.

---

# 23. Working Conclusion

At the current stage of the investigation:

> **The 1284.261 nm feature in the M51 JWST/NIRSpec spectrum is strongly
> favored to be Pa beta rather than the proposed highly excited Cs II
> transition.**

The Cs II interpretation requires an anomalous velocity relative to
the surrounding M51 nebular gas and is overwhelmingly disfavored by
the constrained spectral comparison.

The strongest independent supporting evidence is the spatially resolved
correlation between the 1284 nm feature interpreted as Pa beta and the
Pa gamma hydrogen recombination line, including the analysis using the
propagated JWST S3D uncertainty cube.

The investigation remains open until the final propagated-error
Pa beta/Cs II comparison is completed.

## Major Milestone — Final Propagated-Error Pa β vs Cs II Test

### Date

2026-08-18

### Purpose

The final propagated-error experiment was performed to compare the
**Pa β** and **Cs II** interpretations of the 1284 nm spectral feature
using the JWST/NIRSpec Level-3 S3D cube and its propagated uncertainty
cube.

This test was designed to provide the strongest comparison yet by
using:

- the same local spectral window;
- the same local continuum model;
- the actual NIRSpec instrumental resolution;
- the independently measured local M51 nebular velocity;
- the propagated JWST/NIRSpec S3D uncertainty cube;
- identical statistical treatment of the two competing hypotheses.

The analysis therefore tests whether the observed 1284 nm emission is
consistent with Pa β or with Cs II when both transitions are subjected
to the same observational constraints.

---

### Data

JWST/NIRSpec Level-3 S3D cube:

`data/m51_jwst_level3/jw03435-o006_t010_nirspec_g140m-f100lp_s3d.fits`

The S3D product contains:

- SCI cube: `(1447, 97, 125)`
- propagated ERR cube: `(1447, 97, 125)`
- DQ cube: `(1447, 97, 125)`
- WMAP cube: `(1447, 97, 125)`

The spectral axis was reconstructed directly from the SCI-extension
spectral WCS keywords:

- `CRPIX3 = 1.0`
- `CRVAL3 = 0.970318028616020 um`
- `CDELT3 = 0.000636000011582 um`
- `CTYPE3 = WAVE`
- `CUNIT3 = um`

The resulting wavelength range is:

**970.318–1889.974 nm**

with a spectral sampling of:

**0.636000 nm**

The wavelength-axis construction was independently validated before
performing the spectral comparison.

---

### Instrumental resolution

The adopted NIRSpec resolving power was:

**R = 916.3**

At approximately 1284 nm this corresponds to:

- Instrument FWHM: **1.401586 nm**
- Instrument Gaussian sigma: **0.595199 nm**

The same instrumental resolution was used for the competing
hypotheses.

---

### Observed feature

The measured 1284 nm feature has a fitted wavelength of:

**1284.26130440 nm**

with an uncertainty of approximately:

**±0.00135 nm**

---

### Laboratory wavelengths

#### Pa β

Laboratory wavelength:

**1281.80700000 nm**

At the independently measured local M51 velocity of:

**+573.72 km/s**

the predicted observed wavelength is:

**1284.26237643 nm**

The velocity required for Pa β to reproduce the observed feature is:

**+573.470 km/s**

This differs from the independently measured local [Fe II] velocity
reference by only approximately:

**−0.25 km/s**

---

#### Cs II

NIST air wavelength:

**1284.264060 nm**

NIST vacuum wavelength:

**1284.61537587 nm**

The Cs II comparison uses the vacuum wavelength.

At the independently measured M51 velocity of:

**+573.72 km/s**

the predicted observed Cs II wavelength is:

**1287.07613191 nm**

The velocity required for Cs II to reproduce the observed
1284.26130440 nm feature is:

**−82.642 km/s**

This is radically different from the velocities measured from the
independently identified nebular emission lines.

---

### Local spectral window

The propagated-error spatial analysis used the spectral interval:

**1281.322–1287.046 nm**

containing:

**10 S3D spectral planes**

The same spectral window was used for the competing Pa β and Cs II
models.

---

## Spatial propagated-error model comparison

The Pa β and Cs II hypotheses were evaluated independently in the
spatial pixels of the S3D cube.

Only sufficiently strong spatial pixels satisfying the analysis
quality criteria were retained.

### Accepted spatial pixels

**2,234**

### Pa β model

Median chi-square:

**15.737**

Median amplitude S/N:

**9.55**

### Cs II model

Median chi-square:

**96.843**

Median amplitude S/N:

**−3.56**

### Difference in chi-square

The model comparison quantity was defined as:

**Δχ² = χ²(Cs II) − χ²(Pa β)**

A positive value therefore favors Pa β.

Results:

- Median Δχ²: **+80.919**
- Mean Δχ²: **+517.608**
- Minimum Δχ²: **−54.447**
- Maximum Δχ²: **+13,655.417**

---

### Spatial model preference

Of the 2,234 accepted spatial pixels:

- **2,231 favored Pa β**
- **3 favored Cs II**
- **0 were ties**

Thus:

**99.87% of accepted spatial pixels favored Pa β.**

Only approximately:

**0.13%**

favored Cs II.

The overwhelmingly positive spatial Δχ² distribution provides strong
spatially resolved evidence in favor of the Pa β model under the
adopted fixed-velocity and propagated-error assumptions.

---

## Integrated spectral comparison

The integrated spectrum provides an additional independent comparison
using the same propagated-error framework.

### Pa β

Amplitude:

**29,876.214**

Amplitude uncertainty:

**34.759523**

Amplitude S/N:

**859.51**

Chi-square:

**3,667.499**

### Cs II

Amplitude:

**−7,885.3167**

Amplitude uncertainty:

**29.086443**

Amplitude S/N:

**−271.10**

Chi-square:

**668,932.832**

### Integrated Δχ²

**Δχ² = 665,265.333**

in favor of Pa β.

The Cs II model therefore does not produce a positive emission
amplitude at the wavelength predicted for Cs II at the local M51
velocity. Instead, the fitted amplitude is strongly negative.

---

## Interpretation

This is the strongest Pa β versus Cs II comparison performed so far.

The evidence converges from several independent analyses:

1. The observed 1284.2613 nm feature requires approximately
   **+573.47 km/s** if interpreted as Pa β.

2. This agrees extremely closely with the independently measured
   local nebular velocity of approximately **+573.72 km/s**.

3. Interpreting the same feature as Cs II requires approximately
   **−82.64 km/s**, which is strongly inconsistent with the local
   nebular velocity field.

4. The propagated-error spatial comparison favors Pa β in
   **2,231 of 2,234** accepted spatial pixels.

5. The median spatial Δχ² is **+80.919** in favor of Pa β.

6. The integrated propagated-error comparison gives a
   Δχ² of **+665,265.333** in favor of Pa β.

7. The integrated Cs II model has a strongly negative fitted
   amplitude, with S/N **−271.10**, whereas Pa β has a very strong
   positive amplitude with S/N **859.51**.

Together these results strongly disfavor Cs II as the identification
of the 1284.261 nm feature.

---

## Important statistical qualification

The absolute chi-square values should not be interpreted as formal
goodness-of-fit probabilities.

The analysis is primarily a **comparative model-discrimination test**:
both hypotheses are subjected to the same spectral window, continuum
treatment, instrumental resolution, velocity constraint, and
propagated uncertainty model.

The very large Δχ² values therefore demonstrate a strong difference
between the two models under these assumptions, but they do not by
themselves constitute a complete formal statistical proof of the
atomic identification.

---

## Current scientific conclusion

### **Cs II is strongly disfavored as the identification of the 1284 nm feature.**

The available evidence strongly favors the interpretation that the
1284.261 nm feature is associated with **Pa β / ordinary nebular
hydrogen recombination emission** rather than the proposed highly
excited Cs II transition.

This conclusion is substantially stronger than the original
X1D-only result because the analysis now incorporates:

- spatially resolved S3D spectroscopy;
- independent nebular velocity measurements;
- NIRSpec instrumental resolution;
- propagated JWST uncertainty information;
- pixel-by-pixel model comparison;
- and an integrated-spectrum comparison.

Nevertheless, the result should still be described as a strong
**identification preference**, rather than an absolute exclusion of
every possible alternative atomic transition near 1284.26 nm.

---

## Significance for the Cs II investigation

The original motivation was to determine whether multiple Cs II
transitions arising from the same highly excited upper configuration
could provide a convincing identification in the M51 JWST/NIRSpec
spectrum.

The 1284.264 nm Cs II transition provided a particularly useful test
because it lies extremely close to the observed 1284 nm feature in
air wavelength.

However, after correcting the wavelength reference to vacuum and
incorporating the independently measured M51 nebular velocity, the
Cs II interpretation becomes kinematically inconsistent with the
observed feature.

The subsequent spatially resolved propagated-error comparison
strengthens this conclusion.

The Cs II hypothesis should therefore be retained in the research
record as a **tested and strongly disfavored hypothesis**, rather than
being silently removed from the investigation.

---

## Status

**Major milestone reached.**

The 1284.264 nm Cs II hypothesis has been tested using integrated
spectroscopy, local velocity measurements, velocity-grid analysis,
S3D spatial extraction, free-centroid velocity mapping, hydrogen-line
kinematic comparison, and finally propagated-error Pa β versus Cs II
model discrimination.

### Current status:

**Cs II: strongly disfavored**

**Pa β: strongly preferred**

---

## Recommended next analysis

The next stage should move beyond the specific Cs II-versus-Pa β
question and examine the broader atomic-identification problem.

Potential next steps include:

1. Search the NIST database for other transitions near the measured
   1284.2613 nm wavelength.

2. Examine whether any plausible alternative transitions share the
   observed M51 velocity.

3. Compare the 1284 nm feature quantitatively with other hydrogen
   recombination lines, especially Pa γ.

4. Examine the propagated-error velocity field and spatial morphology
   in greater detail.

5. Investigate the Level-3 S3D cube for possible contamination,
   blending, or continuum-subtraction effects.

6. Preserve the Cs II same-upper-level transition analysis as a
   documented negative result.

The investigation should therefore continue with the broader question:

> **What is the most physically plausible identification of the
> 1284.2613 nm feature after Cs II has been strongly disfavored?**
