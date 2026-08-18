from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit


from tools.m51_spectral_analysis import (
    load_x1d_spectrum,
    prepare_spectrum,
)


# ============================================================
# M51 JWST/NIRSpec spectrum
# ============================================================

X1D_PATH = Path(
    "data/m51_jwst_level3/"
    "jw03435-o006_t010_nirspec_g140m-f100lp_x1d.fits"
)


# ============================================================
# Constants
# ============================================================

C_KMS = 299792.458


# ============================================================
# NIRSpec G140M/F100LP
#
# JWST nominal resolving power:
#
#     R ~ 1000
#
# We use the actual wavelength-dependent resolution file
# supplied in this project when available.
# ============================================================

RESOLUTION_FILE = Path(
    "data/instrument/"
    "jwst_nirspec_g140m_disp.fits"
)


NOMINAL_R = 1000.0


# ============================================================
# Competing identifications
# ============================================================

LINES = {
    "Pa beta": {
        "rest_wavelength_nm": 1281.80700,
        "species": "H I",
    },
    "Cs II": {
        "rest_wavelength_nm": 1284.61537587,
        "species": "Cs II",
    },
}


# ============================================================
# Local velocity reference
#
# We deliberately do NOT use Pa beta itself to establish
# the velocity reference.
#
# These are independent nebular lines measured previously
# from the same X1D spectrum.
#
# Values are observed velocities in km/s.
# ============================================================

LOCAL_VELOCITY_REFERENCE = [
    {
        "name": "He I 1.0833",
        "velocity_kms": 505.00,
        "uncertainty_kms": 1.36,
    },
    {
        "name": "Pa gamma",
        "velocity_kms": 583.14,
        "uncertainty_kms": 1.01,
    },
    {
        "name": "[Fe II] 1.257",
        "velocity_kms": 573.72,
        "uncertainty_kms": 2.11,
    },
    {
        "name": "Brackett 17",
        "velocity_kms": 569.37,
        "uncertainty_kms": 6.41,
    },
    {
        "name": "Brackett 15",
        "velocity_kms": 539.69,
        "uncertainty_kms": 8.60,
    },
    {
        "name": "Brackett 13",
        "velocity_kms": 478.40,
        "uncertainty_kms": 5.41,
    },
    {
        "name": "[Fe II] 1.644",
        "velocity_kms": 510.82,
        "uncertainty_kms": 2.33,
    },
    {
        "name": "Brackett 10",
        "velocity_kms": 461.75,
        "uncertainty_kms": 3.96,
    },
]


# ============================================================
# Robust local velocity estimate
# ============================================================


def calculate_local_velocity_reference():
    """
    Calculate a robust local velocity reference.

    We use the median rather than a formal inverse-variance
    weighted mean because the formal line-center errors are
    much smaller than the systematic uncertainties associated
    with an extended IFU spectrum and imperfect line profiles.
    """

    velocities = np.array(
        [
            item["velocity_kms"]
            for item in LOCAL_VELOCITY_REFERENCE
        ],
        dtype=float,
    )

    median = np.median(
        velocities
    )

    mad = np.median(
        np.abs(
            velocities - median
        )
    )

    return median, mad


# ============================================================
# Relativistic Doppler shift
# ============================================================


def velocity_to_wavelength(
    rest_wavelength_nm,
    velocity_kms,
):
    """
    Convert rest wavelength to observed wavelength using
    the relativistic Doppler relation.
    """

    beta = velocity_kms / C_KMS

    doppler_factor = np.sqrt(
        (1.0 + beta)
        / (1.0 - beta)
    )

    return (
        rest_wavelength_nm
        * doppler_factor
    )


# ============================================================
# Instrument resolution
# ============================================================


def estimate_instrument_sigma(
    wavelength_nm,
):
    """
    Estimate the instrumental Gaussian sigma.

    Primary method:
        use the project's G140M dispersion/resolution FITS
        file if it contains a resolving-power column.

    Fallback:
        use nominal R = 1000.

    Returns
    -------
    sigma_nm : float
        Approximate instrumental Gaussian sigma in nm.
    """

    if RESOLUTION_FILE.exists():

        try:

            from astropy.io import fits

            with fits.open(
                RESOLUTION_FILE
            ) as hdul:

                table = hdul[1].data

                names = [
                    name.upper()
                    for name in (
                        table.names or []
                    )
                ]

                wavelength_column = None
                resolving_column = None

                for name in names:

                    if name in {
                        "WAVELENGTH",
                        "WAVE",
                    }:

                        wavelength_column = name

                    if name in {
                        "R",
                        "RESOLUTION",
                        "RESOLVING_POWER",
                    }:

                        resolving_column = name

                if (
                    wavelength_column
                    is not None
                    and resolving_column
                    is not None
                ):

                    wave = np.asarray(
                        table[
                            wavelength_column
                        ],
                        dtype=float,
                    )

                    resolving_power = np.asarray(
                        table[
                            resolving_column
                        ],
                        dtype=float,
                    )

                    # Resolution file wavelength is
                    # expected to be in microns.

                    if np.nanmedian(wave) < 10:

                        target = (
                            wavelength_nm
                            / 1000.0
                        )

                    else:

                        target = wavelength_nm

                    valid = (
                        np.isfinite(wave)
                        & np.isfinite(
                            resolving_power
                        )
                        & (
                            resolving_power
                            > 0
                        )
                    )

                    if np.sum(valid) > 2:

                        R = np.interp(
                            target,
                            wave[valid],
                            resolving_power[
                                valid
                            ],
                        )

                        fwhm = (
                            wavelength_nm
                            / R
                        )

                        sigma = (
                            fwhm
                            / 2.354820045
                        )

                        print()
                        print(
                            "Resolution source:"
                        )
                        print(
                            RESOLUTION_FILE
                        )
                        print(
                            f"Interpolated R: "
                            f"{R:.1f}"
                        )

                        return sigma

        except Exception as exc:

            print()
            print(
                "WARNING: could not read"
            )
            print(
                "resolution FITS file:"
            )
            print(exc)

    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    fwhm = (
        wavelength_nm
        / NOMINAL_R
    )

    sigma = (
        fwhm
        / 2.354820045
    )

    print()
    print(
        "Resolution source:"
    )
    print(
        "Nominal G140M R = "
        f"{NOMINAL_R:.0f}"
    )

    return sigma


# ============================================================
# Gaussian + local linear continuum
# ============================================================


def gaussian_with_continuum(
    wavelength_nm,
    amplitude,
    continuum,
    slope,
    sigma_nm,
    center_nm,
):
    """
    Gaussian emission profile with a local linear continuum.

    The center and sigma are fixed by the hypothesis and
    instrumental resolution.

    Only amplitude and continuum parameters are fitted.
    """

    return (
        continuum
        + slope
        * (
            wavelength_nm
            - center_nm
        )
        + amplitude
        * np.exp(
            -0.5
            * (
                (
                    wavelength_nm
                    - center_nm
                )
                / sigma_nm
            ) ** 2
        )
    )


# ============================================================
# Fit fixed-center hypothesis
# ============================================================


def fit_fixed_hypothesis(
    wavelength_nm,
    flux,
    uncertainty,
    center_nm,
    sigma_nm,
    window_nm=4.0,
):
    """
    Fit one fixed-center spectral hypothesis.

    The local continuum is identical for both hypotheses.

    Parameters fitted:
        amplitude
        continuum
        continuum slope

    Fixed:
        line center
        instrumental sigma
    """

    mask = (
        np.abs(
            wavelength_nm
            - center_nm
        )
        <= window_nm
    )

    if np.sum(mask) < 7:

        return None

    x = wavelength_nm[
        mask
    ]

    y = flux[
        mask
    ]

    err = uncertainty[
        mask
    ]

    # --------------------------------------------------------
    # Initial guesses
    # --------------------------------------------------------

    continuum_guess = np.median(
        y
    )

    amplitude_guess = (
        np.max(y)
        - continuum_guess
    )

    slope_guess = 0.0

    p0 = [
        amplitude_guess,
        continuum_guess,
        slope_guess,
    ]

    # --------------------------------------------------------
    # Model with fixed center and sigma
    # --------------------------------------------------------

    def model(
        x,
        amplitude,
        continuum,
        slope,
    ):

        return gaussian_with_continuum(
            x,
            amplitude,
            continuum,
            slope,
            sigma_nm,
            center_nm,
        )

    # --------------------------------------------------------
    # Fit
    # --------------------------------------------------------

    try:

        popt, pcov = curve_fit(
            model,
            x,
            y,
            p0=p0,
            sigma=err,
            absolute_sigma=True,
            maxfev=20000,
        )

    except (
        RuntimeError,
        ValueError,
    ):

        return None

    model_flux = model(
        x,
        *popt,
    )

    residuals = (
        y - model_flux
    )

    chi_squared = np.sum(
        (
            residuals
            / err
        ) ** 2
    )

    n = len(x)

    k = len(popt)

    dof = n - k

    if dof > 0:

        reduced_chi_squared = (
            chi_squared
            / dof
        )

    else:

        reduced_chi_squared = np.nan

    # --------------------------------------------------------
    # AIC
    #
    # For Gaussian errors:
    #
    # AIC = chi² + 2k
    #
    # Same data and same number of parameters make this a
    # particularly clean comparison.
    # --------------------------------------------------------

    aic = (
        chi_squared
        + 2.0 * k
    )

    # --------------------------------------------------------
    # BIC
    # --------------------------------------------------------

    bic = (
        chi_squared
        + k * np.log(n)
    )

    # --------------------------------------------------------
    # Amplitude uncertainty
    # --------------------------------------------------------

    try:

        errors = np.sqrt(
            np.diag(pcov)
        )

        amplitude_error = errors[0]

    except Exception:

        amplitude_error = np.nan

    if (
        np.isfinite(
            amplitude_error
        )
        and amplitude_error > 0
    ):

        amplitude_snr = (
            popt[0]
            / amplitude_error
        )

    else:

        amplitude_snr = np.nan

    return {
        "center_nm": center_nm,
        "sigma_nm": sigma_nm,
        "fwhm_nm": (
            2.354820045
            * sigma_nm
        ),
        "amplitude": popt[0],
        "amplitude_error": (
            amplitude_error
        ),
        "amplitude_snr": (
            amplitude_snr
        ),
        "continuum": popt[1],
        "slope": popt[2],
        "chi_squared": (
            chi_squared
        ),
        "dof": dof,
        "reduced_chi_squared": (
            reduced_chi_squared
        ),
        "aic": aic,
        "bic": bic,
        "n_points": n,
    }


# ============================================================
# Main program
# ============================================================


print("=" * 70)
print(
    "M51: Pa BETA vs Cs II"
)
print(
    "FIXED-VELOCITY / INSTRUMENT-RESOLUTION TEST"
)
print("=" * 70)


# ============================================================
# Local velocity
# ============================================================

local_velocity, velocity_mad = (
    calculate_local_velocity_reference()
)

print()
print(
    "Independent local nebular velocity reference"
)

print(
    f"Median velocity: "
    f"{local_velocity:+.2f} km/s"
)

print(
    f"MAD: "
    f"{velocity_mad:.2f} km/s"
)

print()
print(
    "Reference lines:"
)

for item in (
    LOCAL_VELOCITY_REFERENCE
):

    print(
        f"  "
        f"{item['name']:20s}"
        f"{item['velocity_kms']:+8.2f}"
        f" km/s"
    )


# ============================================================
# Load spectrum
# ============================================================

print()
print("=" * 70)
print("LOADING SPECTRUM")
print("=" * 70)

spectrum = load_x1d_spectrum(
    X1D_PATH
)

clean = prepare_spectrum(
    spectrum
)

wavelength_nm = (
    clean["wavelength"]
    * 1000.0
)

flux = clean[
    "flux"
]

uncertainty = clean[
    "uncertainty"
]

print()
print(
    f"Valid points: "
    f"{clean['valid_points']}"
)

print(
    f"Wavelength range: "
    f"{wavelength_nm.min():.3f}"
    f" - "
    f"{wavelength_nm.max():.3f} nm"
)


# ============================================================
# Instrument resolution
# ============================================================

target_wavelength = 1284.3

sigma_inst = (
    estimate_instrument_sigma(
        target_wavelength
    )
)

fwhm_inst = (
    2.354820045
    * sigma_inst
)

print()
print("=" * 70)
print("NIRSPEC INSTRUMENT RESOLUTION")
print("=" * 70)

print()
print(
    f"At wavelength: "
    f"{target_wavelength:.3f} nm"
)

print(
    f"Instrument sigma: "
    f"{sigma_inst:.6f} nm"
)

print(
    f"Instrument FWHM: "
    f"{fwhm_inst:.6f} nm"
)


# ============================================================
# Calculate predicted wavelengths
# ============================================================

print()
print("=" * 70)
print("HYPOTHESIS PREDICTIONS")
print("=" * 70)

predictions = {}


for name, line in (
    LINES.items()
):

    rest = line[
        "rest_wavelength_nm"
    ]

    predicted = (
        velocity_to_wavelength(
            rest,
            local_velocity,
        )
    )

    predictions[name] = predicted

    print()
    print(
        f"{name}"
    )

    print(
        f"  Rest wavelength: "
        f"{rest:.8f} nm"
    )

    print(
        f"  Velocity used: "
        f"{local_velocity:+.2f} km/s"
    )

    print(
        f"  Predicted observed wavelength: "
        f"{predicted:.8f} nm"
    )

    print(
        f"  Distance from observed feature:"
        f" "
        f"{predicted - 1284.26130440:+.6f}"
        f" nm"
    )


# ============================================================
# Fit both hypotheses
# ============================================================

print()
print("=" * 70)
print(
    "COMPETING FIXED-CENTER FITS"
)
print("=" * 70)

results = {}


for name in (
    "Pa beta",
    "Cs II",
):

    center = predictions[
        name
    ]

    result = fit_fixed_hypothesis(
        wavelength_nm,
        flux,
        uncertainty,
        center,
        sigma_inst,
        window_nm=4.0,
    )

    results[name] = result

    print()
    print(
        f"{name}"
    )

    if result is None:

        print(
            "  FIT FAILED"
        )

        continue

    print(
        f"  Fixed center: "
        f"{result['center_nm']:.6f} nm"
    )

    print(
        f"  Fixed sigma: "
        f"{result['sigma_nm']:.6f} nm"
    )

    print(
        f"  Fixed FWHM: "
        f"{result['fwhm_nm']:.6f} nm"
    )

    print(
        f"  Fitted amplitude: "
        f"{result['amplitude']:.8g}"
    )

    print(
        f"  Amplitude S/N: "
        f"{result['amplitude_snr']:.2f}"
    )

    print(
        f"  Chi squared: "
        f"{result['chi_squared']:.3f}"
    )

    print(
        f"  Degrees of freedom: "
        f"{result['dof']}"
    )

    print(
        f"  Reduced chi squared: "
        f"{result['reduced_chi_squared']:.3f}"
    )

    print(
        f"  AIC: "
        f"{result['aic']:.3f}"
    )

    print(
        f"  BIC: "
        f"{result['bic']:.3f}"
    )


# ============================================================
# Direct comparison
# ============================================================

print()
print("=" * 70)
print(
    "PA BETA vs Cs II COMPARISON"
)
print("=" * 70)

pa_beta = results[
    "Pa beta"
]

csii = results[
    "Cs II"
]

if (
    pa_beta is not None
    and csii is not None
):

    delta_chi_squared = (
        csii["chi_squared"]
        - pa_beta["chi_squared"]
    )

    delta_aic = (
        csii["aic"]
        - pa_beta["aic"]
    )

    delta_bic = (
        csii["bic"]
        - pa_beta["bic"]
    )

    print()
    print(
        f"Pa beta chi²: "
        f"{pa_beta['chi_squared']:.3f}"
    )

    print(
        f"Cs II chi²:   "
        f"{csii['chi_squared']:.3f}"
    )

    print()
    print(
        f"Delta chi² "
        f"(Cs II - Pa beta): "
        f"{delta_chi_squared:+.3f}"
    )

    print(
        f"Delta AIC "
        f"(Cs II - Pa beta): "
        f"{delta_aic:+.3f}"
    )

    print(
        f"Delta BIC "
        f"(Cs II - Pa beta): "
        f"{delta_bic:+.3f}"
    )

    print()

    if (
        delta_aic > 10
        and delta_bic > 10
    ):

        print(
            "RESULT: Pa beta is strongly"
        )

        print(
            "preferred over Cs II under"
        )

        print(
            "the adopted local-velocity"
        )

        print(
            "and instrumental-resolution"
        )

        print(
            "constraints."
        )

    elif (
        delta_aic > 2
        and delta_bic > 2
    ):

        print(
            "RESULT: Pa beta is favored,"
        )

        print(
            "but the evidence is not"
        )

        print(
            "decisive."
        )

    elif (
        delta_aic < -10
        and delta_bic < -10
    ):

        print(
            "RESULT: Cs II is strongly"
        )

        print(
            "preferred under the adopted"
        )

        print(
            "constraints."
        )

    elif (
        delta_aic < -2
        and delta_bic < -2
    ):

        print(
            "RESULT: Cs II is favored,"
        )

        print(
            "but the evidence is not"
        )

        print(
            "decisive."
        )

    else:

        print(
            "RESULT: Neither hypothesis"
        )

        print(
            "is decisively preferred by"
        )

        print(
            "this constrained fit."
        )


# ============================================================
# Direct wavelength sanity check
# ============================================================

print()
print("=" * 70)
print(
    "DIRECT WAVELENGTH CHECK"
)
print("=" * 70)

observed_feature = (
    1284.26130440
)

print()
print(
    f"Observed feature: "
    f"{observed_feature:.8f} nm"
)

for name in (
    "Pa beta",
    "Cs II",
):

    predicted = predictions[
        name
    ]

    difference = (
        observed_feature
        - predicted
    )

    velocity_equivalent = (
        C_KMS
        * difference
        / observed_feature
    )

    print()
    print(
        f"{name}"
    )

    print(
        f"  Predicted: "
        f"{predicted:.8f} nm"
    )

    print(
        f"  Difference: "
        f"{difference:+.8f} nm"
    )

    print(
        f"  Equivalent velocity: "
        f"{velocity_equivalent:+.2f}"
        f" km/s"
    )


# ============================================================
# Important interpretation
# ============================================================

print()
print("=" * 70)
print("INTERPRETATION")
print("=" * 70)

print()
print(
    "This test does NOT allow the Pa beta or Cs II"
)

print(
    "velocity to float independently."
)

print()
print(
    "Both hypotheses are constrained to the same"
)

print(
    "independently measured M51 nebular velocity."
)

print()
print(
    "Both hypotheses also use the same local continuum"
)

print(
    "and the same NIRSpec instrumental line width."
)

print()
print(
    "Therefore the comparison asks which laboratory"
)

print(
    "transition predicts the observed feature at the"
)

print(
    "velocity actually measured in the surrounding"
)

print(
    "nebular spectrum."
)

print()
print(
    "A large chi-square difference in favor of Pa beta"
)

print(
    "would strongly disfavor Cs II as the identification."
)

print()
print(
    "A caveat: the X1D spectrum is an integrated"
)

print(
    "spectrum from an extended NIRSpec IFU target."
)

print(
    "Spatially resolved kinematics should ultimately"
)

print(
    "be checked in the Level-3 S3D cube."
)

print()
print("=" * 70)
print(
    "TEST COMPLETE"
)
print("=" * 70)
