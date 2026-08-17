from tools.m51_spectral_analysis import (
    load_x1d_spectrum,
    prepare_spectrum,
    detect_candidate_features,
    fit_gaussian_line,
)

X1D_PATH = "data/m51_jwst_level3/" "jw03435-o006_t010_nirspec_g140m-f100lp_x1d.fits"


print("=" * 60)
print("M51 SPECTRAL ANALYSIS TEST")
print("=" * 60)


# ------------------------------------------------------------
# 1. Load the JWST X1D spectrum
# ------------------------------------------------------------

spectrum = load_x1d_spectrum(X1D_PATH)

print("\nLoaded spectrum:")
print(f"  File: {spectrum['path']}")
print(f"  Points: {len(spectrum['wavelength'])}")


# ------------------------------------------------------------
# 2. Prepare the spectrum
# ------------------------------------------------------------

prepared = prepare_spectrum(spectrum)

print("\nPrepared spectrum:")
print(f"  Valid points: {prepared['valid_points']}")
print(f"  Rejected points: {prepared['rejected_points']}")


# ------------------------------------------------------------
# 3. Detect candidate features
# ------------------------------------------------------------


def detect_candidate_features(
    spectrum,
    prominence=None,
    distance=5,
    snr_threshold=3.0,
    continuum_window=51,
):
    """
    Detect candidate emission features above a local continuum.

    A candidate must satisfy two conditions:

    1. Its peak prominence exceeds the detection threshold.
    2. Its flux is actually above the estimated local continuum.

    This prevents local maxima that occur at or below the continuum
    from being reported as emission-feature candidates.

    Parameters
    ----------
    spectrum : dict
        Prepared spectrum from prepare_spectrum().

    prominence : float or None, optional
        Minimum peak prominence in flux units.

        If None, the threshold is estimated from the median supplied
        flux uncertainty multiplied by snr_threshold.

    distance : int, optional
        Minimum number of samples between detected peaks.

    snr_threshold : float, optional
        Number of estimated-noise units used to establish the
        automatic prominence threshold.

    continuum_window : int, optional
        Width of the median-filter window used to estimate the
        local continuum. Must be an odd integer.

    Returns
    -------
    dict
        Candidate feature information containing:

        wavelength
            Wavelength of each candidate.

        flux
            Observed flux at each candidate.

        continuum
            Estimated local continuum at each candidate.

        excess_flux
            Flux above the local continuum.

        indices
            Array indices of detected candidates.

        prominence
            Peak prominence.

        snr
            Diagnostic prominence-to-noise ratio.

        properties
            Full scipy find_peaks properties dictionary.

        count
            Number of candidates.

        estimated_noise
            Median supplied flux uncertainty.

        prominence_threshold
            Detection threshold used.
    """

    wavelength = spectrum["wavelength"]
    flux = spectrum["flux"]
    uncertainty = spectrum["uncertainty"]

    # --------------------------------------------------------
    # Handle an empty spectrum.
    # --------------------------------------------------------

    if len(flux) == 0:

        return {
            "wavelength": np.array([]),
            "flux": np.array([]),
            "continuum": np.array([]),
            "excess_flux": np.array([]),
            "indices": np.array([], dtype=int),
            "prominence": np.array([]),
            "snr": np.array([]),
            "properties": {},
            "count": 0,
            "estimated_noise": np.nan,
            "prominence_threshold": np.nan,
        }

    # --------------------------------------------------------
    # Validate continuum window.
    # --------------------------------------------------------

    if continuum_window < 3:

        raise ValueError("continuum_window must be at least 3.")

    if continuum_window % 2 == 0:

        raise ValueError("continuum_window must be an odd integer.")

    # --------------------------------------------------------
    # Estimate the noise from the supplied uncertainties.
    # --------------------------------------------------------

    finite_uncertainty = uncertainty[np.isfinite(uncertainty) & (uncertainty > 0)]

    if len(finite_uncertainty) == 0:

        raise ValueError(
            "No valid uncertainty values are available "
            "for candidate-feature detection."
        )

    estimated_noise = float(np.median(finite_uncertainty))

    # --------------------------------------------------------
    # Estimate the local continuum.
    #
    # Median filtering is relatively robust against narrow
    # spectral features.
    # --------------------------------------------------------

    from scipy.ndimage import median_filter

    continuum = median_filter(
        flux,
        size=continuum_window,
        mode="nearest",
    )

    # --------------------------------------------------------
    # Subtract the local continuum.
    # --------------------------------------------------------

    excess_flux = flux - continuum

    # --------------------------------------------------------
    # Establish the prominence threshold.
    # --------------------------------------------------------

    if prominence is None:

        prominence_threshold = snr_threshold * estimated_noise

    else:

        prominence_threshold = float(prominence)

    # --------------------------------------------------------
    # Find peaks in the continuum-subtracted spectrum.
    # --------------------------------------------------------

    peaks, properties = find_peaks(
        excess_flux,
        prominence=prominence_threshold,
        distance=distance,
    )

    peak_prominence = properties.get(
        "prominences",
        np.array([]),
    )

    # --------------------------------------------------------
    # Require positive excess flux.
    #
    # A prominence value alone is not enough. A candidate
    # emission feature must actually lie above its estimated
    # local continuum.
    # --------------------------------------------------------

    positive_excess = excess_flux[peaks] > 0

    peaks = peaks[positive_excess]

    peak_prominence = peak_prominence[positive_excess]

    # --------------------------------------------------------
    # Diagnostic S/N.
    #
    # Use prominence rather than total flux because the
    # continuum level itself should not inflate the signal.
    # --------------------------------------------------------

    peak_snr = peak_prominence / estimated_noise

    # --------------------------------------------------------
    # Rebuild the properties dictionary for the retained
    # candidates.
    # --------------------------------------------------------

    filtered_properties = dict(properties)

    for key, value in properties.items():

        if hasattr(value, "__len__"):

            try:
                filtered_properties[key] = value[positive_excess]

            except (IndexError, TypeError):

                pass

    filtered_properties["prominences"] = peak_prominence

    return {
        "wavelength": wavelength[peaks],
        "flux": flux[peaks],
        "continuum": continuum[peaks],
        "excess_flux": excess_flux[peaks],
        "indices": peaks,
        "prominence": peak_prominence,
        "snr": peak_snr,
        "properties": filtered_properties,
        "count": len(peaks),
        "estimated_noise": estimated_noise,
        "prominence_threshold": prominence_threshold,
    }


# ------------------------------------------------------------
# 4. Test a Gaussian fit
# ------------------------------------------------------------

# Pa-beta region identified in our previous M51 analysis.
center_guess = 1.284282

print("\nGaussian fit near Pa-beta:")

try:

    fit = fit_gaussian_line(
        prepared,
        center_guess=center_guess,
        window=0.003,
    )

    for key, value in fit.items():
        print(f"  {key}: {value}")

except Exception as exc:

    print(f"  Fit failed: {exc}")


print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
