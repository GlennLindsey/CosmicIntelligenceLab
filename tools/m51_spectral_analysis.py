"""
Reusable JWST/NIRSpec spectroscopy tools.

Initial implementation developed and validated on the M51
JWST NIRSpec G140M/F100LP X1D spectrum.

The functions are intentionally data-oriented so that the same
analysis machinery can eventually be reused by the Cosmic
Intelligence Lab research agent, Einstein, on other targets.

Current capabilities
--------------------
1. Load a JWST/NIRSpec X1D spectrum.
2. Apply basic quality filtering.
3. Detect candidate spectral features using a noise-based
   prominence threshold.
4. Fit a Gaussian emission feature with a local linear continuum.
5. Calculate diagnostic line measurements.

Scientific caution
------------------
The measurements returned by this module are diagnostic
measurements. They should not automatically be interpreted as
intrinsic physical line widths, final line fluxes, or confirmed
astrophysical identifications.

Instrumental resolution, sampling, calibration, and the choice
of fitting window must be considered before physical
interpretation.
"""

from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

# ============================================================
# Gaussian model
# ============================================================


def gaussian(x, amplitude, center, sigma):
    """
    Return a Gaussian profile.

    Parameters
    ----------
    x : array-like
        Independent variable, normally wavelength in microns.

    amplitude : float
        Gaussian amplitude.

    center : float
        Gaussian center.

    sigma : float
        Gaussian standard deviation.

    Returns
    -------
    numpy.ndarray
        Gaussian values.
    """

    return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)


# ============================================================
# Load JWST/NIRSpec X1D spectrum
# ============================================================


def load_x1d_spectrum(path):
    """
    Load a JWST/NIRSpec X1D spectrum.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the JWST X1D FITS product.

    Returns
    -------
    dict
        Dictionary containing:

        path
            Original file path.

        wavelength
            Wavelength array in microns.

        flux
            Extracted flux array.

        uncertainty
            Flux uncertainty array.

        data_quality
            JWST data-quality flags.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"X1D file not found: {path}")

    with fits.open(path) as hdul:

        if len(hdul) < 2:
            raise ValueError(
                "X1D FITS file does not contain "
                "the expected spectral table in extension 1."
            )

        table = hdul[1].data

        required_columns = {
            "WAVELENGTH",
            "FLUX",
            "FLUX_ERROR",
            "DQ",
        }

        available_columns = set(table.names or [])

        missing_columns = required_columns - available_columns

        if missing_columns:
            raise ValueError(
                "X1D spectrum is missing required columns: "
                + ", ".join(sorted(missing_columns))
            )

        wavelength = np.asarray(
            table["WAVELENGTH"],
            dtype=float,
        )

        flux = np.asarray(
            table["FLUX"],
            dtype=float,
        )

        uncertainty = np.asarray(
            table["FLUX_ERROR"],
            dtype=float,
        )

        data_quality = np.asarray(
            table["DQ"],
            dtype=np.uint32,
        )

    return {
        "path": str(path),
        "wavelength": wavelength,
        "flux": flux,
        "uncertainty": uncertainty,
        "data_quality": data_quality,
    }


# ============================================================
# Prepare spectrum
# ============================================================


def prepare_spectrum(
    spectrum,
    background_level=0.0,
):
    """
    Prepare an extracted spectrum for analysis.

    The current preparation step:

    - removes non-finite wavelength values;
    - removes non-finite flux values;
    - removes non-finite uncertainty values;
    - requires positive uncertainty;
    - removes points with non-zero DQ flags;
    - subtracts an optional constant background.

    Parameters
    ----------
    spectrum : dict
        Output from load_x1d_spectrum().

    background_level : float, optional
        Constant background level to subtract from the flux.

    Returns
    -------
    dict
        Cleaned spectrum.
    """

    wavelength = spectrum["wavelength"]
    flux = spectrum["flux"]
    uncertainty = spectrum["uncertainty"]
    data_quality = spectrum["data_quality"]

    valid = (
        np.isfinite(wavelength)
        & np.isfinite(flux)
        & np.isfinite(uncertainty)
        & (uncertainty > 0)
        & (data_quality == 0)
    )

    wavelength_clean = wavelength[valid]

    flux_clean = flux[valid] - background_level

    uncertainty_clean = uncertainty[valid]

    return {
        "path": spectrum["path"],
        "wavelength": wavelength_clean,
        "flux": flux_clean,
        "uncertainty": uncertainty_clean,
        "valid_points": int(np.sum(valid)),
        "rejected_points": int(np.sum(~valid)),
    }


# ============================================================
# Candidate spectral-feature detection
# ============================================================


def detect_candidate_features(
    spectrum,
    prominence=None,
    distance=5,
    snr_threshold=3.0,
    continuum_window=51,
):
    """
    Detect candidate emission features above a local continuum.

    The spectrum is first smoothed to estimate the local continuum.
    Candidate peaks are then searched for in the continuum-subtracted
    spectrum.

    This is a first-pass diagnostic detector. It is not a formal
    statistical line-detection algorithm.

    Parameters
    ----------
    spectrum : dict
        Prepared spectrum from prepare_spectrum().

    prominence : float or None, optional
        Minimum peak prominence in flux units.

        If None, estimate the threshold from the median supplied
        flux uncertainty and snr_threshold.

    distance : int, optional
        Minimum number of samples between detected peaks.

    snr_threshold : float, optional
        Threshold in units of the median supplied uncertainty.

    continuum_window : int, optional
        Width of the smoothing window used to estimate the local
        continuum. Must be an odd integer.

    Returns
    -------
    dict
        Candidate peak information.
    """

    wavelength = spectrum["wavelength"]
    flux = spectrum["flux"]
    uncertainty = spectrum["uncertainty"]

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

    if continuum_window < 3:
        raise ValueError("continuum_window must be at least 3.")

    if continuum_window % 2 == 0:
        raise ValueError("continuum_window must be an odd integer.")

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
    # A median filter is deliberately used here because it is
    # relatively robust against narrow emission features.
    # --------------------------------------------------------

    from scipy.ndimage import median_filter

    continuum = median_filter(
        flux,
        size=continuum_window,
        mode="nearest",
    )

    # --------------------------------------------------------
    # Remove the local continuum.
    # --------------------------------------------------------

    excess_flux = flux - continuum

    # --------------------------------------------------------
    # Determine the prominence threshold.
    # --------------------------------------------------------

    if prominence is None:

        prominence_threshold = snr_threshold * estimated_noise

    else:

        prominence_threshold = float(prominence)

    # --------------------------------------------------------
    # Search for peaks in the continuum-subtracted spectrum.
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
    # Diagnostic peak S/N.
    #
    # This uses peak prominence rather than total flux so that
    # a high continuum level does not artificially inflate S/N.
    # --------------------------------------------------------

    peak_snr = peak_prominence / estimated_noise

    return {
        "wavelength": wavelength[peaks],
        "flux": flux[peaks],
        "continuum": continuum[peaks],
        "excess_flux": excess_flux[peaks],
        "indices": peaks,
        "prominence": peak_prominence,
        "snr": peak_snr,
        "properties": properties,
        "count": len(peaks),
        "estimated_noise": estimated_noise,
        "prominence_threshold": prominence_threshold,
    }


# ============================================================
# Gaussian line fitting
# ============================================================


def fit_gaussian_line(
    spectrum,
    center_guess,
    window=0.003,
):
    """
    Fit a Gaussian emission feature plus a local linear continuum.

    Parameters
    ----------
    spectrum : dict
        Prepared spectrum from prepare_spectrum().

    center_guess : float
        Approximate wavelength of the feature in microns.

    window : float, optional
        Half-width of the fitting region in microns.

    Returns
    -------
    dict
        Diagnostic Gaussian-fit measurements:

        center_um
            Fitted line center.

        center_error_um
            Estimated uncertainty in fitted center.

        amplitude
            Gaussian amplitude.

        amplitude_error
            Estimated amplitude uncertainty.

        sigma_um
            Gaussian sigma.

        sigma_error_um
            Estimated sigma uncertainty.

        fwhm_um
            Gaussian FWHM.

        fwhm_error_um
            Estimated FWHM uncertainty.

        integrated_area
            Integrated Gaussian area in the native
            wavelength-flux units.

        n_points
            Number of spectral samples used.

        fit_window_um
            Half-width of fitting region.

    Notes
    -----
    These are diagnostic fit measurements.

    They should not automatically be interpreted as intrinsic
    physical line widths or final calibrated line fluxes.
    """

    wavelength = spectrum["wavelength"]
    flux = spectrum["flux"]
    uncertainty = spectrum["uncertainty"]

    mask = (wavelength >= center_guess - window) & (wavelength <= center_guess + window)

    x = wavelength[mask]
    y = flux[mask]
    yerr = uncertainty[mask]

    if len(x) < 5:
        raise ValueError("Insufficient data points for Gaussian fit.")

    # --------------------------------------------------------
    # Estimate local linear continuum from the edges.
    # --------------------------------------------------------

    edge_width = max(
        2,
        len(x) // 10,
    )

    edge_x = np.concatenate(
        [
            x[:edge_width],
            x[-edge_width:],
        ]
    )

    edge_y = np.concatenate(
        [
            y[:edge_width],
            y[-edge_width:],
        ]
    )

    continuum_coefficients = np.polyfit(
        edge_x,
        edge_y,
        1,
    )

    continuum = np.polyval(
        continuum_coefficients,
        x,
    )

    line_flux = y - continuum

    # --------------------------------------------------------
    # Initial Gaussian parameters.
    # --------------------------------------------------------

    amplitude_guess = float(np.max(line_flux))

    center_initial = float(x[np.argmax(line_flux)])

    sigma_guess = max(
        np.median(np.diff(x)) * 2,
        window / 10,
    )

    # --------------------------------------------------------
    # Gaussian fit.
    # --------------------------------------------------------

    parameters, covariance = curve_fit(
        gaussian,
        x,
        line_flux,
        p0=[
            amplitude_guess,
            center_initial,
            sigma_guess,
        ],
        sigma=yerr,
        absolute_sigma=True,
        maxfev=10000,
    )

    amplitude, center, sigma = parameters

    parameter_errors = np.sqrt(np.diag(covariance))

    amplitude_error = parameter_errors[0]
    center_error = parameter_errors[1]
    sigma_error = parameter_errors[2]

    # --------------------------------------------------------
    # Convert sigma to FWHM.
    # --------------------------------------------------------

    gaussian_factor = 2.0 * np.sqrt(2.0 * np.log(2.0))

    fwhm = gaussian_factor * abs(sigma)

    fwhm_error = gaussian_factor * abs(sigma_error)

    # --------------------------------------------------------
    # Integrated Gaussian area.
    # --------------------------------------------------------

    area = amplitude * abs(sigma) * np.sqrt(2.0 * np.pi)

    return {
        "center_um": float(center),
        "center_error_um": float(center_error),
        "amplitude": float(amplitude),
        "amplitude_error": float(amplitude_error),
        "sigma_um": float(abs(sigma)),
        "sigma_error_um": float(abs(sigma_error)),
        "fwhm_um": float(fwhm),
        "fwhm_error_um": float(fwhm_error),
        "integrated_area": float(area),
        "n_points": int(len(x)),
        "fit_window_um": float(window),
    }


# ============================================================
# Candidate feature characterization
# ============================================================


def characterize_candidate_feature(
    spectrum,
    center_guess,
    window=0.003,
    continuum_window=51,
):
    """
    Characterize a candidate spectral feature.

    This function combines local-continuum estimation with the
    existing Gaussian fitting routine.

    Parameters
    ----------
    spectrum : dict
        Prepared spectrum from prepare_spectrum().

    center_guess : float
        Approximate wavelength of the candidate in microns.

    window : float, optional
        Half-width of the Gaussian fitting region in microns.

    continuum_window : int, optional
        Width of the median-filter window used to estimate the
        local continuum.

    Returns
    -------
    dict
        Structured diagnostic measurements for the candidate.

    Notes
    -----
    This function does not identify the physical origin of the
    feature. It only characterizes the observed spectral feature.

    The resulting measurements should not automatically be
    interpreted as intrinsic physical line properties.
    """

    wavelength = spectrum["wavelength"]
    flux = spectrum["flux"]

    if continuum_window < 3:
        raise ValueError("continuum_window must be at least 3.")

    if continuum_window % 2 == 0:
        raise ValueError("continuum_window must be an odd integer.")

    # --------------------------------------------------------
    # Estimate local continuum.
    # --------------------------------------------------------

    from scipy.ndimage import median_filter

    continuum = median_filter(
        flux,
        size=continuum_window,
        mode="nearest",
    )

    # --------------------------------------------------------
    # Find the spectrum sample closest to the requested
    # candidate wavelength.
    # --------------------------------------------------------

    nearest_index = int(np.argmin(np.abs(wavelength - center_guess)))

    measured_wavelength = float(wavelength[nearest_index])

    measured_flux = float(flux[nearest_index])

    local_continuum = float(continuum[nearest_index])

    excess_flux = measured_flux - local_continuum

    # --------------------------------------------------------
    # Perform Gaussian characterization.
    # --------------------------------------------------------

    fit = fit_gaussian_line(
        spectrum,
        center_guess=center_guess,
        window=window,
    )

    # --------------------------------------------------------
    # Combine the direct measurement and Gaussian-fit results.
    # --------------------------------------------------------

    return {
        "candidate_wavelength_um": float(center_guess),
        "measured_wavelength_um": measured_wavelength,
        "measured_flux": measured_flux,
        "local_continuum": local_continuum,
        "excess_flux": float(excess_flux),
        "gaussian_fit": fit,
    }


# ============================================================
# Spectral Evidence Record
# ============================================================


def create_spectral_evidence(
    characterization,
    source="JWST/NIRSpec",
    object_name="M51",
    dataset=None,
):
    """
    Convert a spectral-feature characterization into a
    standardized evidence record.

    This function records measured quantities only. It does not
    identify the physical or atomic origin of the spectral feature.

    Parameters
    ----------
    characterization : dict
        Output from characterize_candidate_feature().

    source : str, optional
        Observational source or instrument.

    object_name : str, optional
        Astronomical object being investigated.

    dataset : str or None, optional
        Dataset or FITS product identifier.

    Returns
    -------
    dict
        Standardized astronomical evidence record.
    """

    from tools.evidence import create_evidence

    gaussian_fit = characterization["gaussian_fit"]

    facts = {
        "analysis_type": "spectral_feature_characterization",
        "source": source,
        "dataset": dataset,
        "candidate_wavelength_um": (characterization["candidate_wavelength_um"]),
        "measured_wavelength_um": (characterization["measured_wavelength_um"]),
        "measured_flux": (characterization["measured_flux"]),
        "local_continuum": (characterization["local_continuum"]),
        "excess_flux": (characterization["excess_flux"]),
        "gaussian_center_um": (gaussian_fit["center_um"]),
        "gaussian_center_error_um": (gaussian_fit["center_error_um"]),
        "gaussian_amplitude": (gaussian_fit["amplitude"]),
        "gaussian_amplitude_error": (gaussian_fit["amplitude_error"]),
        "gaussian_sigma_um": (gaussian_fit["sigma_um"]),
        "gaussian_sigma_error_um": (gaussian_fit["sigma_error_um"]),
        "gaussian_fwhm_um": (gaussian_fit["fwhm_um"]),
        "gaussian_fwhm_error_um": (gaussian_fit["fwhm_error_um"]),
        "integrated_area": (gaussian_fit["integrated_area"]),
        "fit_points": (gaussian_fit["n_points"]),
        "fit_window_um": (gaussian_fit["fit_window_um"]),
        "identification_status": ("not_identified"),
    }

    return create_evidence(
        evidence_type="spectral_analysis",
        source=source,
        object_name=object_name,
        facts=facts,
    )
