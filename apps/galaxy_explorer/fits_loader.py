from astropy.io import fits


def load_fits_image(filename):
    """
    Load the first 2D image found in a FITS file.

    Returns
    -------
    image : numpy.ndarray
        Image data.
    header : astropy.io.fits.Header
        FITS header.
    """

    with fits.open(filename) as hdul:

        # Look for the first HDU containing 2D image data
        for hdu in hdul:

            if hdu.data is None:
                continue

            if len(hdu.data.shape) == 2:
                return hdu.data, hdu.header

        raise RuntimeError("No 2D image found in FITS file.")
