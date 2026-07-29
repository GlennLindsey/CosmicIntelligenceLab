import sys

from PyQt6.QtWidgets import QApplication, QMainWindow, QStatusBar

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from astropy.visualization import (
    ImageNormalize,
    PercentileInterval,
    AsinhStretch,
)

from fits_loader import load_fits_image


class GalaxyExplorer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Galaxy Explorer")
        self.resize(900, 700)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.setCentralWidget(self.canvas)

        self.status = QStatusBar()
        self.setStatusBar(self.status)

        fits_file = (
            "/home/glenn/Projects/CosmicIntelligenceLab/notebooks/"
            "mastDownload/JWST/"
            "jw01180-o013_t011_nircam_clear-f200w/"
            "jw01180-o013_t011_nircam_clear-f200w_i2d.fits"
        )

        image, header = load_fits_image(fits_file)

        ax = self.figure.add_subplot(111)

        norm = ImageNormalize(
            image,
            interval=PercentileInterval(99.5),
            stretch=AsinhStretch(),
        )

        ax.imshow(
            image,
            origin="lower",
            cmap="gray",
            norm=norm,
        )

        ax.set_title("JWST NIRCam")
        ax.set_xlabel("X Pixels")
        ax.set_ylabel("Y Pixels")

        self.status.showMessage("JWST image loaded")

        self.canvas.draw()


def main():
    app = QApplication(sys.argv)

    window = GalaxyExplorer()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
