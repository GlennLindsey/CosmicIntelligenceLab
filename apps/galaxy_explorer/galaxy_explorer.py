import sys

from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QMainWindow,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT,
)
from matplotlib.figure import Figure
from astropy.wcs import WCS

from astropy.visualization import (
    AsinhStretch,
    ImageNormalize,
    PercentileInterval,
)

from fits_loader import load_fits_image

DEFAULT_FITS_FILE = (
    "/home/glenn/Projects/CosmicIntelligenceLab/notebooks/"
    "mastDownload/JWST/"
    "jw01180-o013_t011_nircam_clear-f200w/"
    "jw01180-o013_t011_nircam_clear-f200w_i2d.fits"
)


class GalaxyExplorer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Galaxy Explorer")
        self.resize(900, 700)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        self.setCentralWidget(central)

        self.status = QStatusBar()
        self.setStatusBar(self.status)

        self.image = None
        self.header = None

        self.create_menu()

        # Connect the mouse once.
        self.canvas.mpl_connect(
            "motion_notify_event",
            self.mouse_moved,
        )

        # Load the default image.
        self.load_image(DEFAULT_FITS_FILE)

    def create_menu(self):
        menu = self.menuBar()

        file_menu = menu.addMenu("&File")

        open_action = file_menu.addAction("Open FITS...")
        open_action.triggered.connect(self.open_fits_file)

        file_menu.addSeparator()

        exit_action = file_menu.addAction("Exit")
        exit_action.triggered.connect(self.close)

    def load_image(self, filename):
        self.image, self.header = load_fits_image(filename)
        self.wcs = WCS(self.header).celestial
        print(self.wcs)

        self.figure.clear()

        ax = self.figure.add_subplot(111)

        norm = ImageNormalize(
            self.image,
            interval=PercentileInterval(99.5),
            stretch=AsinhStretch(),
        )

        ax.imshow(
            self.image,
            origin="lower",
            cmap="gray",
            norm=norm,
        )

        ax.set_title("JWST NIRCam")
        ax.set_xlabel("X Pixels")
        ax.set_ylabel("Y Pixels")

        self.status.showMessage(f"Loaded: {filename}")

        self.canvas.draw()

    def open_fits_file(self):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open FITS Image",
            "",
            "FITS Files (*.fits *.fit *.fts *.fits.gz)",
        )

        if filename:
            self.load_image(filename)

    def mouse_moved(self, event):
        """Update the status bar while moving the mouse."""

        if self.image is None:
            return

        if event.inaxes is None:
            return

        if event.xdata is None or event.ydata is None:
            return

        x = int(event.xdata)
        y = int(event.ydata)

        sky = self.wcs.pixel_to_world(x, y)

        ra = sky.ra.to_string(
            unit="hour",
            sep=":",
            precision=2,
        )

        dec = sky.dec.to_string(
            unit="degree",
            sep=":",
            precision=1,
            alwayssign=True,
        )

        if 0 <= x < self.image.shape[1] and 0 <= y < self.image.shape[0]:
            value = self.image[y, x]

            sky = self.wcs.pixel_to_world(x, y)

            ra = sky.ra.to_string(
                unit="hour",
                sep=":",
                precision=2,
            )

            dec = sky.dec.to_string(
                unit="degree",
                sep=":",
                precision=1,
                alwayssign=True,
            )

            self.status.showMessage(
                f"RA {ra}    "
                f"Dec {dec}    "
                f"X {x}    "
                f"Y {y}    "
                f"Pixel {value:.2f}"
            )


def main():
    app = QApplication(sys.argv)

    window = GalaxyExplorer()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
