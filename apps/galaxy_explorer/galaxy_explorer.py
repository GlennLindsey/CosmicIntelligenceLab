import sys

from PyQt6.QtGui import QAction

from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QInputDialog,
    QMainWindow,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from astropy.coordinates import SkyCoord
import astropy.units as u

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

        self.wcs = None
        self.ax = None
        self.norm = None
        self.target_marker = None

        self.wcs = None
        self.ax = None
        self.norm = None

        self.create_menu()

        # Connect the mouse once.
        self.canvas.mpl_connect(
            "motion_notify_event",
            self.mouse_moved,
        )

        self.canvas.mpl_connect(
            "button_press_event",
            self.mouse_clicked,
        )
        # Load the default image.
        self.load_image(DEFAULT_FITS_FILE)

    def create_menu(self):
        menu = self.menuBar()

        file_menu = menu.addMenu("&File")
        navigate_menu = menu.addMenu("&Navigate")

        open_action = file_menu.addAction("Open FITS...")
        open_action.triggered.connect(self.open_fits_file)

        file_menu.addSeparator()

        goto_action = QAction("Go To Coordinates...", self)
        goto_action.triggered.connect(self.goto_coordinates)
        navigate_menu.addAction(goto_action)

        exit_action = file_menu.addAction("Exit")
        exit_action.triggered.connect(self.close)

    def load_image(self, filename):
        self.image, self.header = load_fits_image(filename)
        self.wcs = WCS(self.header).celestial
        print(self.wcs)

        self.figure.clear()

        self.ax = self.figure.add_subplot(111)

        self.norm = ImageNormalize(
            self.image,
            interval=PercentileInterval(99.5),
            stretch=AsinhStretch(),
        )

        self.ax.imshow(
            self.image,
            origin="lower",
            cmap="gray",
            norm=self.norm,
        )

        self.ax.set_title("JWST NIRCam")
        self.ax.set_xlabel("X Pixels")
        self.ax.set_ylabel("Y Pixels")

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

    def goto_coordinates(self):
        """Go to celestial coordinates."""

        if self.image is None:
            return

        ra, ok = QInputDialog.getDouble(
            self,
            "Go To Coordinates",
            "Right Ascension (degrees):",
            decimals=6,
        )

        if not ok:
            return

        dec, ok = QInputDialog.getDouble(
            self,
            "Go To Coordinates",
            "Declination (degrees):",
            decimals=6,
        )

        if not ok:
            return

        sky = SkyCoord(
            ra=ra * u.deg,
            dec=dec * u.deg,
        )

        x, y = self.wcs.world_to_pixel(sky)

        half_width = 250
        half_height = 250

        self.ax.set_xlim(x - half_width, x + half_width)
        self.ax.set_ylim(y - half_height, y + half_height)

        # Remove the previous marker, if one exists.
        if hasattr(self, "target_marker") and self.target_marker is not None:
            self.target_marker.remove()

        # Draw a new marker at the target coordinates.
        (self.target_marker,) = self.ax.plot(
            x,
            y,
            marker="+",
            color="red",
            markersize=20,
            markeredgewidth=2,
        )

        self.canvas.draw()

        print("X limits:", self.ax.get_xlim())
        print("Y limits:", self.ax.get_ylim())

        self.ax.figure.canvas.draw_idle()

        print(f"RA      = {ra:.6f}")
        print(f"Dec     = {dec:.6f}")
        print(f"Pixel X = {x:.1f}")
        print(f"Pixel Y = {y:.1f}")

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

    def mouse_clicked(self, event):
        """Handle a mouse click on the image."""

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

        print("\n========== Selected Object ==========")
        print(f"RA     : {ra}")
        print(f"Dec    : {dec}")
        print(f"Pixel X: {x}")
        print(f"Pixel Y: {y}")
        print("=====================================\n")


def main():
    app = QApplication(sys.argv)

    window = GalaxyExplorer()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
