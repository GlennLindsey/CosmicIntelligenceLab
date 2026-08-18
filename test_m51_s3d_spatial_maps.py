from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits


# ============================================================
# M51 JWST/NIRSpec S3D spatial-map investigation
# ============================================================

S3D_PATH = Path(
    "data/m51_jwst_level3/"
    "jw03435-o006_t010_nirspec_g140m-f100lp_s3d.fits"
)


# ============================================================
# Spectral targets
# ============================================================

FEATURE_WAVELENGTH_NM = 1284.26130440

PA_BETA_WAVELENGTH_NM = 1284.26002473

FEII_WAVELENGTH_NM = 1259.08723


# ============================================================
# Spectral integration settings
# ============================================================

# The cube has approximately 0.636 nm spectral sampling.
#
# We use several neighboring planes rather than a single
# plane so that the map represents integrated line emission.

HALF_WIDTH_NM = 1.5


# ============================================================
# Utility
# ============================================================

def extract_map(
    cube,
    wavelength_nm,
    cube_wavelength_nm,
    half_width_nm,
):
    """
    Integrate the S3D cube over a wavelength window.
    """

    mask = (
        np.abs(
            cube_wavelength_nm
            - wavelength_nm
        )
        <= half_width_nm
    )

    indices = np.where(mask)[0]

    if len(indices) == 0:

        raise ValueError(
            f"No spectral planes found near "
            f"{wavelength_nm:.6f} nm"
        )

    selected = cube[indices, :, :]

    # Integrate using a simple sum because the spectral
    # sampling is uniform.

    line_map = np.nansum(
        selected,
        axis=0,
    )

    return line_map, indices


# ============================================================
# Main
# ============================================================

print("=" * 70)
print("M51 JWST/NIRSpec S3D SPATIAL MAP INVESTIGATION")
print("=" * 70)

print()
print("S3D:")
print(S3D_PATH)


# ============================================================
# Load cube
# ============================================================

with fits.open(S3D_PATH) as hdul:

    cube = np.asarray(
        hdul["SCI"].data,
        dtype=float,
    )

    header = hdul["SCI"].header


# ============================================================
# Construct wavelength array
# ============================================================

n_wave = cube.shape[0]

crval3 = header["CRVAL3"]
crpix3 = header["CRPIX3"]
cdelt3 = header["CDELT3"]

pixel = np.arange(
    1,
    n_wave + 1,
    dtype=float,
)

wavelength_um = (
    crval3
    + (
        pixel
        - crpix3
    )
    * cdelt3
)

wavelength_nm = (
    wavelength_um
    * 1000.0
)


# ============================================================
# Basic information
# ============================================================

print()
print("=" * 70)
print("CUBE INFORMATION")
print("=" * 70)

print(
    f"Cube shape: {cube.shape}"
)

print(
    f"Wavelength range: "
    f"{wavelength_nm.min():.3f} - "
    f"{wavelength_nm.max():.3f} nm"
)

print(
    f"Spectral sampling: "
    f"{cdelt3 * 1000.0:.6f} nm"
)

print(
    f"Spatial dimensions: "
    f"{cube.shape[2]} x {cube.shape[1]}"
)

print(
    f"BUNIT: "
    f"{header.get('BUNIT')}"
)


# ============================================================
# Target wavelength diagnostics
# ============================================================

targets = {
    "1284 feature": FEATURE_WAVELENGTH_NM,
    "Pa beta": PA_BETA_WAVELENGTH_NM,
    "[Fe II] 1.257": FEII_WAVELENGTH_NM,
}


print()
print("=" * 70)
print("TARGET SPECTRAL PLANES")
print("=" * 70)

for label, target in targets.items():

    index = np.argmin(
        np.abs(
            wavelength_nm
            - target
        )
    )

    print()
    print(label)

    print(
        f"  Target wavelength: "
        f"{target:.6f} nm"
    )

    print(
        f"  Nearest plane: "
        f"{index}"
    )

    print(
        f"  Cube wavelength: "
        f"{wavelength_nm[index]:.6f} nm"
    )

    print(
        f"  Difference: "
        f"{wavelength_nm[index] - target:+.6f} nm"
    )


# ============================================================
# Extract maps
# ============================================================

maps = {}

for label, target in targets.items():

    line_map, indices = extract_map(
        cube,
        target,
        wavelength_nm,
        HALF_WIDTH_NM,
    )

    maps[label] = line_map

    print()
    print(
        f"{label}: "
        f"{len(indices)} spectral planes"
    )

    print(
        "  Plane indices:",
        indices,
    )

    print(
        "  Wavelengths:",
        wavelength_nm[indices],
    )

    finite = np.isfinite(line_map)

    if np.any(finite):

        print(
            f"  Map minimum: "
            f"{np.nanmin(line_map):.6g}"
        )

        print(
            f"  Map maximum: "
            f"{np.nanmax(line_map):.6g}"
        )

        print(
            f"  Map median: "
            f"{np.nanmedian(line_map):.6g}"
        )


# ============================================================
# Spatial morphology comparison
# ============================================================

print()
print("=" * 70)
print("SPATIAL PEAK LOCATIONS")
print("=" * 70)

for label, line_map in maps.items():

    finite = np.isfinite(line_map)

    if not np.any(finite):
        continue

    safe_map = np.where(
        finite,
        line_map,
        -np.inf,
    )

    y, x = np.unravel_index(
        np.argmax(safe_map),
        safe_map.shape,
    )

    print()
    print(label)

    print(
        f"  Peak pixel: "
        f"x={x}, y={y}"
    )

    print(
        f"  Peak value: "
        f"{line_map[y, x]:.6g}"
    )


# ============================================================
# Generate individual spatial maps
# ============================================================

print()
print("=" * 70)
print("GENERATING SPATIAL MAPS")
print("=" * 70)


for label, line_map in maps.items():

    plt.figure(
        figsize=(10, 7)
    )

    plt.imshow(
        line_map,
        origin="lower",
        aspect="equal",
    )

    plt.colorbar(
        label="Integrated flux (MJy/sr × planes)"
    )

    plt.xlabel(
        "Spatial pixel X"
    )

    plt.ylabel(
        "Spatial pixel Y"
    )

    plt.title(
        f"M51 NIRSpec — {label}"
    )

    filename = (
        "m51_s3d_"
        + label.replace(" ", "_")
        .replace("[", "")
        .replace("]", "")
        .replace(".", "")
        + ".png"
    )

    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=150,
    )

    plt.close()

    print(
        f"Saved: {filename}"
    )


# ============================================================
# Difference / morphology comparison
# ============================================================

feature_map = maps[
    "1284 feature"
]

pabeta_map = maps[
    "Pa beta"
]

feii_map = maps[
    "[Fe II] 1.257"
]


# ============================================================
# Correlation analysis
# ============================================================

print()
print("=" * 70)
print("SPATIAL CORRELATION")
print("=" * 70)


def spatial_correlation(
    map_a,
    map_b,
):

    valid = (
        np.isfinite(map_a)
        & np.isfinite(map_b)
    )

    if np.sum(valid) < 10:

        return np.nan

    a = map_a[valid]
    b = map_b[valid]

    if (
        np.std(a) == 0
        or np.std(b) == 0
    ):

        return np.nan

    return np.corrcoef(
        a,
        b,
    )[0, 1]


corr_feature_pabeta = spatial_correlation(
    feature_map,
    pabeta_map,
)

corr_feature_feii = spatial_correlation(
    feature_map,
    feii_map,
)


print(
    "1284 feature vs Pa beta:"
)

print(
    f"  Pearson r = "
    f"{corr_feature_pabeta:.5f}"
)


print(
    "1284 feature vs [Fe II]:"
)

print(
    f"  Pearson r = "
    f"{corr_feature_feii:.5f}"
)


# ============================================================
# Combined comparison plot
# ============================================================

plt.figure(
    figsize=(10, 7)
)

plt.imshow(
    feature_map,
    origin="lower",
    aspect="equal",
)

plt.colorbar(
    label="1284 nm integrated flux"
)

plt.xlabel(
    "Spatial pixel X"
)

plt.ylabel(
    "Spatial pixel Y"
)

plt.title(
    "M51 — 1284 nm Spatial Emission Map"
)

plt.tight_layout()

plt.savefig(
    "m51_1284_spatial_map.png",
    dpi=150,
)

plt.close()


# ============================================================
# Final summary
# ============================================================

print()
print("=" * 70)
print("INTERPRETATION")
print("=" * 70)

print(
    """
This analysis tests whether the 1284 nm emission has
a spatial morphology similar to independently identified
nebular emission.

A strong spatial correlation with Pa beta and/or
[Fe II] would support the interpretation that the
1284 nm feature arises from the same physical nebular
region.

A substantially different spatial morphology would
motivate a more detailed investigation of an additional
emission component.

This is a spatial test only. It does not by itself
identify the atomic transition.
"""
)

print()
print("=" * 70)
print("S3D SPATIAL ANALYSIS COMPLETE")
print("=" * 70)
