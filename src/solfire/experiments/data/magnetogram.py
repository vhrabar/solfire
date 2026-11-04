"""
GONG site magnetograms ("bzi") resampled onto the MAGFiLO H-alpha pixel grid.
"""

from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy import ndimage

HA_SIZE = 2048
HA_CENTRE = 1024.0
HA_RADIUS = 900.0
ROTATION_DEG = 90.0
# the magnetogram is unreliable in a thin ring at the limb
LIMB_FRACTION = 0.97
SOLAR_RADIUS_MM = 696.0
# full-frame grid the auxiliary targets are stored on, about 3 Mm per pixel
TARGET_GRID = 512


def read_magnetogram(path) -> tuple[np.ndarray, fits.Header]:
    """
    Load one bzi FITS file.
    :param path: Path to a ``*bzi*.fits.gz`` file
    :return: Tuple of (line-of-sight field in gauss with NaN set to 0, header)
    """
    with fits.open(path) as hdus:
        data = np.nan_to_num(hdus[0].data.astype(np.float32))
        return data, hdus[0].header


def disk_coordinates(size, window=None) -> tuple[np.ndarray, np.ndarray]:
    """
    Disk-normalised coordinates of an H-alpha pixel grid.
    :param size: Edge length of the output grid
    :param window: ``(x0, y0, edge)`` in native 2048 px H-alpha pixels, or ``None`` for the full frame
    :return: Tuple of (u, v), right and down in units of the disk radius
    """
    x0, y0, edge = window if window is not None else (0, 0, HA_SIZE)
    steps = (np.arange(size) + 0.5) * edge / size
    v, u = np.meshgrid((y0 + steps - HA_CENTRE) / HA_RADIUS, (x0 + steps - HA_CENTRE) / HA_RADIUS, indexing="ij")
    return u, v


def magnetogram_on_grid(field, header, size, window=None, rotation_deg=ROTATION_DEG) -> tuple[np.ndarray, np.ndarray]:
    """
    Resample a bzi magnetogram onto (part of) the H-alpha pixel grid.
    :param field: Magnetogram from :func:`read_magnetogram`
    :param header: Its FITS header
    :param size: Edge length of the output grid
    :param window: ``(x0, y0, edge)`` in native H-alpha pixels, or ``None`` for the full frame
    :param rotation_deg: Turn from H-alpha to bzi orientation
    :return: Tuple of (field in gauss, boolean mask of on-disk pixels away from the limb)
    """
    u, v = disk_coordinates(size, window)
    angle = np.deg2rad(rotation_deg)
    turned_u = np.cos(angle) * u - np.sin(angle) * v
    turned_v = np.sin(angle) * u + np.cos(angle) * v
    radius = (header["FNDLMBMI"] + header["FNDLMBMA"]) / 2
    x = header["FNDLMBXC"] + radius * turned_u
    y = header["FNDLMBYC"] + radius * turned_v
    resampled = ndimage.map_coordinates(field, [y, x], order=1, cval=0.0).astype(np.float32)
    valid = u**2 + v**2 < LIMB_FRACTION**2
    resampled[~valid] = 0.0
    return resampled, valid


def plage_correlation(field, header, jpeg_path, rotation_deg=ROTATION_DEG, size=256) -> float:
    """
    How well |B| lines up with H-alpha activity, as a registration and timing check.
    :param field: Magnetogram from :func:`read_magnetogram`
    :param header: Its FITS header
    :param jpeg_path: The matching MAGFiLO H-alpha JPEG
    :param rotation_deg: Turn from H-alpha to bzi orientation
    :param size: Working grid edge length
    :return: Pearson correlation over the inner disk
    """
    from PIL import Image

    with Image.open(jpeg_path) as handle:
        image = np.asarray(handle.convert("L").resize((size, size), Image.Resampling.BILINEAR), np.float32)
    activity = np.abs(image - ndimage.median_filter(image, 15))
    resampled, _ = magnetogram_on_grid(field, header, size, rotation_deg=rotation_deg)
    u, v = disk_coordinates(size)
    inner = u**2 + v**2 < 0.9**2
    a = ndimage.gaussian_filter(np.abs(resampled), 2)[inner]
    b = ndimage.gaussian_filter(activity, 2)[inner]
    return float(np.corrcoef(a, b)[0, 1])


def mm_per_pixel(size) -> float:
    """
    :param size: Edge length of a grid covering the whole H-alpha frame
    :return: Megametres per grid pixel at disk centre
    """
    return float(SOLAR_RADIUS_MM / HA_RADIUS * HA_SIZE / size)


def pil_map(field, valid, sigma, gradient) -> np.ndarray:
    """
    Polarity-inversion pixels of a smoothed magnetogram.
    :param field: Line-of-sight field on the grid, gauss
    :param valid: On-disk mask
    :param sigma: Gaussian smoothing in grid pixels
    :param gradient: Least field gradient across the line, gauss per grid pixel
    :return: Boolean PIL mask
    """
    smooth = ndimage.gaussian_filter(field, sigma)
    sign = smooth > 0
    crossing = np.zeros_like(sign)
    crossing[:, :-1] |= sign[:, :-1] != sign[:, 1:]
    crossing[:-1, :] |= sign[:-1, :] != sign[1:, :]
    steep = np.hypot(*np.gradient(smooth)) >= gradient
    inner = ndimage.binary_erosion(valid, iterations=int(2 * sigma) + 2)
    mask: np.ndarray = crossing & steep & inner
    return mask


def magnetic_targets(field, header, size=TARGET_GRID, sigma_mm=12.0, gradient=1.0, length_mm=10.0, strong_gauss=100.0):
    """
    Auxiliary training targets of one observation on a coarse full-frame grid.
    :param field: Magnetogram from :func:`read_magnetogram`
    :param header: Its FITS header
    :param size: Grid edge length
    :param sigma_mm: Smoothing before the inversion lines are traced
    :param gradient: Least field gradient across a line, gauss per grid pixel
    :param length_mm: Decay length of the PIL proximity
    :param strong_gauss: Field scale of the strong-flux target
    :return: ``(3, size, size)`` float32: PIL proximity ``exp(-d / length)``, strong flux
        ``tanh(|B| / strong_gauss)`` at about 3 Mm smoothing, and the valid-pixel mask
    """
    resampled, valid = magnetogram_on_grid(field, header, size)
    scale = mm_per_pixel(size)
    pil = pil_map(resampled, valid, sigma_mm / scale, gradient)
    distance = ndimage.distance_transform_edt(~pil) * scale if pil.any() else np.full(pil.shape, np.inf)
    proximity = np.exp(-distance / length_mm)
    strong = np.tanh(np.abs(ndimage.gaussian_filter(resampled, 3.0 / scale)) / strong_gauss)
    inner = ndimage.binary_erosion(valid, iterations=int(2 * sigma_mm / scale) + 2)
    return np.stack([proximity * inner, strong * inner, inner]).astype(np.float32)


class MagneticTargets:
    """
    Windows of the precomputed auxiliary targets (``scripts/experiments/magnetic_targets.py``).
    """

    def __init__(self, directory) -> None:
        """
        :param directory: Directory of ``<image stem>.npy`` files, ``(3, G, G)`` uint8
        """
        self.directory = Path(directory)
        self.available = {path.stem for path in self.directory.glob("*.npy")}
        if not self.available:
            raise FileNotFoundError(
                f"no magnetic targets in {self.directory}; build them with scripts/experiments/magnetic_targets.py"
            )

    def window(self, file_name, x, y, crop, size) -> np.ndarray:
        """
        The targets under one crop, upsampled to the crop's pixels.
        :param file_name: Observation file name
        :param x: Left edge of the crop in the decoded observation
        :param y: Top edge of the crop
        :param crop: Edge length of the crop
        :param size: Edge length the observation is decoded at
        :return: ``(3, crop, crop)`` float32; all zeros, valid included, without a magnetogram
        """
        stem = Path(file_name).stem
        if stem not in self.available:
            return np.zeros((3, crop, crop), dtype=np.float32)
        grid = np.load(self.directory / f"{stem}.npy", mmap_mode="r")
        factor = grid.shape[-1] / size
        steps = (np.arange(crop) + 0.5) * factor - 0.5
        rows, columns = np.meshgrid(y * factor + steps, x * factor + steps, indexing="ij")
        out = np.stack([ndimage.map_coordinates(np.asarray(channel, np.float32), [rows, columns], order=1) for channel in grid])
        out /= 255.0
        out[2] = out[2] > 0.5
        return out


def magnetogram_path(directory, image_name) -> Path:
    """
    Where the magnetogram matched to one H-alpha image is stored.
    :param directory: Magnetogram directory
    :param image_name: H-alpha file name such as ``20110109104734Ch.jpeg``
    :return: Path of the matched ``.fits.gz``
    """
    return Path(directory) / f"{Path(image_name).stem}.fits.gz"
