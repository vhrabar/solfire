"""
Synthetic filaments for the tests
"""

import numpy as np

SIZE = 128


def filament_polygon(rng, size=SIZE) -> list[float]:
    """
    A thin, slightly bent quadrilateral strip, the rough shape of a filament.
    :param rng: numpy random generator
    :param size: Canvas edge length
    :return: Flat ``[x0, y0, x1, y1, ...]`` polygon
    """
    x0, y0 = rng.uniform(10, size - 40, 2)
    length, width = rng.uniform(15, 30), rng.uniform(3, 7)
    angle = rng.uniform(0, np.pi)
    dx, dy = np.cos(angle) * length, np.sin(angle) * length
    nx, ny = -np.sin(angle) * width, np.cos(angle) * width
    points = [(x0, y0), (x0 + dx, y0 + dy), (x0 + dx + nx, y0 + dy + ny), (x0 + nx, y0 + ny)]
    return [float(v) for point in points for v in point]
