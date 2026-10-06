"""
Physically-based weather on XP12 frames, using the exact scene geometry.

Fog / haze follows Koschmieder's law per pixel:

    I = J · t + A · (1 − t),     t = exp(−β · d),     β = 3.912 / V

with ``d`` the camera-to-ground distance along the pixel ray (known exactly from
the pose and the runway plane — no depth estimator), ``V`` the meteorological
visibility (m, 2 % contrast threshold) and ``A`` the airlight colour. Pixels
whose ray does not hit the ground (sky) get ``d = ∞``. Geometry is untouched,
so pose and corner labels remain valid.
"""

from __future__ import annotations

import cv2
import numpy as np

from pose_estimation.runway_model import CameraPoseLocal
from runway_detection.lard.intrinsics import CameraIntrinsics
from runway_detection.lard.projection import apply_rotations_intrinsic

KOSCHMIEDER = 3.912  # −ln(0.02)


def _fill_rows(img: np.ndarray, x0: int, x1: int, y0: int, y1: int) -> None:
    """Replace a sky rectangle by a per-row linear blend of its left / right borders."""
    left = img[y0:y1, x0 - 6 : x0 - 1].astype(np.float32).mean(axis=1, keepdims=True)
    right = img[y0:y1, x1 + 1 : x1 + 6].astype(np.float32).mean(axis=1, keepdims=True)
    w = np.linspace(0.0, 1.0, x1 - x0)[None, :, None]
    img[y0:y1, x0:x1] = np.clip(left * (1 - w) + right * w, 0, 255).astype(np.uint8)


def remove_hud(bgr: np.ndarray) -> np.ndarray:
    """Erase the X-Plane overlay (time-speed label + tinted targeting box); the sky is a smooth gradient."""
    out = bgr.copy()
    _fill_rows(out, 575, 705, 30, 56)
    _fill_rows(out, 518, 762, 58, 302)
    return out


def ground_distance_map(
    pose: CameraPoseLocal,
    intrinsics: CameraIntrinsics,
    *,
    runway_azimuth_deg: float = 0.0,
) -> np.ndarray:
    """Distance (m) from the camera to the runway plane z = 0 along each pixel ray; ∞ above the horizon."""
    x_cam, y_cam, z_cam = apply_rotations_intrinsic(
        *np.radians([runway_azimuth_deg - pose.yaw_cam_deg, 90.0 - pose.pitch_cam_deg, pose.roll_cam_deg])
    )
    u = np.arange(intrinsics.width) + 0.5
    v = np.arange(intrinsics.height) + 0.5
    a = (intrinsics.cx - u)[None, :] / intrinsics.fx  # left component
    b = (intrinsics.cy - v)[:, None] / intrinsics.fy  # up component
    dir_z = x_cam[2] + a * y_cam[2] + b * z_cam[2]
    norm = np.sqrt(1.0 + a**2 + b**2)
    with np.errstate(divide="ignore", invalid="ignore"):
        dist = np.where(dir_z < -1e-9, pose.height_m / -dir_z * norm, np.inf)
    return dist


def estimate_airlight(bgr: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Fog colour: bright, slightly desaturated version of the sky just above the horizon."""
    sky = ~np.isfinite(dist)
    rows = np.nonzero(sky.any(axis=1))[0]
    if rows.size == 0:
        return np.array([215.0, 215.0, 215.0])
    band = bgr[max(rows.max() - 40, 0) : rows.max() + 1][sky[max(rows.max() - 40, 0) : rows.max() + 1]]
    base = band.reshape(-1, 3).mean(axis=0) if band.size else np.array([215.0, 215.0, 215.0])
    gray = base.mean()
    return np.clip(0.4 * base + 0.6 * gray + 12.0, 0, 255)


def apply_fog(
    bgr: np.ndarray,
    pose: CameraPoseLocal,
    intrinsics: CameraIntrinsics,
    visibility_m: float,
    *,
    airlight: np.ndarray | None = None,
    sky_distance_m: float = 30_000.0,
) -> np.ndarray:
    """Homogeneous fog with meteorological visibility ``visibility_m``."""
    dist = ground_distance_map(pose, intrinsics)
    A = estimate_airlight(bgr, dist) if airlight is None else np.asarray(airlight, dtype=float)
    d = np.where(np.isfinite(dist), dist, sky_distance_m)
    # Soften the hard horizon line (trees / terrain above the ideal plane).
    d = cv2.GaussianBlur(np.minimum(d, sky_distance_m).astype(np.float32), (0, 0), 2.0)
    t = np.exp(-KOSCHMIEDER / visibility_m * d)[..., None]
    out = bgr.astype(np.float32) * t + A[None, None, :] * (1.0 - t)
    return np.clip(out, 0, 255).astype(np.uint8)
