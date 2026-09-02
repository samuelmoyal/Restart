# Homography-based 2-DOF readout (experimental)

## Idea

With **4 DOF fixed** (along-track, height, pitch, roll), build a nominal
homography `H_nom` from runway plane (meters) → image pixels assuming
**zero heading error** and **zero lateral offset**.

Rectify the image with `H_nom⁻¹`. The runway appears as a **rotated + shifted**
OBB in the plane; the rotation encodes **heading error**.

## Validation

```bash
PYTHONPATH=. python scripts/validate_homography_dof.py --data-root data/LARD -n 8
```

Overlays: `data/LARD/homography_overlays/`

## Results (GT corners, CYEG flsim)

| DOF | Readout in rectified plane | Works? |
|-----|---------------------------|--------|
| **Heading error** | Angle of runway centerline vs +X | **Yes** — \|ΔHE\| ≈ 0.7° mean on 30 frames |
| **Lateral offset** | Y of threshold in plane coords | **No** — not equal to metric CTE (perspective) |

Heading error is validated by comparing centerline angle after `H_nom⁻¹` to the
camera forward direction projected on the ground plane.

Lateral offset is **not** a pure 2D translation in the rectified plane: moving
the camera cross-track changes perspective convergence, not just OBB position.
For metric CTE, keep `g_dof` or combine HE from homography + scale from height.

## Next steps

- Detect OBB on YOLO mask in rectified view (not GT corners)
- HE-only homography + 1D `g_dof` sweep on lateral
- Optional: branch-and-bound if greedy coarse-to-fine misses basins
