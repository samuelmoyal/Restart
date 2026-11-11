"""Night control: add runway light points to a clip's edge control (edge.mp4 -> edge_lights.mp4), at their true positions.

    python cosmos_lights.py <clip_dir> [...]

Series 2 lesson (000170 night v1/v2): with the day Canny edges as control, the prompt alone does not create edge-light rows;
Cosmos renders a dark runway and one orange bar at the far end. Here the lights are drawn into the control as white dots,
so they are placed by the geometry, not by the model. The runway plane is mapped from the unit square (u across, v along,
v=0 at the threshold) to the image by the homography of the 4 GT corners (NL, FL, FR, NR), so points are exact in
perspective; v < 0 extends the plane before the threshold (approach lights).
- edge lights: 2 rows (u=0, u=1), N_EDGE points per side over the runway length;
- threshold: a row of points across v=0;
- approach lights: centre line of bars before the threshold (v in [-APP, 0)), with one crossbar.
Dot radius grows with the local scale (runway width in px), 1 to 4 px.
"""
import sys, os, json, numpy as np, cv2

N_EDGE, N_THR, APP, N_APP = int(os.environ.get("N_EDGE", 40)), 14, float(os.environ.get("APP", 0.3)), 12
# v2 (000162 night2_L: dots drawn on the runway edge lines merged with them, no distinct lights): bigger dots (DIV), each
# isolated by a black ring so it reads as a blob, not as part of a line. Output name OUT (default edge_lights).
DIV, RMAX, OUT = float(os.environ.get("DIV", 25)), int(os.environ.get("RMAX", 4)), os.environ.get("OUT", "edge_lights")


def pts_unit():
    p = [(0.0, v) for v in np.linspace(0, 1, N_EDGE)] + [(1.0, v) for v in np.linspace(0, 1, N_EDGE)]
    p += [(u, 0.0) for u in np.linspace(0, 1, N_THR)]
    for v in np.linspace(-APP, -APP / N_APP, N_APP):
        p += [(u, v) for u in np.linspace(0.42, 0.58, 4)]
    p += [(u, -APP / 3) for u in np.linspace(0.15, 0.85, 10)]                                   # crossbar
    return np.array(p, np.float32)


U = pts_unit()
for clip in sys.argv[1:]:
    gt = json.load(open(f"{clip}/gt.json")); cap = cv2.VideoCapture(f"{clip}/edge.mp4")
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = cv2.VideoWriter(f"{clip}/{OUT}_raw.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 16, (W, H)); i = 0
    while True:
        ok, f = cap.read()
        if not ok: break
        g = gt[min(i, len(gt) - 1)]
        if g.get("labelled", False):
            q = np.array(g["corners"], np.float32)                                                # NL, FL, FR, NR
            Hm = cv2.getPerspectiveTransform(np.float32([[0, 0], [0, 1], [1, 1], [1, 0]]), q)
            def proj(uv):
                h = np.c_[uv, np.ones(len(uv))] @ Hm.T
                sgn = np.sign((np.c_[[[0, 0], [1, 1]], np.ones(2)] @ Hm.T)[0, 2])      # w sign of points in front of the camera
                return h[:, :2] / h[:, 2:3], h[:, 2] * sgn > 1e-6
            P, front = proj(U)
            # local scale: runway width in px at the same v (distance) as each point
            A, _ = proj(np.c_[np.zeros(len(U)), U[:, 1]]); B, _ = proj(np.c_[np.ones(len(U)), U[:, 1]])
            wpx = np.linalg.norm(A - B, axis=1)
            for (x, y), w, ok_ in zip(P, wpx, front):
                if ok_ and 0 <= x < W and 0 <= y < H and np.isfinite(w):
                    r = int(np.clip(w / DIV, 1, RMAX)); c = (int(round(x)), int(round(y)))
                    if OUT != "edge_lights": cv2.circle(f, c, r + 2, (0, 0, 0), -1)
                    cv2.circle(f, c, r, (255, 255, 255), -1, cv2.LINE_AA)
        vw.write(f); i += 1
    vw.release()
    os.system(f"ffmpeg -loglevel error -y -i {clip}/{OUT}_raw.mp4 -c:v libx264 -pix_fmt yuv420p -crf 12 {clip}/{OUT}.mp4 && rm {clip}/{OUT}_raw.mp4")
    print(clip, i, "frames", flush=True)
