"""Input and edge-control videos of X-Plane approaches for Cosmos-Transfer2.5 (one clip = consecutive frames of one episode).

    python cosmos_prep.py <episodes_dir> <out_dir> <n_frames> <episode>:<last_frame>[:<stride>] [...]

<episodes_dir>/<ep>/{images,corners,position} come from the xp12_flights archive. A clip is the last <n_frames> frames up to
<last_frame>, every <stride> frames (sim frames are ~9 per second, ~5.5 m apart; the video is written at 16 fps). For each clip, writes
<out>/<ep>_<last>/input.mp4 (sim frames, HUD text band and gaze box removed), edge.mp4 (Canny of the sim frame + GT runway quad, white on black),
mask.mp4 (white on the runway dilated by 30 % of its width, 8 to 40 px: where the edge control applies when given as mask_path)
edge_light.mp4 (same inside the mask; outside it only the edges of the blurred frame longer than LMIN px (default 60): horizon, roads,
large buildings, so the layout stays fixed but the sim texture is free) and gt.json (runway corners in pixels, distance to threshold and whether the frame can be labelled, for every frame).
A frame is labelled when both far corners are in view and the near corners project below them (near corners may be off-screen).
"""
import sys, os, json, numpy as np, cv2

SRC, OUT, NF = sys.argv[1], sys.argv[2], int(sys.argv[3]); W, H, FPS = 1280, 720, 16; LMIN = int(os.environ.get("LMIN", 60))


def fill_rows(img, x0, x1, y0, y1):
    l = img[y0:y1, x0-6:x0-1].astype(np.float32).mean(1, keepdims=True); r = img[y0:y1, x1+1:x1+6].astype(np.float32).mean(1, keepdims=True)
    w = np.linspace(0, 1, x1-x0)[None, :, None]; img[y0:y1, x0:x1] = np.clip(l*(1-w)+r*w, 0, 255).astype(np.uint8)


BOX = np.zeros((H, W), np.uint8)                                                            # X-Plane gaze box: thin square + centre cross
cv2.rectangle(BOX, (529, 69), (750, 290), 255, 5); cv2.rectangle(BOX, (627, 167), (652, 192), 255, -1)


def writer(path):
    return cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))


for spec in sys.argv[4:]:
    ep, last, *st = spec.split(":"); last, st = int(last), int(st[0]) if st else 1; d = f"{SRC}/{ep}"
    frames = sorted(int(f.split(".")[0]) for f in os.listdir(f"{d}/images"))
    frames = [f for f in frames if f <= last and (last - f) % st == 0][-NF:]
    o = f"{OUT}/{ep}_{last}" + (f"s{st}" if st > 1 else ""); os.makedirs(o, exist_ok=True); (vi, ve, vm, vl), gt = [writer(f"{o}/{k}_raw.mp4") for k in ("input", "edge", "mask", "edge_light")], []
    for f in frames:
        im = cv2.imread(f"{d}/images/{f:06d}.jpg"); im = cv2.resize(im, (W, H)) if im.shape[:2] != (H, W) else im
        fill_rows(im, 405, 876, 28, 58); im = cv2.inpaint(im, BOX, 3, cv2.INPAINT_TELEA)         # X-Plane static overlays: HUD text band, gaze box
        c = np.array(open(f"{d}/corners/{f:06d}.txt").read().split(), float).reshape(4, 3); q = c[:, :2] * [W, H]
        e = cv2.Canny(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), 60, 160)
        mk = np.zeros((H, W), np.uint8); ok = bool(c[1, 2] == 1 and c[2, 2] == 1 and q[0, 1] > q[1, 1] and q[3, 1] > q[2, 1])     # NL, FL, FR, NR
        if ok:
            qi = np.round(q).astype(np.int32); cv2.polylines(e, [qi], True, 255, 2); cv2.fillPoly(mk, [qi], 255)
            r = int(np.clip(0.3 * np.ptp(q[:, 0]), 8, 40)); mk = cv2.dilate(mk, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
        el = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (0, 0), 1.5), 40, 100)            # layout edges only
        n, lab, st_, _ = cv2.connectedComponentsWithStats(el, connectivity=8)
        el = np.where(np.isin(lab, [i for i in range(1, n) if max(st_[i, 2], st_[i, 3]) >= LMIN]), 255, 0).astype(np.uint8)
        el[mk > 0] = e[mk > 0]                                                                   # full detail on the runway and around it
        vl.write(cv2.cvtColor(el, cv2.COLOR_GRAY2BGR)); vi.write(im); ve.write(cv2.cvtColor(e, cv2.COLOR_GRAY2BGR)); vm.write(cv2.cvtColor(mk, cv2.COLOR_GRAY2BGR))
        pos = float(open(f"{d}/position/{f:06d}.txt").read().split()[0]) if os.path.exists(f"{d}/position/{f:06d}.txt") else None
        gt.append({"frame": f, "corners": q.round(2).tolist(), "visible": bool(c[:, 2].sum() == 4), "labelled": ok, "along_m": pos})
    vi.release(); ve.release(); vm.release(); vl.release()
    for k in ("input", "edge", "mask", "edge_light"):                                                                      # re-encode in H.264 for the decoder
        os.system(f"ffmpeg -loglevel error -y -i {o}/{k}_raw.mp4 -c:v libx264 -pix_fmt yuv420p -crf 12 {o}/{k}.mp4 && rm {o}/{k}_raw.mp4")
    json.dump(gt, open(f"{o}/gt.json", "w"))
    print(ep, last, len(frames), "frames, along_m", gt[0]["along_m"], "->", gt[-1]["along_m"], flush=True)
