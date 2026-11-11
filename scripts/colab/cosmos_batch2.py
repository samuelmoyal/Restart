"""Cosmos-Transfer2.5 specs, series 2: full edge control at reduced weight, no style image, condition set by the prompt only.

    python cosmos_batch2.py <in_dir> <spec_dir> <seed> <clip>:<condition> [...]
    condition in: night, night2, snow, fog, heavy_rain, dusk, overcast

Lessons from series 1 (2026-10-20, cosmos/long_light): light edges let the scene drift far from the runway (runway turned
into a road, clearings into asphalt); a real style image imposes its lens (fisheye, blur) and both night videos failed with
one; frames after touchdown are unusable (clips are cut at the threshold by cosmos_prep.py's <last_frame>).
Here: edge.mp4 (Canny of the sim frame + GT quad) over the whole frame, control weight CW (default 0.65) so that textures,
light and weather stay free while the layout is kept; time of day and weather come from the prompt alone.
"""
import sys, os, json

IN, SPECS, SEED = sys.argv[1], sys.argv[2], int(sys.argv[3]); ITEMS = sys.argv[4:]; CW = float(os.environ.get("CW", 0.65))
VIS, VISW, NSTEPS = os.environ.get("VIS", ""), float(os.environ.get("VISW", 0.5)), int(os.environ.get("NSTEPS", 0))
# VIS=<file stem in the clip dir> adds a vis (blur) control from that video (multi-control edge + vis), e.g. night_vis;
# NSTEPS=4 writes num_steps for the distilled edge model (run with --model=edge/distilled, 93-frame clips only).
EDGE = os.environ.get("EDGEFILE", "edge")            # edge (Canny + quad) or edge_lights (+ runway light points, cosmos_lights.py)
NEG = ("video game, flight simulator, computer graphics, 3d render, cartoon, painting, text, watermark, distorted runway, blurry, low quality, "
       "fisheye lens, curved horizon, aircraft nose, propeller, wing, engine, cockpit interior, dashboard, aircraft on the runway, "
       "neon colors, fluorescent green, oversaturated, second runway, road leading to the horizon, fire, flames, smoke, glowing embers, wide dark asphalt area")
CAM = ("A realistic video recorded by a forward-looking camera fixed on an aircraft during its final approach to an airport runway, "
       "seen from the air. No part of the aircraft is visible. ")
RWY_DAY = ("The runway ahead is a grey asphalt strip with a white centerline and white threshold markings; it is small near the horizon "
           "at first and grows as the aircraft gets closer. ")
COND = {
    "night": ("It is night. The runway is outlined by two rows of white edge lights, with green threshold lights at its near end and the "
              "white approach lighting system leading to it; the runway surface itself is dark. The surroundings are dark with scattered "
              "orange street lights and lit buildings. Natural night photograph, low light, slight sensor noise. "),
    # v2 (after 000170 night v1: lights collapsed into one orange bar at the far end, threshold stripes painted green,
    # muddy black ground): describe each light system as rows of point lights, keep the painted markings white.
    "night2": ("It is night. Two straight parallel rows of evenly spaced white runway edge lights run along both sides of the runway, "
               "from its near end to its far end, converging towards the horizon. A row of small green threshold lights crosses the "
               "near end of the runway, in front of the white painted threshold stripes. Before the runway, a line of white approach "
               "light bars with a crossbar leads to it. The runway surface between the lights is dark grey asphalt with faint white "
               "markings. Around the airport, roads lined with small point-like warm white street lights, lit windows and dark fields. Natural night photograph, "
               "low light, slight sensor noise. "),
    # v2 (after 000128 snow v1: "the cleared runway stays darker" turned a grass clearing into a giant dark runway)
    "snow": (RWY_DAY + "It is winter: white snow covers the grass, the fields, the clearings, the roofs and the trees around the airport. "
             "Only the narrow runway strip itself is cleared. Overcast white sky, diffuse light. "),
    "fog": (RWY_DAY + "Dense fog: visibility is low, the far landscape and the horizon disappear into a uniform grey haze, contrast is low "
            "and the runway emerges from the fog as the aircraft approaches. "),
    "heavy_rain": (RWY_DAY + "Heavy rain under a dark overcast sky: wet surfaces reflect the light, the distance is grey and washed out, "
                   "raindrops streak across the windshield. "),
    "dusk": (RWY_DAY + "It is dusk: warm low sunlight, long shadows, the sky glows orange near the horizon and the runway edge lights "
             "are on. "),
    "overcast": (RWY_DAY + "Overcast grey sky, flat diffuse light, muted natural colors of grass, fields, roads and buildings. "),
}

os.makedirs(SPECS, exist_ok=True)
for it in ITEMS:
    clip, cond = it.split(":"); d = f"{IN}/{clip}"; name = f"{clip}_{cond}_cw{int(CW * 100)}_s{SEED}" + {"edge": "", "edge_lights": "_L"}.get(EDGE, "_" + EDGE.replace("edge_", ""))
    spec = {"name": name, "prompt": CAM + COND[cond] + "The aircraft descends smoothly toward the runway. Real footage, natural colors.",
            "negative_prompt": NEG, "video_path": f"{d}/input.mp4", "guidance": 3, "seed": SEED,
            "edge": {"control_path": f"{d}/{EDGE}.mp4", "control_weight": CW}}
    if VIS:
        spec["vis"] = {"control_path": f"{d}/{VIS}.mp4", "control_weight": VISW}; t = VIS.replace("night_vis", "").replace("_vis", "").replace("snow", "").replace("rain", "").replace("fog46", "").replace("fog120", "").replace("fog", "")   # variant tag, e.g. night_vis3 -> v3
        spec["name"] = name = name + f"_vis{int(VISW * 100)}" + (f"v{t}" if t else "")
    if NSTEPS:
        spec["num_steps"] = NSTEPS; spec["name"] = name = name + f"_d{NSTEPS}"
        open(f"{SPECS}/{name}.model", "w").write("edge/distilled")
    json.dump(spec, open(f"{SPECS}/{name}.json", "w"), indent=1)
    print(name, flush=True)
