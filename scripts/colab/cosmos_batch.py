"""Cosmos-Transfer2.5 spec files: runway-only edge control + a random real image as style reference, prompt built from its tags.

    python cosmos_batch.py <in_dir> <spec_dir> <captions.json> <seed> <clip> [...]

For each clip of cosmos_prep.py, a real image is drawn from the LoRA training captions (HPA, playlist and the LARD airports
not held out), and the prompt describes the same time of day, sky and weather, so the text and the style image agree.
EDGE=mask (default): the edge control applies only inside mask.mp4 (runway + margin), the rest of the frame is free.
EDGE=light: edge_light.mp4 over the whole frame (full detail on the runway, only the strong layout edges elsewhere), no mask.
"""
import sys, os, json, random

IN, SPECS, CAPS, SEED = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]); CLIPS = sys.argv[5:]; EDGE = os.environ.get("EDGE", "mask")
NEG = ("video game, flight simulator, computer graphics, 3d render, cartoon, painting, text, watermark, distorted runway, blurry, low quality, "
       "aircraft nose, propeller, wing, engine, cockpit interior, dashboard, aircraft on the runway")
CAM = ("A realistic video recorded by a forward-looking camera fixed on an aircraft during its approach to an airport runway, "
       "seen from the air. No part of the aircraft is visible. ")
TIME = {"in daylight": "It is daytime", "at dusk with warm sunset light": "It is dusk, with warm and low sunset light", "at night with runway lights": "It is night"}
SKY = {"under a clear blue sky": " under a clear blue sky", "in haze or fog with low visibility": ", in haze and fog with low visibility",
       "under an overcast grey sky": " under an overcast grey sky", "with scattered clouds": " with scattered clouds", "": ""}


def prompt(c):
    night = "night" in c["time"]
    p = CAM + TIME[c["time"]] + SKY[c["sky"]] + ". "
    p += ("The distant runway is outlined by rows of bright edge lights, green threshold lights and the approach lighting system, while the "
          "surroundings are dark with scattered city lights. " if night else
          "A distant runway lies ahead, small near the horizon at first and growing larger as the aircraft gets closer: a grey asphalt "
          "strip with a white centerline and threshold markings, surrounded by grass, roads, trees, fields and buildings seen from above. ")
    if "fog" in c["sky"]: p += "The distant landscape fades into the mist. "
    if c["wx"] == "snow covered ground": p += "Snow covers the ground around the runway. "
    if c["wx"] == "rain drops on the windshield": p += "Raindrops are visible on the windshield. "
    return p + "The aircraft descends smoothly toward the runway. Real photograph, natural colors."


rng = random.Random(SEED); caps = [c for c in json.load(open(CAPS)) if os.path.exists(c["file"])]; os.makedirs(SPECS, exist_ok=True)
print(len(caps), "real style images available", flush=True)
for clip in CLIPS:
    c = rng.choice(caps); d = f"{IN}/{clip}"; name = f"{clip}_real{SEED}" + ("_light" if EDGE == "light" else "")
    spec = {"name": name, "prompt": prompt(c), "negative_prompt": NEG, "video_path": f"{d}/input.mp4", "image_context_path": c["file"], "guidance": 3,
            "seed": SEED, "edge": {"control_path": f"{d}/edge_light.mp4", "control_weight": 1.0} if EDGE == "light" else
                                  {"control_path": f"{d}/edge.mp4", "mask_path": f"{d}/mask.mp4", "control_weight": 1.0}}
    json.dump(spec, open(f"{SPECS}/{name}.json", "w"), indent=1)
    print(name, "|", c["source"], os.path.basename(c["file"]), "|", c["time"], "|", c["sky"], "|", c["wx"], flush=True)
