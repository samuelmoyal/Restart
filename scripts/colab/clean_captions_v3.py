import json, re, collections
R = "/content/real_all"; D = "/content/drive/MyDrive/capstone_sim2real"
rows = json.load(open(f"{R}/train_captions_v3.json")); n_fix = collections.Counter()
for r in rows:
    b = r["blip"]
    b2 = re.sub(r"\s*(with|and) (a|an|several|many|two|some)?\s*(large\s+)?(jet|air)?(plane|planes|airplane|airplanes|cars|car|truck|trucks|traffic)\b[^,]*", "", b).strip()
    b2 = re.sub(r"\s*(from|through) (a|an|the)?\s*(plane|airplane|rainy)?\s*window", "", b2).strip(); b2 = re.sub(r"\s+", " ", b2)
    if b2 != b: n_fix["invented objects / window removed"] += 1
    low = b2.lower()
    t = "at night with runway lights" if "night" in low else ("at dusk with warm sunset light" if re.search(r"dusk|sunset|sunrise", low) else r["time"])
    if t != r["time"]: n_fix["time taken from the caption"] += 1
    s = "" if "night" in t else r["sky"]
    wx = "rain drops on the windshield" if re.search(r"\brain|rainy|wet\b", low) else ("snow covered ground" if (re.search(r"snow|winter", low) or r["wx"].startswith("snow")) else "")
    if wx != r["wx"]: n_fix["weather tag changed"] += 1
    r.update(blip_clean=b2, time=t, sky=s, wx=wx, caption="rwyreal photo, " + b2 + ", " + ", ".join(x for x in (t, s, wx) if x))
json.dump(rows, open(f"{R}/train_captions_v3.json", "w")); json.dump(rows, open(f"{D}/real_all_captions_v3.json", "w"))
print(dict(n_fix))
for k in ("time", "sky", "wx"): print(k, dict(collections.Counter(r[k] for r in rows).most_common()))
print("distinct captions:", len({r["caption"] for r in rows}))
