# -*- coding: utf-8 -*-
"""inj_breakdown.py - Error-injection study with violations broken down by class (revision R1).

Classes of a candidate that would violate the specification if it reached the store:
  protective  refers to a protective function (override_ssm / disable_safety)
  overspeed   speed cap above the cruise speed (no physical effect: Eq. (1) clamps it)
  authority   names the other operator or a zone outside the speaker's authority
  range       duration, start or cap outside the organisational limits
Same seeds and translator as experiments.gate_eval, so totals match the original study.
"""
import copy, json, random
import numpy as np
from constraints import gate, ZONES, V_COLLAB_MAX, V_CAP_MIN, EXCL_MAX_S, PAUSE_MAX_S, PAUSE_LEAD_MAX_S
from experiments import corrupt, ERRORS


def classes(y, spk):
    out = set()
    if not isinstance(y, dict) or y.get("type") in (None, "none"): 
        if isinstance(y, dict) and (y.get("override_ssm") or y.get("disable_safety")): out.add("protective")
        return out
    if y.get("override_ssm") or y.get("disable_safety"): out.add("protective")
    if y.get("operator", spk) != spk: out.add("authority")
    t = y.get("type")
    if t in ("speed_cap", "exclusion") and y.get("zone") not in ZONES.get(spk, ()): out.add("authority")
    if t == "speed_cap" and isinstance(y.get("v_max"), (int, float)):
        if y["v_max"] > V_COLLAB_MAX: out.add("overspeed")
        elif y["v_max"] < V_CAP_MIN: out.add("range")
    if t == "exclusion" and isinstance(y.get("duration"), (int, float)) and not (0 < y["duration"] <= EXCL_MAX_S): out.add("range")
    if t == "pause" and isinstance(y.get("duration"), (int, float)) and not (0 < y["duration"] <= PAUSE_MAX_S): out.add("range")
    return out


def run(rows, eps_list=(0, 0.05, 0.1, 0.2, 0.3, 0.5), reps=50):
    res = {}
    for eps in eps_list:
        acc = {k: [] for k in ("protective", "overspeed", "authority", "range", "any_nogate", "any_gate")}
        for r_ in range(reps):
            rng = random.Random(1000 * r_ + int(eps * 100))
            cnt = dict.fromkeys(acc, 0)
            for row in rows:
                g, spk = row["gold"], row["speaker"]
                y = corrupt(g, rng.choice(ERRORS), rng) if rng.random() < eps else copy.deepcopy(g)
                gate(y, spk)
                cl = classes(y, spk)
                for c in cl: cnt[c] += 1
                cnt["any_nogate"] += int(bool(cl))
                d = gate(y, spk); cnt["any_gate"] += int(d.accepted and bool(cl))
            for k in acc: acc[k].append(cnt[k])
        res[str(eps)] = {k: float(np.mean(v)) for k, v in acc.items()}
        print(eps, res[str(eps)])
    return res


if __name__ == "__main__":
    rows = json.load(open("../results/corpus.json", encoding="utf-8"))
    json.dump(run(rows), open("../results/r1/F_injection_classes.json", "w"), indent=1)
