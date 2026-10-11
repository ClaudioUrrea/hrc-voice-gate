# -*- coding: utf-8 -*-
"""experiments.py - Runs every experiment reported in the manuscript and stores raw results."""
import json, copy, random, statistics as st
import numpy as np
from cell_sim import Cell
from constraints import Constraint as C, gate, safety_violating, V_COLLAB_MAX

OUT = "../results/"
SEEDS = list(range(30))

SCEN = {
    "S0 Baseline": [],
    "S1 Fragile to operator A": [C("handling", "Op_A", part_class="fragile", agent="operator")],
    "S2 Speed cap B (150 mm/s)": [C("speed_cap", "Op_B", zone="kit_B", v_max=150)],
    "S3 Exclusion A (4 min)": [C("exclusion", "Op_A", zone="kit_A", start_in=900, duration=240)],
    "S4 Pause A (10 min)": [C("pause", "Op_A", start_in=900, duration=600)],
    "S5 Priority B": [C("priority", "Op_B")],
    "S6 Combined (S1+S2+S4)": [C("handling", "Op_A", part_class="fragile", agent="operator"),
                               C("speed_cap", "Op_B", zone="kit_B", v_max=150),
                               C("pause", "Op_A", start_in=900, duration=600)],
}
KEYS = ["throughput", "kits_A", "kits_B", "jain", "robot_idle", "human_idle", "wait_A", "wait_B",
        "stop_frac", "reduced_frac", "min_sep_moving", "fragile_to_operator", "excl_viol"]


def scenarios():
    res = {}
    for name, cons in SCEN.items():
        runs = [Cell(copy.deepcopy(cons), seed=s).run() for s in SEEDS]
        res[name] = {k: [float(r.get(k, np.nan)) for r in runs] for k in KEYS}
        print(name, "thr %.2f±%.2f" % (np.mean(res[name]["throughput"]), np.std(res[name]["throughput"], ddof=1)))
    return res


def trace():
    c = Cell([C("speed_cap", "Op_B", zone="kit_B", v_max=150)], seed=3, horizon=600)
    c.run(); return c.log


# ---------------------------------------------------------------- translation errors
ERRORS = ["wrong_type", "other_zone", "speed_inflate", "missing_field", "override_flag", "wrong_part", "duration_x10"]


def corrupt(g, kind, rng):
    o = copy.deepcopy(g); spk = g["operator"]; other = "Op_B" if spk == "Op_A" else "Op_A"
    if kind == "wrong_type":
        o = dict(type=rng.choice([t for t in ("priority", "pause", "none") if t != g["type"]]), operator=spk)
        if o["type"] == "pause": o.update(start_in=0, duration=300)
    elif kind == "other_zone":
        if g["type"] in ("speed_cap", "exclusion"): o["zone"] = "kit_B" if spk == "Op_A" else "kit_A"
        else: o["operator"] = other
    elif kind == "speed_inflate":
        if g["type"] == "speed_cap": o["v_max"] = g["v_max"] * 4
        else: o = dict(type="speed_cap", operator=spk, zone="kit_A" if spk == "Op_A" else "kit_B", v_max=rng.choice([600, 800, 1000]))
    elif kind == "missing_field":
        for f in ("v_max", "duration", "part_class", "zone", "agent"):
            if f in o: del o[f]; break
        else: o["type"] = "handling"
    elif kind == "override_flag":
        o["override_ssm"] = True
    elif kind == "wrong_part":
        if g["type"] == "handling": o["part_class"] = rng.choice([p for p in ("fragile", "housing", "standard") if p != g["part_class"]])
        else: o = dict(type="handling", operator=spk, part_class="standard", agent="operator")
    elif kind == "duration_x10":
        if "duration" in o: o["duration"] = g["duration"] * 10
        else: o["type"] = "exclusion"; o["zone"] = "kit_A" if spk == "Op_A" else "kit_B"; o["duration"] = 3000
    return o


def same_intent(a, b):
    keys = ("type", "operator", "part_class", "agent", "zone", "v_max", "duration", "start_in")
    return all(a.get(k) == b.get(k) for k in keys if (k in a or k in b))


def gate_eval(rows, eps_list=(0, 0.05, 0.1, 0.2, 0.3, 0.5), reps=50):
    out = {}
    for eps in eps_list:
        agg = dict(unsafe_exec_gate=[], unsafe_exec_nogate=[], fidelity=[], false_reject=[], semantic_pass=[])
        for r_ in range(reps):
            rng = random.Random(1000 * r_ + int(eps * 100))
            ue_g = ue_n = ok = fr = sp = nb = 0
            for row in rows:
                g, spk = row["gold"], row["speaker"]
                y = corrupt(g, rng.choice(ERRORS), rng) if rng.random() < eps else copy.deepcopy(g)
                dec = gate(y, spk)
                unsafe = safety_violating(y, spk)
                ue_g += int(dec.accepted and unsafe)
                ue_n += int(unsafe and y.get("type") not in (None, "none") or bool(y.get("disable_safety") or y.get("override_ssm")))
                if row["category"] not in ("adversarial", "no_op"):
                    nb += 1
                    ok += int(dec.accepted and same_intent(y, g))
                    fr += int((not dec.accepted) and same_intent(y, g))
                    sp += int(dec.accepted and not same_intent(y, g))
            agg["unsafe_exec_gate"].append(ue_g); agg["unsafe_exec_nogate"].append(ue_n)
            agg["fidelity"].append(ok / nb); agg["false_reject"].append(fr / nb); agg["semantic_pass"].append(sp / nb)
        out[str(eps)] = agg
        print("eps", eps, {k: round(float(np.mean(v)), 4) for k, v in agg.items()})
    return out


def gate_faithful(rows):
    """Faithful translator (gold): what does the gate do with each category?"""
    from collections import defaultdict
    tab = defaultdict(lambda: dict(n=0, accepted=0, rejected=0, reasons={}))
    for row in rows:
        d = gate(row["gold"], row["speaker"]); t = tab[row["category"]]
        t["n"] += 1; t["accepted" if d.accepted else "rejected"] += 1
        if not d.accepted: t["reasons"][d.category] = t["reasons"].get(d.category, 0) + 1
    return dict(tab)


if __name__ == "__main__":
    rows = json.load(open(OUT + "corpus.json", encoding="utf-8"))
    res = dict(scenarios=scenarios(), trace=trace(), gate_faithful=gate_faithful(rows), gate_eval=gate_eval(rows))
    json.dump(res, open(OUT + "results.json", "w"), indent=1)
    print(json.dumps(res["gate_faithful"], indent=1))
