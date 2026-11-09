# -*- coding: utf-8 -*-
"""
experiments_r1.py - Additional experiments for the revised manuscript (review round 1).

  A  scenarios     S0-S6 with the joint-consistent motion model (CellJ), the full-arm safety
                   oracle, idle-time decomposition and Jain's index; also with link-aware SSM
  B  surrogate     the original Cartesian surrogate (CellX) checked with the same oracle
  C  sensitivity   measurement latency and acceleration limits
  D  quotes        cost of a request as a function of its magnitude (cost-quotation tables)
  E  dynamic       requests arriving, expiring and conflicting during the shift, translated by a
                   faithful, a corrupted or a real language model, with and without the gate

Usage:  python experiments_r1.py [A B C D E]      (results -> ../results/r1/*.json)
"""
import copy, json, math, os, random, sys
from multiprocessing import Pool
import numpy as np
from cell_ext import CellX, CellJ
from constraints import Constraint as C, gate, safety_violating, V_COLLAB_MAX, CONSTRAINT_TYPES
from experiments import SCEN, corrupt, ERRORS, same_intent

OUT = "../results/r1/"
SEEDS = list(range(30))
KEYS = ["throughput", "kits_A", "kits_B", "jain", "robot_idle", "human_idle", "wait_A", "wait_B",
        "stop_frac", "reduced_frac", "min_sep_moving", "min_sep_link_moving", "oracle_viol",
        "max_point_speed", "max_kappa", "assemble_A", "assemble_B", "reach_tray_A", "reach_tray_B",
        "fetch_A", "fetch_B", "absent_A", "absent_B", "rearrange_A", "rearrange_B",
        "believed_excl_exposure", "believed_cap_overspeed", "interference"]


def _run(args):
    cls, cons, seed, kw = args
    m = {"CellX": CellX, "CellJ": CellJ}[cls](copy.deepcopy(cons), seed=seed, **kw).run()
    return {k: float(m.get(k, np.nan)) for k in KEYS}


def batch(pool, cls, cons, kw, seeds=SEEDS):
    runs = pool.map(_run, [(cls, cons, s, kw) for s in seeds])
    return {k: [r[k] for r in runs] for k in KEYS}


def A(pool):
    out = {}
    for law in ("tcp", "link"):
        for name, cons in SCEN.items():
            out[f"{law}|{name}"] = batch(pool, "CellJ", cons, dict(link_aware=(law == "link")))
            r = out[f"{law}|{name}"]
            print(law, name, "thr %.2f  stop %.4f  oracle %.2f  vmax %.0f" % (
                np.mean(r["throughput"]), np.mean(r["stop_frac"]), np.mean(r["oracle_viol"]), max(r["max_point_speed"])), flush=True)
    return out


def B(pool):
    out = {}
    for name, cons in SCEN.items():
        out[name] = batch(pool, "CellX", cons, {})
        print("surrogate", name, "oracle %.2f s  kappa %.2f" % (np.mean(out[name]["oracle_viol"]), max(out[name]["max_kappa"])), flush=True)
    return out


def Csens(pool):
    out = {}
    for scen in ("S0 Baseline", "S1 Fragile to operator A"):
        cons = SCEN[scen]
        for lat in (0.0, 0.1, 0.2):
            for tr_ok in (False, True):
                if lat == 0.0 and tr_ok: continue
                kw = dict(latency=lat, Tr_model=(0.1 + lat) if tr_ok else 0.1, accel=1500.0)
                key = f"{scen}|lat={lat}|Tr={'0.1+lat' if tr_ok else '0.1'}|accel"
                out[key] = batch(pool, "CellJ", cons, kw)
                r = out[key]
                print(key, "thr %.2f stop %.4f oracle %.3f minsep %.0f" % (np.mean(r["throughput"]), np.mean(r["stop_frac"]),
                      np.mean(r["oracle_viol"]), min(r["min_sep_link_moving"])), flush=True)
    return out


def D(pool):
    out = {}
    kw = dict(oracle=False)
    for v in (50, 100, 150, 200, 250, 300, 400, 500):
        out[f"cap_B={v}"] = batch(pool, "CellJ", [C("speed_cap", "Op_B", zone="kit_B", v_max=v)], kw)
    for dur in (60, 120, 180, 240, 300):
        out[f"excl_A={dur}"] = batch(pool, "CellJ", [C("exclusion", "Op_A", zone="kit_A", start_in=900, duration=dur)], kw)
    for dur in (300, 600, 900, 1200):
        out[f"pause_A={dur}"] = batch(pool, "CellJ", [C("pause", "Op_A", start_in=900, duration=dur)], kw)
    out["base"] = batch(pool, "CellJ", [], kw)
    for k, r in out.items():
        print(k, "A %.2f B %.2f" % (np.mean(r["kits_A"]), np.mean(r["kits_B"])), flush=True)
    return out


# --------------------------------------------------------------------- dynamic requests (E)
LAMBDA_PER_OP = 4.0          # requests per operator per hour
STANDING = ("handling", "speed_cap", "priority")


def to_constraint(y, speaker, t_a, t_off_standing):
    """What a constraint store without a gate would hold: the candidate's own fields, with no
    check of authority, safety or range. Fields the scheduler cannot read are ignored."""
    if not isinstance(y, dict) or y.get("type") not in CONSTRAINT_TYPES or y.get("type") == "none":
        return None
    op = y.get("operator", speaker)
    if op not in ("Op_A", "Op_B"): return None
    c = C(y["type"], op, raw=y); c.speaker = speaker; c.t_on = t_a
    if c.ctype == "handling":
        c.part_class, c.agent = y.get("part_class"), y.get("agent")
        if c.part_class not in ("fragile", "housing", "standard") or c.agent not in ("operator", "robot"): return None
        c.t_off = t_off_standing
    elif c.ctype == "speed_cap":
        c.zone, c.v_max = y.get("zone"), y.get("v_max")
        if c.zone not in ("kit_A", "kit_B", "handover") or not isinstance(c.v_max, (int, float)) or c.v_max <= 0: return None
        c.t_off = t_off_standing
    elif c.ctype == "priority":
        c.t_off = t_off_standing
    else:
        dur = y.get("duration"); st = y.get("start_in", 0) or 0
        if not isinstance(dur, (int, float)) or dur <= 0 or not isinstance(st, (int, float)): return None
        c.zone = y.get("zone") if c.ctype == "exclusion" else None
        if c.ctype == "exclusion" and c.zone not in ("kit_A", "kit_B", "handover"): return None
        c.start_in = t_a + float(st); c.duration = float(dur); c.t_off = c.start_in + c.duration
    return c


def make_requests(rows, seed):
    rng = random.Random(10_000 + seed); reqs = []
    for spk in ("Op_A", "Op_B"):
        pool_rows = [r for r in rows if r["speaker"] == spk]
        t = rng.expovariate(LAMBDA_PER_OP / 3600.0)
        while t < 3300.0:
            reqs.append(dict(row=rng.choice(pool_rows), t=t, life=rng.uniform(300.0, 1200.0)))
            t += rng.expovariate(LAMBDA_PER_OP / 3600.0)
    return reqs


def _dyn(args):
    rows, seed, translator, arm = args
    reqs = make_requests(rows, seed)
    trng = random.Random(77_000 + seed)
    store, intents, outcome = [], [], dict(n=0, actionable=0, realised=0, rejected=0, wrong_admitted=0,
                                             caught_by_readback=0, unsafe_admitted=0)
    for q in reqs:
        row, t_a = q["row"], q["t"]; spk = row["speaker"]; g = row["gold"]
        t_off = min(3600.0, t_a + q["life"])
        actionable = row["category"] not in ("adversarial", "no_op")
        if translator["kind"] == "gold":
            y = copy.deepcopy(g)
        elif translator["kind"] == "eps":
            y = corrupt(g, trng.choice(ERRORS), trng) if trng.random() < translator["eps"] else copy.deepcopy(g)
        else:
            y = copy.deepcopy(translator["table"].get(row["id"]))
        outcome["n"] += 1; outcome["actionable"] += int(actionable)
        if arm == "nogate":
            c = to_constraint(y, spk, t_a, t_off); admitted = c is not None
            outcome["unsafe_admitted"] += int(admitted and safety_violating(y, spk))
        else:
            d = gate(y, spk) if y is not None else None
            admitted = bool(d and d.accepted); c = None
            if admitted:
                if arm == "gate+readback" and actionable and not same_intent(y, g) and trng.random() < 0.9:
                    admitted = False; outcome["caught_by_readback"] += 1
                else:
                    c = to_constraint(y, spk, t_a, t_off)
                    outcome["unsafe_admitted"] += int(safety_violating(y, spk))
        if c is not None: store.append(c)
        if actionable:
            ok = admitted and same_intent(y, g)
            outcome["realised"] += int(ok); outcome["rejected"] += int(not admitted)
            outcome["wrong_admitted"] += int(admitted and not ok)
            # physical behaviour: the operator acts on what they asked for once the terminal
            # reports the request as accepted (pause is taken regardless of the answer)
            gi = to_constraint(g, spk, t_a, t_off)
            if gi is not None and (admitted or gi.ctype == "pause"):
                intents.append(gi)
    m = CellJ(store, seed=seed, dynamic=True, intents=intents, link_aware=True).run()
    r = {k: float(m.get(k, np.nan)) for k in KEYS}; r.update(outcome)
    return r


def E(pool, translators, arms=("gate", "nogate", "gate+readback")):
    rows = json.load(open("../results/corpus.json", encoding="utf-8"))
    out = {}
    for tname, tr in translators.items():
        for arm in arms:
            runs = pool.map(_dyn, [(rows, s, tr, arm) for s in SEEDS])
            out[f"{tname}|{arm}"] = {k: [r[k] for r in runs] for k in runs[0]}
            r = out[f"{tname}|{arm}"]
            print(tname, arm, "thr %.2f  realised %.3f  wrong %.3f  unsafe %.2f  oracle %.2f  exposure %.1f  interf %.1f" % (
                np.mean(r["throughput"]), np.sum(r["realised"]) / np.sum(r["actionable"]),
                np.sum(r["wrong_admitted"]) / np.sum(r["actionable"]), np.mean(r["unsafe_admitted"]),
                np.mean(r["oracle_viol"]), np.mean(r["believed_excl_exposure"]), np.mean(r["interference"])), flush=True)
    return out


def llm_table(path):
    recs = json.load(open(path, encoding="utf-8"))["records"]
    return {x["id"]: x["parsed"] for x in recs}


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    which = sys.argv[1:] or list("ABCDE")
    with Pool(int(os.environ.get("NPROC", os.cpu_count() or 1))) as pool:
        if "A" in which: json.dump(A(pool), open(OUT + "A_scenarios.json", "w"))
        if "B" in which: json.dump(B(pool), open(OUT + "B_surrogate.json", "w"))
        if "C" in which: json.dump(Csens(pool), open(OUT + "C_sensitivity.json", "w"))
        if "D" in which: json.dump(D(pool), open(OUT + "D_quotes.json", "w"))
        if "E" in which:
            tr = {"faithful": dict(kind="gold"), "eps0.2": dict(kind="eps", eps=0.2), "eps0.5": dict(kind="eps", eps=0.5),
                  "llama3.1-8b-fp16": dict(kind="table", table=llm_table("../results/llm_eval.json"))}
            d = "../results/llm_compare/"
            if os.path.isdir(d):
                for f in sorted(os.listdir(d)):
                    if f.endswith("__greedy.json"):
                        tr[f[:-5]] = dict(kind="table", table=llm_table(d + f))
            for name, t in tr.items():
                path = OUT + f"E_{name}.json"
                if os.path.exists(path): continue
                arms = ("gate", "nogate", "gate+readback")
                json.dump(E(pool, {name: t}, arms), open(path, "w"))
