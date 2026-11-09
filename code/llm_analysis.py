# -*- coding: utf-8 -*-
"""llm_analysis.py - Summary tables of the multi-model study (results/llm_compare -> results/r1/G_llm.json)."""
import json, glob, os, sys
import numpy as np
from collections import Counter

D = "../results/llm_compare/"
USED = {"handling": ("type", "operator", "part_class", "agent"), "speed_cap": ("type", "operator", "zone", "v_max"),
        "exclusion": ("type", "operator", "zone", "start_in", "duration"), "pause": ("type", "operator", "start_in", "duration"),
        "priority": ("type", "operator"), "none": ("type",)}
corpus = {r["id"]: r for r in json.load(open("../results/corpus.json", encoding="utf-8"))}


def effective(y, g):
    """Agreement on the fields the scheduler actually reads (extra, unused fields ignored)."""
    if not isinstance(y, dict): return False
    if y.get("type") != g.get("type"): return False
    for k in USED.get(g["type"], ()):
        a, b = y.get(k, 0 if k == "start_in" else None), g.get(k, 0 if k == "start_in" else None)
        if a != b: return False
    return True


def outcome(r):
    if r["category"] in ("adversarial", "no_op"): return None
    return "exact" if r["exact"] else ("wrong" if r["accepted"] else "rejected")


def fleiss(table):
    """table: list of dicts category -> count (same number of raters per item)."""
    cats = sorted({c for row in table for c in row}); n = sum(table[0].values()); N = len(table)
    M = np.array([[row.get(c, 0) for c in cats] for row in table], float)
    p = M.sum(0) / (N * n); P = ((M * M).sum(1) - n) / (n * (n - 1))
    Pbar, Pe = P.mean(), (p * p).sum()
    return float((Pbar - Pe) / (1 - Pe)) if Pe < 1 else 1.0


def summarise(recs):
    act = [r for r in recs if r["category"] not in ("adversarial", "no_op")]
    lat = sorted(r["latency_s"] for r in recs)
    return dict(valid=np.mean([r["valid_json"] for r in recs]),
                exact=np.mean([r["exact"] for r in act]),
                exact_en=np.mean([r["exact"] for r in act if r["lang"] == "en"]),
                exact_es=np.mean([r["exact"] for r in act if r["lang"] == "es"]),
                effective=np.mean([effective(r["parsed"], corpus[r["id"]]["gold"]) and r["accepted"] for r in act]),
                rejected=np.mean([not r["accepted"] for r in act]),
                wrong=np.mean([r["accepted"] and not r["exact"] for r in act]),
                adv_admitted=int(sum(r["accepted"] for r in recs if r["category"] == "adversarial")),
                noop_admitted=int(sum(r["accepted"] for r in recs if r["category"] == "no_op")),
                violations=int(sum(r["unsafe_accepted"] for r in recs)),
                rej_reasons=dict(Counter(r.get("gate_category", "?") for r in act if not r["accepted"])),
                lat_med=lat[len(lat) // 2], lat_p95=lat[int(0.95 * (len(lat) - 1))])


def main():
    models = sorted({os.path.basename(f).split("__")[0] for f in glob.glob(D + "*.json")})
    out = {}
    fp16 = json.load(open("../results/llm_eval.json", encoding="utf-8"))["records"]
    out["llama3.1-8b-fp16 (submitted)"] = dict(greedy=summarise(fp16))
    for m in models:
        res = {}
        for cond in ("greedy", "constrained", "asr"):
            f = D + f"{m}__{cond}.json"
            if os.path.exists(f): res[cond] = summarise(json.load(open(f, encoding="utf-8"))["records"])
        samp = [json.load(open(f, encoding="utf-8"))["records"] for f in sorted(glob.glob(D + f"{m}__sampled*.json"))]
        if samp:
            S = [summarise(s) for s in samp]
            res["sampled"] = {k: (float(np.mean([s[k] for s in S])), float(np.std([s[k] for s in S], ddof=1)) if len(S) > 1 else 0.0)
                              for k in ("exact", "effective", "rejected", "wrong", "lat_med")}
            res["sampled"]["violations"] = int(sum(s["violations"] for s in S)); res["sampled"]["runs"] = len(S)
            ids = [r["id"] for r in samp[0]]
            ident = np.mean([len({json.dumps(s[i]["parsed"], sort_keys=True) for s in samp}) == 1 for i in range(len(ids))])
            act = [i for i, r in enumerate(samp[0]) if r["category"] not in ("adversarial", "no_op")]
            allc = np.mean([all(s[i]["exact"] for s in samp) for i in act]); none = np.mean([not any(s[i]["exact"] for s in samp) for i in act])
            maj = np.mean([sum(s[i]["exact"] for s in samp) > len(samp) / 2 for i in act])
            kappa = fleiss([Counter(outcome(s[i]) for s in samp) for i in act]) if len(samp) > 1 else None
            res["repeat"] = dict(identical_all=float(ident), always_correct=float(allc), never_correct=float(none),
                                 majority_correct=float(maj), fleiss_kappa=kappa)
        out[m] = res
    json.dump(out, open("../results/r1/G_llm.json", "w"), indent=1, default=float)
    for m, r in out.items():
        g = r.get("greedy", {}); print(m, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in g.items()})
        for c in ("constrained", "asr"):
            if c in r: print("   ", c, "exact %.3f eff %.3f rej %.3f wrong %.3f viol %d" % (r[c]["exact"], r[c]["effective"], r[c]["rejected"], r[c]["wrong"], r[c]["violations"]))
        if "sampled" in r: print("    sampled", r["sampled"], r["repeat"])


if __name__ == "__main__":
    main()
