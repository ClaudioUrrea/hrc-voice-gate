import json, glob, numpy as np, os
rows = []
order = ["faithful", "eps0.2", "eps0.5", "llama3.1-8b-fp16"] + sorted(os.path.basename(f)[2:-5] for f in glob.glob("../results/r1/E_*__greedy.json"))
for t in order:
    f = f"../results/r1/E_{t}.json"
    if not os.path.exists(f): continue
    E = json.load(open(f))
    for arm in ("gate", "nogate", "gate+readback"):
        r = E.get(f"{t}|{arm}")
        if not r: continue
        act = np.sum(r["actionable"])
        rows.append(dict(t=t, arm=arm, thr=np.mean(r["throughput"]), thr_sd=np.std(r["throughput"], ddof=1),
            realised=np.sum(r["realised"]) / act, wrong=np.sum(r["wrong_admitted"]) / act, rejected=np.sum(r["rejected"]) / act,
            viol=np.mean(r["unsafe_admitted"]), interf=np.mean(r["interference"]), expo=np.mean(r["believed_excl_exposure"]),
            over=np.mean(r["believed_cap_overspeed"]), oracle=np.sum(r["oracle_viol"]), stop=100 * np.mean(r["stop_frac"]),
            n=int(np.sum(r["n"])), nact=int(act), minsep=min(r["min_sep_link_moving"])))
for x in rows: print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in x.items()})
json.dump(rows, open("../results/r1/E_table.json", "w"), indent=1, default=float)
