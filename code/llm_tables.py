import json
G = json.load(open("../results/r1/G_llm.json"))
order = [("llama3.1-8b-fp16 (submitted)", "Llama 3.1 8B, 16-bit (submitted)"), ("llama3.2_3b", "Llama 3.2 3B"), ("llama3.1_8b", "Llama 3.1 8B"),
         ("qwen2.5_7b", "Qwen2.5 7B"), ("mistral_7b", "Mistral 7B v0.3"), ("gemma2_9b", "Gemma 2 9B"), ("qwen2.5_14b", "Qwen2.5 14B")]
order = [(k, l) for k, l in order if k in G]
P = lambda v: f"{100*v:.1f}"
L = []
for k, l in order:
    g = G[k]["greedy"]; c = G[k].get("constrained"); a = G[k].get("asr")
    lat = f"{g['lat_med']:.1f} / {g['lat_p95']:.1f}" if g['lat_med'] > 5 else f"{g['lat_med']:.2f} / {g['lat_p95']:.2f}"
    L.append(f"{l} & {P(g['exact'])} & {P(g['exact_en'])} / {P(g['exact_es'])} & {P(g['effective'])} & {P(g['rejected'])} & {P(g['wrong'])} & {g['adv_admitted']} & {g['violations']} & {P(c['exact']) if c else '---'} & {P(a['exact']) if a else '---'} & {lat} \\\\")
t1 = "\n".join(L)
R = []
for k, l in order:
    if "repeat" not in G[k]: continue
    s = G[k]["sampled"]; r = G[k]["repeat"]
    R.append(f"{l} & {P(G[k]['greedy']['exact'])} & {P(s['exact'][0])} $\\pm$ {P(s['exact'][1])} & {P(r['identical_all'])} & {P(r['always_correct'])} & {P(r['never_correct'])} & {P(r['majority_correct'])} & {r['fleiss_kappa']:.2f} & {s['violations']} \\\\")
t2 = "\n".join(R)
open("../results/r1/llm_rows1.tex", "w").write(t1); open("../results/r1/llm_rows2.tex", "w").write(t2)
print(t1); print(t2)
