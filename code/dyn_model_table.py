import json
rows = json.load(open("../results/r1/E_table.json"))
names = {"llama3.1-8b-fp16": "Llama 3.1 8B, 16-bit (submitted)", "llama3.2_3b__greedy": "Llama 3.2 3B", "llama3.1_8b__greedy": "Llama 3.1 8B",
         "qwen2.5_7b__greedy": "Qwen2.5 7B", "mistral_7b__greedy": "Mistral 7B v0.3", "gemma2_9b__greedy": "Gemma 2 9B", "qwen2.5_14b__greedy": "Qwen2.5 14B"}
by = {(r["t"], r["arm"]): r for r in rows}
L = []
for k, lab in names.items():
    g, n, b = by.get((k, "gate")), by.get((k, "nogate")), by.get((k, "gate+readback"))
    if not g: continue
    L.append(f"{lab} & {100*g['realised']:.1f} & {100*g['wrong']:.1f} & {g['thr']:.1f} & {g['over']:.1f} & {100*b['wrong']:.1f} & {b['over']:.1f} & {n['interf']:.0f} & {n['thr']:.1f} & {g['oracle']+n['oracle']+b['oracle']:.0f} \\\\")
open("../results/r1/dyn_model_rows.tex", "w").write("\n".join(L)); print("\n".join(L))
