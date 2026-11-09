# -*- coding: utf-8 -*-
"""
llm_compare.py - Multi-model, repeated-run evaluation of the interpretation layer.

Runs several locally hosted, quantised open models (served by Ollama on the workstation GPU)
on the 264-utterance bilingual benchmark under three conditions:

  greedy        three-shot prompt of llm_translate.py, temperature 0 (deterministic)
  constrained   same prompt, temperature 0, decoding constrained to the OCL JSON schema
  sampled       same prompt, temperature 0.7, top-p 0.9, five seeds (repeatability)
  asr           greedy, on the benchmark with synthetic transcription noise (asr_noise.py)

Every output is scored with the same gate and the same metrics as llm_translate.py. Results
are written incrementally (one JSON file per model/condition/repetition), so an interrupted
run resumes where it stopped. Only the Python standard library is used.

Usage (PowerShell, from this folder, with Ollama running):
    python llm_compare.py --models llama3.1:8b qwen2.5:7b mistral:7b gemma2:9b qwen2.5:14b
"""
import argparse, json, os, time, urllib.request
from constraints import gate, safety_violating
from llm_translate import SYSTEM, FEWSHOT, parse

HOST = "http://127.0.0.1:11434"
KEYS = ("type", "operator", "part_class", "agent", "zone", "v_max", "duration", "start_in")

OCL_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {"type": "string", "enum": ["handling", "speed_cap", "exclusion", "pause", "priority", "none"]},
        "operator": {"type": "string", "enum": ["Op_A", "Op_B"]},
        "part_class": {"type": "string", "enum": ["fragile", "housing", "standard"]},
        "agent": {"type": "string", "enum": ["operator", "robot"]},
        "zone": {"type": "string", "enum": ["kit_A", "kit_B", "handover"]},
        "v_max": {"type": "number"},
        "start_in": {"type": "number"},
        "duration": {"type": "number"},
        "override_ssm": {"type": "boolean"},
    },
    "required": ["type", "operator"],
    "additionalProperties": False,
}


def same_intent(a, b):
    return all(a.get(k) == b.get(k) for k in KEYS if (k in a or k in b))


def messages(row, field="utterance"):
    msgs = [{"role": "system", "content": SYSTEM}]
    for spk, u, g in FEWSHOT:
        msgs += [{"role": "user", "content": f"Speaker: {spk}\nRequest: {u}"},
                 {"role": "assistant", "content": json.dumps(g, ensure_ascii=False)}]
    msgs.append({"role": "user", "content": f"Speaker: {row['speaker']}\nRequest: {row[field]}"})
    return msgs


def chat(model, msgs, temperature, seed, schema=None):
    body = {"model": model, "messages": msgs, "stream": False, "keep_alive": "30m",
            "options": {"temperature": temperature, "top_p": 0.9 if temperature > 0 else 1.0,
                        "seed": seed, "num_predict": 96, "num_ctx": 2048}}
    if schema is not None:
        body["format"] = schema
    req = urllib.request.Request(HOST + "/api/chat", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.loads(r.read().decode("utf-8"))
    return out["message"]["content"], time.perf_counter() - t0, out


def run(model, rows, cond, temperature, seed, schema, path, field="utterance"):
    recs = []
    for row in rows:
        text, lat, raw = chat(model, messages(row, field), temperature, seed, schema)
        y = parse(text)
        d = gate(y, row["speaker"]) if y is not None else None
        recs.append(dict(id=row["id"], category=row["category"], lang=row["lang"], raw=text, parsed=y,
                         latency_s=lat, eval_tokens=raw.get("eval_count"),
                         gate_category=(d.category if d else "unparsed"),
                         valid_json=y is not None, exact=bool(y is not None and same_intent(y, row["gold"])),
                         accepted=bool(d and d.accepted),
                         unsafe_accepted=bool(d and d.accepted and safety_violating(y, row["speaker"]))))
    act = [x for x in recs if x["category"] not in ("adversarial", "no_op")]
    lat = sorted(x["latency_s"] for x in recs)
    summ = dict(model=model, condition=cond, temperature=temperature, seed=seed, n=len(recs),
                valid_json=sum(x["valid_json"] for x in recs) / len(recs),
                exact_actionable=sum(x["exact"] for x in act) / len(act),
                exact_en=sum(x["exact"] for x in act if x["lang"] == "en") / max(1, sum(x["lang"] == "en" for x in act)),
                exact_es=sum(x["exact"] for x in act if x["lang"] == "es") / max(1, sum(x["lang"] == "es" for x in act)),
                rejected_actionable=sum(not x["accepted"] for x in act) / len(act),
                wrong_but_admitted=sum(x["accepted"] and not x["exact"] for x in act) / len(act),
                adversarial_admitted=sum(x["accepted"] for x in recs if x["category"] == "adversarial"),
                noop_admitted=sum(x["accepted"] for x in recs if x["category"] == "no_op"),
                unsafe_executed=sum(x["unsafe_accepted"] for x in recs),
                latency_median_s=lat[len(lat) // 2], latency_p95_s=lat[int(0.95 * (len(lat) - 1))])
    json.dump(dict(summary=summ, records=recs), open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps(summ), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--corpus", default="../results/corpus_asr.json")
    ap.add_argument("--outdir", default="../results/llm_compare")
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    rows = json.load(open(a.corpus, encoding="utf-8"))
    for model in a.models:
        tag = model.replace(":", "_").replace("/", "_")
        plan = [("greedy", 0.0, 0, None, "utterance"), ("constrained", 0.0, 0, OCL_SCHEMA, "utterance"),
                ("asr", 0.0, 0, None, "utterance_asr")] + \
               [(f"sampled{k}", 0.7, 100 + k, None, "utterance") for k in range(a.reps)]
        chat(model, messages(rows[0]), 0.0, 0)          # warm-up: load weights before timing
        for cond, temp, seed, schema, field in plan:
            path = os.path.join(a.outdir, f"{tag}__{cond}.json")
            if os.path.exists(path):
                continue
            print(f"== {model} {cond}", flush=True)
            run(model, rows, cond, temp, seed, schema, path, field)


if __name__ == "__main__":
    main()
