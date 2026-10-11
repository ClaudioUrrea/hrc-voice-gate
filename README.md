# hrc-voice-gate

**Giving Operators a Voice Without Giving Up Safety: A Safety-Gated Sociotechnical Architecture for Natural-Language Preferences in Human–Robot Collaborative Kitting**

Claudio Urrea · Electrical Engineering Department, University of Santiago of Chile (USACH)

*Submitted to MDPI Systems, October 2026.*

---

## Abstract

Operators of collaborative cells have no run-time channel to ask the robot to slow down, prioritise fragile parts, or reorder its work. Language models can interpret such requests, but one inside a safety-critical loop has unbounded failure modes. This paper separates interpretation from authority: a local model renders each utterance as a candidate in a small typed language, and a deterministic gate admits only candidates that are well formed, raised at the speaker's own terminal, confined to their zones, and tightening rather than relaxing the robot's behaviour. Speed-and-separation monitoring runs below the scheduler, beyond any admitted request. In a two-operator kitting cell with a Doosan M1013, cost tracks poorly with how modest a request sounds: routing fragile parts costs 0.9% of throughput but raises protective stops sevenfold; a speed cap costs 7.3% and cuts a colleague's output by 6.5%. On a 264-utterance bilingual benchmark the gate admitted all 236 legitimate requests and rejected all 28 adversarial or non-actionable ones. Under 50% injected corruption, and with a model conveying only half the requests exactly, no unsafe command reached the robot. The residual risk of a language channel is a misunderstood preference; operator voice in a shared cell belongs to scheduling policy, not interface design.

---

## Repository contents

| Path | Description |
|------|-------------|
| `code/constraints.py` | Operator Constraint Language (OCL) and the deterministic admission gate |
| `code/cell_sim.py` | Time-stepped cell simulation with ISO/TS 15066 speed-and-separation monitoring |
| `code/corpus.py` | Builds the 264-utterance bilingual (EN/ES) benchmark, including 16 adversarial requests |
| `code/experiments.py` | Scenarios S0–S6, gate evaluation and the error-injection study |
| `code/llm_translate.py` | Evaluates the local Llama 3.1 8B Instruct translator on the benchmark |
| `code/figures.py` | Generates Figures 2–6 and the Table 3 rows |
| `robodk/build_cell_systems.py` | Builds the two-operator kitting cell in RoboDK (Doosan M1013), solves 26 targets and checks all 31 motion segments for collisions |
| `robodk/cell_layout.json` | Cell configuration, joint solutions and collision report (0 collisions) |
| `results/corpus.json` | The benchmark with gold annotations |
| `results/results.json` | Scenario outcomes, SSM trace, gate decisions and injection study |
| `results/llm_eval.json` | Per-utterance translator output and summary |
| `results/table_scenarios.json` | Table 3 rows as emitted by `figures.py` |
| `figures/` | Figures 1–6 (PDF and PNG); `figures/raw/` holds the RoboDK renders behind Figures 1 and 2 |

---

## Reproducing the results

### Requirements

```bash
pip install -r requirements.txt
```

`numpy`, `scipy`, `matplotlib` and `pillow` are enough for the simulation, the gate and the
figures. `robodk` is needed only to rebuild the cell, and `torch`, `transformers` and
`accelerate` only to re-run the local translator.

### Running

> **The scripts resolve `../results/` and `../figures/` relative to their own directory, so
> run them from inside `code/`.**

```bash
cd code

python corpus.py        # -> ../results/corpus.json      (264 utterances)
python experiments.py   # -> ../results/results.json     (scenarios, gate, injection)
python figures.py       # -> ../figures/Figure2..6.{pdf,png}, ../results/table_scenarios.json
```

`experiments.py` needs `corpus.json`, and `figures.py` needs `results.json` and the renders in
`figures/raw/`, so keep that order. All three are deterministic and need no GPU; the committed
JSON files under `results/` are the outputs of exactly these commands.

Re-running the local translator additionally requires the Llama 3.1 8B Instruct weights and
takes far longer (median 80.9 s per utterance on the evaluation hardware):

```bash
python llm_translate.py --model meta-llama/Llama-3.1-8B-Instruct \
                        --corpus ../results/corpus.json \
                        --out ../results/llm_eval.json
```

Rebuilding the RoboDK cell requires a RoboDK licence and is run from inside RoboDK:
`robodk/build_cell_systems.py` writes `cell_layout.json`.

### Where each result lives

| Paper item | File | Key |
|------------|------|-----|
| Table 3 — scenario outcomes | `results/results.json` | `scenarios` |
| Table 4 — gate decisions, faithful translator | `results/results.json` | `gate_faithful` |
| Table 5 — local Llama 3.1 translator | `results/llm_eval.json` | — |
| Figure 4 — SSM speed law and trace | `results/results.json` | `trace` |
| Figure 6 — robustness to injected error | `results/results.json` | `gate_eval` |

---

## Data availability

The cell, benchmark, constraint grammar and all experimental data are archived at:

> **Figshare** · [https://doi.org/10.6084/m9.figshare.33961735](https://doi.org/10.6084/m9.figshare.33961735)

All data were produced in simulation; no human participant contributed data.

---

## Keywords

human–robot collaboration · sociotechnical systems · large language models · Industry 5.0 · worker autonomy · speed and separation monitoring · ISO/TS 15066 · runtime assurance · human-centric manufacturing · natural-language interfaces

---

## Citation

```bibtex
@article{urrea2026systems,
  author  = {Urrea, Claudio},
  title   = {Giving Operators a Voice Without Giving Up Safety: A Safety-Gated
             Sociotechnical Architecture for Natural-Language Preferences in
             Human--Robot Collaborative Kitting},
  journal = {Systems},
  year    = {2026},
  note    = {Submitted}
}
```

---

## License

Code: [MIT License](LICENSE) · Data, figures and cell record: [CC BY 4.0](LICENSE-DATA)

`LICENSE-DATA` states which files fall under each licence and gives the attribution string to use.

---

*Author contact: claudio.urrea@usach.cl*
