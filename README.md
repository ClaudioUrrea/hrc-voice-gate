# hrc-voice-gate

**Giving Operators a Voice Without Giving Up Safety: A Safety-Gated Sociotechnical Architecture for Natural-Language Preferences in Human–Robot Collaborative Kitting**

Claudio Urrea · Electrical Engineering Department, University of Santiago of Chile (USACH)

*Submitted to MDPI Systems, October 2026.*

---

## Abstract

Operators of collaborative cells lack a run-time channel to tell the robot to slow down, prioritise fragile parts, or reorder its work. Language models can interpret such requests, but placing one inside a safety-critical loop introduces a component whose failure modes cannot be bounded. This paper separates interpretation from authority: a local language model renders each utterance as a candidate in a small typed language, and a deterministic gate admits only candidates that are well formed, originate at the speaker's terminal, concern the speaker's assigned zones, and tighten rather than relax the robot's behaviour. Speed-and-separation monitoring runs below the scheduler, beyond any admitted request. Experiments in a two-operator kitting cell with a Doosan M1013 show that preference cost is poorly predicted by how modest a request sounds: routing fragile parts costs 0.9% of throughput but raises protective stops sevenfold; a speed cap costs 7.3% and reduces a colleague's output by 6.5%. On a 264-utterance bilingual benchmark the gate admitted all 236 legitimate requests and rejected all 28 adversarial ones. Under 50% injected corruption, and under a model conveying only half the requests exactly, no unsafe command reached the robot. The residual risk of a language channel is a misunderstood preference; operator voice in a shared cell belongs to scheduling policy, not interface design.

---

## Repository Contents

| Path | Description |
|------|-------------|
| `build_cell_systems.py` | Builds the RoboDK simulation cell (Doosan M1013, two-operator layout) |
| `cell_layout.json` | Parametrised cell geometry and zone assignments |
| `experiments/` | Experiment scripts: throughput, protective-stop and fluency measurements |
| `constraints/` | OCL (Operator Constraint Language) grammar and admission-gate implementation |
| `llm_eval.json` | 264-utterance bilingual benchmark (Spanish/English) with ground-truth labels |
| `fig_simplex.pdf` | Figure: preference-cost simplex (throughput × fluency × safety margin) |
| `figures/` | Cell screenshots and additional figures used in the paper |

---

## Reproducing the Experiments

### Requirements

- Python ≥ 3.10
- [RoboDK](https://robodk.com/) (trial or full licence)
- `pip install robodk llama-cpp-python` (or Ollama with `llama3.1:8b-instruct-q4_K_M`)

### Quick start

```bash
# 1. Build the cell
python build_cell_systems.py

# 2. Run the admission-gate benchmark
python experiments/run_benchmark.py --model llama3.1-8b --utterances llm_eval.json

# 3. Run the throughput/safety experiment (all six preference scenarios)
python experiments/run_scenarios.py --cell cell_layout.json --output results/
```

Results are written to `results/` as CSV files that reproduce Tables 3–5 of the paper.

---

## Data Availability

The simulation cell, benchmark, constraint grammar and all experimental data are also archived at:

> **Figshare** · [https://doi.org/10.6084/m9.figshare.33961735](https://doi.org/10.6084/m9.figshare.33961735)

---

## Keywords

human–robot collaboration · sociotechnical systems · large language models · Industry 5.0 · worker autonomy · speed and separation monitoring · ISO/TS 15066 · runtime assurance · natural-language interfaces

---

## Citation

If you use this repository, please cite:

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

Code: [MIT License](LICENSE) · Data and figures: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

---

*Author contact: claudio.urrea@usach.cl*
