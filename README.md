# hrc-voice-gate

**Giving Operators a Voice Without Giving Up Safety: A Safety-Gated Sociotechnical Architecture for Natural-Language Preferences in Human–Robot Collaborative Cells**

Claudio Urrea · Electrical Engineering Department, University of Santiago of Chile (USACH)

*MDPI Systems, manuscript systems-4653532 — submitted October 2026, revised November 2026 (release 1.1.0).*

---

## Abstract

Operators of collaborative cells have no run-time channel to ask the robot to slow down, leave fragile parts to them or serve them first. Language models can interpret such requests, but their failure modes cannot be listed in advance. This work separates interpretation from authority: a local model renders each request as a candidate in a small typed language; a deterministic gate admits only candidates that are well formed, raised at the speaker's own terminal, confined to their zones and tightening; and speed-and-separation monitoring runs below the scheduler, beyond any request. The problem is formalised, and a hazard analysis locates the main residual risk in mode confusion. In a simulated two-operator cell with a Doosan M1013, costs bore little relation to how modest a request sounded: routing fragile parts to an operator cost 1.3% of throughput but raised protective stops sevenfold, and a speed cap cost 6.4%, almost half of it borne by the colleague. Six local open-weight models responded within 1.2 s on one GPU and rendered 41–81% of requests exactly. No candidate outside the speaker's authority or the plant's limits was admitted, and, under the recommended speed law, a full-arm oracle found no breach of the protective distance.

---

## Repository contents

| Path | Description |
|------|-------------|
| `code/constraints.py` | Operator Constraint Language (OCL) and the deterministic admission gate |
| `code/cell_sim.py` | Time-stepped cell model with speed-and-separation monitoring (SSM) |
| `code/cell_ext.py` | Revised cell model: joint-consistent transits (`CellJ`), approximate M1013 kinematics, separation from every link, full-arm safety oracle, link-aware SSM, latency and acceleration limits, time-varying constraints, operator intent vs. constraint store (mode confusion), operator time accounting |
| `code/corpus.py` | Builds the 264-utterance bilingual (EN/ES) benchmark |
| `code/asr_noise.py` | Builds the benchmark copy with synthetic transcription noise (WER ≈ 15%) |
| `code/experiments.py` | Experiments of the submitted version (scenarios, gate, error injection) |
| `code/experiments_r1.py` | Revised experiments A–E: fixed scenarios (TCP and link-aware law), original-surrogate check, latency/acceleration sensitivity, cost-quotation sweep, dynamic requests (gate / no gate / gate + read-back) |
| `code/inj_breakdown.py` | Error-injection study by class of specification violation (F) |
| `code/llm_translate.py` | Llama 3.1 8B Instruct, 16-bit, via `transformers` (submitted version) |
| `code/llm_compare.py` | Six local models via Ollama: greedy, schema-constrained, five sampled seeds, noisy text |
| `code/llm_analysis.py`, `code/llm_tables.py` | Model summaries, repeatability statistics (Fleiss' κ) and table rows (G) |
| `code/dyn_table.py`, `code/dyn_model_table.py` | Tables of the dynamic and closed-loop studies |
| `code/figures_systems_r1.py` | **Every figure of the revised article (Figures 1–8)** |
| `code/graphical_abstract.py` | Graphical abstract (2750 × 1400 px) |
| `code/figures_submitted.py` | Figures of the submitted version, written to `figures/submitted/` (traceability only) |
| `robodk/build_cell_systems.py` | Builds the cell in RoboDK, solves 26 targets and checks the 31 motion segments for collisions |
| `robodk/cell_layout.json` | Cell configuration, joint solutions and collision report (0 collisions) |
| `results/corpus.json`, `results/corpus_asr.json` | Benchmark with gold annotations, and its noisy copy |
| `results/results.json`, `results/llm_eval.json`, `results/table_scenarios.json` | Results of the submitted version |
| `results/r1/` | Results of the revision: `A_scenarios`, `B_surrogate`, `C_sensitivity`, `D_quotes`, `E_*` (dynamic study per translator), `E_table`, `F_injection_classes`, `G_llm`, and table rows |
| `results/llm_compare/` | Per-utterance outputs of six models × 8 runs (264 utterances each) |
| `figures/` | Figures 1–8 (PDF/PNG), graphical abstract, and the RoboDK renders in `figures/raw/` |

---

## Reproducing the results

```bash
pip install -r requirements.txt
cd code                                   # all scripts resolve ../results and ../figures
```

**Submitted-version experiments** (deterministic, CPU only):

```bash
python corpus.py && python experiments.py
```

**Revision** (deterministic, CPU only; about 1 h on two cores):

```bash
python asr_noise.py                       # results/corpus_asr.json
python experiments_r1.py A B C D          # scenarios, surrogate, sensitivity, quotes
python inj_breakdown.py                   # violation classes
python experiments_r1.py E                # dynamic study (uses results/llm_compare/ if present)
python dyn_table.py && python dyn_model_table.py
```

**Local language models** (GPU; Ollama 0.40.15, default 4-bit builds: Q4_K_M, Q4_0 for Gemma 2 9B; about 2.5 h on an RTX 3060):

```bash
ollama pull llama3.2:3b llama3.1:8b qwen2.5:7b mistral:7b gemma2:9b qwen2.5:14b
python llm_compare.py --models llama3.2:3b llama3.1:8b qwen2.5:7b mistral:7b gemma2:9b qwen2.5:14b
python llm_analysis.py && python llm_tables.py
```

**Figures**:

```bash
python figures_systems_r1.py              # Figures 1-8 of the revised article
python graphical_abstract.py
```

The committed JSON files under `results/` are the outputs of exactly these commands. RoboDK (licence required) is needed only to rebuild the cell with `robodk/build_cell_systems.py`.

### Where each result lives

| Article item | Source |
|--------------|--------|
| Table 5 — SSM sensitivity | `results/r1/C_sensitivity.json` |
| Tables 6–7, Figure 5 — fixed scenarios, time decomposition | `results/r1/A_scenarios.json` (keys prefixed `tcp`; prefix `link` for the link-aware law) |
| Figure 6 — cost quotation | `results/r1/D_quotes.json` |
| Table 8 — dynamic study | `results/r1/E_faithful.json`, `E_eps0.2.json`, `E_eps0.5.json`, `E_table.json` |
| Table 9 — faithful translator | `results/results.json` (`gate_faithful`) |
| Figure 7 — error injection | `results/results.json` (`gate_eval`), `results/r1/F_injection_classes.json` |
| Tables 10–11, Figure 8 — local models, repeatability | `results/r1/G_llm.json`, `results/llm_compare/`, `results/llm_eval.json` |
| Table 12 — closed loop per model | `results/r1/E_*__greedy.json`, `E_llama3.1-8b-fp16.json`, `E_table.json` |
| Appendix A — original vs. revised motion model | `results/results.json` (`scenarios`), `results/r1/A_scenarios.json`, `B_surrogate.json` |

---

## Data availability

Archived at Figshare: [https://doi.org/10.6084/m9.figshare.33961735](https://doi.org/10.6084/m9.figshare.33961735). All data were produced in simulation; no human participant contributed data.

## Citation

```bibtex
@article{urrea2026systems,
  author  = {Urrea, Claudio},
  title   = {Giving Operators a Voice Without Giving Up Safety: A Safety-Gated
             Sociotechnical Architecture for Natural-Language Preferences in
             Human--Robot Collaborative Cells},
  journal = {Systems},
  year    = {2026},
  note    = {Under review (revised version)}
}
```

## License

Code: [MIT License](LICENSE) · Data, figures and cell record: [CC BY 4.0](LICENSE-DATA)

*Contact: claudio.urrea@usach.cl*
