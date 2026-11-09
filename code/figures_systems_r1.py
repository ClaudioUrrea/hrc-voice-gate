# -*- coding: utf-8 -*-
"""figures_systems_r1.py - Every figure of the revised manuscript (Systems, round 1).

    Figure 1  safety-gated control structure (drawn here)
    Figure 2  cell overview (RoboDK render, figures/raw/)
    Figure 3  annotated top view of the cell (RoboDK render, figures/raw/)
    Figure 4  SSM speed law and a 90-s trace under an admitted cap (simulated here)
    Figure 5  fixed-window scenarios            <- results/r1/A_scenarios.json
    Figure 6  cost of a request vs its size     <- results/r1/D_quotes.json
    Figure 7  error-injection study             <- results/results.json, results/r1/F_injection_classes.json
    Figure 8  local language models             <- results/r1/G_llm.json

Run from inside code/:   python figures_systems_r1.py            (all figures)
                         python figures_systems_r1.py 1 5 8      (selected figures)
Output: ../figures/FigureN_*.pdf and .png. Requires numpy, scipy, matplotlib, pillow;
the TeX Gyre Pagella font is used if installed (otherwise matplotlib falls back silently).
"""
import json, math, os, sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.ticker import FuncFormatter, FixedFormatter
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from PIL import Image
import shutil
from scipy import stats
for f in fm.findSystemFonts():
    if "pagella" in f.lower(): fm.fontManager.addfont(f)
plt.rcParams.update({"font.family": "TeX Gyre Pagella", "mathtext.fontset": "custom",
                     "mathtext.rm": "TeX Gyre Pagella", "mathtext.it": "TeX Gyre Pagella:italic",
                     "mathtext.bf": "TeX Gyre Pagella:bold", "font.size": 10, "axes.titlesize": 11,
                     "axes.labelsize": 10.5, "legend.fontsize": 9, "savefig.dpi": 400,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "mathtext.fallback": "cm", "axes.unicode_minus": True})
MINUS = "−"
def _fmt(x, pos=None): return f"{x:g}".replace("-", MINUS)
def math_ticks(fig, x=True):
    for a in fig.axes:
        for ax_ in ((a.xaxis, a.yaxis) if x else (a.yaxis,)):
            if not isinstance(ax_.get_major_formatter(), FixedFormatter):
                ax_.set_major_formatter(FuncFormatter(_fmt))
FIG = "../figures/"; R1 = "../results/r1/"; UP = "../figures/raw/"
C1, C2, C3, C4, C5 = "#1f4e79", "#c55a11", "#548235", "#7f6000", "#7030a0"
which = sys.argv[1:] or ["1", "2", "3", "4", "5", "6", "7", "8"]

if "1" in which:
    HUM, INT, EXE, SAF = "#fbe5d6", "#e4dbf1", "#dae3f3", "#f8cbad"
    CH, CI, CE, CS = "#c55a11", "#7030a0", "#1f4e79", "#c00000"
    fig, ax = plt.subplots(figsize=(7.4, 5.6)); ax.set_xlim(0, 100); ax.set_ylim(-2.5, 76); ax.axis("off")

    def band(y0, h, fc, label, col):
        ax.add_patch(Rectangle((0.3, y0), 99.4, h, fc=fc, ec="none", alpha=0.35, zorder=0))
        ax.text(1.2, y0 + h - 1.3, label, fontsize=8.6, fontweight="bold", color=col, va="top", zorder=5)

    def box(x, y, w, h, t, fc, ec="0.25", lw=0.8, fs=7.4, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35,rounding_size=1.0",
                                    fc=fc, ec=ec, lw=lw, zorder=3))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, zorder=4,
                fontweight="bold" if bold else "normal", linespacing=1.15)

    def arr(p, q, t="", c="0.2", ls="-", lw=0.9, off=(0, 0), ha="center", rad=0.0, fs=6.6):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, lw=lw, color=c, linestyle=ls,
                                     connectionstyle=f"arc3,rad={rad}", zorder=2))
        if t:
            ax.text((p[0] + q[0]) / 2 + off[0], (p[1] + q[1]) / 2 + off[1], t, fontsize=fs, ha=ha, va="center",
                    color=c, style="italic", zorder=6, bbox=dict(fc="white", ec="none", pad=0.4, alpha=0.85))

    # bands
    band(54, 21.5, HUM, "Human layer", CH)
    band(39, 14, INT, "Interpretation layer (untrusted, on premises)", CI)
    band(0.5, 37.5, EXE, "Execution layer (deterministic)", CE)

    # human layer
    box(3, 59, 17, 9.5, "Operator A or B\nspeaks or types\na preference", HUM)
    box(27, 59, 19, 9.5, "Interaction terminal\nbound to one station\n(push-to-talk, display)", HUM)
    box(54, 59, 20, 9.5, "Read-back, reason\nand cost quote\nshown to the operator", HUM)
    box(81, 59, 16.5, 9.5, "Andon beacon\n(request active)", HUM)
    arr((20.6, 63.7), (26.6, 63.7), "utterance", off=(0, 1.8))

    # interpretation layer
    box(27, 42, 19, 8.5, "Local LLM $\\tau$\n(open-weight, quantised)", INT)
    box(54, 42, 20, 8.5, "OCL candidate $y$\n(JSON, not trusted)", INT)
    arr((36.5, 58.6), (36.5, 51.0), "text $u$", c=CI, off=(-3.2, 0.3))
    arr((46.4, 46.2), (53.6, 46.2), c=CI)

    # execution layer
    box(54, 25, 20, 9.5, "Safety gate $G(y,s)$\nschema · safety ·\nauthority · range", EXE, ec=CE, lw=1.3, bold=False)
    box(27, 25, 19, 9.5, "Constraint store $\\mathcal{C}(t)$\n(time-windowed,\nadmitted only)", EXE)
    box(3, 25, 17, 9.5, "Scheduler $\\pi$\nselects next job\n(part, drop zone)", EXE)
    box(3, 8, 17, 10, "Verified motion\nprogramme (31\njoint-space segments)", EXE)
    box(27, 8, 19, 10, "Robot controller\n$v_{\\mathrm{cmd}}=\\min(v_{\\mathrm{seg}},v_{\\mathrm{cap}},v_{\\mathrm{SSM}})$", EXE, fs=7.0)
    box(54, 8, 20, 10, "SSM (ISO/TS 15066)\nsafety-rated, not\nreachable from OCL", SAF, ec=CS, lw=1.3)
    box(81, 25, 16.5, 9.5, "Laser scanner and\nbody tracking\n(separation $d$, $v_h$)", SAF, ec=CS)
    box(81, 8, 16.5, 10, "Doosan M1013\nworking beside\nthe operators", "#ededed")

    arr((64, 41.6), (64, 35.0), "candidate", c=CI, off=(3.6, 0))
    # trusted identity path: terminal -> gate, outside the interpretation layer's boxes
    idc = CE
    ax.plot([42, 42, 76.6, 76.6], [58.6, 53.2, 53.2, 31.0], color=idc, lw=1.7, zorder=2, solid_capstyle="butt")
    ax.add_patch(FancyArrowPatch((76.6, 31.0), (74.5, 31.0), arrowstyle="-|>", mutation_scale=10, lw=1.7, color=idc, zorder=2))
    ax.text(59.5, 53.2, "terminal identity $s$ (hardware path, bypasses the LLM)", fontsize=6.6, color=idc,
            style="italic", ha="center", va="center", zorder=6, bbox=dict(fc="white", ec="none", pad=0.5))
    # decision, reason and status back to the human layer
    ax.plot([74.5, 79.0, 79.0], [27.0, 27.0, 56.0], color=CE, lw=0.9, ls="--", zorder=2)
    ax.plot([64, 89.2], [56.0, 56.0], color=CE, lw=0.9, ls="--", zorder=2)
    ax.add_patch(FancyArrowPatch((64, 56.0), (64, 58.6), arrowstyle="-|>", mutation_scale=9, lw=0.9, color=CE, zorder=2))
    ax.add_patch(FancyArrowPatch((89.2, 56.0), (89.2, 58.6), arrowstyle="-|>", mutation_scale=9, lw=0.9, color=CE, zorder=2))
    ax.text(86.5, 44.5, "decision, reason\nand status", fontsize=6.6, color=CE, style="italic", ha="center",
            va="center", zorder=6, bbox=dict(fc="white", ec="none", pad=0.4, alpha=0.9))
    arr((53.6, 29.8), (46.4, 29.8), "admitted $c$", c=CE, off=(0, 1.9))
    arr((26.6, 29.8), (20.4, 29.8), c=CE)
    arr((11.5, 24.6), (11.5, 18.4), "job", c=CE, off=(2.6, 0))
    arr((20.4, 13), (26.6, 13), "$v_{\\mathrm{seg}}$", c=CE, off=(0, 1.8))
    arr((36.5, 24.6), (36.5, 18.4), "$v_{\\mathrm{cap}}$", c=CE, off=(2.8, 0))
    arr((53.6, 13), (46.4, 13), "$v_{\\mathrm{SSM}}$", c=CS, off=(0, 1.8), lw=1.2)
    arr((82.5, 24.6), (73.0, 18.4), "$d,\\,v_h$", c=CS, off=(-1.2, 2.2), lw=1.2)
    arr((92.5, 18.4), (92.5, 24.6), "sensed", c=CS, off=(-3.3, 0))
    arr((36.5, 7.6), (88.0, 7.6), "", c="0.35", rad=0.22)
    ax.text(62, -1.2, "joint motion commanded by the controller", fontsize=6.6, style="italic", color="0.3", ha="center")
    fig.savefig(FIG + "Figure1_Architecture.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(FIG + "Figure1_Architecture.png", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

if "2" in which or "3" in which:
    # ---------------------------------------------------------------- Figure 2: cell overview (RoboDK render)
    shutil.copy(UP + "Escena__22_09_2026_.png", FIG + "Figure2_Cell_Overview.png")

    # ---------------------------------------------------------------- Figure 3: annotated top view of the cell
    im = Image.open(UP + "Escena_Vista_Cenital__22_09_2026_.png"); W, H = im.size; k = W / 2133.0
    fig, ax = plt.subplots(figsize=(7.2, 7.2 * H / W)); ax.imshow(im); ax.axis("off")
    lab = [("Collaborative robot\n(Doosan M1013,\npedestal 780 mm)", (1110, 830), (1500, 680)),
           ("Fragile-part rack\n(PCB, ESD trays)", (700, 940), (420, 1060)),
           ("Standard-part bins", (1440, 940), (1720, 1060)),
           ("Housing infeed conveyor", (1070, 500), (1070, 300)),
           ("Handover zone", (1075, 1100), (1300, 1170)),
           ("Kit tray A", (875, 1225), (600, 1260)), ("Kit tray B", (1265, 1225), (1540, 1260)),
           ("Operator A", (870, 1540), (870, 1700)), ("Operator B", (1290, 1540), (1290, 1700)),
           ("Interaction\nterminal A", (620, 1480), (330, 1560)), ("Interaction\nterminal B", (1530, 1480), (1800, 1560))]
    for txt, p, q in lab:
        ax.annotate(txt, xy=(p[0]*k, p[1]*k), xytext=(q[0]*k, q[1]*k), ha="center", va="center", fontsize=8.5,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.3", lw=0.6, alpha=0.93),
                    arrowprops=dict(arrowstyle="-", lw=0.7, color="0.15"))
    fig.savefig(FIG + "Figure3_Cell_Layout.png", bbox_inches="tight", pad_inches=0.02); plt.close(fig)


if "4" in which:
    import cell_sim
    from cell_ext import CellJ
    from constraints import Constraint as _C
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.2), gridspec_kw=dict(width_ratios=[1, 1.5]))
    d = np.linspace(0, 1200, 400)
    for vh, c in ((200, C1), (800, C2), (1600, C3)):
        a.plot(d, [min(500, cell_sim.ssm_speed(x, vh)) for x in d], color=c, lw=1.5, label=f"$v_h$ = {vh} mm/s")
    a.set_xlabel("Separation $d$ (mm)"); a.set_ylabel("Admissible speed (mm/s)"); a.legend(frameon=False, loc="lower right", fontsize=8)
    _c = CellJ([_C("speed_cap", "Op_B", zone="kit_B", v_max=150)], seed=3, horizon=600, oracle=False); _c.log_every = 1; _c.run()
    t = np.array(_c.log["t"]); dd = np.array(_c.log["d"]); v = np.array(_c.log["v"]); vc = np.array(_c.log["vcap"])
    on = np.where(vc < 499)[0]; t0 = max(0.0, t[on[0]] - 25) if len(on) else 100.0
    m = (t >= t0) & (t <= t0 + 90)
    b.fill_between(t[m], 0, 520, where=vc[m] < 499, color=C2, alpha=0.15, lw=0, label="Cap active (zone B)")
    b.plot(t[m], v[m], color=C1, lw=1.0, label="TCP speed command")
    b.set_xlabel("Time (s)"); b.set_ylabel("Speed (mm/s)"); b.set_ylim(0, 520); b.set_xlim(t0, t0 + 90)
    b2 = b.twinx(); b2.plot(t[m], dd[m], color="0.55", lw=0.8, label="Separation (right axis)")
    b2.set_ylabel("Separation (mm)"); b2.spines["top"].set_visible(False); b2.spines["right"].set_visible(True)
    b.spines["right"].set_visible(True)
    h1, l1 = b.get_legend_handles_labels(); h2, l2 = b2.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, frameon=False, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.62, -0.02))
    b.set_title("(b) Speed under SSM and an admitted cap", loc="left"); a.set_title("(a) SSM speed law", loc="left")
    math_ticks(fig); fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(FIG + "Figure4_SSM.pdf"); fig.savefig(FIG + "Figure4_SSM.png"); plt.close(fig)

if "5" in which:
    A = json.load(open(R1 + "A_scenarios.json"))
    names = [k for k in A if k.startswith("tcp|")]; S = {k.split("|", 1)[1]: A[k] for k in names}
    names = list(S); short = [n.split(" ")[0] for n in names]
    def ms(key, scale=1.0): return [np.mean(S[n][key]) * scale for n in names], [np.std(S[n][key], ddof=1) * scale for n in names]
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.8)); x = np.arange(len(names))
    mA, sA = ms("kits_A"); mB, sB = ms("kits_B")
    axs[0].bar(x - 0.18, mA, 0.36, yerr=sA, color=C1, capsize=1.5, label="Operator A", error_kw=dict(lw=0.6))
    axs[0].bar(x + 0.18, mB, 0.36, yerr=sB, color=C2, capsize=1.5, label="Operator B", error_kw=dict(lw=0.6))
    axs[0].set_ylim(8, 17.5); axs[0].set_ylabel("Kits per hour"); axs[0].set_title("(a) Output per operator", loc="left")
    axs[0].legend(frameon=False, fontsize=7.6, ncol=2, loc="upper right", columnspacing=0.8, handlelength=1.2)
    mu1, _ = ms("robot_idle", 100); mu2, _ = ms("human_idle", 100)
    axs[1].bar(x - 0.18, mu1, 0.36, color=C1, label="Robot idle"); axs[1].bar(x + 0.18, mu2, 0.36, color=C2, label="Human idle")
    axs[1].set_ylabel("% of shift"); axs[1].set_title("(b) Idle time", loc="left"); axs[1].set_ylim(0, 60)
    axs[1].legend(frameon=False, fontsize=7.6, ncol=2, loc="upper right", columnspacing=0.8, handlelength=1.2); axs[1].set_yticks([0, 10, 20, 30, 40, 50])
    mu3, _ = ms("reduced_frac", 100); mu4, _ = ms("stop_frac", 100)
    axs[2].bar(x, mu3, 0.6, color=C3, label="Reduced speed"); axs[2].bar(x, mu4, 0.6, bottom=mu3, color="#c00000", label="Protective stop")
    axs[2].set_ylabel("% of shift"); axs[2].set_title("(c) Slowdown and stops", loc="left"); axs[2].set_ylim(0, 22)
    axs[2].legend(frameon=False, fontsize=7.6, loc="upper left")
    for a_ in axs: a_.set_xticks(x); a_.set_xticklabels(short, fontsize=8.5)
    math_ticks(fig, x=False); fig.tight_layout(); fig.savefig(FIG + "Figure5_Scenarios.pdf"); fig.savefig(FIG + "Figure5_Scenarios.png"); plt.close(fig)

if "7" in which:
    G = json.load(open("../results/results.json"))["gate_eval"]; F = json.load(open(R1 + "F_injection_classes.json"))
    eps = sorted(float(e) for e in G)
    key = lambda e: [k for k in G if abs(float(k) - e) < 1e-9][0]
    fkey = lambda e: [k for k in F if abs(float(k) - e) < 1e-9][0]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 2.9))
    x = np.arange(len(eps)); bottom = np.zeros(len(eps))
    cls = [("protective", "Protective function (no effect)", "#bfbfbf"), ("overspeed", "Cap above $\\bar v$ (clamped)", "#8faadc"),
           ("authority", "Colleague or their zone", "#c00000"), ("range", "Outside policy range", C2)]
    for k, lab, col in cls:
        vals = np.array([F[fkey(e)][k] for e in eps]); a.bar(x, vals, 0.62, bottom=bottom, color=col, label=lab); bottom += vals
    a.plot(x, [F[fkey(e)]["any_gate"] for e in eps], "s-", color=C1, ms=4, lw=1.4, label="Admitted with the gate")
    a.set_xticks(x); a.set_xticklabels([f"{e:g}" for e in eps]); a.set_xlabel("Injected error rate $\\varepsilon$")
    a.set_ylabel("Violations per 264-utterance pass"); a.set_title("(a) Specification violations", loc="left")
    a.legend(frameon=False, fontsize=7.0, loc="upper left")
    fid = [100*np.mean(G[key(e)]["fidelity"]) for e in eps]; sp = [100*np.mean(G[key(e)]["semantic_pass"]) for e in eps]
    lost = [100*np.mean(G[key(e)]["false_reject"]) for e in eps]
    b.plot(eps, fid, "o-", color=C3, lw=1.4, ms=4, label="Intent executed as meant")
    b.plot(eps, sp, "^-", color=C2, lw=1.4, ms=4, label="Wrong but admitted")
    b.plot(eps, lost, "v-", color=C5, lw=1.4, ms=4, label="Correct intent withheld")
    b.set_xlabel("Injected error rate $\\varepsilon$"); b.set_ylabel("% of actionable utterances")
    b.legend(frameon=False, fontsize=7.6); b.set_title("(b) Preference fidelity", loc="left")
    math_ticks(fig); a.xaxis.set_major_formatter(FixedFormatter([f"{e:g}" for e in eps]))
    fig.tight_layout(); fig.savefig(FIG + "Figure7_Gate_Robustness.pdf"); fig.savefig(FIG + "Figure7_Gate_Robustness.png"); plt.close(fig)

if "6" in which:
    Q = json.load(open(R1 + "D_quotes.json")); b = Q["base"]
    bA, bB = np.mean(b["kits_A"]), np.mean(b["kits_B"])
    def dd(k): return np.mean(Q[k]["kits_A"]) - bA, np.mean(Q[k]["kits_B"]) - bB
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.7))
    caps = [50, 100, 150, 200, 250, 300, 400, 500]
    a = np.array([dd(f"cap_B={v}") for v in caps])
    axs[0].plot(caps, -a[:, 1], "o-", color=C2, ms=3.5, lw=1.4, label="Requester (B)")
    axs[0].plot(caps, -a[:, 0], "s-", color=C1, ms=3.5, lw=1.4, label="Colleague (A)")
    axs[0].set_xlabel("Speed cap on tray B (mm/s)"); axs[0].set_ylabel("Kits per hour lost"); axs[0].set_title("(a) Speed cap", loc="left")
    axs[0].legend(frameon=False, fontsize=7.6)
    ex = [60, 120, 180, 240, 300]; e = np.array([dd(f"excl_A={v}") for v in ex])
    axs[1].plot(np.array(ex) / 60, -e[:, 0], "o-", color=C2, ms=3.5, lw=1.4, label="Requester (A)")
    axs[1].plot(np.array(ex) / 60, -e[:, 1], "s-", color=C1, ms=3.5, lw=1.4, label="Colleague (B)")
    axs[1].set_xlabel("Exclusion of tray A (min)"); axs[1].set_title("(b) Exclusion", loc="left")
    pz = [300, 600, 900, 1200]; p = np.array([dd(f"pause_A={v}") for v in pz])
    axs[2].plot(np.array(pz) / 60, -p[:, 0], "o-", color=C2, ms=3.5, lw=1.4, label="Requester (A)")
    axs[2].plot(np.array(pz) / 60, -p[:, 1], "s-", color=C1, ms=3.5, lw=1.4, label="Colleague (B)")
    axs[2].set_xlabel("Pause of operator A (min)"); axs[2].set_title("(c) Pause", loc="left")
    for a_ in axs: a_.axhline(0, color="0.6", lw=0.6)
    math_ticks(fig); fig.tight_layout(); fig.savefig(FIG + "Figure6_Quotes.pdf"); fig.savefig(FIG + "Figure6_Quotes.png"); plt.close(fig)

if "8" in which:
    G = json.load(open(R1 + "G_llm.json"))
    order = [("llama3.2_3b", "Llama 3.2\n3B"), ("llama3.1_8b", "Llama 3.1\n8B"), ("qwen2.5_7b", "Qwen2.5\n7B"),
             ("mistral_7b", "Mistral\n7B"), ("gemma2_9b", "Gemma 2\n9B"), ("qwen2.5_14b", "Qwen2.5\n14B")]
    order = [(k, l) for k, l in order if k in G]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.0), gridspec_kw=dict(width_ratios=[2.1, 1]))
    x = np.arange(len(order)); w = 0.2
    g = [100 * G[k]["greedy"]["exact"] for k, _ in order]
    c = [100 * G[k].get("constrained", G[k]["greedy"])["exact"] for k, _ in order]
    s = [100 * G[k]["sampled"]["exact"][0] if "sampled" in G[k] else np.nan for k, _ in order]
    se = [100 * G[k]["sampled"]["exact"][1] if "sampled" in G[k] else 0 for k, _ in order]
    n = [100 * G[k]["asr"]["exact"] if "asr" in G[k] else np.nan for k, _ in order]
    a.bar(x - 1.5 * w, g, w, color=C1, label="Greedy")
    a.bar(x - 0.5 * w, c, w, color=C3, label="Schema-constrained")
    a.bar(x + 0.5 * w, s, w, yerr=se, color=C2, label="Sampled (5 seeds)", capsize=1.5, error_kw=dict(lw=0.6))
    a.bar(x + 1.5 * w, n, w, color="#a5a5a5", label="Noisy text, greedy")
    a.axhline(100 * G["llama3.1-8b-fp16 (submitted)"]["greedy"]["exact"], color="#c00000", lw=0.9, ls="--", label="Submitted (Llama 3.1 8B, 16-bit)")
    a.set_xticks(x); a.set_xticklabels([l for _, l in order], fontsize=8); a.set_ylim(0, 118); a.set_yticks([0, 20, 40, 60, 80, 100])
    a.set_ylabel("Exact intent, actionable (%)"); a.set_title("(a) Fidelity by model and decoding", loc="left")
    a.legend(frameon=False, fontsize=6.6, ncol=3, loc="upper left", columnspacing=0.8, handlelength=1.4)
    med = [G[k]["greedy"]["lat_med"] for k, _ in order]; p95 = [G[k]["greedy"]["lat_p95"] for k, _ in order]
    b.bar(x, med, 0.55, color=C1, label="Median"); b.scatter(x, p95, marker="_", s=120, color="#c00000", label="95th percentile", zorder=5)
    b.set_xticks(x); b.set_xticklabels([l.replace("\n", " ") for _, l in order], fontsize=6.6, rotation=50, ha="right"); b.set_ylabel("Latency per request (s)")
    b.set_title("(b) Latency on the GPU", loc="left"); b.set_ylim(0, 1.5); b.legend(frameon=False, fontsize=7, loc="upper left")
    math_ticks(fig, x=False); fig.tight_layout(); fig.savefig(FIG + "Figure8_Models.pdf"); fig.savefig(FIG + "Figure8_Models.png"); plt.close(fig)
