# -*- coding: utf-8 -*-
"""figures.py - All manuscript figures, typeset in TeX Gyre Pagella (Palatino Linotype clone)."""
import json, math, shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image
from scipy import stats
import cell_sim

for f in fm.findSystemFonts():
    if "pagella" in f.lower(): fm.fontManager.addfont(f)
plt.rcParams.update({"font.family": "TeX Gyre Pagella", "mathtext.fontset": "custom",
                     "mathtext.rm": "TeX Gyre Pagella", "mathtext.it": "TeX Gyre Pagella:italic",
                     "mathtext.bf": "TeX Gyre Pagella:bold", "font.size": 10, "axes.titlesize": 11,
                     "axes.labelsize": 10.5, "legend.fontsize": 9, "savefig.dpi": 400,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "mathtext.fallback": "cm", "axes.unicode_minus": True})
from matplotlib.ticker import FuncFormatter

MINUS = "\u2212"          # U+2212, the minus sign required by MDPI (never a hyphen)

def _fmt(x, pos=None):
    """Digits in Pagella; negatives with the MDPI/LaTeX minus sign (U+2212), not a hyphen."""
    return f"{x:g}".replace("-", MINUS)

def math_ticks(fig, x=True):
    for a in fig.axes:
        from matplotlib.ticker import FixedFormatter
        for ax_ in ((a.xaxis, a.yaxis) if x else (a.yaxis,)):
            if not isinstance(ax_.get_major_formatter(), FixedFormatter):
                ax_.set_major_formatter(FuncFormatter(_fmt))

FIG = "../figures/"; UP = "../figures/raw/"   # RoboDK renders used for Figures 1 and 2
R = json.load(open("../results/results.json"))
C1, C2, C3, C4, C5 = "#1f4e79", "#c55a11", "#548235", "#7f6000", "#7030a0"

# ---------------------------------------------------------------- Figure 1: overview (RoboDK render)
shutil.copy(UP + "Escena__22_09_2026_.png", FIG + "Figure1_Cell_Overview.png")

# ---------------------------------------------------------------- Figure 2: annotated top view
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
zones = R["scenarios"] and None
fig.savefig(FIG + "Figure2_Cell_Layout.png", bbox_inches="tight", pad_inches=0.02); plt.close(fig)

# ---------------------------------------------------------------- Figure 3: architecture
fig, ax = plt.subplots(figsize=(7.4, 4.4)); ax.set_xlim(0, 100); ax.set_ylim(-6, 60); ax.axis("off")
def box(x, y, w, h, t, fc, fs=8.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec="0.25", lw=0.8))
    ax.text(x + w/2, y + h/2, t, ha="center", va="center", fontsize=fs)
def arr(p, q, t="", c="0.2", ls="-", off=(0, 0), ha="center"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=10, lw=0.9, color=c, linestyle=ls))
    if t: ax.text((p[0]+q[0])/2 + off[0], (p[1]+q[1])/2 + off[1], t, fontsize=7.0, ha=ha, va="center", color=c, style="italic")
W_ = 21.5
box(2, 45, W_, 9, "Operator A / B\nutterance (EN / ES)", "#fbe5d6")
box(27.5, 45, W_, 9, "Interaction terminal\n(authenticated identity)", "#fbe5d6")
box(53, 45, W_, 9, "Explanation and\nconfirmation", "#fbe5d6")
box(78.5, 45, 19.5, 9, "Andon beacon\n(blue: request active)", "#fbe5d6")
box(2, 24, W_, 9, "Local LLM (Llama 3.1 8B)\nwith OCL prompt", "#e4dbf1")
box(27.5, 24, W_, 9, "OCL candidate\n(JSON object)", "#e4dbf1")
box(2, 2, W_, 10, "Safety gate\nschema · authority\n· invariants", "#dae3f3")
box(27.5, 2, W_, 10, "Constraint store\n(time-windowed)", "#dae3f3")
box(53, 2, W_, 10, "Preference-aware\nscheduler", "#dae3f3")
box(78.5, 2, 19.5, 10, "Robot controller\n+ SSM (ISO/TS 15066)", "#dae3f3")
arr((23.9, 49.5), (27.3, 49.5))
arr((31, 44.6), (20, 33.4), "text + speaker id", off=(-1.5, 1.2), ha="right")
arr((23.9, 28.5), (27.3, 28.5))
arr((31, 23.6), (20, 12.4), "untrusted candidate", c=C5, off=(-1.5, 1.2), ha="right")
arr((23.9, 7), (27.3, 7)); arr((49.4, 7), (52.8, 7)); arr((74.9, 7), (78.3, 7))
arr((63.75, 12.4), (63.75, 44.6), "decision + reason", c=C1, ls="--", off=(-1.2, 10), ha="right")
arr((88, 12.4), (88, 44.6), "status", c=C1, ls="--", off=(1.2, -9), ha="left")
ax.text(99, 58, "Human layer", ha="right", fontsize=9.2, fontweight="bold", color=C2)
ax.text(76, 28.5, "Interpretation layer\n(untrusted)", ha="center", va="center", fontsize=9.2, fontweight="bold", color=C5)
ax.text(99, -3.8, "Execution layer (trusted, deterministic); SSM cannot be modified by any request",
        ha="right", fontsize=9.2, fontweight="bold", color=C1)
fig.savefig(FIG + "Figure3_Architecture.pdf", bbox_inches="tight", pad_inches=0.03); fig.savefig(FIG + "Figure3_Architecture.png", bbox_inches="tight", pad_inches=0.03); plt.close(fig)

# ---------------------------------------------------------------- Figure 4: SSM law + trace
fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.2), gridspec_kw=dict(width_ratios=[1, 1.5]))
d = np.linspace(0, 1200, 400)
for vh, c in ((200, C1), (800, C2), (1600, C3)):
    a.plot(d, [min(500, cell_sim.ssm_speed(x, vh)) for x in d], color=c, lw=1.5, label=f"$v_h$ = {vh} mm/s")
a.set_xlabel("Separation $d$ (mm)"); a.set_ylabel("Admissible TCP speed (mm/s)"); a.legend(frameon=False)
a.set_title("(a) SSM speed law", loc="left")
from constraints import Constraint as _C
_c = cell_sim.Cell([_C("speed_cap", "Op_B", zone="kit_B", v_max=150)], seed=3, horizon=600); _c.log_every = 1; _c.run()
t = np.array(_c.log["t"]); dd = np.array(_c.log["d"]); v = np.array(_c.log["v"]); vc = np.array(_c.log["vcap"])
on = np.where(vc < 499)[0]; t0 = max(0.0, t[on[0]] - 25) if len(on) else 100.0
m = (t >= t0) & (t <= t0 + 90)
b.fill_between(t[m], 0, 520, where=vc[m] < 499, color=C2, alpha=0.15, lw=0, label="Cap active (zone B)")
b.plot(t[m], v[m], color=C1, lw=1.0, label="TCP speed command")
b.set_xlabel("Time (s)"); b.set_ylabel("Speed (mm/s)"); b.set_ylim(0, 520); b.set_xlim(t0, t0 + 90)
b2 = b.twinx(); b2.plot(t[m], dd[m], color="0.55", lw=0.8, label="Separation (right axis)")
b2.set_ylabel("Separation (mm)"); b2.spines["top"].set_visible(False)
b2.spines["right"].set_visible(True)   # visible right axis: its ticks are not read as minus signs
b.spines["right"].set_visible(True)
h1, l1 = b.get_legend_handles_labels(); h2, l2 = b2.get_legend_handles_labels()
fig.legend(h1 + h2, l1 + l2, frameon=False, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.62, -0.02))
b.set_title("(b) Speed under SSM and an admitted cap", loc="left")
a.set_title("(a) SSM speed law", loc="left")
math_ticks(fig); fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(FIG + "Figure4_SSM.pdf"); fig.savefig(FIG + "Figure4_SSM.png"); plt.close(fig)

# ---------------------------------------------------------------- Figure 5: scenario outcomes
S = R["scenarios"]; names = list(S.keys()); short = [n.split(" ")[0] for n in names]
def ms(key, scale=1.0):
    return [np.mean(S[n][key]) * scale for n in names], [np.std(S[n][key], ddof=1) * scale for n in names]
fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.8)); x = np.arange(len(names))
mu, sd = ms("throughput")
axs[0].bar(x, mu, yerr=sd, color=[C1] + [C2] * (len(names) - 1), capsize=2, width=0.65)
axs[0].set_ylim(20, 33); axs[0].set_ylabel("Kits per hour"); axs[0].set_title("(a) Throughput", loc="left")
mu1, _ = ms("robot_idle", 100); mu2, _ = ms("human_idle", 100)
axs[1].bar(x - 0.18, mu1, 0.36, color=C1, label="Robot idle"); axs[1].bar(x + 0.18, mu2, 0.36, color=C2, label="Human idle")
axs[1].set_ylabel("% of shift"); axs[1].set_title("(b) Idle time", loc="left"); axs[1].set_ylim(0, 55)
axs[1].legend(frameon=False, fontsize=7.8, ncol=2, loc="upper right", columnspacing=0.8, handlelength=1.2); axs[1].set_yticks([0, 10, 20, 30, 40])
mu3, _ = ms("reduced_frac", 100); mu4, _ = ms("stop_frac", 100)
axs[2].bar(x, mu3, 0.6, color=C3, label="Reduced speed"); axs[2].bar(x, mu4, 0.6, bottom=mu3, color="#c00000", label="Protective stop")
axs[2].set_ylabel("% of shift"); axs[2].set_title("(c) Slowdown and stops", loc="left"); axs[2].set_ylim(0, 22)
axs[2].legend(frameon=False, fontsize=7.8, loc="upper left")
for a_ in axs: a_.set_xticks(x); a_.set_xticklabels(short, fontsize=8.5)
math_ticks(fig, x=False); fig.tight_layout(); fig.savefig(FIG + "Figure5_Scenarios.pdf"); fig.savefig(FIG + "Figure5_Scenarios.png"); plt.close(fig)

# ---------------------------------------------------------------- Figure 6: gate under translation errors
G = R["gate_eval"]; eps = sorted(float(e) for e in G)
def gm(k): return [np.mean(G[str(e) if str(e) in G else repr(e)][k]) for e in eps]
def key(e):
    for kk in G:
        if abs(float(kk) - e) < 1e-9: return kk
fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 2.8))
ue_n = [np.mean(G[key(e)]["unsafe_exec_nogate"]) for e in eps]; ue_g = [np.mean(G[key(e)]["unsafe_exec_gate"]) for e in eps]
a.plot(eps, ue_n, "o-", color="#c00000", lw=1.4, ms=4, label="Without gate")
a.plot(eps, ue_g, "s-", color=C1, lw=1.4, ms=4, label="With safety gate")
a.set_xlabel("Injected translation error rate $\\varepsilon$"); a.set_ylabel("Unsafe commands executed\n(per 264-utterance pass)")
a.legend(frameon=False); a.set_title("(a) Safety", loc="left")
fid = [100*np.mean(G[key(e)]["fidelity"]) for e in eps]; sp = [100*np.mean(G[key(e)]["semantic_pass"]) for e in eps]
lost = [100*np.mean(G[key(e)]["false_reject"]) for e in eps]
b.plot(eps, fid, "o-", color=C3, lw=1.4, ms=4, label="Intent executed as meant")
b.plot(eps, sp, "^-", color=C2, lw=1.4, ms=4, label="Wrong but safe, admitted")
b.plot(eps, lost, "v-", color=C5, lw=1.4, ms=4, label="Correct intent withheld")
b.set_xlabel("Injected translation error rate $\\varepsilon$"); b.set_ylabel("% of actionable utterances")
b.legend(frameon=False, fontsize=7.8); b.set_title("(b) Preference fidelity", loc="left")
math_ticks(fig); fig.tight_layout(); fig.savefig(FIG + "Figure6_Gate_Robustness.pdf"); fig.savefig(FIG + "Figure6_Gate_Robustness.png"); plt.close(fig)

# ---------------------------------------------------------------- statistics table for the manuscript
base = S[names[0]]
rows = []
for n in names:
    r = dict(name=n)
    for kk in ("throughput", "kits_A", "kits_B", "jain", "robot_idle", "human_idle", "reduced_frac", "stop_frac", "min_sep_moving", "fragile_to_operator"):
        r[kk] = (float(np.mean(S[n][kk])), float(np.std(S[n][kk], ddof=1)))
    if n != names[0]:
        tt = stats.ttest_ind(S[n]["throughput"], base["throughput"], equal_var=False)
        dcoh = (np.mean(S[n]["throughput"]) - np.mean(base["throughput"])) / math.sqrt(
            (np.var(S[n]["throughput"], ddof=1) + np.var(base["throughput"], ddof=1)) / 2)
        r["p"], r["d"] = float(tt.pvalue), float(dcoh)
    rows.append(r)
json.dump(rows, open("../results/table_scenarios.json", "w"), indent=1)
for r in rows: print(r["name"], {k: (round(v[0], 3) if isinstance(v, tuple) else round(v, 4)) for k, v in r.items() if k != "name"})
