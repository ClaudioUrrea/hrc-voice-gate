"""
Graphical Abstract — Systems (MDPI)
"Giving Operators a Voice Without Giving Up Safety: A Safety-Gated Sociotechnical
 Architecture for Natural-Language Preferences in Human-Robot Collaborative Cells" (revised)
C. Urrea, 2026.

MDPI specification: minimum 560 x 1100 px (HEIGHT x WIDTH) -> landscape banner,
aspect ratio 1100/560 = 1.964. Rendered at 11.0 x 5.6 in, 250 dpi = 2750 x 1400 px
(2.5x the minimum, exact ratio). No bbox_inches='tight': that would break the ratio.
Fonts restricted to the MDPI-approved list. Check marks are drawn as polylines so the
figure never depends on a glyph the font may lack.

Every number comes from the manuscript:
  Table 3 (scenarios), Table 4 (gate decisions), Table 5 (local translator),
  Section 4.2 (error injection), Section 4.1 (spillover).
Nothing is interpolated or estimated.
"""

import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Polygon, FancyArrowPatch
from matplotlib import font_manager
import os, glob

# ----------------------------------------------------------------------------- fonts
# MDPI recommends Times, Arial, Courier, Helvetica, Ubuntu or Calibri.
# Liberation Sans / Nimbus Sans are metric-compatible Arial / Helvetica clones.
CANDIDATES = [
    ('Arial', [r'C:\Windows\Fonts\arial.ttf', r'C:\Windows\Fonts\arialbd.ttf',
               r'C:\Windows\Fonts\ariali.ttf', r'C:\Windows\Fonts\arialbi.ttf',
               '/Library/Fonts/Arial.ttf', '/System/Library/Fonts/Supplemental/Arial.ttf']),
    ('Helvetica', ['/System/Library/Fonts/Helvetica.ttc']),
    ('Liberation Sans', glob.glob('/usr/share/fonts/truetype/liberation/LiberationSans-*.ttf')),
    ('Nimbus Sans', glob.glob('/usr/share/fonts/**/NimbusSans-*.*', recursive=True)),
]
FONT = 'DejaVu Sans'
for name, files in CANDIDATES:
    found = [f for f in files if os.path.isfile(f)]
    if found:
        for f in found:
            font_manager.fontManager.addfont(f)
        FONT = name
        break
print('Font used:', FONT)
matplotlib.rcParams.update({'font.family': FONT, 'mathtext.fontset': 'custom',
                            'mathtext.rm': FONT, 'mathtext.it': FONT + ':italic',
                            'mathtext.bf': FONT + ':bold', 'axes.unicode_minus': True})

# ---------------------------------------------------------------------------- palette
RED   = '#C1121F'    # untrusted / unsafe
GREEN = '#1E7A4C'    # verified / admitted
BLUE  = '#1F4788'    # structure
AMBER = '#B26A00'    # cost
SLATE = '#44546A'    # panel header
GREY  = '#F4F5F7'
EDGE  = '#C9CED6'
DARK  = '#1D2B3A'
MUTED = '#5B6676'
PALE  = '#9AA3B0'

W, H = 11.0, 5.6
fig = plt.figure(figsize=(W, H), dpi=250)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')
ax.add_patch(Rectangle((0, 0), W, H, fc='white', ec='none', zorder=-10))

# --------------------------------------------------------------------------- helpers
def panel(x, y, w, h, title, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.11',
                                fc=GREY, ec=EDGE, lw=1.1, zorder=1))
    ax.add_patch(FancyBboxPatch((x, y + h - 0.42), w, 0.42,
                                boxstyle='round,pad=0.02,rounding_size=0.11',
                                fc=color, ec='none', zorder=2))
    ax.add_patch(Rectangle((x - 0.02, y + h - 0.42), w + 0.04, 0.18, fc=color, ec='none', zorder=2))
    ax.text(x + w / 2, y + h - 0.21, title, ha='center', va='center', color='white',
            fontsize=11.4, fontweight='bold', zorder=3)

def tick(x, y, s=1.0, color=GREEN, lw=1.9):
    """Check mark as a polyline: never depends on a font glyph."""
    ax.plot([x - 0.052 * s, x - 0.014 * s, x + 0.060 * s],
            [y + 0.004 * s, y - 0.042 * s, y + 0.052 * s],
            color=color, lw=lw, solid_capstyle='round', zorder=9)

def vflow(x, y0, y1, color=MUTED, lw=1.5):
    ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle='-|>', mutation_scale=11,
                                 color=color, lw=lw, shrinkA=0, shrinkB=0, zorder=6))

def chip(x, y, w, h, label, fc, ec, fs=8.6, weight='bold', tc=DARK, dashed=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.015,rounding_size=0.06',
                                fc=fc, ec=ec, lw=1.3, zorder=4,
                                linestyle=(0, (3.5, 2.2)) if dashed else '-'))
    ax.text(x + w / 2, y + h / 2, label, ha='center', va='center', fontsize=fs,
            fontweight=weight, color=tc, zorder=7, linespacing=1.15)

# ========================================================================== headline
ax.text(W / 2, H - 0.29,
        'Operator voice without giving up safety: Interpretation by a language model, '
        'authority by a deterministic gate',
        ha='center', va='center', fontsize=13.4, fontweight='bold', color=DARK)

PY, PH = 0.58, 4.46                      # panels span y = 0.58 .. 5.04
TOP = PY + PH - 0.42                     # first usable y inside a panel = 4.62
AX, AW = 0.16, 3.30
BX, BW = 3.72, 3.30
CX, CW = 7.28, 3.56

for x0 in (AX + AW + 0.03, BX + BW + 0.03):
    ax.add_patch(FancyArrowPatch((x0, 2.70), (x0 + 0.20, 2.70),
                                 arrowstyle='simple,head_width=9,head_length=8',
                                 color=PALE, lw=0, zorder=3))

# ============================================== PANEL A — the mechanism
panel(AX, PY, AW, PH, 'Interpretation is separated from authority', SLATE)
cxA = AX + AW / 2

# --- operator utterance (speech bubble; text sits above the bubble in z-order)
ax.add_patch(FancyBboxPatch((AX + 0.22, 4.12), 2.86, 0.44,
                            boxstyle='round,pad=0.02,rounding_size=0.13',
                            fc='white', ec=BLUE, lw=1.3, zorder=4))
ax.add_patch(Polygon([(cxA - 0.10, 4.14), (cxA + 0.10, 4.14), (cxA - 0.01, 3.98)],
                     fc='white', ec=BLUE, lw=1.3, zorder=4))
ax.add_patch(Polygon([(cxA - 0.085, 4.16), (cxA + 0.085, 4.16), (cxA - 0.01, 4.01)],
                     fc='white', ec='white', lw=0, zorder=5))
ax.text(cxA, 4.45, 'Operator B, speaking at terminal B   (EN / ES)',
        ha='center', va='center', fontsize=7.3, color=MUTED, zorder=7)
ax.text(cxA, 4.26, '"Baja la velocidad cerca de mi bandeja"',
        ha='center', va='center', fontsize=9.3, fontweight='bold',
        color=BLUE, style='italic', zorder=7)

vflow(cxA, 3.96, 3.80)

# --- untrusted translator
chip(AX + 0.22, 3.34, 2.86, 0.44, 'Local open-weight LLM   (6 tested)',
     fc='#FBEAEA', ec=RED, fs=9.1, dashed=True)
ax.text(cxA, 3.27, 'Untrusted  ·  Proposes a candidate, never commands',
        ha='center', va='top', fontsize=7.2, color=RED, style='italic', zorder=7)

vflow(cxA, 3.10, 2.94)

# --- candidate in the typed language
ax.text(AX + 0.38, 2.92, 'Candidate in OCL',
        ha='left', va='bottom', fontsize=7.0, color=MUTED, zorder=7)
chip(AX + 0.38, 2.56, 2.54, 0.32, 'speed_cap(B, kit_B, 150 mm/s)',
     fc='#EEF2F9', ec=BLUE, fs=8.5, weight='normal')

vflow(cxA, 2.52, 2.36)

# --- the gate
ax.add_patch(FancyBboxPatch((AX + 0.15, 1.52), 3.00, 0.84,
                            boxstyle='round,pad=0.02,rounding_size=0.08',
                            fc='#E8F3ED', ec=GREEN, lw=2.1, zorder=4))
ax.text(cxA, 2.26, 'DETERMINISTIC ADMISSION GATE', ha='center', va='center',
        fontsize=9.1, fontweight='bold', color=GREEN, zorder=7)
for i, (k, v) in enumerate([('Schema', 'Well-typed'),
                            ('Origin', "Speaker's terminal"),
                            ('Authority', "Speaker's zones"),
                            ('Monotone', 'Tightens only')]):
    col, row = i % 2, i // 2
    xk, yk = AX + 0.33 + col * 1.49, 2.03 - row * 0.29
    tick(xk, yk, s=0.85)
    ax.text(xk + 0.11, yk + 0.012, k, ha='left', va='center', fontsize=8.0,
            fontweight='bold', color=DARK, zorder=7)
    ax.text(xk + 0.11, yk - 0.135, v, ha='left', va='center', fontsize=6.9,
            color=MUTED, zorder=7)

# --- admit / reject
vflow(cxA - 0.70, 1.50, 1.38, color=GREEN)
vflow(cxA + 0.70, 1.50, 1.38, color=RED)
chip(cxA - 1.17, 1.06, 0.94, 0.32, 'Admitted', fc='#DCEFE4', ec=GREEN, fs=8.3, tc=GREEN)
chip(cxA + 0.23, 1.06, 0.94, 0.32, 'Rejected', fc='#FBE4E4', ec=RED, fs=8.3, tc=RED)

# --- the bedrock: SSM sits below the scheduler and nothing above can reach it
ax.add_patch(FancyBboxPatch((AX + 0.15, 0.64), 3.00, 0.34,
                            boxstyle='round,pad=0.015,rounding_size=0.05',
                            fc=DARK, ec='none', zorder=3))
ax.text(cxA, 0.87, 'Admitted requests reach the scheduler', ha='center', va='center',
        fontsize=7.0, color='white', fontweight='bold', zorder=7)
ax.text(cxA, 0.73, 'Speed-and-separation monitoring (after ISO/TS 15066) runs below it',
        ha='center', va='center', fontsize=6.7, color='#AEB9C6', zorder=7)

# ============================================== PANEL B — safety result
panel(BX, PY, BW, PH, 'What the gate guarantees, for any model', GREEN)

ax.text(BX + 0.20, 4.44, 'Gate decisions on the 264-utterance EN/ES benchmark',
        ha='left', va='center', fontsize=8.2, fontweight='bold', color=DARK, zorder=7)

bx0, bw, by0, bh = BX + 0.20, 2.90, 3.94, 0.30
for n, c in [(236, GREEN), (12, PALE), (10, RED), (6, '#7A0C14')]:
    wseg = bw * n / 264
    ax.add_patch(Rectangle((bx0, by0), wseg, bh, fc=c, ec='white', lw=0.8, zorder=4))
    if n >= 20:                       # narrow segments are named in the caption instead
        ax.text(bx0 + wseg / 2, by0 + bh / 2, str(n), ha='center', va='center',
                fontsize=8.3, fontweight='bold', color='white', zorder=7)
    bx0 += wseg
ax.text(BX + 0.20, 3.84,
        'Faithful translator  ·  236 admitted  ·  28 rejected (12 no-op, 10 safety, 6 authority)',
        ha='left', va='top', fontsize=6.0, color=MUTED, zorder=7)
ax.text(BX + 0.20, 3.68, 'No legitimate request refused, no adversarial request admitted',
        ha='left', va='top', fontsize=7.3, color=GREEN, fontweight='bold', zorder=7)

ax.text(BX + 0.20, 3.44, 'Specification violations entering the store, per pass',
        ha='left', va='center', fontsize=8.2, fontweight='bold', color=DARK, zorder=7)

ax2 = fig.add_axes([(BX + 0.48) / W, 1.66 / H, 2.52 / W, 1.54 / H])
ax2.set_facecolor('white')
groups = ['Faithful translator\n(no corruption)', 'Half the fields corrupted\n(50% injection)']
nogate = [16.0, 80.5]
xpos = np.arange(2)
ax2.bar(xpos - 0.19, nogate, width=0.34, color=RED, edgecolor='none', label='No gate')
ax2.bar(xpos + 0.19, [1.0, 1.0], width=0.34, color=GREEN, edgecolor='none', label='With gate')
for i, v in enumerate(nogate):
    ax2.text(i - 0.19, v + 2.2, f'{v:.1f}', ha='center', va='bottom', fontsize=8.8,
             fontweight='bold', color=RED)
    ax2.text(i + 0.19, 3.4, '0', ha='center', va='bottom', fontsize=11.0,
             fontweight='bold', color=GREEN)
ax2.set_xticks(xpos); ax2.set_xticklabels(groups, fontsize=7.2, color=DARK)
ax2.set_ylim(0, 96); ax2.set_yticks([0, 20, 40, 60, 80])
ax2.tick_params(axis='y', labelsize=7.1, length=2.5, color=EDGE)
ax2.tick_params(axis='x', length=0)
for s_ in ('top', 'right'):
    ax2.spines[s_].set_visible(False)
for s_ in ('left', 'bottom'):
    ax2.spines[s_].set_color(EDGE)
ax2.legend(fontsize=7.1, frameon=False, loc='upper left', handlelength=1.0,
           handleheight=0.8, borderpad=0.0, labelspacing=0.22)

ax.add_patch(FancyBboxPatch((BX + 0.20, 0.66), 2.90, 0.68,
                            boxstyle='round,pad=0.02,rounding_size=0.07',
                            fc='white', ec=GREEN, lw=1.4, zorder=4))
ax.text(BX + 0.33, 1.19, 'Six local models: fidelity varies, guarantees hold',
        ha='left', va='center', fontsize=8.0, fontweight='bold', color=DARK, zorder=7)
ax.text(BX + 0.33, 0.90,
        'Exact intent 41-81% across models, under 1.2 s per request;\n'
        'none of 12,936 translations (49 runs) admitted a violation.',
        ha='left', va='center', fontsize=7.3, color=MUTED, zorder=7, linespacing=1.35)

# ============================================== PANEL C — cost result
panel(CX, PY, CW, PH, 'Preference cost is not what it sounds like', AMBER)

ax.text(CX + 0.20, 4.46, 'Every preference priced on two axes at once',
        ha='left', va='center', fontsize=8.5, fontweight='bold', color=DARK, zorder=7)
ax.text(CX + 0.20, 4.29,
        'S1 Fragile parts handled by the operator  ·  S2 Speed cap  ·  S3 Exclusion\n'
        'S4 Ten-minute break  ·  S5 Priority  ·  S6 S1, S2 and S4 at once',
        ha='left', va='top', fontsize=6.5, color=MUTED, zorder=7, linespacing=1.3)

ax3 = fig.add_axes([(CX + 0.58) / W, 2.46 / H, 2.82 / W, 1.56 / H])
ax3.set_facecolor('white')
# scenario, throughput lost (% of the 29.23 kits/h baseline), protective stop (% of shift)
scen = [('S1', 1.3, 2.59, RED), ('S2', 6.4, 0.40, BLUE), ('S3', 3.5, 0.34, PALE),
        ('S4', 8.6, 0.30, BLUE), ('S5', 0.9, 0.36, PALE), ('S6', 14.6, 2.08, '#7A0C14')]
ax3.axhline(1.0, color=EDGE, lw=1.0, ls=(0, (4, 3)), zorder=1)
ax3.text(16.1, 0.58, 'Baseline', ha='right', va='center', fontsize=6.4, color=MUTED)
for lab, dT, stop, col in scen:
    ax3.scatter([dT], [stop / 0.37], s=150, color=col, edgecolor='white', lw=1.3, zorder=5)
    ax3.text(dT, stop / 0.37, lab, ha='center', va='center', fontsize=6.3,
             fontweight='bold', color='white', zorder=6)
ax3.annotate('', xy=(1.75, 6.75), xytext=(3.4, 5.6),
             arrowprops=dict(arrowstyle='->', lw=1.3, color=RED,
                             connectionstyle='arc3,rad=0.22'))
ax3.text(3.6, 5.6, 'Cheapest in throughput,\nyet stops rise sevenfold', ha='left',
         va='center', fontsize=7.2, color=RED, fontweight='bold', linespacing=1.3)
ax3.text(11.4, 2.05, 'Expensive,\nno extra stops', ha='center', va='bottom',
         fontsize=6.9, color=BLUE, linespacing=1.25)
ax3.annotate('', xy=(8.8, 1.35), xytext=(10.6, 2.0),
             arrowprops=dict(arrowstyle='->', lw=1.0, color=BLUE,
                             connectionstyle='arc3,rad=-0.2'))
ax3.set_xlim(-1.3, 16.4); ax3.set_ylim(0, 8.0)
ax3.set_xticks([0, 5, 10, 15]); ax3.set_xticklabels(['0', '5', '10', '15'], fontsize=7.1)
ax3.set_yticks([0, 2, 4, 6]); ax3.set_yticklabels(['0', '2x', '4x', '6x'], fontsize=7.1)
ax3.set_xlabel('Throughput lost (% of baseline)', fontsize=7.6, color=DARK, labelpad=1.5)
ax3.set_ylabel('Protective stops\n(multiple of baseline)', fontsize=7.6, color=DARK,
               linespacing=1.15, labelpad=1.5)
for s_ in ('top', 'right'):
    ax3.spines[s_].set_visible(False)
for s_ in ('left', 'bottom'):
    ax3.spines[s_].set_color(EDGE)
ax3.tick_params(length=2.5, color=EDGE)

# --- spillover: drawn in figure coordinates, no nested axes, no overlap
ax.add_patch(FancyBboxPatch((CX + 0.20, 0.66), CW - 0.40, 1.42,
                            boxstyle='round,pad=0.02,rounding_size=0.07',
                            fc='white', ec=AMBER, lw=1.4, zorder=4))
ax.text(CX + 0.34, 1.92, 'A preference does not stay with the operator',
        ha='left', va='center', fontsize=8.0, fontweight='bold', color=DARK, zorder=7)
ax.text(CX + 0.34, 1.73, 'S2: Cap on zone B only.  Kits/h, baseline to capped.',
        ha='left', va='center', fontsize=6.9, color=MUTED, style='italic', zorder=7)

def track(v):                                   # 14.0 -> 9.06 ,  15.6 -> 10.26
    return CX + 1.85 + (v - 13.4) * (1.10 / 1.5)

for i, (lab, v0, v1, col) in enumerate(
        [('Operator A  (asked for nothing)', 14.53, 13.63, RED),
         ('Operator B  (asked for the cap)', 14.70, 13.73, MUTED)]):
    y = 1.40 - i * 0.34
    ax.text(CX + 0.34, y, lab, ha='left', va='center', fontsize=7.1, color=DARK, zorder=7)
    ax.plot([track(v1), track(v0)], [y, y], color=col, lw=2.4,
            solid_capstyle='round', zorder=6)
    ax.scatter([track(v0)], [y], s=40, color='white', edgecolor=PALE, lw=1.4, zorder=7)
    ax.scatter([track(v1)], [y], s=52, color=col, edgecolor='white', lw=1.2, zorder=8)
    ax.text(track(v0) + 0.09, y, '\u2212%.1f%%' % (100 * (v0 - v1) / v0), ha='left',
            va='center', fontsize=7.8, fontweight='bold', color=col, zorder=7)

# ============================================================================ footer
ax.add_patch(FancyBboxPatch((0.16, 0.09), W - 0.32, 0.36,
                            boxstyle='round,pad=0.01,rounding_size=0.07',
                            fc=DARK, ec='none', zorder=4))
ax.text(W / 2, 0.27,
        'Doosan M1013 two-operator collaborative cell (RoboDK)   ·   Fixed and dynamic requests, 30-shift runs'
        '   ·   264-utterance EN/ES benchmark   ·   6 local LLMs   ·   STPA   ·   Open code and data',
        ha='center', va='center', color='white', fontsize=7.4, zorder=7)

# ============================================================================= output
fig.savefig('../figures/Graphical_Abstract.png', dpi=250, facecolor='white')
from PIL import Image                                  # MDPI: RGB, no transparency
im = Image.open('../figures/Graphical_Abstract.png').convert('RGB')
im.save('../figures/Graphical_Abstract.png', dpi=(250, 250))
im.save('../figures/Graphical_Abstract.tiff', compression='tiff_lzw', dpi=(250, 250))
print('size: %d x %d px  |  ratio %.3f  (MDPI target 1.964, minimum 1100 x 560)'
      % (im.size[0], im.size[1], im.size[0] / im.size[1]))
