# -*- coding: utf-8 -*-
"""
cell_sim.py - Time-stepped simulation of the collaborative kitting cell.

Geometry is taken from the RoboDK cell (build_cell_systems.py / cell_layout.json).
The robot follows the validated pick/place sequence as a Cartesian TCP surrogate at the
collaborative speed; its speed is scaled online by speed-and-separation monitoring (SSM,
ISO/TS 15066) and by the operator constraints admitted by the safety gate.
"""
import math, random
import numpy as np
from constraints import Constraint

DT = 0.1
BASE = np.array([0.0, 0.0, 780.0])
SRC = {"fragile": [np.array([100 + 150*i, -800.0, 940.0]) for i in range(3)],
       "standard": [np.array([90 + 160*i, 800.0, 1010.0]) for i in range(3)],
       "housing": [np.array([-700.0, -600 + 400*i, 925.0]) for i in range(4)]}
DROP = {"kit_A": np.array([870.0, -420.0, 940.0]), "kit_B": np.array([870.0, 420.0, 940.0]),
        "handover": np.array([600.0, 0.0, 1000.0])}
KIT_OF = {"Op_A": "kit_A", "Op_B": "kit_B"}
KIT_RECIPE = ("fragile", "fragile", "housing", "housing", "standard", "standard")

V_NOM, V_APP, APP = 500.0, 150.0, 150.0     # mm/s cruise, mm/s approach, mm approach height
GRIP_T = 0.8                                # s per grasp/release
SSM = dict(Tr=0.10, Ts=0.30, a=1500.0, C=120.0, Zd=50.0, Zr=20.0, vh_floor=200.0)
ZONE_R = 350.0                              # radius of a zone for speed caps / exclusions


def ssm_speed(d, vh):
    """Largest TCP speed that keeps d >= S_p(v) (ISO/TS 15066, braking B = v^2/2a)."""
    K = vh * (SSM["Tr"] + SSM["Ts"]) + SSM["C"] + SSM["Zd"] + SSM["Zr"]
    if d <= K:
        return 0.0
    a, Tr = SSM["a"], SSM["Tr"]
    return a * (-Tr + math.sqrt(Tr * Tr + 2.0 * (d - K) / a))


def transit(p):
    th = math.atan2(p[1], p[0])
    return np.array([350*math.cos(th), 350*math.sin(th), 1500.0])


class Operator:
    def __init__(self, name, y, rng):
        self.name, self.y, self.rng = name, y, rng
        self.shoulder = np.array([1500.0, y, 1400.0])
        self.rest = np.array([1180.0, y, 980.0])
        self.fixture = np.array([1150.0, y, 900.0])
        self.tray = np.array([900.0, y, 900.0])
        self.hand = self.rest.copy(); self.hand_v = 0.0
        self.state, self.t_left = "wait", 0.0
        self.reach_target, self.reach_t = None, 0.0
        self.pending_fetch = 0          # fragile parts waiting at the handover for this operator
        self.present = True
        self.kits_done, self.idle_t, self.busy_t = 0, 0.0, 0.0

    def step(self, dt, kit_complete, handover_has):
        target = self.rest
        if not self.present:
            target = self.shoulder + np.array([600.0, 0, -400.0])   # away from the table
        elif self.reach_target is not None:
            target = self.reach_target; self.reach_t -= dt
            if self.reach_t <= 0: self.reach_target = None
        elif self.state == "assemble":
            target = self.fixture
            self.t_left -= dt; self.busy_t += dt
            if self.rng.random() < dt / 8.0:                 # occasional reach into the tray
                self.reach_target, self.reach_t = self.tray, 1.5
            if self.t_left <= 0:
                self.state = "wait"; self.kits_done += 1
                return "kit_done"
        else:
            if handover_has and self.pending_fetch > 0:     # fetch a fragile part from the handover
                self.reach_target, self.reach_t = DROP["handover"] + np.array([140.0, 0, -80.0]), 2.5
                self.pending_fetch -= 1
                return "fetched"
            if kit_complete:
                self.state, self.t_left = "assemble", self.rng.triangular(90, 150, 120)
            else:
                self.idle_t += dt
        step = target - self.hand; dist = np.linalg.norm(step); vmax = 800.0
        mv = min(dist, vmax * dt)
        if dist > 1e-6: self.hand = self.hand + step / dist * mv
        self.hand_v = mv / dt
        return None

    def points(self):
        torso = np.array([1500.0, self.y, 1230.0]) if self.present else np.array([2200.0, self.y, 1230.0])
        return [self.hand, torso]


class Cell:
    def __init__(self, constraints=(), seed=0, horizon=3600.0):
        self.rng = random.Random(seed); self.T = horizon; self.t = 0.0
        self.ops = {"Op_A": Operator("Op_A", -420.0, self.rng), "Op_B": Operator("Op_B", 420.0, self.rng)}
        self.kit = {k: [] for k in ("kit_A", "kit_B")}
        self.handover = []                               # parts waiting for pickup (owner names)
        self.cons = list(constraints)
        self.tcp = transit(np.array([500.0, 0, 0])); self.path = []; self.job = None
        self.log = dict(t=[], d=[], v=[], vcap=[])
        self.m = dict(move=0.0, reduced=0.0, stop=0.0, idle=0.0, min_sep=1e9, parts=0, fragile_handover=0,
                      fragile_total=0, excl_viol=0.0, cap_viol=0.0)

    # ------------------------------------------------------------- constraints
    def active(self, ctype, t=None):
        t = self.t if t is None else t
        out = []
        for c in self.cons:
            if c.ctype != ctype: continue
            if c.ctype in ("pause", "exclusion"):
                if c.start_in <= t < c.start_in + c.duration: out.append(c)
            else:
                out.append(c)
        return out

    def handling_agent(self, op, part):
        for c in self.active("handling"):
            if c.operator == op and c.part_class == part: return c.agent
        return "robot"

    def excluded(self, zone, horizon=30.0):
        """True if zone is excluded now or within the look-ahead needed to finish a job."""
        for c in self.cons:
            if c.ctype == "exclusion" and c.zone == zone and \
               c.start_in - horizon <= self.t < c.start_in + c.duration:
                return True
        return False

    # ------------------------------------------------------------- robot jobs
    def next_job(self):
        pri = [c.operator for c in self.active("priority")]
        cands = []
        for op, kit in KIT_OF.items():
            o = self.ops[op]
            if o.state == "assemble": continue
            have = list(self.kit[kit]) + ["fragile"] * (o.pending_fetch + sum(1 for h in self.handover if h == op))
            need = list(KIT_RECIPE)
            for p in have:
                if p in need: need.remove(p)
            if not need: continue
            for part in dict.fromkeys(need):
                drop = "handover" if (part == "fragile" and self.handling_agent(op, part) == "operator") else kit
                if self.excluded(drop): continue
                cands.append((0 if op in pri else 1, len(need), self.rng.random(), op, part, drop))
        if not cands: return None
        cands.sort()
        _, _, _, op, part, drop = cands[0]
        return dict(op=op, part=part, drop=drop)

    def plan(self, job):
        src = self.rng.choice(SRC[job["part"]]); dst = DROP[job["drop"]]
        up = np.array([0, 0, APP])
        wp = [(transit(src), V_NOM), (src + up, V_NOM), (src, V_APP), ("grip", 0), (src + up, V_APP),
              (transit(src), V_NOM), (transit(dst), V_NOM), (dst + up, V_NOM), (dst, V_APP), ("release", 0),
              (dst + up, V_APP), (transit(dst), V_NOM)]
        self.path = wp; job["dst"] = dst

    # ------------------------------------------------------------- main step
    def step(self):
        for c in self.active("pause"):
            self.ops[c.operator].present = False
        for name, o in self.ops.items():
            if not any(c.operator == name for c in self.active("pause")): o.present = True
        for name, o in self.ops.items():
            kit = KIT_OF[name]
            complete = sorted(self.kit[kit]) == sorted(KIT_RECIPE)
            ev = o.step(DT, complete and o.present, name in self.handover) if o.present else o.step(DT, False, False)
            if ev == "kit_done": self.kit[kit] = []
            if ev == "fetched":
                self.handover.remove(name); self.kit[kit].append("fragile")
        # separation and SSM speed
        d, vh = 1e9, SSM["vh_floor"]
        for o in self.ops.values():
            for p in o.points():
                dd = float(np.linalg.norm(self.tcp - p))
                if dd < d: d, vh = dd, max(SSM["vh_floor"], o.hand_v)
        v_ssm = ssm_speed(d, vh)
        v_cap = V_NOM
        for c in self.active("speed_cap"):
            if np.linalg.norm(self.tcp[:2] - DROP[c.zone][:2]) < ZONE_R: v_cap = min(v_cap, c.v_max)
        self.m["min_sep"] = min(self.m["min_sep"], d)
        self._d_now = d
        # robot motion
        if not self.path:
            self.job = self.next_job()
            if self.job: self.plan(self.job)
        if not self.path:
            self.m["idle"] += DT; vcmd = 0.0
        else:
            tgt, vseg = self.path[0]
            if isinstance(tgt, str):
                self.grip_t = getattr(self, "grip_t", 0.0) + DT; vcmd = 0.0; self.m["move"] += DT
                if self.grip_t >= GRIP_T:
                    self.grip_t = 0.0; self.path.pop(0)
                    if tgt == "release":
                        j = self.job; self.m["parts"] += 1
                        if j["part"] == "fragile":
                            self.m["fragile_total"] += 1
                            if j["drop"] == "handover": self.m["fragile_handover"] += 1
                        if j["drop"] == "handover":
                            self.handover.append(j["op"]); self.ops[j["op"]].pending_fetch += 1
                        else:
                            self.kit[j["drop"]].append(j["part"])
            else:
                vcmd = min(vseg, v_cap, v_ssm)
                if vcmd >= 1.0: self.m["min_sep_moving"] = min(self.m.get("min_sep_moving", 1e9), self._d_now)
                if vcmd < 1.0: self.m["stop"] += DT
                elif vcmd < vseg - 1e-6: self.m["reduced"] += DT; self.m["move"] += DT
                else: self.m["move"] += DT
                step = tgt - self.tcp; dist = np.linalg.norm(step); mv = min(dist, vcmd * DT)
                if dist > 1e-9: self.tcp = self.tcp + step / dist * mv
                if dist - mv < 0.5: self.path.pop(0)
                # oracle checks (should stay at zero: constraints are enforced by construction)
                for c in self.active("exclusion"):
                    if np.linalg.norm(self.tcp[:2] - DROP[c.zone][:2]) < 120: self.m["excl_viol"] += DT
        if int(round(self.t / DT)) % getattr(self, "log_every", 5) == 0:
            self.log["t"].append(self.t); self.log["d"].append(d); self.log["v"].append(vcmd); self.log["vcap"].append(min(v_cap, V_NOM))
        self.t += DT

    def run(self):
        while self.t < self.T: self.step()
        return self.metrics()

    def metrics(self):
        m = dict(self.m); T = self.T
        kA, kB = self.ops["Op_A"].kits_done, self.ops["Op_B"].kits_done
        m["kits_A"], m["kits_B"] = kA, kB
        m["throughput"] = (kA + kB) * 3600.0 / T
        wA = self.ops["Op_A"].idle_t / T; wB = self.ops["Op_B"].idle_t / T
        m["wait_A"], m["wait_B"] = wA, wB
        x = np.array([kA, kB], float)
        m["jain"] = float(x.sum() ** 2 / (2 * (x ** 2).sum())) if x.sum() > 0 else 1.0
        m["robot_idle"] = self.m["idle"] / T
        m["stop_frac"], m["reduced_frac"] = self.m["stop"] / T, self.m["reduced"] / T
        m["human_idle"] = (wA + wB) / 2
        m["fragile_to_operator"] = self.m["fragile_handover"] / max(1, self.m["fragile_total"])
        return m
