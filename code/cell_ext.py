# -*- coding: utf-8 -*-
"""
cell_ext.py - Extensions of the cell model used in the revised manuscript (round 1).

All extensions are opt-in. With default arguments CellX reproduces Cell exactly (same random
stream, same metrics), so the original scenario results remain valid; the extensions add:

  * time accounting for each operator (assembling, reaching into the tray, fetching from the
    handover, waiting, absent), so that "human idle" can be decomposed;
  * an approximate kinematic model of the Doosan M1013 (elbow-up, vertical tool) that places
    the shoulder, elbow, wrist and gripper along the Cartesian TCP surrogate, so that the
    separation can be computed from every link to every tracked body point;
  * an independent safety oracle: at each step it checks whether any point of the arm is closer
    to a body point than the ISO/TS 15066 protective distance S_p evaluated with that point's
    own speed - this oracle does not use the gate or the OCL;
  * a link-aware SSM variant that bounds the speed of every point of the arm, not only the TCP;
  * measurement latency and acceleration limits on the commanded speed (sensitivity study);
  * time-varying constraints (arrival and expiry) and a separation between what an operator
    physically does (their intent) and what the scheduler has admitted (the constraint store),
    which is what exposes mode confusion when the two differ.
"""
import math
import numpy as np
from cell_sim import (Cell, Operator, DT, DROP, KIT_OF, KIT_RECIPE, V_NOM, SSM, ZONE_R, ssm_speed)

# ------------------------------------------------------------------ approximate arm model
BASE_TOP = np.array([0.0, 0.0, 780.0])
SHOULDER_H = 152.5          # J1-J2 offset (M1013 d1)
L_UP, L_FORE = 620.0, 559.0 # upper arm, forearm to wrist centre (M1013 a2, d4)
TOOL_LEN = 121.0 + 150.0    # wrist centre to flange (d6) + gripper, tool pointing down


def arm_points(tcp):
    """Shoulder, elbow, wrist centre and TCP for an elbow-up posture with a vertical tool."""
    S = BASE_TOP + np.array([0.0, 0.0, SHOULDER_H])
    W = tcp + np.array([0.0, 0.0, TOOL_LEN])
    r = math.hypot(W[0], W[1]); th = math.atan2(W[1], W[0])
    dz = W[2] - S[2]; L = math.hypot(r, dz)
    L = min(L, L_UP + L_FORE - 1e-6)
    cos_b = (L_UP ** 2 + L ** 2 - L_FORE ** 2) / (2 * L_UP * L)
    b = math.acos(max(-1.0, min(1.0, cos_b)))
    phi = math.atan2(dz, r) + b                       # elbow up: rotate towards +z
    er, ez = L_UP * math.cos(phi), L_UP * math.sin(phi)
    E = np.array([er * math.cos(th), er * math.sin(th), S[2] + ez])
    return [BASE_TOP, S, E, W, tcp]


def seg_dist(p, a, b):
    ab = b - a; t = float(np.dot(p - a, ab) / max(1e-9, np.dot(ab, ab)))
    t = min(1.0, max(0.0, t)); return float(np.linalg.norm(p - (a + t * ab))), t


def sp_distance(vr, vh, Tr=None):
    Tr = SSM["Tr"] if Tr is None else Tr
    return vh * (Tr + SSM["Ts"]) + vr * Tr + vr * vr / (2 * SSM["a"]) + SSM["C"] + SSM["Zd"] + SSM["Zr"]


# ------------------------------------------------------------------ operators with accounting
class OperatorX(Operator):
    """Same behaviour and random stream as Operator, plus time accounting and, optionally,
    the physical behaviour implied by an exclusion the operator asked for (hand in own tray)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.acct = dict(assemble=0.0, reach_tray=0.0, fetch=0.0, wait=0.0, absent=0.0, rearrange=0.0)
        self.reach_kind = None
        self.rearranging = False            # set by the cell during an intended exclusion

    def step(self, dt, kit_complete, handover_has):
        target = self.rest
        if not self.present:
            self.acct["absent"] += dt
            target = self.shoulder + np.array([600.0, 0, -400.0])
        elif self.rearranging and self.reach_target is None:
            self.acct["rearrange"] += dt
            target = self.tray + np.array([0.0, 0.0, 30.0])
            if self.state == "assemble":    # rearranging interrupts assembly
                pass
        elif self.reach_target is not None:
            self.acct["fetch" if self.reach_kind == "fetch" else "reach_tray"] += dt
            target = self.reach_target; self.reach_t -= dt
            if self.reach_t <= 0: self.reach_target = None; self.reach_kind = None
        elif self.state == "assemble":
            self.acct["assemble"] += dt
            target = self.fixture
            self.t_left -= dt; self.busy_t += dt
            if self.rng.random() < dt / 8.0:
                self.reach_target, self.reach_t = self.tray, 1.5; self.reach_kind = "tray"
            if self.t_left <= 0:
                self.state = "wait"; self.kits_done += 1
                return "kit_done"
        else:
            if handover_has and self.pending_fetch > 0:
                self.acct["fetch"] += dt
                self.reach_target, self.reach_t = DROP["handover"] + np.array([140.0, 0, -80.0]), 2.5
                self.reach_kind = "fetch"
                self.pending_fetch -= 1
                return "fetched"
            if kit_complete:
                self.acct["assemble"] += dt
                self.state, self.t_left = "assemble", self.rng.triangular(90, 150, 120)
            else:
                self.acct["wait"] += dt
                self.idle_t += dt
        step = target - self.hand; dist = np.linalg.norm(step); vmax = 800.0
        mv = min(dist, vmax * dt)
        if dist > 1e-6: self.hand = self.hand + step / dist * mv
        self.hand_v = mv / dt
        return None


# ------------------------------------------------------------------ cell
class CellX(Cell):
    def __init__(self, constraints=(), seed=0, horizon=3600.0, link_aware=False, oracle=True,
                 latency=0.0, accel=None, Tr_model=None, intents=None, dynamic=False):
        super().__init__(constraints, seed, horizon)
        self.ops = {"Op_A": OperatorX("Op_A", -420.0, self.rng), "Op_B": OperatorX("Op_B", 420.0, self.rng)}
        self.link_aware, self.oracle_on = link_aware, oracle
        self.latency, self.accel = latency, accel
        self.Tr_model = SSM["Tr"] if Tr_model is None else Tr_model
        self.dynamic = dynamic               # constraints carry t_on / t_off
        self.intents = list(intents) if intents is not None else None
        self.v_act = 0.0
        self.hist = []                       # (t, list of body points, vh) for latency
        self.prev_pts = None
        self.x = dict(oracle_viol=0.0, oracle_viol_tcp_law=0.0, min_sep_link_moving=1e9,
                      max_point_speed=0.0, max_kappa=0.0, believed_excl_exposure=0.0,
                      believed_excl_stops=0.0, believed_cap_overspeed=0.0, interference=0.0,
                      store_match=0.0, intent_time=0.0)

    # ---------------- constraints with arrival / expiry
    def _live(self, c, t):
        if self.dynamic:
            return c.t_on <= t < c.t_off
        return True

    def active(self, ctype, t=None):
        t = self.t if t is None else t
        out = []
        for c in self.cons:
            if c.ctype != ctype or not self._live(c, t): continue
            if c.ctype in ("pause", "exclusion"):
                if c.start_in <= t < c.start_in + c.duration: out.append(c)
            else:
                out.append(c)
        return out

    def excluded(self, zone, horizon=30.0):
        for c in self.cons:
            if c.ctype == "exclusion" and c.zone == zone and (not self.dynamic or c.t_on <= self.t) and \
               c.start_in - horizon <= self.t < c.start_in + c.duration:
                return True
        return False

    def _intent_active(self, ctype, op=None):
        if self.intents is None: return []
        out = []
        for c in self.intents:
            if c.ctype != ctype or (op and c.operator != op): continue
            if not (c.t_on <= self.t < c.t_off): continue
            if ctype in ("pause", "exclusion") and not (c.start_in <= self.t < c.start_in + c.duration): continue
            out.append(c)
        return out

    # ---------------- main step (same structure as Cell.step, with extensions)
    def step(self):
        if self.intents is None:
            for c in self.active("pause"): self.ops[c.operator].present = False
            for name, o in self.ops.items():
                if not any(c.operator == name for c in self.active("pause")): o.present = True
        else:                                 # physical behaviour follows what the operator meant
            for name, o in self.ops.items():
                o.present = not self._intent_active("pause", name)
                o.rearranging = any(c.zone == KIT_OF[name] for c in self._intent_active("exclusion", name))
        for name, o in self.ops.items():
            kit = KIT_OF[name]
            complete = sorted(self.kit[kit]) == sorted(KIT_RECIPE)
            ev = o.step(DT, complete and o.present, name in self.handover) if o.present else o.step(DT, False, False)
            if ev == "kit_done": self.kit[kit] = []
            if ev == "fetched":
                self.handover.remove(name); self.kit[kit].append("fragile")

        body_now = [(o.name, p, max(SSM["vh_floor"], o.hand_v)) for o in self.ops.values() for p in o.points()]
        self.hist.append((self.t, body_now))
        lag = int(round(self.latency / DT))
        body_meas = self.hist[max(0, len(self.hist) - 1 - lag)][1]
        if len(self.hist) > 50: self.hist.pop(0)

        # TCP-based separation (as in the original model), on the measured body points
        d, vh = 1e9, SSM["vh_floor"]
        for _, p, v in body_meas:
            dd = float(np.linalg.norm(self.tcp - p))
            if dd < d: d, vh = dd, v
        v_ssm = ssm_speed_tr(d, vh, self.Tr_model)

        pts = arm_points(self.tcp)
        segs = list(zip(pts[:-1], pts[1:]))
        if self.link_aware or self.oracle_on:
            # speed ratio of every arm point to the TCP for the current direction of motion
            kap = self._kappa(pts)
        if self.link_aware:
            lim = v_ssm
            for _, p, v in body_meas:
                for (a, b), (ka, kb) in zip(segs, zip(kap[:-1], kap[1:])):
                    ds, tt = seg_dist(p, a, b)
                    k = max(1e-3, ka + tt * (kb - ka))
                    lim = min(lim, ssm_speed_tr(ds, v, self.Tr_model) / k)
            v_ssm = lim

        v_cap = V_NOM
        for c in self.active("speed_cap"):
            if np.linalg.norm(self.tcp[:2] - DROP[c.zone][:2]) < ZONE_R: v_cap = min(v_cap, c.v_max)
        self.m["min_sep"] = min(self.m["min_sep"], d)
        self._d_now = d

        self._tcp_prev = self.tcp.copy()
        if not self.path:
            self.job = self.next_job()
            if self.job: self.plan(self.job)
        moving = False
        if not self.path:
            self.m["idle"] += DT; vcmd = 0.0
        else:
            tgt, vseg = self.path[0]
            if isinstance(tgt, str):
                self.grip_t = getattr(self, "grip_t", 0.0) + DT; vcmd = 0.0; self.m["move"] += DT
                if self.grip_t >= 0.8:
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
                if self.accel is not None:
                    up = self.v_act + self.accel * DT
                    vcmd = min(vcmd, up)
                    # braking is bounded by the deceleration assumed in S_p
                    vcmd = max(vcmd, self.v_act - SSM["a"] * DT) if vcmd < self.v_act else vcmd
                if vcmd >= 1.0: self.m["min_sep_moving"] = min(self.m.get("min_sep_moving", 1e9), self._d_now)
                if vcmd < 1.0: self.m["stop"] += DT
                elif vcmd < vseg - 1e-6: self.m["reduced"] += DT; self.m["move"] += DT
                else: self.m["move"] += DT
                step = tgt - self.tcp; dist = np.linalg.norm(step); mv = min(dist, vcmd * DT)
                if dist > 1e-9: self.tcp = self.tcp + step / dist * mv
                if dist - mv < 0.5: self.path.pop(0)
                moving = vcmd >= 1.0
                for c in self.active("exclusion"):
                    if np.linalg.norm(self.tcp[:2] - DROP[c.zone][:2]) < 120: self.m["excl_viol"] += DT
        self.v_act = vcmd

        if self.oracle_on and moving:
            self._oracle(pts, segs, kap, body_now, vcmd)
        if self.intents is not None:
            self._mode_confusion(body_now, vcmd, moving)
        if int(round(self.t / DT)) % getattr(self, "log_every", 5) == 0:
            self.log["t"].append(self.t); self.log["d"].append(d); self.log["v"].append(vcmd); self.log["vcap"].append(min(v_cap, V_NOM))
        self.t += DT

    def _kappa(self, pts):
        if not self.path or isinstance(self.path[0][0], str):
            return [0.0, 0.0, 1.0, 1.0, 1.0]
        tgt = self.path[0][0]; dvec = tgt - self.tcp; n = np.linalg.norm(dvec)
        if n < 1e-6: return [0.0, 0.0, 1.0, 1.0, 1.0]
        h = 1.0
        p2 = arm_points(self.tcp + dvec / n * h)
        return [float(np.linalg.norm(b - a)) / h for a, b in zip(pts, p2)]

    def _oracle(self, pts, segs, kap, body_now, vcmd):
        """Independent check: every arm point at least S_p(own speed) from every body point."""
        viol = False; dmin = 1e9
        for _, p, vh in body_now:
            for (a, b), (ka, kb) in zip(segs, zip(kap[:-1], kap[1:])):
                ds, tt = seg_dist(p, a, b)
                vp = vcmd * (ka + tt * (kb - ka))
                dmin = min(dmin, ds)
                if ds < sp_distance(vp, vh, SSM["Tr"]) - 1e-6 and vp > 1.0:
                    viol = True
        self.x["min_sep_link_moving"] = min(self.x["min_sep_link_moving"], dmin)
        self.x["max_kappa"] = max(self.x["max_kappa"], max(kap))
        self.x["max_point_speed"] = max(self.x["max_point_speed"], vcmd * max(kap))
        if viol: self.x["oracle_viol"] += DT

    def _mode_confusion(self, body_now, vcmd, moving):
        # exposure: an operator is rearranging their tray because they asked for an exclusion,
        # and the robot is moving inside that tray zone (the store does not protect them)
        for name, o in self.ops.items():
            if o.rearranging:
                tray = DROP[KIT_OF[name]]
                near = np.linalg.norm(self._tcp_prev[:2] - tray[:2]) < ZONE_R
                if near and moving: self.x["believed_excl_exposure"] += DT
                if near and not moving and self.path and not isinstance(self.path[0][0], str):
                    self.x["believed_excl_stops"] += DT
            for c in self._intent_active("speed_cap", name):
                if np.linalg.norm(self._tcp_prev[:2] - DROP[c.zone][:2]) < ZONE_R and vcmd > c.v_max + 1e-6:
                    self.x["believed_cap_overspeed"] += DT
        # interference: a stored constraint that touches a zone or a person other than its speaker
        for c in self.cons:
            if not self._live(c, self.t): continue
            if getattr(c, "speaker", c.operator) != c.operator or \
               (c.ctype in ("speed_cap", "exclusion") and c.zone not in
                    (KIT_OF[getattr(c, "speaker", c.operator)], "handover")):
                if c.ctype not in ("pause", "exclusion") or c.start_in <= self.t < c.start_in + (c.duration or 0):
                    self.x["interference"] += DT

    def metrics(self):
        m = super().metrics()
        m.update(self.x)
        for name, o in self.ops.items():
            for k, v in o.acct.items(): m[f"{k}_{name[-1]}"] = v / self.T
        return m


def ssm_speed_tr(d, vh, Tr):
    K = vh * (Tr + SSM["Ts"]) + SSM["C"] + SSM["Zd"] + SSM["Zr"]
    if d <= K: return 0.0
    a = SSM["a"]
    return a * (-Tr + math.sqrt(Tr * Tr + 2.0 * (d - K) / a))


# ------------------------------------------------------------------ joint-interpolated transits
def _arc(p, q, step_deg=5.0):
    """Waypoints from transit pose p to transit pose q by base rotation at constant radius,
    which is how the retracted transit moves of the RoboDK programme (joint moves) behave."""
    a0, a1 = math.atan2(p[1], p[0]), math.atan2(q[1], q[0])
    da = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
    n = max(1, int(abs(math.degrees(da)) / step_deg))
    r, z = math.hypot(q[0], q[1]), q[2]
    return [np.array([r * math.cos(a0 + da * k / n), r * math.sin(a0 + da * k / n), z]) for k in range(1, n + 1)]


class CellJ(CellX):
    """CellX whose transit moves follow the base rotation (joint space) instead of a chord."""

    def plan(self, job):
        super().plan(job)
        wp = []; cur = self.tcp.copy()
        for tgt, v in self.path:
            if isinstance(tgt, str):
                wp.append((tgt, v)); continue
            if abs(tgt[2] - 1500.0) < 1.0 and abs(cur[2] - 1500.0) < 1.0:
                wp += [(p, v) for p in _arc(cur, tgt)]
            else:
                wp.append((tgt, v))
            cur = tgt
        self.path = wp
