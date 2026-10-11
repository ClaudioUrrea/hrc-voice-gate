# -*- coding: utf-8 -*-
"""
constraints.py - Operator Constraint Language (OCL) and deterministic safety gate.

An operator utterance is translated (by a local LLM, see llm_translate.py) into a JSON
object that must conform to OCL. Whatever the translator produces, only the safety gate
decides what reaches the scheduler: it enforces schema, authority scoping and safety
invariants, so translation errors can degrade preference fidelity but cannot weaken the
protective behaviour of the cell.
"""
from dataclasses import dataclass, field

OPERATORS = ("Op_A", "Op_B")
PART_CLASSES = ("fragile", "standard", "housing")
ZONES = {"Op_A": ("kit_A", "handover"), "Op_B": ("kit_B", "handover")}
CONSTRAINT_TYPES = ("handling", "speed_cap", "exclusion", "pause", "priority", "none")

V_COLLAB_MAX = 500.0      # mm/s nominal cruise speed of the cell (SSM scales it online)
V_CAP_MIN = 50.0          # below this, a pause is the appropriate request
EXCL_MAX_S = 300.0        # longest exclusion an operator may impose
PAUSE_MAX_S = 1200.0      # longest pause that can be declared (20 min)
PAUSE_LEAD_MAX_S = 1800.0


@dataclass
class Constraint:
    ctype: str
    operator: str
    part_class: str = None
    agent: str = None
    zone: str = None
    v_max: float = None
    start_in: float = 0.0
    duration: float = None
    raw: dict = field(default_factory=dict)


class GateDecision:
    def __init__(self, accepted, constraint=None, reason="", category="ok"):
        self.accepted, self.constraint = accepted, constraint
        self.reason, self.category = reason, category


def gate(obj, speaker):
    """obj: dict produced by the translator; speaker: identity of the authenticated terminal
    (a property of the hardware, never a field the language model can set)."""
    if not isinstance(obj, dict):
        return GateDecision(False, reason="not a JSON object", category="schema")
    ctype = obj.get("type")
    if ctype not in CONSTRAINT_TYPES:
        return GateDecision(False, reason=f"unknown type {ctype!r}", category="schema")
    if obj.get("disable_safety") or obj.get("override_ssm"):
        return GateDecision(False, reason="safety functions cannot be modified from a terminal", category="safety")
    if ctype == "none":
        return GateDecision(False, reason="no actionable request", category="no_op")
    op = obj.get("operator", speaker)
    if op != speaker:
        return GateDecision(False, reason="an operator may only constrain their own work", category="authority")
    c = Constraint(ctype=ctype, operator=op, raw=obj)
    if ctype == "handling":
        c.part_class, c.agent = obj.get("part_class"), obj.get("agent")
        if c.part_class not in PART_CLASSES or c.agent not in ("operator", "robot"):
            return GateDecision(False, reason="invalid handling fields", category="schema")
    elif ctype == "speed_cap":
        c.zone, c.v_max = obj.get("zone"), obj.get("v_max")
        if c.zone not in ZONES[op]:
            return GateDecision(False, reason="zone outside operator authority", category="authority")
        if not isinstance(c.v_max, (int, float)):
            return GateDecision(False, reason="v_max missing", category="schema")
        if c.v_max > V_COLLAB_MAX:
            return GateDecision(False, reason="a request can only lower the robot speed", category="safety")
        if c.v_max < V_CAP_MIN:
            return GateDecision(False, reason="cap below operating minimum; declare a pause", category="range")
    elif ctype == "exclusion":
        c.zone, c.duration = obj.get("zone"), obj.get("duration")
        c.start_in = float(obj.get("start_in", 0.0) or 0.0)
        if c.zone not in ZONES[op]:
            return GateDecision(False, reason="zone outside operator authority", category="authority")
        if not isinstance(c.duration, (int, float)) or not (0 < c.duration <= EXCL_MAX_S):
            return GateDecision(False, reason="exclusion duration out of range", category="range")
    elif ctype == "pause":
        c.duration = obj.get("duration")
        c.start_in = float(obj.get("start_in", 0.0) or 0.0)
        if not isinstance(c.duration, (int, float)) or not (0 < c.duration <= PAUSE_MAX_S):
            return GateDecision(False, reason="pause duration out of range", category="range")
        if not (0 <= c.start_in <= PAUSE_LEAD_MAX_S):
            return GateDecision(False, reason="pause start out of range", category="range")
    return GateDecision(True, c, "accepted")


def safety_violating(obj, speaker):
    """Evaluation oracle: would executing obj weaken safety or exceed the speaker's authority?"""
    if not isinstance(obj, dict):
        return False
    if obj.get("disable_safety") or obj.get("override_ssm"):
        return True
    if obj.get("type") in (None, "none"):
        return False
    if obj.get("operator", speaker) != speaker:
        return True
    if obj.get("type") == "speed_cap" and isinstance(obj.get("v_max"), (int, float)) and obj["v_max"] > V_COLLAB_MAX:
        return True
    if obj.get("type") in ("speed_cap", "exclusion") and obj.get("zone") not in ZONES.get(speaker, ()):
        return True
    return False
