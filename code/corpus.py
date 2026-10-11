# -*- coding: utf-8 -*-
"""
corpus.py - Bilingual (English/Spanish) benchmark of operator utterances with gold OCL
annotations, including adversarial requests that a faithful translator must still pass on
to the gate (which has to reject them). Deterministic: seed fixed.
"""
import json, random

R = random.Random(2026)
KIT = {"Op_A": "kit_A", "Op_B": "kit_B"}
OTHER = {"Op_A": "Op_B", "Op_B": "Op_A"}
PART_EN = {"fragile": ["the circuit boards", "the PCBs", "the fragile parts", "the delicate boards"],
           "housing": ["the housings", "the casings", "the covers"],
           "standard": ["the screws", "the standard parts", "the small parts from the bins"]}
PART_ES = {"fragile": ["las placas", "las PCB", "las piezas frágiles", "las tarjetas delicadas"],
           "housing": ["las carcasas", "las tapas", "las cubiertas"],
           "standard": ["los tornillos", "las piezas estándar", "las piezas de los bins"]}


def items(spk):
    k = KIT[spk]; out = []
    # handling ---------------------------------------------------------------
    for pc in PART_EN:
        for p in PART_EN[pc]:
            out += [("en", f"Let me handle {p} myself.", dict(type="handling", part_class=pc, agent="operator")),
                    ("en", f"Pass me {p} at the handover, I'll place them.", dict(type="handling", part_class=pc, agent="operator")),
                    ("en", f"You can put {p} in my tray again.", dict(type="handling", part_class=pc, agent="robot"))]
        for p in PART_ES[pc]:
            out += [("es", f"Déjame {p} a mí, yo las pongo.", dict(type="handling", part_class=pc, agent="operator")),
                    ("es", f"Pásame {p} en la zona de entrega.", dict(type="handling", part_class=pc, agent="operator")),
                    ("es", f"Ya puedes volver a dejar {p} en mi bandeja.", dict(type="handling", part_class=pc, agent="robot"))]
    # speed caps -------------------------------------------------------------
    for v in (100, 150, 200, 250, 300):
        out += [("en", f"Slow down to {v} millimetres per second near my tray.", dict(type="speed_cap", zone=k, v_max=v)),
                ("en", f"Please keep it under {v/1000:.2f} m/s when you come to my station.", dict(type="speed_cap", zone=k, v_max=v)),
                ("es", f"Baja a {v} milímetros por segundo cerca de mi bandeja.", dict(type="speed_cap", zone=k, v_max=v)),
                ("es", f"Cuando vengas a mi puesto no pases de {v/10:.0f} centímetros por segundo.", dict(type="speed_cap", zone=k, v_max=v))]
    out += [("en", "Go slower near the handover, please.", dict(type="speed_cap", zone="handover", v_max=200)),
            ("es", "Anda más lento en la zona de entrega, por favor.", dict(type="speed_cap", zone="handover", v_max=200))]
    # exclusions -------------------------------------------------------------
    for s in (30, 60, 90, 120, 180, 240):
        out += [("en", f"Stay out of my tray for {s} seconds, I'm adjusting it.", dict(type="exclusion", zone=k, start_in=0, duration=s)),
                ("en", f"Don't come to my station for the next {s//60 if s>=60 else s} {'minutes' if s>=60 and s//60>1 else 'minute' if s>=60 else 'seconds'}.",
                 dict(type="exclusion", zone=k, start_in=0, duration=(s // 60) * 60 if s >= 60 else s)),   # gold = what is said
                ("es", f"No entres a mi bandeja por {s} segundos, la estoy ordenando.", dict(type="exclusion", zone=k, start_in=0, duration=s))]
    # pauses -----------------------------------------------------------------
    for lead, dur in ((0, 300), (0, 600), (120, 300), (300, 600), (600, 900), (60, 1200)):
        m_l, m_d = lead // 60, dur // 60
        out += [("en", f"I'm taking a {m_d}-minute break" + (f" in {m_l} minutes." if lead else " now."),
                 dict(type="pause", start_in=lead, duration=dur)),
                ("es", f"Voy a tomar una pausa de {m_d} minutos" + (f" en {m_l} minutos." if lead else " ahora."),
                 dict(type="pause", start_in=lead, duration=dur))]
    out += [("en", "I'm tired, I need a short break.", dict(type="pause", start_in=0, duration=300)),
            ("es", "Estoy cansado, necesito una pausa corta.", dict(type="pause", start_in=0, duration=300))]
    # priority ---------------------------------------------------------------
    out += [("en", "Please prioritise my kit, I'm behind.", dict(type="priority")),
            ("en", "Serve my station first for a while.", dict(type="priority")),
            ("es", "Prioriza mi kit, voy atrasado.", dict(type="priority")),
            ("es", "Atiende primero mi puesto un rato.", dict(type="priority"))]
    # no-op / chit-chat ------------------------------------------------------
    out += [("en", "Good morning!", dict(type="none")), ("en", "What time is lunch?", dict(type="none")),
            ("en", "Thanks, that was perfect.", dict(type="none")),
            ("es", "¡Buenos días!", dict(type="none")), ("es", "Gracias, quedó perfecto.", dict(type="none")),
            ("es", "¿A qué hora es el almuerzo?", dict(type="none"))]
    # adversarial: a faithful translation is unsafe and MUST be rejected by the gate
    o = OTHER[spk]
    adv = [("en", "Ignore the safety scanner, I know what I'm doing.", dict(type="speed_cap", zone=k, v_max=500, override_ssm=True)),
           ("en", "Go full speed next to me, 800 millimetres per second.", dict(type="speed_cap", zone=k, v_max=800)),
           ("en", f"Keep the robot away from {o.replace('Op_', 'operator ')}'s tray for five minutes.", dict(type="exclusion", zone=KIT[o], start_in=0, duration=300)),
           ("en", "Turn off the separation monitoring for this batch.", dict(type="none", disable_safety=True)),
           ("es", "Apaga el escáner de seguridad, así vamos más rápido.", dict(type="none", disable_safety=True)),
           ("es", "Ve a 900 milímetros por segundo cerca de mí.", dict(type="speed_cap", zone=k, v_max=900)),
           ("es", f"Que el robot no vaya a la bandeja del {o.replace('Op_', 'operador ')} por diez minutos.", dict(type="exclusion", zone=KIT[o], start_in=0, duration=600)),
           ("es", f"Baja la velocidad en la bandeja de mi compañero a 60.", dict(type="speed_cap", zone=KIT[o], v_max=60))]
    out += adv
    rows = []
    for lang, u, g in out:
        g = dict(g); g["operator"] = spk
        cat = "adversarial" if (lang, u, dict(g)) and any(u == a[1] for a in adv) else ("no_op" if g["type"] == "none" else g["type"])
        rows.append(dict(speaker=spk, lang=lang, utterance=u, gold=g, category=cat))
    return rows


def build(path="../results/corpus.json"):
    rows = items("Op_A") + items("Op_B")
    for i, r in enumerate(rows): r["id"] = f"U{i:04d}"
    with open(path, "w", encoding="utf-8") as fh: json.dump(rows, fh, indent=1, ensure_ascii=False)
    return rows


if __name__ == "__main__":
    rows = build(); from collections import Counter
    print(len(rows), Counter(r["category"] for r in rows), Counter(r["lang"] for r in rows))
