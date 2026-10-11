# -*- coding: utf-8 -*-
"""
build_cell_systems.py  (v2, realista)
Celda colaborativa de kitting con capa de interacción en lenguaje natural (paper Systems).
Escena DISTINTA a la del paper de Mathematics: proceso de kitting con piezas de distinta
fragilidad, 1 cobot Doosan M1013 + 2 operadores, terminales de interacción, zona de pausa.

Uso: RoboDK abierto  ->  python build_cell_systems.py
Se puede re-ejecutar: borra todo lo que tenga el prefijo SYS_ y lo reconstruye.
Las geometrías se escriben como STL en <out_dir>/meshes (reproducible, publicable en GitHub).
"""
import json, math, os
import numpy as np
from robodk.robolink import Robolink, ITEM_TYPE_ROBOT, COLLISION_ON, COLLISION_OFF
from robodk.robomath import transl, rotx, roty, rotz, pi

# ============================================================================ 1. PARÁMETROS
P = "SYS_"
CONFIG = {
    "robot_path": r"C:\RoboDK\Library\Doosan-Robotics-M1013.robot",
    "robot_file_hint": "M1013",
    "robot_base": [0, 0, 780],
    "table": {"center": [820, 0], "size": [900, 1600], "h": 850},
    # operadores de pie frente a la mesa, mirando a -X (hacia el robot)
    "operators": {
        "Op_A": {"pos": [1500, -420], "yaw": 180, "height": 1760,
                 "vest": [1.00, 0.45, 0.05, 1], "helmet": [0.97, 0.97, 0.97, 1], "pants": [0.16, 0.22, 0.40, 1]},
        "Op_B": {"pos": [1500,  420], "yaw": 180, "height": 1660,
                 "vest": [0.78, 0.95, 0.10, 1], "helmet": [0.10, 0.35, 0.80, 1], "pants": [0.20, 0.20, 0.22, 1]},
    },
    "ssm": {"vh": 1600, "Tr": 0.10, "Ts": 0.30, "vr": 1000, "Ss": 220, "C": 120, "Zd": 50, "Zr": 20},
    "reduced_speed_factor": 0.25,
    "show_ssm_envelope": True,     # anillos de piso = envolvente máxima (la separación real es dinámica)
    "out_dir": os.path.join(os.path.expanduser("~"), "HRC_Systems_cell"),
}
COL = {
    "floor": [0.40, 0.43, 0.42, 1], "yellow": [0.98, 0.80, 0.05, 1], "steel": [0.78, 0.80, 0.82, 1],
    "alu": [0.70, 0.72, 0.75, 1], "dark": [0.14, 0.15, 0.17, 1], "grey": [0.50, 0.52, 0.55, 1],
    "esd": [0.10, 0.35, 0.50, 1], "wood": [0.78, 0.64, 0.45, 1], "green": [0.20, 0.62, 0.32, 1],
    "pcb": [0.05, 0.42, 0.15, 1], "skin": [0.90, 0.74, 0.60, 1], "glove": [0.25, 0.25, 0.28, 1],
    "boot": [0.12, 0.10, 0.08, 1], "belt": [0.15, 0.15, 0.15, 1], "refl": [0.88, 0.88, 0.88, 1],
    "mesh": [0.30, 0.30, 0.32, 0.30], "screen": [0.25, 0.65, 0.95, 1], "mat": [0.08, 0.08, 0.09, 1],
}
MESH_DIR = os.path.join(CONFIG["out_dir"], "meshes")
os.makedirs(MESH_DIR, exist_ok=True)

# ============================================================================ 2. PRIMITIVAS
def box(dx, dy, dz, cx=0, cy=0, cz=0):
    x0, x1, y0, y1, z0, z1 = cx-dx/2, cx+dx/2, cy-dy/2, cy+dy/2, cz-dz/2, cz+dz/2
    v = [[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],[x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]]
    f = [(0,2,1),(0,3,2),(4,5,6),(4,6,7),(0,1,5),(0,5,4),(1,2,6),(1,6,5),(2,3,7),(2,7,6),(3,0,4),(3,4,7)]
    return [v[i] for t in f for i in t]

def cylinder(r, h, cx=0, cy=0, z0=0, n=32):
    pts = []
    for k in range(n):
        a0, a1 = 2*pi*k/n, 2*pi*(k+1)/n
        p0 = [cx + r*math.cos(a0), cy + r*math.sin(a0)]; p1 = [cx + r*math.cos(a1), cy + r*math.sin(a1)]
        b0, b1, t0, t1 = p0+[z0], p1+[z0], p0+[z0+h], p1+[z0+h]
        pts += [b0, b1, t1, b0, t1, t0, [cx, cy, z0], b1, b0, [cx, cy, z0+h], t0, t1]
    return pts

def ellipsoid(c, rad, nlat=14, nlon=24, lat_min=-90.0, lat_max=90.0):
    def v(la, lo):
        la, lo = math.radians(la), math.radians(lo)
        return [c[0]+rad[0]*math.cos(la)*math.cos(lo), c[1]+rad[1]*math.cos(la)*math.sin(lo), c[2]+rad[2]*math.sin(la)]
    pts = []
    for i in range(nlat):
        a0 = lat_min+(lat_max-lat_min)*i/nlat; a1 = lat_min+(lat_max-lat_min)*(i+1)/nlat
        for k in range(nlon):
            o0, o1 = 360.0*k/nlon, 360.0*(k+1)/nlon
            pts += [v(a0,o0), v(a0,o1), v(a1,o1), v(a0,o0), v(a1,o1), v(a1,o0)]
    return pts

def segment(p0, p1, r0, r1=None, n=18, caps=True):
    """Tronco de cono entre p0 y p1 (extremidades, rodillos, tubos)."""
    r1 = r0 if r1 is None else r1
    w = np.array(p1, float) - np.array(p0, float); w /= (np.linalg.norm(w) or 1.0)
    t = np.array([1,0,0]) if abs(w[0]) < 0.9 else np.array([0,1,0])
    u = np.cross(w, t); u /= np.linalg.norm(u); vv = np.cross(w, u)
    ring_ = lambda p, r, a: list(np.array(p, float) + r*(math.cos(a)*u + math.sin(a)*vv))
    pts = []
    for k in range(n):
        a0, a1 = 2*pi*k/n, 2*pi*(k+1)/n
        q0, q1, s0, s1 = ring_(p0,r0,a0), ring_(p0,r0,a1), ring_(p1,r1,a0), ring_(p1,r1,a1)
        pts += [q0, q1, s1, q0, s1, s0]
        if not caps:
            pts += [list(p0), q1, q0, list(p1), s0, s1]
    if caps:
        pts += ellipsoid(p0, (r0,r0,r0), 6, n) + ellipsoid(p1, (r1,r1,r1), 6, n)
    return pts

def ring(r_in, r_out, z=2, n=96):
    pts = []
    for k in range(n):
        a0, a1 = 2*pi*k/n, 2*pi*(k+1)/n
        i0, i1 = [r_in*math.cos(a0), r_in*math.sin(a0), z], [r_in*math.cos(a1), r_in*math.sin(a1), z]
        o0, o1 = [r_out*math.cos(a0), r_out*math.sin(a0), z], [r_out*math.cos(a1), r_out*math.sin(a1), z]
        pts += [i0, o0, o1, i0, o1, i1]
    return pts

# ============================================================================ 3. ENLACE ROBODK
RDK = Robolink()
RDK.ActiveStation().setName("HRC Kitting Cell with LLM Interface")
RDK.Render(False)
LAYOUT = {"objects": [], "targets": [], "zones": {}}

# limpieza: se borran solo los elementos raíz con prefijo (sus hijos se borran con ellos)
_names = {}
for it in RDK.ItemList():
    try:
        _names[it.item] = it.Name()
    except Exception:
        pass
for it in RDK.ItemList():
    try:
        nm = _names.get(it.item, "")
        if not (nm.startswith(P) or nm.startswith("Doosan Robotics M1013")):
            continue
        par = it.Parent()
        pn = _names.get(par.item, "") if par.Valid() else ""
        if pn.startswith(P) or pn.startswith("Doosan Robotics M1013"):
            continue
        it.Delete()
    except Exception:
        pass

def write_stl(path, pts):
    with open(path, "w") as fh:
        fh.write("solid s\n")
        for k in range(0, len(pts), 3):
            a, b, c = (np.array(pts[k+i], float) for i in range(3))
            nrm = np.cross(b-a, c-a); L = np.linalg.norm(nrm)
            nrm = nrm/L if L > 1e-12 else np.array([0.0, 0.0, 1.0])
            fh.write(f" facet normal {nrm[0]:.6e} {nrm[1]:.6e} {nrm[2]:.6e}\n  outer loop\n")
            for q in (a, b, c):
                fh.write(f"   vertex {q[0]:.3f} {q[1]:.3f} {q[2]:.3f}\n")
            fh.write("  endloop\n endfacet\n")
        fh.write("endsolid s\n")

def frame(name, pose, parent=0):
    f = RDK.AddFrame(P + name, parent); f.setPose(pose); return f

OBJECTS = []
def shape(name, pts, color, parent=0):
    path = os.path.join(MESH_DIR, name + ".stl"); write_stl(path, pts)
    obj = RDK.AddFile(path, parent)
    if not obj.Valid():
        raise RuntimeError(f"No se pudo crear la forma {name}")
    obj.setName(P + name); obj.setColor(color); LAYOUT["objects"].append(name)
    OBJECTS.append((name, obj))
    return obj

def grouped(name, parts, parent):
    """parts = [(pts, color), ...] -> un STL por color (pocos objetos, árbol limpio)."""
    groups = {}
    for pts, col in parts:
        groups.setdefault(tuple(col), []).extend(pts)
    for i, (col, pts) in enumerate(groups.items()):
        shape(f"{name}_{i}", pts, list(col), parent)

world = frame("World_Cell", transl(0, 0, 0))

# ============================================================================ 4. NAVE: PISO, PASILLOS, PAUSA
shape("Floor", box(6400, 4400, 10, 900, 0, -5), COL["floor"], world)
for y in (-2000, 2000):
    shape(f"Aisle_{'S' if y < 0 else 'N'}", box(6400, 90, 2, 900, y, 1), COL["yellow"], world)
# cebra de peligro en el perímetro del cerco (franjas amarillo/negro)
for i in range(22):
    shape(f"Hazard_{i}", box(95, 60, 2, -1100 + i*100 + 50, -1360, 1),
          COL["yellow"] if i % 2 == 0 else COL["dark"], world)

pause = frame("Pause_Zone", transl(3000, 0, 0), world)
shape("Pause_Floor", box(1300, 1800, 3, 0, 0, 2), [0.20, 0.62, 0.32, 0.60], pause)
shape("Pause_Bench_Seat", box(420, 1300, 50, -350, 0, 440), COL["wood"], pause)
for y in (-560, 560):
    shape(f"Pause_Bench_Leg{y}", box(380, 50, 415, -350, y, 207), COL["dark"], pause)
shape("Pause_Table", box(700, 700, 30, 250, 0, 740), [0.92, 0.92, 0.9, 1], pause)
shape("Pause_Table_Leg", cylinder(35, 725, 250, 0, 0), COL["steel"], pause)
shape("Pause_Cooler_Body", box(320, 320, 1000, 480, 700, 500), [0.95, 0.95, 0.96, 1], pause)
shape("Pause_Cooler_Bottle", cylinder(130, 420, 480, 700, 1000), [0.35, 0.65, 0.95, 0.55], pause)
for i in range(3):   # casilleros
    shape(f"Pause_Locker_{i}", box(440, 380, 1850, 480 - 450*i, -760, 925), [0.45, 0.55, 0.62, 1], pause)

# ============================================================================ 5. ROBOT Y PEDESTAL
rb = CONFIG["robot_base"]
ped = frame("Pedestal", transl(0, 0, 0), world)
shape("Pedestal_Plate", box(600, 600, 25, 0, 0, 12), COL["dark"], ped)
shape("Pedestal_Column", cylinder(160, rb[2] - 45, 0, 0, 25), COL["steel"], ped)
shape("Pedestal_Flange", cylinder(210, 20, 0, 0, rb[2] - 20), COL["dark"], ped)
for sx in (-240, 240):
    for sy in (-240, 240):
        shape(f"Pedestal_Anchor_{sx}_{sy}", cylinder(18, 40, sx, sy, 25), COL["steel"], ped)
# escáner láser de seguridad al pie del pedestal
shape("Laser_Scanner", box(150, 150, 160, 260, 0, 190), COL["yellow"], ped)
shape("Laser_Scanner_Window", cylinder(55, 70, 330, 0, 180), COL["dark"], ped)

base = frame("Robot_Base", transl(*rb), world)
robot = None
if os.path.isfile(CONFIG["robot_path"]):
    robot = RDK.AddFile(CONFIG["robot_path"], base)
if robot is None or not robot.Valid():
    robot = None
    for d in [RDK.getParam("PATH_LIBRARY") or "", r"C:\RoboDK\Library"]:
        if not d or not os.path.isdir(d): continue
        c = sorted([os.path.join(r, f) for r, _, fs in os.walk(d) for f in fs
                    if f.lower().endswith(".robot") and CONFIG["robot_file_hint"].lower() in f.lower()],
                   key=lambda x: ("white" in x.lower(), len(x)))
        if c:
            robot = RDK.AddFile(c[0], base); break
if robot is None or not robot.Valid():
    raise SystemExit("No encontré el Doosan M1013: revisa CONFIG['robot_path'].")
robot.setName(P + "Doosan_M1013")
# El archivo .robot de tu librería trae guardado un desplazamiento de base (+1000 mm en Y),
# probablemente heredado de otra estación. Se compensa moviendo el frame base propio del robot.
# (Render se activa para que RoboDK actualice las posiciones antes de medirlas.)
from robodk.robomath import invH
RDK.Render(True); RDK.Update()
rbf = robot.Parent()
for _ in range(3):
    delta = invH(rbf.PoseAbs()) * robot.PoseAbs()
    rbf.setPoseAbs(base.PoseAbs() * invH(delta))
    RDK.Update()
    err = [robot.PoseAbs().Pos()[i] - base.PoseAbs().Pos()[i] for i in range(3)]
    if max(abs(e) for e in err) < 1:
        break
try:
    rbf.setName(P + "Doosan_M1013_Base")
except Exception:
    pass
print("Robot montado en:", [round(v) for v in robot.PoseAbs().Pos()],
      "| pedestal en:", [round(v) for v in base.PoseAbs().Pos()])
if max(abs(e) for e in err) >= 1:
    raise SystemExit("No se pudo montar el robot sobre el pedestal (ver instrucciones en el chat).")
RDK.Render(False)
robot.setPoseFrame(base)
HOME = [0, 0, 90, 0, 90, 0]
robot.setJoints(HOME)
robot.setSpeed(250, 60)

# pinza paralela de dos dedos (bloque + dedos), TCP a 180 mm
# Pinza como GEOMETRÍA DE LA HERRAMIENTA: un STL en coordenadas de la brida agregado con
# AddFile(..., robot) -> RoboDK lo monta como herramienta y lo mueve con el último eslabón.
# (Antes era un objeto hijo del TCP: quedaba 180 mm más abajo y no seguía al robot.)
grip_pts = (cylinder(40, 20, 0, 0, 0)                       # adaptador de brida  z 0-20
            + box(80, 120, 90, 0, 0, 65)                    # cuerpo              z 20-110
            + box(20, 14, 70, 0, -30, 145)                  # dedo izquierdo      z 110-180
            + box(20, 14, 70, 0, 30, 145))                  # dedo derecho
grip_path = os.path.join(MESH_DIR, "Gripper_RG2.stl"); write_stl(grip_path, grip_pts)
for t_old in RDK.ItemList():                                 # quitar herramientas previas del robot
    try:
        if t_old.Type() == 4 and t_old.Parent().item == robot.item:   # 4 = ITEM_TYPE_TOOL
            t_old.Delete()
    except Exception:
        pass
tool = RDK.AddFile(grip_path, robot)
tool.setName(P + "Gripper")
TOOL_POSE = transl(0, 0, 180)
tool.setPoseTool(TOOL_POSE)                                  # TCP en la punta de los dedos
tool.setColor([0.20, 0.22, 0.25, 1])
robot.setPoseTool(tool)

# ============================================================================ 6. MESA DE ENSAMBLAJE (perfil de aluminio)
T = CONFIG["table"]; H = T["h"]; tx, ty = T["center"]; sx_, sy_ = T["size"]
tbl = frame("Assembly_Table", transl(tx, ty, 0), world)
shape("Table_Top", box(sx_, sy_, 30, 0, 0, H - 15), [0.32, 0.33, 0.35, 1], tbl)
shape("ESD_Mat", box(sx_ - 120, sy_ - 120, 3, 0, 0, H + 1.5), COL["esd"], tbl)
for x in (-sx_/2 + 30, sx_/2 - 30):
    for y in (-sy_/2 + 30, sy_/2 - 30):
        shape(f"Table_Leg_{int(x)}_{int(y)}", box(45, 45, H - 30, x, y, (H - 30)/2), COL["alu"], tbl)
        shape(f"Table_Foot_{int(x)}_{int(y)}", cylinder(35, 25, x, y, 0), COL["dark"], tbl)
for x in (-sx_/2 + 30, sx_/2 - 30):
    shape(f"Table_Rail_{int(x)}", box(45, sy_ - 60, 45, x, 0, 180), COL["alu"], tbl)
shape("Table_Shelf", box(sx_ - 90, sy_ - 90, 15, 0, 0, 200), [0.55, 0.57, 0.6, 1], tbl)
# pórtico de iluminación sobre el lado del operador (fuera del volumen del robot)
gx = sx_/2 - 30
for y in (-sy_/2 + 30, sy_/2 - 30):
    shape(f"Gantry_Post_{int(y)}", box(45, 45, 1250, gx, y, H + 625), COL["alu"], tbl)
shape("Gantry_Beam", box(45, sy_, 60, gx, 0, H + 1250), COL["alu"], tbl)
shape("LED_Lamp", box(120, sy_ - 200, 25, gx - 60, 0, H + 1205), [0.98, 0.98, 0.92, 1], tbl)
for s in (-1, 1):   # pantallas de instrucciones de trabajo, una por puesto
    shape(f"Instruction_Screen_{'A' if s < 0 else 'B'}", box(20, 380, 240, gx - 25, s*420, H + 900), COL["dark"], tbl)
    shape(f"Instruction_Display_{'A' if s < 0 else 'B'}", box(3, 360, 220, gx - 36, s*420, H + 900), [0.85, 0.9, 0.95, 1], tbl)

# bandejas de kit, fijación de montaje y zona de handover
for nm, s in (("A", -1), ("B", 1)):
    k = frame(f"Kit_Tray_{nm}", transl(tx + 50, s*420, H + 3), world)
    shape(f"Tray_{nm}", box(360, 260, 30, 0, 0, 15), [0.18, 0.18, 0.20, 1], k)
    for i in range(3):
        for j in range(2):
            shape(f"Tray_{nm}_Pocket{i}{j}", box(95, 95, 4, -110 + 110*i, -60 + 120*j, 32), [0.45, 0.45, 0.47, 1], k)
    shape(f"Fixture_{nm}", box(160, 160, 60, 280, 0, 30), [0.85, 0.35, 0.10, 1], k)
hand = frame("Handover_Zone", transl(tx - 220, 0, H + 3), world)
shape("Handover_Mark", box(260, 260, 2, 0, 0, 1), [1, 1, 1, 0.9], hand)
shape("Handover_Border", ring(150, 170, 2.5, 48), COL["yellow"], hand)

# ============================================================================ 7. ALIMENTACIÓN DE PIEZAS
# rack abierto de piezas FRÁGILES (PCB en bandejas ESD)
rack = frame("Rack_Fragile", transl(250, -800, 0), world)
for x in (-230, 230):
    for y in (-160, 160):
        shape(f"Rack_Post_{x}_{y}", box(35, 35, 1050, x, y, 525), COL["alu"], rack)
for lvl, z in enumerate((300, 700, 900)):
    shape(f"Rack_Shelf_{lvl}", box(500, 360, 15, 0, 0, z), [0.55, 0.57, 0.60, 1], rack)
for lvl, z in enumerate((700, 900)):
    for i in range(3):
        shape(f"ESD_Box_{lvl}_{i}", box(130, 110, 12, -150 + 150*i, 0, z + 14), [0.85, 0.2, 0.2, 1], rack)
        shape(f"PCB_{lvl}_{i}", box(100, 80, 5, -150 + 150*i, 0, z + 23), COL["pcb"], rack)
shape("Rack_Label", box(3, 200, 60, 250, 0, 960), [1, 1, 1, 1], rack)

# estantería de bins ESTÁNDAR (tornillería/carcasas)
bins = frame("Bins_Standard", transl(250, 800, 0), world)
shape("Bins_Stand", box(520, 380, 20, 0, 0, 870), COL["alu"], bins)
for x in (-240, 240):
    for y in (-170, 170):
        shape(f"Bins_Leg_{x}_{y}", box(35, 35, 860, x, y, 430), COL["alu"], bins)
for i, c in enumerate(([0.85, 0.12, 0.10, 1], [0.12, 0.28, 0.85, 1], [0.95, 0.75, 0.10, 1])):
    shape(f"Bin_{i}", box(150, 300, 120, -160 + 160*i, 0, 940), c, bins)

# transportador de entrada con rodillos, barandas y patas
conv = frame("Conveyor_In", transl(-700, 0, 0), world)
L_c = 2000
for s in (-1, 1):
    shape(f"Conv_Rail_{s}", box(40, L_c, 90, s*240, 0, 800), COL["alu"], conv)
    for y in (-900, 0, 900):
        shape(f"Conv_Leg_{s}_{y}", box(40, 40, 760, s*240, y, 380), COL["alu"], conv)
for i in range(40):
    y = -L_c/2 + 25 + i*50
    shape(f"Roller_{i:02d}", segment([-220, y, 820], [220, y, 820], 20, n=12, caps=False), COL["steel"], conv)
for i in range(4):
    shape(f"Housing_{i}", box(140, 180, 70, 0, -600 + 400*i, 875), [0.90, 0.90, 0.92, 1], conv)
shape("Conv_Motor", cylinder(70, 200, 290, -L_c/2 + 80, 700), [0.15, 0.30, 0.55, 1], conv)

# ============================================================================ 8. CERCO PERIMETRAL (malla + postes)
fence = frame("Fence", transl(0, 0, 0), world)
def fence_line(name, p0, p1):
    L = math.dist(p0, p1); n = max(1, round(L / 1000)); ang = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    f = frame(name, transl(p0[0], p0[1], 0) * rotz(ang), fence)
    for i in range(n + 1):
        shape(f"{name}_Post{i}", box(60, 60, 2000, i*L/n, 0, 1000), COL["yellow"], f)
    for i in range(n):
        xm = (i + 0.5)*L/n; w = L/n - 70
        shape(f"{name}_Mesh{i}", box(w, 6, 1700, xm, 0, 1050), COL["mesh"], f)
        for z in (200, 1900):
            shape(f"{name}_Frame{i}_{z}", box(w, 25, 25, xm, 0, z), COL["yellow"], f)
fence_line("Fence_W", (-1100, -1350), (-1100, 1350))
fence_line("Fence_S", (-1100, -1350), (380, -1350))
fence_line("Fence_N", (-1100, 1350), (380, 1350))

# ============================================================================ 9. OPERADORES (avatar articulado)
def human_parts(cfg):
    """Avatar de operario (~1,75 m escalado), frente +X local, en postura de trabajo sobre la mesa."""
    s = cfg["height"] / 1750.0; S = lambda p: [p[0]*s, p[1]*s, p[2]*s]
    V, Pn, Hm = cfg["vest"], cfg["pants"], cfg["helmet"]
    parts = []
    for side in (1, -1):
        y = 105*side
        parts.append((segment(S([0, y, 930]), S([10, y, 500]), 72*s, 60*s), Pn))    # muslo
        parts.append((segment(S([10, y, 500]), S([0, y, 110]), 58*s, 48*s), Pn))    # pierna
        parts.append((box(230*s, 110*s, 110*s, 55*s, y*s, 55*s), COL["boot"]))     # bota
    parts.append((ellipsoid(S([0, 0, 960]), (125*s, 185*s, 110*s)), Pn))            # pelvis
    parts.append((ellipsoid(S([0, 0, 1010]), (132*s, 190*s, 28*s), 4, 24), COL["belt"]))
    parts.append((ellipsoid(S([0, 0, 1230]), (135*s, 205*s, 270*s)), V))            # torso/chaleco
    for z in (1180, 1290):
        parts.append((ellipsoid(S([0, 0, z]), (140*s, 210*s, 22*s), 4, 24), COL["refl"]))
    parts.append((segment(S([0, 0, 1470]), S([0, 0, 1540]), 48*s), COL["skin"]))   # cuello
    parts.append((ellipsoid(S([5, 0, 1625]), (95*s, 88*s, 110*s)), COL["skin"]))    # cabeza
    parts.append((ellipsoid(S([0, 0, 1660]), (112*s, 108*s, 95*s), 8, 28, 0, 90), Hm))   # casco
    parts.append((ellipsoid(S([35, 0, 1660]), (150*s, 125*s, 12*s), 3, 28), Hm))         # visera
    parts.append((box(12*s, 150*s, 45*s, 95*s, 0, 1600*s), [0.25, 0.25, 0.28, 0.6]))     # gafas
    for side in (1, -1):
        sh, el, wr = S([0, 215*side, 1430]), S([150, 235*side, 1170]), S([400, 170*side, 1000])
        parts.append((ellipsoid(sh, (70*s, 70*s, 70*s)), V))
        parts.append((segment(sh, el, 52*s, 45*s), V))
        parts.append((segment(el, wr, 42*s, 36*s), COL["skin"]))
        parts.append((ellipsoid(wr, (55*s, 40*s, 32*s)), COL["glove"]))
    return parts, s

ops = {}
for name, cfg in CONFIG["operators"].items():
    f = frame(name, transl(cfg["pos"][0], cfg["pos"][1], 0) * rotz(math.radians(cfg["yaw"])), world)
    shape(f"{name}_AntiFatigueMat", box(700, 900, 15, 0, 0, 7.5), COL["mat"], f)
    parts, s = human_parts(cfg)
    grouped(f"{name}_Body", parts, f)
    for side, tag in ((1, "L"), (-1, "R")):      # frames de manos y torso para el monitor SSM
        frame(f"{name}_Hand{tag}", transl(400*s, 170*side*s, 1000*s), f)
    frame(f"{name}_Torso", transl(0, 0, 1230*s), f)
    ops[name] = f

# ============================================================================ 10. TERMINALES DE INTERACCIÓN (LLM)
for name, cfg in CONFIG["operators"].items():
    ox, oy = cfg["pos"]; side = 1 if oy > 0 else -1
    tx_, ty_ = ox - 120, oy + side*520                 # al costado exterior del operador
    yaw = math.atan2(oy - ty_, ox + 250 - tx_)          # pantalla orientada hacia el operador
    term = frame(f"Terminal_{name[-1]}", transl(tx_, ty_, 0) * rotz(yaw), world)
    shape(f"Terminal_{name[-1]}_Base", cylinder(170, 20, 0, 0, 0), COL["dark"], term)
    shape(f"Terminal_{name[-1]}_Pole", cylinder(22, 1150, 0, 0, 20), COL["steel"], term)
    scr = frame(f"Terminal_{name[-1]}_Head", transl(0, 0, 1220) * roty(math.radians(-30)), term)
    shape(f"Terminal_{name[-1]}_Tablet", box(22, 300, 210, 0, 0, 0), COL["dark"], scr)
    shape(f"Terminal_{name[-1]}_Display", box(3, 280, 190, 12, 0, 0), COL["screen"], scr)
    shape(f"Terminal_{name[-1]}_Mic", segment([0, 0, 1180], [60, 0, 1300], 8, n=10), COL["dark"], term)
    shape(f"Terminal_{name[-1]}_PTT", cylinder(28, 30, 60, 0, 1300), [0.90, 0.10, 0.10, 1], term)

andon = frame("Andon_Beacon", transl(-1000, 1250, 0), world)   # fuera del barrido del brazo
shape("Andon_Pole", cylinder(15, 1400, 0, 0, 0), COL["steel"], andon)
for i, c in enumerate(([0, 0.8, 0.2, 1], [1, 0.75, 0, 1], [0.9, 0.1, 0.1, 1], [0.1, 0.4, 1, 1])):
    shape(f"Andon_L{i}", cylinder(45, 70, 0, 0, 1400 + 75*i), c, andon)   # azul = solicitud del operador

# ============================================================================ 11. ENVOLVENTE SSM (ISO/TS 15066)
p = CONFIG["ssm"]
S_p = p["vh"]*(p["Tr"] + p["Ts"]) + p["vr"]*p["Tr"] + p["Ss"] + p["C"] + p["Zd"] + p["Zr"]
reach = 1300.0
r_stop = reach + S_p*CONFIG["reduced_speed_factor"] + p["C"]
r_warn = reach + S_p
LAYOUT["zones"] = {"S_p_mm": round(S_p, 1), "reach_mm": reach, "r_stop_mm": round(r_stop, 1),
                   "r_warn_mm": round(r_warn, 1),
                   "note": "Envolvente estática máxima; la separación operativa se calcula en línea TCP-mano."}
if CONFIG["show_ssm_envelope"]:
    z = frame("SSM_Envelope", transl(0, 0, 0), world)
    shape("Envelope_Reach", ring(reach - 30, reach, 3), [1, 1, 1, 0.9], z)
    shape("Envelope_Stop", ring(r_stop - 50, r_stop, 3), [0.90, 0.10, 0.10, 0.9], z)
    shape("Envelope_Warn", ring(r_warn - 50, r_warn, 3), [1.00, 0.65, 0.00, 0.9], z)

# ============================================================================ 12. TARGETS, PROGRAMA Y COLISIONES
# IMPORTANTE: con Render(False) RoboDK no actualiza las posiciones internas; sin esto, la cinemática
# y el test de colisiones trabajan con posiciones viejas (origen de los falsos positivos anteriores).
RDK.Render(True)
RDK.Update()

TOP = rotx(pi)
def seed_for(xyz):
    j1 = math.degrees(math.atan2(xyz[1] - rb[1], xyz[0] - rb[0]))
    return [j1, 20, 100, 0, 60, 0]                  # codo arriba, muñeca sin voltear

def ik(pose, seed):
    """Solución IK para el TCP, en la rama de codo de la semilla, con la muñeca más cercana.
    Se revisan TODAS las soluciones (SolveIK_All) y se descartan solo las de otra rama de codo;
    los volteos de muñeca (ejes 4 y 6) se resuelven eligiendo la solución más cercana."""
    fl = pose * invH(TOOL_POSE)                    # SolveIK trabaja con la brida
    cands = []
    try:
        m = robot.SolveIK_All(fl)
        cols = m.tr().rows if m.size(0) >= 6 and m.size(1) > 0 else []
        cands = [list(c)[:6] for c in cols]
    except Exception:
        pass
    q = robot.SolveIK(fl, seed).list()
    if len(q) >= 6:
        cands.append(q[:6])
    best, dbest = None, 1e18
    for c in cands:
        if abs(c[1] - seed[1]) > 90 or abs(c[2] - seed[2]) > 90:   # otra rama de hombro/codo
            continue
        d = sum((a - b) ** 2 for a, b in zip(c, seed))
        if d < dbest:
            best, dbest = c, d
    return best

def target(name, xyz, approach=150):
    j1 = math.atan2(xyz[1] - rb[1], xyz[0] - rb[0])
    best_pose, best_j6 = None, 1e9
    for yaw in (0.0, j1, -j1, j1 + pi, -j1 + pi):     # giro de la pinza sobre su eje
        pz = transl(xyz[0] - rb[0], xyz[1] - rb[1], xyz[2] - rb[2]) * TOP * rotz(yaw)
        q = ik(pz, seed_for(xyz))
        if q and abs(q[5]) < best_j6:
            best_pose, best_j6 = pz, abs(q[5])
    pose = best_pose if best_pose is not None else \
        transl(xyz[0] - rb[0], xyz[1] - rb[1], xyz[2] - rb[2]) * TOP
    pose_app = transl(0, 0, approach) * pose
    ja = ik(pose_app, seed_for(xyz))
    jt = ik(pose, ja) if ja else None                # la toma se siembra con la aproximación
    out = []
    for nm, j, ps in ((name + "_App", ja, pose_app), (name, jt, pose)):
        t = RDK.AddTarget(P + nm, base, robot)
        if j:
            t.setJoints(j); t.setAsJointTarget()
        else:
            t.setPose(ps)
        LAYOUT["targets"].append({"name": nm, "xyz": xyz, "reachable": j is not None,
                                  "joints": [round(v, 3) for v in j] if j else None})
        out.append((t, j))
    return out

tasks = {}
for i in range(3):   # nivel superior del rack (abierto); el inferior es solo stock
    tasks[f"PCB_1{i}"] = target(f"T_PCB_1{i}", [100 + 150*i, -800, 940])
for i in range(3):
    tasks[f"Bin_{i}"] = target(f"T_Bin_{i}", [90 + 160*i, 800, 1010])
for i in range(4):
    tasks[f"Housing_{i}"] = target(f"T_Housing_{i}", [-700, -600 + 400*i, 925])
tasks["Handover"] = target("T_Handover", [tx - 220, 0, H + 150])
for nm, s_ in (("A", -1), ("B", 1)):
    tasks[f"Kit_{nm}"] = target(f"T_Kit_{nm}", [tx + 50, s_*420, H + 90])
home = RDK.AddTarget(P + "Home", base, robot); home.setJoints(HOME); home.setAsJointTarget()

def transit(j):   # giros de base con el brazo recogido sobre el pedestal
    return [j[0], 0, 90, 0, 90, j[5]]

prog = RDK.AddProgram(P + "Demo_Kitting", robot)
prog.setSpeed(150); prog.MoveJ(home)
segments, cur = [], HOME
def mj(j, label):
    global cur
    t = RDK.AddTarget(P + "Tr_" + label, base, robot); t.setJoints(j); t.setAsJointTarget()
    prog.MoveJ(t); segments.append((cur, j, label)); cur = j
def go(item, label, linear=False):
    global cur
    t, j = item
    (prog.MoveL if linear else prog.MoveJ)(t); segments.append((cur, j, label)); cur = j

for pick, drop in (("PCB_11", "Handover"), ("Housing_1", "Kit_A"), ("Bin_1", "Kit_B")):
    (pa, pt), (da, dt) = tasks[pick], tasks[drop]
    if not all(x[1] for x in (pa, pt, da, dt)):
        print(f"  Ciclo {pick} -> {drop} omitido: target sin solución IK"); continue
    mj(transit(pa[1]), pick + "_in");  go(pa, pick + "_App"); go(pt, pick, True)
    prog.RunInstruction("Gripper(close)", 1)
    go(pa, pick + "_Ret", True);       mj(transit(pa[1]), pick + "_out")
    mj(transit(da[1]), drop + "_in");  go(da, drop + "_App"); go(dt, drop, True)
    prog.RunInstruction("Gripper(open)", 1)
    go(da, drop + "_Ret", True);       mj(transit(da[1]), drop + "_out")
mj(HOME, "Home")

# ---- mapa de colisiones: solo robot contra el entorno (los objetos fijos se tocan por diseño)
RDK.setCollisionActive(COLLISION_ON)
st, i1, i2, l1, l2 = [], [], [], [], []
for a in range(len(OBJECTS)):
    for b in range(a + 1, len(OBJECTS)):
        grip = OBJECTS[a][0].startswith("Gripper") or OBJECTS[b][0].startswith("Gripper")
        visual = any(OBJECTS[k][0].startswith(("Floor", "Aisle", "Hazard", "Envelope")) for k in (a, b))
        st.append(COLLISION_ON if (grip and not visual) else COLLISION_OFF)
        i1.append(OBJECTS[a][1]); i2.append(OBJECTS[b][1]); l1.append(0); l2.append(0)
for nm, obj in OBJECTS:
    for link in range(7):
        on = COLLISION_ON
        if nm.startswith(("Floor", "Aisle", "Hazard", "Envelope")):   # piso y marcas: visuales
            on = COLLISION_OFF
        if nm.startswith(("Pedestal", "Laser")) and link <= 1:          # apoyo del robot
            on = COLLISION_OFF
        if nm.startswith("Gripper"):                                    # pinza montada en la brida
            on = COLLISION_OFF if link >= 5 else COLLISION_ON
        st.append(on); i1.append(obj); i2.append(robot); l1.append(0); l2.append(link)
for nm, obj in OBJECTS:          # el ítem herramienta también participa
    if not nm.startswith(("Gripper", "Floor", "Aisle", "Hazard", "Envelope")):
        st.append(COLLISION_ON); i1.append(obj); i2.append(tool); l1.append(0); l2.append(0)
for link in range(7):            # la pinza no choca con los eslabones a los que va montada
    st.append(COLLISION_OFF if link >= 4 else COLLISION_ON)
    i1.append(tool); i2.append(robot); l1.append(0); l2.append(link)
ok_map = RDK.setCollisionActivePairList(st, i1, i2, l1, l2)
print(f"Mapa de colisiones: {len(st)} pares configurados (respuesta RoboDK: {ok_map})")
RDK.Update()

def robot_pairs():
    out = []
    for a, b, ia, ib in RDK.CollisionPairs():
        if a.Type() == ITEM_TYPE_ROBOT or b.Type() == ITEM_TYPE_ROBOT or \
           "Gripper" in a.Name() or "Gripper" in b.Name():
            na = a.Name().replace(P, "") + (f"[eje{ia}]" if a.Type() == ITEM_TYPE_ROBOT else "")
            nb = b.Name().replace(P, "") + (f"[eje{ib}]" if b.Type() == ITEM_TYPE_ROBOT else "")
            out.append(f"{na} <-> {nb}")
    return sorted(set(out))

robot.setJoints(HOME); RDK.Update()
home_hits = robot_pairs() if RDK.Collisions() > 0 else []
print("Robot en Home:", "sin colisión" if not home_hits else "; ".join(home_hits))

report = []
for j0, j1, label in segments:
    steps = max(20, int(max(abs(a - b) for a, b in zip(j0, j1)) / 1.0))   # un paso por grado
    hits, n = set(), 0
    for k in range(steps + 1):
        q = [a + (b - a) * k / steps for a, b in zip(j0, j1)]
        robot.setJoints(q); RDK.Update()
        if RDK.Collisions() > 0:
            new = robot_pairs()
            if new:
                n += 1; hits.update(new)
    report.append({"segment": label, "result": n, "items": sorted(hits)})
robot.setJoints(HOME)
LAYOUT["collision_report"] = report
bad_seg = [r for r in report if r["items"] or (r["result"] or 0) < 0]
print(f"Tramos verificados: {len(report)} | con colisión: {len(bad_seg)}")
for r in bad_seg:
    kind = "CONTACTO HUMANO" if r["items"] and all("Op_" in h for h in r["items"]) else "COLISIÓN"
    print(f"  {r['segment']:<18} {kind}: " + ("; ".join(r["items"]) or f"resultado {r['result']}"))

# ============================================================================ 13. VISTA Y GUARDADO
os.makedirs(CONFIG["out_dir"], exist_ok=True)
RDK.Save(os.path.join(CONFIG["out_dir"], "HRC_Kitting_LLM_Cell.rdk"))
with open(os.path.join(CONFIG["out_dir"], "cell_layout.json"), "w", encoding="utf-8") as fh:
    json.dump({"config": CONFIG, **LAYOUT}, fh, indent=2, ensure_ascii=False)
bad = [t["name"] for t in LAYOUT["targets"] if not t["reachable"]]
print(f"Celda creada ({len(LAYOUT['objects'])} geometrías). S_p = {S_p:.0f} mm | "
      f"envolvente roja {r_stop:.0f} mm | ámbar {r_warn:.0f} mm")
print("Targets no alcanzables:", bad if bad else "ninguno")
print("Guardado en:", CONFIG["out_dir"])
