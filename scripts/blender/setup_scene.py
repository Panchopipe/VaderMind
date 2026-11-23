"""Prepara la escena base de VaderMind 3D: unidades, visor y colecciones.

Uso, dentro de Blender (o via MCP):
    import runpy; runpy.run_path(r"<VaderMind>/scripts/blender/setup_scene.py")
Es seguro repetirlo: no borra lo ya modelado.
"""

import sys
from pathlib import Path

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib  # noqa: E402

import common  # noqa: E402

importlib.reload(common)  # usar siempre la version actual de common.py

COLLECTIONS = [
    "Casco",
    "Electronica/Pi4",
    "Electronica/PowerBank",
    "Electronica/Parlantes",
    "Electronica/Ojos",
    "Electronica/Ventilador",
    "Electronica/Microfono",
    "Electronica/Cableado",
    "Escena",
]

DEFAULT_OBJECTS = ("Cube", "Light", "Camera")

scene = bpy.context.scene

# Unidades: metros a escala real, sin factor de escala.
scene.unit_settings.system = "METRIC"
scene.unit_settings.scale_length = 1.0
scene.unit_settings.length_unit = "MILLIMETERS"  # solo cambia como se muestran

# Borrar los objetos de fabrica de un archivo nuevo.
for name in DEFAULT_OBJECTS:
    obj = bpy.data.objects.get(name)
    in_default = obj is not None and any(
        c.name == "Collection" for c in obj.users_collection
    )
    if in_default:
        bpy.data.objects.remove(obj, do_unlink=True)
default_coll = bpy.data.collections.get("Collection")
if default_coll is not None and not default_coll.all_objects:
    bpy.data.collections.remove(default_coll)

for path in COLLECTIONS:
    common.get_collection(path)

# Visor: recorte cercano de 1 mm para acercarse a piezas pequenas sin que
# desaparezcan, y rejilla de 1 cm.
for window in bpy.context.window_manager.windows:
    for area in window.screen.areas:
        if area.type == "VIEW_3D":
            space = area.spaces.active
            space.clip_start = 0.001
            space.clip_end = 100.0
            space.overlay.grid_scale = 0.01

print("Colecciones:", [c.name for c in bpy.data.collections])
