"""Raspberry Pi 4 Model B a escala real, desde su plano mecanico publico.

Coordenadas en mm sobre la placa: origen en la esquina inferior izquierda
vista desde arriba, con el conector GPIO arriba y los USB/Ethernet a la
derecha. El Empty `Pi4` queda en el centro de la placa, a la altura de su cara
inferior, para mover y girar todo junto.

Fuente: plano "Raspberry Pi 4 Model B mechanical drawing" (85 x 56 mm,
agujeros M2.5 a 3,5 mm de los bordes, separados 58 x 49 mm). Los chips son
aproximados: quedan bajo los disipadores y se ajustan con las fotos del usuario.
Disipadores, ventilador y caja se agregan cuando haya fotos y medidas.
"""

import sys
from pathlib import Path

import bmesh
import bpy

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib  # noqa: E402

import common  # noqa: E402

importlib.reload(common)  # usar siempre la version actual de common.py
from common import mm  # noqa: E402

W, D, T = 85.0, 56.0, 1.4  # placa: ancho, fondo y grosor (mm)
CX, CY = W / 2, D / 2


def at(x, y, z):
    """Coordenadas de la placa (mm) a coordenadas centradas en el Empty."""
    return (x - CX, y - CY, z)


coll = common.get_collection("Electronica/Pi4")
common.clear_collection(coll)

M = {
    "pcb": common.material("Pi4_PCB_verde", (0.015, 0.16, 0.05), roughness=0.45),
    "metal": common.material("Metal_estanado", (0.78, 0.78, 0.8), metallic=1.0, roughness=0.3),
    "negro": common.material("Plastico_negro", (0.012, 0.012, 0.012), roughness=0.45),
    "azul": common.material("USB3_azul", (0.01, 0.1, 0.55), roughness=0.4),
    "oro": common.material("Pines_dorados", (1.0, 0.72, 0.3), metallic=1.0, roughness=0.25),
    "chip": common.material("Chip_negro", (0.03, 0.03, 0.03), roughness=0.6),
    "blanco": common.material("Plastico_blanco", (0.8, 0.8, 0.75), roughness=0.5),
}

root = common.empty("Pi4", coll)


def part(name, size, center, mat):
    return common.box(name, size, at(*center), coll, M[mat], root)


# Placa con esquinas de 3 mm y los cuatro agujeros de montaje.
pcb = common.rounded_plate("Pi4_PCB", W, D, T, 3.0, at(0, 0, 0), coll, M["pcb"], root)
holes = [(3.5, 3.5), (61.5, 3.5), (3.5, 52.5), (61.5, 52.5)]
common.drill(pcb, 1.35, [at(x, y, 0)[:2] for x, y in holes])

# Conector GPIO de 40 pines (2 x 20, paso 2,54 mm), centrado en x = 32,5.
part("Pi4_GPIO_base", (50.8, 5.08, 2.5), (32.5, 52.5, T + 1.25), "negro")
bm = bmesh.new()
for i in range(20):
    for row in (-1.27, 1.27):
        x, y, _ = mm(*at(32.5 + (i - 9.5) * 2.54, 52.5 + row, 0))
        geom = bmesh.ops.create_cube(bm, size=1.0)
        for v in geom["verts"]:
            v.co.x = v.co.x * mm(0.64) + x
            v.co.y = v.co.y * mm(0.64) + y
            v.co.z = v.co.z * mm(8.5) + mm(T + 4.25)
pins = bpy.data.meshes.new("Pi4_GPIO_pines")
bm.to_mesh(pins)
bm.free()
pins_obj = bpy.data.objects.new("Pi4_GPIO_pines", pins)
pins_obj.data.materials.append(M["oro"])
pins_obj.parent = root
coll.objects.link(pins_obj)

# Borde derecho: Ethernet arriba, USB 3 (azul) al centro y USB 2 abajo.
part("Pi4_Ethernet", (21.3, 16.0, 13.5), (76.75, 45.75, T + 6.75), "metal")
part("Pi4_Ethernet_boca", (0.4, 12.0, 9.0), (87.6, 45.75, T + 5.5), "negro")
for name, y, mat in (("Pi4_USB3", 27.0, "azul"), ("Pi4_USB2", 9.0, "negro")):
    part(name, (17.5, 13.1, 15.6), (78.75, y, T + 7.8), "metal")
    for k, z in enumerate((T + 4.0, T + 11.6)):
        part(f"{name}_boca{k}", (0.4, 12.0, 4.5), (87.6, y, z), mat)

# Borde inferior: USB-C (energia), dos micro HDMI y jack de audio de 3,5 mm.
part("Pi4_USBC", (8.94, 7.35, 3.16), (11.2, 2.375, T + 1.58), "metal")
part("Pi4_microHDMI0", (6.5, 7.5, 3.0), (26.0, 2.75, T + 1.5), "metal")
part("Pi4_microHDMI1", (6.5, 7.5, 3.0), (39.5, 2.75, T + 1.5), "metal")
part("Pi4_Audio", (7.0, 12.0, 6.0), (53.5, 7.0, T + 3.0), "negro")
common.cylinder("Pi4_Audio_boca", 3.0, 2.5, at(53.5, -1.25, T + 3.0), coll,
                M["negro"], root, axis="Y")

# Chips principales (posicion aproximada; van bajo los disipadores).
part("Pi4_SoC", (15.0, 15.0, 1.3), (29.0, 32.0, T + 0.65), "metal")
part("Pi4_RAM", (10.0, 15.0, 1.0), (45.5, 32.0, T + 0.5), "chip")
part("Pi4_USB_ctrl", (7.0, 7.0, 1.0), (64.0, 23.0, T + 0.5), "chip")

# Conectores de pantalla (DSI) y camara (CSI), y lector microSD por debajo.
part("Pi4_DSI", (5.0, 21.0, 5.5), (5.0, 28.0, T + 2.75), "blanco")
part("Pi4_CSI", (5.0, 21.0, 5.5), (45.0, 12.0, T + 2.75), "blanco")
part("Pi4_microSD", (12.0, 11.5, 1.4), (7.0, 28.0, -0.7), "metal")

print("Pi4:", len(coll.objects), "objetos")
