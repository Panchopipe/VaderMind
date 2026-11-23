"""Utilidades compartidas para construir el modelo 3D de VaderMind en Blender.

Se ejecutan dentro de Blender (necesitan `bpy`). Las medidas se escriben en
milimetros, como vienen del pie de metro y de los planos, y se convierten a
metros con `mm()`, porque la escena trabaja en metros a escala real.
"""

import math

import bmesh
import bpy

MM = 0.001


def mm(*values):
    """Convierte milimetros a metros (uno o varios valores)."""
    if len(values) == 1:
        return values[0] * MM
    return tuple(v * MM for v in values)


def get_collection(path):
    """Devuelve la coleccion `path` ("Electronica/Pi4"), creandola si falta."""
    parent = bpy.context.scene.collection
    for name in path.split("/"):
        coll = bpy.data.collections.get(name)
        if coll is None:
            coll = bpy.data.collections.new(name)
        if coll.name not in parent.children:
            parent.children.link(coll)
        parent = coll
    return parent


def clear_collection(coll):
    """Borra los objetos de una coleccion para poder reconstruirla."""
    for obj in list(coll.all_objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for mesh in list(bpy.data.meshes):
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)


def material(name, color, metallic=0.0, roughness=0.5, emission=None, strength=0.0):
    """Material Principled reutilizable; el color es (r, g, b) lineal 0-1."""
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness
    if emission is not None:
        bsdf.inputs["Emission Color"].default_value = (*emission, 1.0)
        bsdf.inputs["Emission Strength"].default_value = strength
    mat.diffuse_color = (*color, 1.0)
    return mat


def _new_object(name, bm, coll, mat=None, parent=None):
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    coll.objects.link(obj)
    if mat is not None:
        obj.data.materials.append(mat)
    if parent is not None:
        obj.parent = parent
    return obj


def box(name, size_mm, center_mm, coll, mat=None, parent=None):
    """Caja de `size_mm` (x, y, z) centrada en `center_mm`, ambos en mm."""
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    sx, sy, sz = mm(*size_mm)
    cx, cy, cz = mm(*center_mm)
    for v in bm.verts:
        v.co.x = v.co.x * sx + cx
        v.co.y = v.co.y * sy + cy
        v.co.z = v.co.z * sz + cz
    return _new_object(name, bm, coll, mat, parent)


def cylinder(name, radius_mm, depth_mm, center_mm, coll, mat=None, parent=None,
             axis="Z", segments=32):
    """Cilindro en mm, orientado segun `axis` ("X", "Y" o "Z")."""
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=segments,
                          radius1=mm(radius_mm), radius2=mm(radius_mm),
                          depth=mm(depth_mm))
    if axis == "X":
        bmesh.ops.rotate(bm, verts=bm.verts, cent=(0, 0, 0),
                         matrix=_rot(math.pi / 2, "Y"))
    elif axis == "Y":
        bmesh.ops.rotate(bm, verts=bm.verts, cent=(0, 0, 0),
                         matrix=_rot(math.pi / 2, "X"))
    bmesh.ops.translate(bm, verts=bm.verts, vec=mm(*center_mm))
    return _new_object(name, bm, coll, mat, parent)


def rounded_plate(name, width_mm, depth_mm, thick_mm, radius_mm, corner_mm,
                  coll, mat=None, parent=None, segments=8):
    """Placa con esquinas redondeadas; `corner_mm` es su esquina inferior izquierda."""
    w, d, t, r = mm(width_mm, depth_mm, thick_mm, radius_mm)
    x0, y0, z0 = mm(*corner_mm)
    centers = [(x0 + w - r, y0 + r), (x0 + w - r, y0 + d - r),
               (x0 + r, y0 + d - r), (x0 + r, y0 + r)]
    start = [-math.pi / 2, 0.0, math.pi / 2, math.pi]
    bm = bmesh.new()
    ring = []
    for (cx, cy), a0 in zip(centers, start):
        for i in range(segments + 1):
            a = a0 + (math.pi / 2) * i / segments
            ring.append(bm.verts.new((cx + r * math.cos(a), cy + r * math.sin(a), z0)))
    face = bm.faces.new(ring)
    ext = bmesh.ops.extrude_face_region(bm, geom=[face])
    top = [e for e in ext["geom"] if isinstance(e, bmesh.types.BMVert)]
    bmesh.ops.translate(bm, verts=top, vec=(0, 0, t))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return _new_object(name, bm, coll, mat, parent)


def drill(target, radius_mm, centers_mm, depth_mm=10.0):
    """Perfora agujeros verticales en `target` con un booleano aplicado."""
    coll = target.users_collection[0]
    for i, (x, y) in enumerate(centers_mm):
        cutter = cylinder(f"_corte_{i}", radius_mm, depth_mm, (x, y, 0), coll)
        mod = target.modifiers.new("agujero", "BOOLEAN")
        mod.operation = "DIFFERENCE"
        mod.object = cutter
        with bpy.context.temp_override(object=target, active_object=target):
            bpy.ops.object.modifier_apply(modifier=mod.name)
        cutter_mesh = cutter.data
        bpy.data.objects.remove(cutter, do_unlink=True)
        bpy.data.meshes.remove(cutter_mesh)


def empty(name, coll, location_mm=(0, 0, 0)):
    """Empty que agrupa las piezas de un componente (se mueve todo junto)."""
    obj = bpy.data.objects.new(name, None)
    obj.empty_display_type = "PLAIN_AXES"
    obj.empty_display_size = mm(20)
    obj.location = mm(*location_mm)
    coll.objects.link(obj)
    return obj


def _rot(angle, axis):
    from mathutils import Matrix
    return Matrix.Rotation(angle, 3, axis)
