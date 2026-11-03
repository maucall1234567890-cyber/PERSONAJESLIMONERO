"""Run in the saved v4 blend. Tests never save over the neutral pose."""
import math
import bpy
import bmesh
import numpy as np
from mathutils import Vector

scene = bpy.context.scene
rig = next(o for o in scene.objects if o.type=="ARMATURE")
body = next(o for o in scene.objects if o.name.startswith("Piel continua"))
pb = rig.pose.bones
bm = bmesh.new()
bm.from_mesh(body.data)
assert all(e.is_manifold for e in bm.edges)
bm.free()
assert max(abs(sum(g.weight for g in v.groups)-1) for v in body.data.vertices)<1e-5
rest = np.array([v.co[:] for v in body.data.vertices])
chest = (abs(rest[:,0])<.16)&(rest[:,2]>2.75)&(rest[:,2]<3.18)
feet = rest[:,2]<.25


def refresh():
    rig.update_tag()
    bpy.context.view_layer.update()


def vertices():
    refresh()
    geo = body.evaluated_get(bpy.context.evaluated_depsgraph_get()).data
    xyz = np.empty(len(rest)*3,np.float32)
    geo.vertices.foreach_get("co",xyz)
    return xyz.reshape(-1,3)


def moved(a,b,subset):
    return np.linalg.norm(a[subset]-b[subset],axis=1).max()


def set_rotation(name,x,y,z):
    pb[name].rotation_euler = np.radians((x,y,z))


pb["Cuerpo"]["Brazos en descanso"] = 0.
base = vertices()
set_rotation("Brazo.L",15,10,77)
raised = vertices()
delta = moved(raised,base,chest)
print("ESTERNÓN",int(chest.sum()),"vértices, desplazamiento máximo",delta)
assert delta<1e-5
set_rotation("Cuerpo",15,11,-8)
bent = vertices()
assert moved(raised,bent,feet)<1e-5
print("PIES al inclinar el torso:",moved(raised,bent,feet))
set_rotation("Cuerpo",0,0,0)
set_rotation("Brazo.L",0,0,0)
set_rotation("Índice.1.L",50,0,4)
set_rotation("Índice.2.L",45,0,0)
curled = vertices()
tips = np.linalg.norm(rest-np.array(pb["Medio.2.L"].bone.tail_local),axis=1)<.055
assert tips.any() and moved(curled,base,tips)<1e-5
print("PUNTA del dedo vecino:",moved(curled,base,tips))

# Verify evaluated local transforms, not just the limits displayed in the UI.
def local_degrees(name):
    refresh()
    b = pb[name]
    matrix = rig.convert_space(pose_bone=b,matrix=b.matrix,from_space="POSE",to_space="LOCAL")
    return np.degrees(matrix.to_euler("XYZ"))


set_rotation("Palma.L",80,0,0)
assert abs(local_degrees("Palma.L")[0]-54)<.02
set_rotation("Índice.2.L",100,0,0)
assert abs(local_degrees("Índice.2.L")[0]-76)<.02
pb["Palma.L"]["Giro"] = 55.
for name,angle in [("Torsión1.L",55*.42),("Torsión2.L",55*.82),("Base giro Palma.L",55.)]:
    assert abs(local_degrees(name)[1]-angle)<.02
print("TOPES de muñeca y dedos y torsión escalonada verificados")
set_rotation("Brazo.L",12,3,46)
set_rotation("Antebrazo.L",28,0,34)
set_rotation("Palma.L",18,0,-9)
pb["Palma.L"]["Giro"] = 25.
for digit,angle in (("Pulgar",6),("Índice",12),("Medio",50),("Menor",58)):
    set_rotation(digit+".1.L",angle,0,0)
    set_rotation(digit+".2.L",angle*.84,0,0)
set_rotation("Cuerpo",5,7,-6)
refresh()
scene.render.filepath = "//Prueba_de_movimiento.png"
bpy.ops.render.render(write_still=True,scene=scene.name)
set_rotation("Cuerpo",0,0,0)
set_rotation("Brazo.L",0,0,0)
set_rotation("Antebrazo.L",0,0,0)
set_rotation("Palma.L",21,0,10)
pb["Palma.L"]["Giro"] = 30.
refresh()
target = rig.matrix_world @ ((pb["Palma.L"].head+pb["Medio.2.L"].tail)*.5)
cam = scene.camera
cam.location = target+Vector((.8,-8,1.2))
cam.rotation_euler = (target-cam.location).to_track_quat("-Z","Y").to_euler()
cam.data.ortho_scale = 1.02
scene.render.resolution_x,scene.render.resolution_y = 600,480
scene.render.filepath = "//Mano_flexionada.png"
bpy.ops.render.render(write_still=True,scene=scene.name)
