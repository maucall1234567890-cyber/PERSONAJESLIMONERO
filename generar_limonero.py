"""Ejecutar en Blender mediante --background --python generar_limonero.py."""
import bpy
import math
from pathlib import Path
from mathutils import Vector

folder = Path(__file__).resolve().parent
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
bones = {}
parts = []


def bone(name, a, b, parent=None):
    bones[name] = (a, b, parent)
    return name


def material(name, color):
    m = bpy.data.materials.new(name)
    m.diffuse_color = (*color, 1)
    m.use_nodes = True
    shader = m.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (*color,1)
    shader.inputs["Roughness"].default_value = .78
    return m


lime = material("Cáscara lima", (.55, .82, .027))
gold = material("Amarillo limón", (.98, .87, .022))
leaf = material("Hoja verde", (.10, .38, .019))
vein = material("Nervaduras", (.035, .15, .012))
white = material("Marfil", (1, .97, .84))
dark = material("Pupila", (.005, .025, .010))
mouth = material("Interior boca", (.060, .017, .009))
pink = material("Lengua", (.98, .22, .12))


def oval(name, center, size, mat, joint, form=None):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=20, location=center)
    obj = bpy.context.object
    obj.name = name
    for v in obj.data.vertices:
        x, y, z = v.co
        if form:
            x, y, z = form(x, y, z)
        v.co = (size[0]*x, size[1]*y, size[2]*z)
    obj.data.materials.append(mat)
    for p in obj.data.polygons:
        p.use_smooth = True
    parts.append((obj, joint))
    return obj


def limb(name, a, b, radius, mat, joint):
    axis = Vector(b)-Vector(a)
    obj = oval(name, (Vector(a)+Vector(b))/2,
               (radius, radius, axis.length/2+radius*.65), mat, joint,
               lambda x,y,z: (x,y, math.copysign(abs(z)**.67,z)))
    obj.rotation_euler = axis.to_track_quat("Z","Y").to_euler()
    return obj


root = bone("Raiz", (0,0,0),(0,0,.3))
body = bone("Cuerpo",(0,0,2),(0,0,3.28),root)
head = bone("Cabeza",(0,0,3.28),(0,0,5.53),body)
oval("Tronco",(0,0,2.65),(.34,.28,.79),gold,body)
oval("Cuello",(0,0,3.32),(.19,.19,.25),lime,head)
def citrus(x,y,z):
    taper = 1-.09*z
    return x*taper,y*taper,z+.17*max(z,0)**20
oval("Cabeza de lima",(0,0,4.43),(1,.72,1.04),lime,head,citrus)
for s,side in [(1,".L"),(-1,".R")]:
    def pos(x,z,y=0):
        return (s*x,y,z)
    for chain, points, radii in [
        (["Brazo","Antebrazo"],[pos(.23,3.17),pos(.84,2.83),pos(1.35,2.48)],[.125,.113]),
        (["Muslo","Pierna"],[pos(.19,2.06),pos(.24,1.17),pos(.29,.31)],[.156,.15])]:
        parent = body
        for i,label in enumerate(chain):
            joint = bone(label+side,points[i],points[i+1],parent)
            limb(label+side,points[i],points[i+1],radii[i],gold,joint)
            parent = joint
    foot = bone("Pie"+side,pos(.29,.31),pos(.42,.12,-.25),"Pierna"+side)
    oval("Pie"+side,pos(.40,.17,-.12),(.34,.36,.175),gold,foot)
    palm = bone("Mano"+side,pos(1.35,2.48),pos(1.56,2.37),"Antebrazo"+side)
    oval("Palma"+side,pos(1.48,2.4),(.20,.105,.15),gold,palm)
    for i,(x,z,dx,dz) in enumerate([(1.40,2.34,.032,-.22),(1.48,2.32,.19,-.27),(1.55,2.36,.26,-.21),(1.59,2.43,.27,-.10)]):
        a,b = pos(x,z),pos(x+dx,z+dz)
        joint = bone("Dedo%d%s"%(i+1,side),a,b,palm)
        limb(joint,a,b,.064,gold,joint)
    eye = bone("Ojo"+side,pos(.38,4.45,-.62),pos(.38,4.45,-.85),head)
    oval("Esclerótica"+side,pos(.38,4.45,-.64),(.245,.137,.42),white,head)
    oval("Pupila"+side,pos(.34,4.45,-.756),(.15,.055,.306),dark,eye)
    oval("Brillo"+side,pos(.30,4.57,-.805),(.045,.018,.085),white,eye)
    lid = bone("Parpado"+side,pos(.38,4.88,-.65),pos(.38,4.48,-.65),head)
    obj = oval("Párpado"+side,pos(.38,4.875,-.76),(.253,.092,.435),lime,lid)
    basis = obj.shape_key_add(name="Abierto")
    close = obj.shape_key_add(name="Cerrar")
    for v in basis.data:
        v.co.z *= .018
    for v in close.data:
        v.co.z -= .42
    driver = close.driver_add("value").driver
    var = driver.variables.new()
    var.name = "cierre"
    var.targets[0].id = None
    driver.expression = "cierre"
    obj["driver_bone"] = lid

jaw = bone("Boca",(0,-.63,3.91),(0,-.64,3.65),head)
def smile(x,y,z):
    return (x,y,z*(.32 if z>0 else 1)+.36*x*x)
oval("Sonrisa",(0,-.62,3.91),(.36,.085,.29),mouth,jaw,smile)
oval("Lengua",(0,-.66,3.75),(.19,.054,.078),pink,jaw)
stem = bone("Tallo",(.09,0,5.48),(.43,0,5.61),head)
limb("Pecíolo",bones[stem][0],bones[stem][1],.032,vein,stem)
blade = bone("Hoja",(.40,0,5.61),(1.23,0,5.32),stem)
def blade_form(x,y,z):
    return x,y,z*(1-.8*abs(x)**3)-.60*x+.23*(1-x*x)
oval("Limbo",(.80,0,5.52),(.46,.074,.205),leaf,blade,blade_form)
limb("Nervio central",(.39,-.06,5.64),(1.22,-.045,5.40),.013,vein,blade)

arm = bpy.data.armatures.new("Esqueleto")
rig = bpy.data.objects.new("Limonero • controles FK",arm)
scene.collection.objects.link(rig)
bpy.context.view_layer.objects.active = rig
rig.select_set(True)
bpy.ops.object.mode_set(mode="EDIT")
for name,(a,b,parent) in bones.items():
    eb = arm.edit_bones.new(name)
    eb.head,eb.tail = a,b
    if parent:
        eb.parent = arm.edit_bones[parent]
bpy.ops.object.mode_set(mode="OBJECT")
rig.show_in_front = True
arm.display_type = "STICK"
for obj,joint in parts:
    if "driver_bone" in obj:
        pb = rig.pose.bones[joint]
        pb["Cerrar"] = 0.0
        pb.id_properties_ui("Cerrar").update(min=0,max=1)
        target = obj.data.shape_keys.animation_data.drivers[0].driver.variables[0].targets[0]
        target.id = rig
        target.data_path = pb.path_from_id()+'["Cerrar"]'
    group = obj.vertex_groups.new(name=joint)
    group.add(list(range(len(obj.data.vertices))),1,"REPLACE")
    obj.parent = rig
    mod = obj.modifiers.new("Rig","ARMATURE")
    mod.object = rig

def aim(obj,target):
    obj.rotation_euler = (Vector(target)-obj.location).to_track_quat("-Z","Y").to_euler()
bpy.ops.object.camera_add(location=(0,-12,2.86))
cam = bpy.context.object
aim(cam,(0,0,2.86))
cam.data.type = "ORTHO"
cam.data.ortho_scale = 6.35
scene.camera = cam
for name,loc,power,size in [("Principal",(-4,-5,8),620,5),("Relleno",(4,-4,5),360,5),("Borde",(2,3,7),700,3)]:
    bpy.ops.object.light_add(type="AREA",location=loc)
    light = bpy.context.object
    light.name = name
    light.data.energy,light.data.shape,light.data.size = power,"DISK",size
    aim(light,(0,0,3))
scene.render.engine = "CYCLES"
scene.cycles.samples = 12
scene.render.film_transparent = True
scene.render.resolution_x = scene.render.resolution_y = 512
scene.render.resolution_percentage = 100
scene.render.image_settings.color_mode = "RGBA"
scene.view_settings.view_transform = "Standard"
scene.render.use_freestyle = True
scene.render.line_thickness = 1.35
line_set = scene.view_layers[0].freestyle_settings.linesets[0]
line_set.linestyle = bpy.data.linestyles.new("Tinta verde")
style = line_set.linestyle
style.color = (.027,.072,.009)
style.thickness = 1.2

# Ciclo de reposo. Renderizar los fotogramas 1–24 conserva el bucle.
scene.render.fps = 24
scene.frame_end = 24
for frame,rise,angle in [(1,0,0),(7,.035,.045),(13,0,0),(19,-.022,-.035),(25,0,0)]:
    torso = rig.pose.bones[body]
    torso.location.y = rise
    torso.keyframe_insert(data_path="location",frame=frame)
    tip = rig.pose.bones[blade]
    tip.rotation_mode = "XYZ"
    tip.rotation_euler.z = angle
    tip.keyframe_insert(data_path="rotation_euler",frame=frame)
for side in (".L",".R"):
    lid = rig.pose.bones["Parpado"+side]
    for frame,amount in [(1,0),(15,0),(17,1),(19,0),(25,0)]:
        lid["Cerrar"] = amount
        lid.keyframe_insert(data_path='["Cerrar"]',frame=frame)
rig.animation_data.action.name = "Reposo • 24 fotogramas"
scene.frame_set(1)
for area in bpy.context.screen.areas:
    if area.type == "VIEW_3D":
        area.spaces.active.region_3d.view_perspective = "CAMERA"
bpy.ops.object.select_all(action="DESELECT")
rig.select_set(True)
bpy.context.view_layer.objects.active = rig
scene.render.filepath = str(folder/"Limonero_sprite.png")
bpy.context.preferences.filepaths.save_version = 0
bpy.ops.wm.save_as_mainfile(filepath=str(folder/"Limonero.blend"))
bpy.ops.render.render(write_still=True)
