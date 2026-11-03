"""Run against Limonero_v4.blend. Builds the animated scene in a separate folder."""
from pathlib import Path
import math
import bpy
import numpy as np

out = Path(__file__).resolve().parent/"Limonero_v5"
out.mkdir(exist_ok=True)
sc = bpy.context.scene
sprite = next(s for s in bpy.data.scenes if s.render.film_transparent)
rig = next(o for o in sc.objects if o.type=="ARMATURE")
skin = next(o for o in sc.objects if o.name.startswith("Piel continua"))
pb = rig.pose.bones


def refresh():
    rig.update_tag()
    bpy.context.view_layer.update()


def coords(ob):
    vs = ob.evaluated_get(bpy.context.evaluated_depsgraph_get()).data.vertices
    data = np.empty(3*len(vs),np.float32)
    vs.foreach_get("co",data)
    return data.reshape(-1,3)


def smooth(x):
    u = np.clip(x,0.,1.)
    return u*u*(3-2*u)


rest = np.array([v.co[:] for v in skin.data.vertices])
x,y,z = rest.T
edges = np.array([e.vertices[:] for e in skin.data.edges])
edges = edges[np.linalg.norm(rest[edges[:,0]]-rest[edges[:,1]],axis=1)>.004]
rest_length = np.linalg.norm(rest[edges[:,0]]-rest[edges[:,1]],axis=1)

# A compact delta-smoothing pass relaxes the *deformed* surface around moving
# joints while preserving rest detail. Keep the sternum, mouth and soles outside.
mask = np.zeros(len(rest))
ux = abs(x)
def region(cx,cz,rx,rz):
    return smooth(1-np.sqrt(((ux-cx)/rx)**2+((z-cz)/rz)**2))
for center in [(.253,3.12,.28,.29),(.89,2.80,.35,.24),(1.40,2.44,.24,.21),
               (.215,2.07,.255,.28),(.225,1.15,.21,.30),(.28,.29,.20,.19)]:
    mask = np.maximum(mask,region(*center))
mask *= np.where(z>2.7,smooth((ux-.16)/.085),1.)
mask[z<.15] = 0.
group = skin.vertex_groups.new(name="Corrección articular • volumen")
for i in np.where(mask>.001)[0]:
    group.add([int(i)],float(mask[i]),"REPLACE")
correction = skin.modifiers.new("Corrección de volumen en flexión","CORRECTIVE_SMOOTH")
correction.vertex_group = group.name
correction.smooth_type = "LENGTH_WEIGHTED"
correction.factor,correction.iterations = .65,10
# The rest-detail filter is useful near idle. In deeper flexion the longer
# bone blends should carry the bend: repeated surface smoothing otherwise
# pinches a few short edges on the inside of the knee and wrist.
driver = correction.driver_add("factor").driver
terms = []
for side in (".L",".R"):
    for name,axis in (("Brazo",2),("Antebrazo",0),("Palma",0),("Muslo",0),("Pierna",0)):
        var = driver.variables.new()
        var.name = "a"+str(len(terms))
        var.targets[0].id = rig
        var.targets[0].data_path = pb[name+side].path_from_id()+f".rotation_euler[{axis}]"
        terms.append("abs("+var.name+")")
driver.expression = ".65-.62*min(1,max(0,(max("+",".join(terms)+")-.18)/.45))"


def stress_metrics(points):
    ratio = np.linalg.norm(points[edges[:,0]]-points[edges[:,1]],axis=1)/rest_length
    use = mask[edges].mean(axis=1)>.3
    return np.percentile(abs(np.log(np.maximum(ratio[use],1e-6))),[90,99])


def tune_joint_weights():
    # Use a longer transition across the deltoid and the elbow/knee creases.
    # Preserve all unrelated/end-bone influences and the original total.
    groups = skin.vertex_groups
    def weights(name):
        idx = groups[name].index
        return np.array([next((g.weight for g in v.groups if g.group==idx),0.)
                         for v in skin.data.vertices])
    def transfer(first,second,share,selection):
        wa,wb = weights(first),weights(second)
        total = wa+wb
        for i in np.flatnonzero(selection & (total>.00001)):
            for group,w in ((first,1-share[i]),(second,share[i])):
                groups[group].add([int(i)],float(total[i]*w),"REPLACE")
    for side,sign in ((".L",1),(".R",-1)):
        u = sign*x
        selection = (u>=0)&(u<.49)&(z>2.70)&(z<3.29)
        lateral = smooth((u-.24)/.16)
        height = smooth((z-2.84)/.25)
        share = smooth((u-.163)/.258)*(height+(1-height)*lateral)
        transfer("Cuerpo","Brazo"+side,share,selection)
        for upper,lower,span in [("Brazo","Antebrazo",.30),("Muslo","Pierna",.35)]:
            a,m,c = (np.array(v) for v in (pb[upper+side].bone.head_local,
                                            pb[lower+side].bone.head_local,
                                            pb[lower+side].bone.tail_local))
            along = (rest-m) @ ((c-a)/np.linalg.norm(c-a))
            transfer(upper+side,lower+side,smooth(.5+along/(2*span)),abs(along)<span)


def scapula_follow():
    for side,sign in ((".L",1),(".R",-1)):
        helper = pb["Base Brazo"+side]
        for channel,amount in ((0,sign*.033),(2,.016)):
            drv = helper.driver_add("location",channel).driver
            for varname,bone,prop in [("arm","Brazo"+side,".rotation_euler[2]"),
                                      ("rest","Cuerpo",'["Brazos en descanso"]')]:
                var = drv.variables.new()
                var.name = varname
                var.targets[0].id = rig
                var.targets[0].data_path = pb[bone].path_from_id()+prop
            # The animator still owns the FK bone. Track only elevation above the bind pose.
            drv.expression = f"{amount:.6f}*min(1,max(0,({sign}*arm-.75049*rest)/1.29))**1.4"


def rig_pose(values):
    for name,angles in values.items():
        pb[name].rotation_euler = np.radians(angles)
    refresh()


pb["Cuerpo"]["Brazos en descanso"] = 0.
stress = {"Brazo.L":(16,11,74),"Antebrazo.L":(42,2,42),"Palma.L":(33,0,15),
          "Muslo.R":(-35,9,12),"Pierna.R":(62,0,0),"Cuerpo":(14,9,-6)}
rig_pose(stress)
correction.show_viewport = False
bpy.context.view_layer.update()
before = stress_metrics(coords(skin))
tune_joint_weights()
scapula_follow()
correction.show_viewport = True
bpy.context.view_layer.update()
after = stress_metrics(coords(skin))
print("Log edge strain P90/P99, before -> after:",before,after,flush=True)
rig_pose({name:(0,0,0) for name in stress})
pb["Cuerpo"]["Brazos en descanso"] = 1.


def prop_driver(target,channel,label,path,expression,rotation_axis=None):
    curve = target.driver_add(channel)
    driver = curve.driver
    variable = driver.variables.new()
    variable.name = "valor"
    variable.targets[0].id = rig
    variable.targets[0].data_path = pb[label].path_from_id()+'["'+path+'"]'
    if rotation_axis:
        rot = driver.variables.new()
        rot.name,rot.type = "angulo","TRANSFORMS"
        rot.targets[0].id = rig
        rot.targets[0].bone_target,rot.targets[0].transform_type = rotation_axis
        rot.targets[0].transform_space = "LOCAL_SPACE"
    driver.expression = expression


for prop in ("Mirada horizontal","Mirada vertical"):
    pb["Cabeza"][prop] = 0.
    pb["Cabeza"].id_properties_ui(prop).update(min=-1.,max=1.,description="Mirada conjunta; cada hueso Mirada admite un ajuste individual")


def paint_eye(side,sign):
    # Attach the pupil to the deforming sclera in rest UVs. Its entire shape
    # follows the eyelid retraction and is always clipped by the oval outline.
    ob = bpy.data.objects["Óvalo marfil"+side]
    geo = ob.data
    uv = geo.uv_layers.new(name="Mirada")
    xy = np.array([((v.co.x-sign*.365)/.248,(v.co.z-4.365)/.425) for v in geo.vertices])
    ids = [loop.vertex_index for loop in geo.loops]
    uv.data.foreach_set("uv",xy[ids].astype(np.float32).ravel())
    mat = ob.data.materials[0].copy()
    mat.name = "Ojo móvil"+side
    ob.data.materials[0] = mat
    tree = mat.node_tree
    nodes,link = tree.nodes,tree.links.new
    uvnode = nodes.new("ShaderNodeUVMap")
    uvnode.uv_map = "Mirada"
    center = nodes.new("ShaderNodeCombineXYZ")
    for axis,prop,offset,scale,bone_axis in [
            (0,"Mirada horizontal",-.015/.248,.32,"ROT_Z"),
            (1,"Mirada vertical",-.006/.425,.20,"ROT_X")]:
        prop_driver(center.inputs[axis],"default_value","Cabeza",prop,
                    f"{offset}+{scale}*max(-1,min(1,valor+2*angulo))",("Mirada"+side,bone_axis))
    offset = nodes.new("ShaderNodeVectorMath")
    offset.operation = "SUBTRACT"
    link(uvnode.outputs["UV"],offset.inputs[0])
    link(center.outputs[0],offset.inputs[1])

    def oval(source,rx,rz):
        div = nodes.new("ShaderNodeVectorMath")
        div.operation = "DIVIDE"
        link(source,div.inputs[0])
        div.inputs[1].default_value = (rx,rz,1.)
        length = nodes.new("ShaderNodeVectorMath")
        length.operation = "LENGTH"
        link(div.outputs[0],length.inputs[0])
        edge = nodes.new("ShaderNodeMapRange")
        edge.interpolation_type = "SMOOTHSTEP"
        link(length.outputs["Value"],edge.inputs["Value"])
        for name,val in [("From Min",.982),("From Max",1.012),("To Min",1.),("To Max",0.)]:
            edge.inputs[name].default_value = val
        return edge.outputs[0]

    current = nodes.get("Principled BSDF").outputs[0]
    spec = [(offset.outputs[0],.155/.248,.291/.425,"Pupila"+side)]
    highlight = nodes.new("ShaderNodeVectorMath")
    highlight.operation = "SUBTRACT"
    link(offset.outputs[0],highlight.inputs[0])
    highlight.inputs[1].default_value = (-.038/.248,.118/.425,0)
    spec.append((highlight.outputs[0],.043/.248,.079/.425,"Destello"+side))
    for source,rx,rz,name in spec:
        patch = bpy.data.objects[name]
        source_bs = patch.data.materials[0].node_tree.nodes.get("Principled BSDF")
        bs = nodes.new("ShaderNodeBsdfPrincipled")
        for field in ("Base Color","Roughness","Specular IOR Level"):
            bs.inputs[field].default_value = source_bs.inputs[field].default_value
        mix = nodes.new("ShaderNodeMixShader")
        link(oval(source,rx,rz),mix.inputs[0])
        link(current,mix.inputs[1])
        link(bs.outputs[0],mix.inputs[2])
        current = mix.outputs[0]
        patch.hide_render = patch.hide_viewport = True
    link(current,nodes.get("Material Output").inputs["Surface"])


for side,sign in ((".L",1),(".R",-1)):
    paint_eye(side,sign)

# Trace the actual lower-to-upper lip seal on the evaluated closed face.
# Rays sample the camera-facing lip directly; a tapered ribbon lifts out
# of the skin only during the final part of the MBP control.
from mathutils import Vector
pb["Boca"]["MBP"] = 1.
refresh()
evaluated = skin.evaluated_get(bpy.context.evaluated_depsgraph_get())
vs,fs,closed = [],[],[]
for i in range(81):
    t = (i+.02)/80.04
    u = 2*t-1
    xx = -.008+.352*u
    zz = 3.86+.052*((xx+.008)/.365)**2-.002
    width = .012*(math.sin(math.pi*t)**.68)
    for j,edge in enumerate((-1.,0.,1.)):
        zz1 = zz+edge*width
        hit,point,_,_ = evaluated.ray_cast(Vector((xx,-2,zz1)),Vector((0,1,0)))
        assert hit
        # A central ray may hit the inside of the narrow closed cavity.
        # Sample the surrounding skin instead of dipping into that slit.
        _,above,_,_ = evaluated.ray_cast(Vector((xx,-2,zz+.019)),Vector((0,1,0)))
        _,below,_,_ = evaluated.ray_cast(Vector((xx,-2,zz-.019)),Vector((0,1,0)))
        surface_y = min(point.y,(above.y+below.y)*.5+(above.y-below.y)*edge*width/.038)
        vs.append((xx,surface_y+.035,zz))
        closed.append((xx,surface_y-.004-.003*(1-abs(edge)),zz1))
        if i and j:
            n = 3*i+j
            fs.append((n-4,n-1,n,n-3))
mesh = bpy.data.meshes.new("Trazo de la sonrisa")
mesh.from_pydata(vs,[],fs)
mesh.update()
lip = bpy.data.objects.new("Sonrisa • línea en el cierre",mesh)
skin.users_collection[0].objects.link(lip)
mat = bpy.data.materials.new("Sonrisa • marrón oliva")
mat.diffuse_color = (.08,.039,.012,1.)
mat.use_nodes = True
bs = mat.node_tree.nodes.get("Principled BSDF")
bs.inputs["Base Color"].default_value = mat.diffuse_color
bs.inputs["Roughness"].default_value = .82
mesh.materials.append(mat)
lip.parent = rig
lip.vertex_groups.new(name="Cabeza").add(list(range(len(vs))),1.,"REPLACE")
mod = lip.modifiers.new("Acompañar la cabeza","ARMATURE")
mod.object = rig
lip.shape_key_add(name="Oculta")
key = lip.shape_key_add(name="Sonrisa cerrada")
key.data.foreach_set("co",np.array(closed,np.float32).ravel())
prop_driver(key,"value","Boca","MBP","max(0,min(1,(valor-.89)/.11))")


def animate_prop(bone,prop,keys):
    for frame,value in keys:
        # Leave these as float ID properties even at a 0 or 1 endpoint;
        # integer properties make Blender round the in-between frames.
        pb[bone][prop] = float(value)
        pb[bone].keyframe_insert(data_path='["'+prop+'"]',frame=frame,group=bone)


def animate_rotation(name,keys):
    pb[name].rotation_mode = "XYZ"
    for frame,angles in keys:
        pb[name].rotation_euler = np.radians(angles)
        pb[name].keyframe_insert(data_path="rotation_euler",frame=frame,group=name)


# Unhurried holds distinguish the eye saccades, single blink and lip closure.
animate_prop("Cabeza","Mirada horizontal",[(1,-.08),(8,-.08),(13,.78),(25,.78),(31,.64),(35,-.70),(48,-.70),(56,-.12),(66,-.08),(73,-.08)])
animate_prop("Cabeza","Mirada vertical",[(1,0),(8,0),(14,.36),(26,.36),(35,-.15),(46,-.15),(57,0),(73,0)])
animate_prop("Boca","MBP",[(1,1),(7,1),(22,.23),(30,.23),(41,.70),(49,1),(73,1)])
for side,sign in ((".L",1),(".R",-1)):
    animate_prop("Párpado"+side,"Cerrar",[(1,0),(30,0),(33,1),(34,1),(39,0),(73,0)])
    for digit,base,tip in [("Pulgar",4,5),("Índice",7,9),("Medio",13,15),("Menor",19,20)]:
        pb[digit+".1"+side].rotation_euler.x = math.radians(base)
        pb[digit+".2"+side].rotation_euler.x = math.radians(tip)
    animate_rotation("Palma"+side,[(1,(5,0,-2*sign)),(24,(7,0,-3*sign)),(49,(4,0,-2*sign)),(73,(5,0,-2*sign))])
    animate_rotation("Antebrazo"+side,[(1,(3,0,0)),(24,(5,0,sign)),(49,(2,0,0)),(73,(3,0,0))])
animate_rotation("Cuerpo",[(1,(.2,0,0)),(23,(-1.1,.6,.55)),(49,(.8,-.3,-.4)),(73,(.2,0,0))])
animate_rotation("Cabeza",[(1,(0,0,0)),(17,(-.6,2.4,.7)),(36,(.6,-1.7,-.65)),(58,(0,0,0)),(73,(0,0,0))])
animate_rotation("Hoja",[(1,(0,0,0)),(22,(1,0,-2.8)),(44,(-1,0,2.1)),(62,(0,0,-.4)),(73,(0,0,0))])
action = rig.animation_data.action
action.name = "Reposo • mirada, parpadeo y sonrisa"
for layer in action.layers:
    for strip in layer.strips:
        for bag in strip.channelbags:
            for curve in bag.fcurves:
                for k in curve.keyframe_points:
                    k.interpolation = "BEZIER"
                    k.handle_left_type = k.handle_right_type = "AUTO_CLAMPED"
                curve.update()
for scene in (sc,sprite):
    scene.frame_start,scene.frame_end = 1,72
    scene.render.fps = 24
    for frame,name in [(1,"Reposo"),(13,"Mirada"),(33,"Parpadeo"),(49,"Sonrisa cerrada")]:
        scene.timeline_markers.new(name,frame=frame)
    scene.frame_set(1)
    scene.cycles.samples = 16
sc.render.image_settings.media_type = "VIDEO"
sc.render.image_settings.file_format = "FFMPEG"
sc.render.ffmpeg.format = "MPEG4"
sc.render.ffmpeg.codec = "H264"
sc.render.ffmpeg.constant_rate_factor = "HIGH"
sc.render.filepath = "//Limonero_reposo.mp4"
sprite.render.filepath = "//sprites/Reposo_"
for area in bpy.context.screen.areas:
    if area.type=="VIEW_3D":
        area.spaces.active.overlay.show_overlays = False
bpy.context.preferences.filepaths.save_version = 0
bpy.ops.wm.save_as_mainfile(filepath=str(out/"Limonero_v5.blend"))
