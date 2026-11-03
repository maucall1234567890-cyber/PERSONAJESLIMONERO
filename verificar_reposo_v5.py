"""Verify the saved animation, its loop and the encoded movie; sample the movie itself."""
from pathlib import Path
import bpy
import bmesh
import numpy as np

folder = Path(bpy.data.filepath).parent
scene = bpy.context.scene
rig = next(o for o in scene.objects if o.type=="ARMATURE")
body = next(o for o in scene.objects if o.name.startswith("Piel continua"))
p = rig.pose.bones
bm = bmesh.new()
bm.from_mesh(body.data)
assert all(e.is_manifold for e in bm.edges)
bm.free()
deform = {g.index for g in body.vertex_groups
          if (b:=rig.data.bones.get(g.name)) and b.use_deform}
assert all(abs(sum(g.weight for g in v.groups if g.group in deform)-1.)<1.e-5 for v in body.data.vertices)


def geometry():
    vs = body.evaluated_get(bpy.context.evaluated_depsgraph_get()).data.vertices
    xyz = np.empty(len(vs)*3,np.float32)
    vs.foreach_get("co",xyz)
    return xyz.reshape(-1,3)


scene.frame_set(1)
base = geometry()
feet = base[:,2]<.23
delta = 0.
for frame in range(1,74):
    scene.frame_set(frame)
    shape = geometry()
    delta = max(delta,float(np.max(abs(shape[feet]-base[feet]))))
    if frame==22:
        assert .22<p["Boca"]["MBP"]<.24
        assert p["Cabeza"]["Mirada horizontal"]>.7
    if frame==32:
        assert .70<p["Párpado.L"]["Cerrar"]<.80
    if frame==33:
        assert p["Párpado.L"]["Cerrar"]==p["Párpado.R"]["Cerrar"]==1.
    if 49<=frame<=72:
        assert p["Boca"]["MBP"]==1.
        assert bpy.data.objects["Sonrisa • línea en el cierre"].data.shape_keys.key_blocks["Sonrisa cerrada"].value>.999
assert delta<1e-5
assert np.max(abs(shape-base))<1e-5
print("Loop and feet verified; max foot movement",delta,flush=True)

video = folder/"Limonero_reposo.mp4"
assert video.is_file() and video.stat().st_size>2000
check = bpy.data.scenes.new("Comprobar video")
check.render.resolution_x,check.render.resolution_y = 504,686
check.render.resolution_percentage = 100
check.render.fps = 24
check.view_settings.view_transform = "Standard"
clip = check.sequence_editor_create().strips.new_movie("Reposo",str(video),1,1)
assert clip.frame_duration==72 and abs(clip.fps-24)<.01,(clip.frame_duration,clip.fps)
check.render.image_settings.file_format = "PNG"
for frame in (1,20,33,49):
    check.frame_set(frame)
    check.render.filepath = str(folder/f"control_{frame}.png")
    bpy.ops.render.render(write_still=True,scene=check.name)
print("Decoded video:",clip.frame_duration,"frames at",clip.fps,"fps")
