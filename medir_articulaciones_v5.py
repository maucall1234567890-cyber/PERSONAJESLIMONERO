"""Compare joint strain across the same FK poses on the actual v4 skin."""
import bpy
import numpy as np

sc = bpy.context.scene
r = next(o for o in sc.objects if o.type=="ARMATURE")
b = next(o for o in sc.objects if o.name.startswith("Piel continua"))
p = r.pose.bones
if r.animation_data and r.animation_data.action:
    r.animation_data.action = None
for joint in p:
    joint.matrix_basis.identity()
p["Boca"]["MBP"] = 0.
vs = np.array([v.co[:] for v in b.data.vertices])
x,y,z = vs.T
e = np.array([v.vertices[:] for v in b.data.edges])
length = np.linalg.norm(vs[e[:,0]]-vs[e[:,1]],axis=1)
e = e[length>.004]
length = length[length>.004]
regions = {"axila":((x-.26)/.20)**2+((z-3.12)/.20)**2<1,
           "codo":((x-.88)/.24)**2+((z-2.79)/.18)**2<1,
           "muñeca":((x-1.40)/.21)**2+((z-2.44)/.16)**2<1,
           "cadera":((x+.18)/.20)**2+((z-2.03)/.23)**2<1,
           "rodilla":((x+.225)/.17)**2+((z-1.15)/.23)**2<1,
           "tobillo":((x+.28)/.17)**2+((z-.29)/.17)**2<1}
p["Cuerpo"]["Brazos en descanso"]=0
for n,angles in {"Brazo.L":(16,11,74),"Antebrazo.L":(42,2,42),"Palma.L":(33,0,15),
                 "Muslo.R":(-35,9,12),"Pierna.R":(62,0,0),"Cuerpo":(14,9,-6),"Pie.R":(22,0,0)}.items():
    p[n].rotation_euler = np.radians(angles)
r.update_tag()
bpy.context.view_layer.update()
def report():
    xyz = np.array([v.co[:] for v in b.evaluated_get(bpy.context.evaluated_depsgraph_get()).data.vertices])
    ratio = np.linalg.norm(xyz[e[:,0]]-xyz[e[:,1]],axis=1)/length
    for name,mask in regions.items():
        ids=np.where(mask[e].all(axis=1))[0]
        strain=abs(np.log(np.maximum(ratio[ids],1e-6)))
        ix=ids[strain.argmax()]
        print(name,np.round(np.percentile(strain,[90,99]),3),vs[e[ix]].mean(axis=0).round(3))
report()
