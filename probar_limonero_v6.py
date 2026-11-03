"""Read-only pose tests for both versions and render the v6 motion study."""
from pathlib import Path
import math
import bpy
import numpy as np
from mathutils import Matrix,Vector
from mathutils.bvhtree import BVHTree

s=bpy.context.scene
is_v6=Path(bpy.data.filepath).stem=="Limonero_v6"
rig=next(o for o in s.objects if o.type=="ARMATURE")
ob=next(o for o in s.objects if o.name.startswith("Piel continua"))
p=rig.pose.bones
rest=np.array([v.co[:] for v in ob.data.vertices])
x,y,z=rest.T
es=np.array([e.vertices[:] for e in ob.data.edges])
length=np.linalg.norm(rest[es[:,0]]-rest[es[:,1]],axis=1)
es=es[length>.004]
length=length[length>.004]
def mesh():
    rig.update_tag()
    bpy.context.view_layer.update()
    data=ob.evaluated_get(bpy.context.evaluated_depsgraph_get()).data
    points=np.empty(len(rest)*3,np.float32)
    data.vertices.foreach_get("co",points)
    return points.reshape(-1,3)
def pose(name,angles):
    p[name].rotation_euler=np.radians(angles)
def view(file,target,location,span):
    camera=s.camera
    camera.location=location
    camera.rotation_euler=(Vector(target)-camera.location).to_track_quat("-Z","Y").to_euler()
    camera.data.ortho_scale=span
    s.render.image_settings.media_type="IMAGE"
    s.render.image_settings.file_format="PNG"
    s.render.filepath="//"+file+".png"
    bpy.ops.render.render(write_still=True,scene=s.name)

s.frame_set(1)
base=mesh()
foot_mask=rest[:,2]<.18
if is_v6:
    # Check tips throughout the cycle in world space. Test non-neighboring
    # finger surfaces for actual intersections, not overlap in projection.
    polygons=[list(face.vertices) for face in ob.data.polygons]
    subset={}
    for side in (".L",".R"):
        for finger in ("Pulgar","Índice","Medio","Menor"):
            groups=[ob.vertex_groups[finger+"."+n+side].index for n in ("1","2")]
            owned=np.array([sum(g.weight for g in v.groups if g.group in groups)>.68 for v in ob.data.vertices])
            subset[finger+side]=[face for face in polygons if owned[face].all()]
    def tree_for_faces(points,faces):
        ids=np.unique(np.concatenate(faces))
        index=np.full(len(points),-1,int)
        index[ids]=np.arange(len(ids))
        return BVHTree.FromPolygons(points[ids].tolist(),[index[f].tolist() for f in faces])
    hand_faces=[f for f in polygons if (abs(x[f])>1.38).all()]
    thigh_faces=[f for f in polygons if ((abs(x[f])<.50)&(z[f]<2.14)&(z[f]>1.05)).all()]
    for frame in (1,18,33,49,72,73):
        s.frame_set(frame)
        points=mesh()
        assert np.max(abs(points[foot_mask]-base[foot_mask]))<1.e-5
        if frame==73:
            assert np.max(abs(points-base))<1.e-5
        assert not tree_for_faces(points,hand_faces).overlap(tree_for_faces(points,thigh_faces))
        for side in (".L",".R"):
            bounds=[]
            for digit in ("Pulgar","Índice","Medio","Menor"):
                name=digit+side
                faces=subset[name]
                assert faces
                tree=tree_for_faces(points,faces)
                for other,other_tree in bounds:
                    num=len(tree.overlap(other_tree))
                    if num:
                        raise RuntimeError(f"Contacto entre dedos en {frame}: {name}, {other}, {num}")
                bounds.append((name,tree))
            for digit in ("Índice","Medio","Menor"):
                b=p[digit+".2"+side]
                assert (b.tail-b.head).normalized().z<-.88
    s.frame_set(1)
    view("Descanso_frontal",(0,0,2.87),(0,-16,5.45),6.74)
    view("Mano_en_descanso",(.52,-.08,1.57),(2.8,-8,2.05),1.0)
rig.animation_data.action=None
for b in p:
    b.matrix_basis=Matrix.Identity(4)
p["Cuerpo"]["Brazos en descanso"]=0.
p["Boca"]["MBP"]=0.
for side in (".L",".R"):
    p["Palma"+side]["Giro"]=0.
base=mesh()
regions={"hombro":((x-.27)/.22)**2+((z-3.12)/.21)**2<1,
         "codo":((x-.88)/.24)**2+((z-2.79)/.18)**2<1,
         "torso":(abs(x)<.30)&(z>2.01)&(z<2.65),
         "antebrazo":(x>.98)&(x<1.48)&(z>2.34)&(z<2.83)}
tests=[("Elevar",{"Brazo.L":(13,9,74),"Antebrazo.L":(42,1,42)}),
       ("Alcanzar",{"Brazo.L":(52,8,10),"Antebrazo.L":(72,0,0)}),
       ("Inclinar",{"Cuerpo":(18,13,13)}),
       ("Girar antebrazo",{"Antebrazo.L":(18,9,12),"Palma.L":(30,0,12)})]
for title,values in tests:
    for b in p:
        b.matrix_basis=Matrix.Identity(4)
    p["Palma.L"]["Giro"]=50. if title=="Girar antebrazo" else 0.
    for name,angles in values.items():
        pose(name,angles)
    points=mesh()
    strain=abs(np.log(np.maximum(np.linalg.norm(points[es[:,0]]-points[es[:,1]],axis=1)/length,1.e-7)))
    print(title,{n:round(float(np.percentile(strain[mask[es].all(axis=1)],99)),4) for n,mask in regions.items()},flush=True)
    if title=="Elevar":
        sternum=(abs(x)<.154)&(z>2.75)&(z<3.18)
        print("Sternum maximum",abs(points[sternum]-base[sternum]).max(),flush=True)
        if is_v6:
            view("Prueba_brazo_elevado",(0,0,3.0),(5,-15,4.35),6.74)
    if title=="Inclinar":
        assert np.max(abs(points[foot_mask]-base[foot_mask]))<1.e-5
if is_v6:
    for b in p:
        b.matrix_basis=Matrix.Identity(4)
    p["Cuerpo"]["Brazos en descanso"]=1.
    p["Palma.L"]["Giro"]=-12.
    pose("Clavícula.L",(6,0,7))
    pose("Brazo.L",(12,7,82))
    pose("Antebrazo.L",(15,-6,54))
    pose("Cuerpo",(8,-4,-6))
    for digit,a in (("Pulgar",9),("Índice",16),("Medio",28),("Menor",35)):
        pose(digit+".1.L",(a,0,0))
        pose(digit+".2.L",(a*.65,0,0))
    p["Boca"]["MBP"]=.83
    mesh()
    view("Prueba_torso_y_hombro",(0,0,2.87),(6.2,-15,5.48),6.74)
