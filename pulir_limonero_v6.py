"""Build v6 from the saved v5, including its existing idle and facial controls."""
from pathlib import Path
import math
import bpy
import numpy as np
from mathutils import Vector, Matrix
from mathutils.kdtree import KDTree

folder = Path(__file__).resolve().parent/"Limonero_v6"
folder.mkdir(exist_ok=True)
scene = bpy.context.scene
sprite = next(s for s in bpy.data.scenes if s.render.film_transparent)
rig = next(o for o in scene.objects if o.type=="ARMATURE")
skin = next(o for o in scene.objects if o.name.startswith("Piel continua"))
p = rig.pose.bones
rest = np.array([v.co[:] for v in skin.data.vertices])
x,y,z = rest.T
action = rig.animation_data.action
scene.frame_set(1)


def update():
    rig.update_tag()
    bpy.context.view_layer.update()


def xyz():
    ob = skin.evaluated_get(bpy.context.evaluated_depsgraph_get())
    points = np.empty(len(rest)*3,np.float32)
    ob.data.vertices.foreach_get("co",points)
    return points.reshape(-1,3).astype(float)


def ease(v):
    t = np.clip(v,0,1)
    return t*t*(3-2*t)


def variable(driver,name,path):
    v = driver.variables.new()
    v.name = name
    v.targets[0].id = rig
    v.targets[0].data_path = path


def path(name,suffix):
    return p[name].path_from_id()+suffix


def rest_driver(bone,axis,value):
    pose = p[bone]
    pose.driver_remove("rotation_euler",axis)
    d = pose.driver_add("rotation_euler",axis).driver
    variable(d,"rest",path("Cuerpo",'["Brazos en descanso"]'))
    d.expression = f"rest*{value:.10f}"


def evaluated_axis(b):
    return (p[b].tail-p[b].head).normalized()


def aim_helper(helper,child,direction):
    # Change the zero-to-rest helper, leaving the artist's own FK channels free.
    update()
    b = p[helper]
    current = evaluated_axis(child)
    swing = current.rotation_difference(Vector(direction).normalized())
    pose = b.matrix.to_quaternion()
    local = b.matrix_basis.to_quaternion()
    parent_space = pose @ local.inverted()
    desired = (parent_space.inverted() @ swing @ pose).to_euler("XYZ")
    for axis in range(3):
        rest_driver(helper,axis,desired[axis])
    update()


# Hidden deformation supports and two animator-facing clavicles.
bpy.context.view_layer.objects.active = rig
bpy.ops.object.mode_set(mode="EDIT")
eb = rig.data.edit_bones
def new_bone(name,a,b,parent,roll=0.):
    bone = eb.new(name)
    bone.head,bone.tail,bone.roll = a,b,roll
    bone.parent = eb[parent]
    return bone
new_bone("Abdomen • soporte",(0,0,2.08),(0,0,2.49),"Pelvis",eb["Cuerpo"].roll)
for side,s in ((".L",1),(".R",-1)):
    arm = eb["Brazo"+side]
    clav = new_bone("Clavícula"+side,(s*.085,0,3.11),tuple(arm.head),"Cuerpo")
    clav.align_roll(Vector((0,-1,0)))
    eb["Base Brazo"+side].parent = clav
    a,b,roll = tuple(arm.head),tuple(arm.tail),arm.roll
    new_bone("Húmero • torsión"+side,a,b,"Brazo"+side,roll)
bpy.ops.object.mode_set(mode="POSE")
p = rig.pose.bones
abd = p["Abdomen • soporte"]
abd.bone.hide = True
rig.data.collections["Mecanismo"].assign(abd.bone)
c = abd.constraints.new("COPY_ROTATION")
c.name = "Acompañar media inclinación del torso"
c.target,c.subtarget = rig,"Cuerpo"
c.owner_space=c.target_space="LOCAL"
c.influence=.50
for side,s in ((".L",1),(".R",-1)):
    clav = p["Clavícula"+side]
    clav.rotation_mode = "XYZ"
    clav.lock_location = clav.lock_scale = (True,True,True)
    clav.bone.color.palette = "THEME04" if s==1 else "THEME03"
    rig.data.collections["Brazos y manos"].assign(clav.bone)
    c=clav.constraints.new("LIMIT_ROTATION")
    c.name,c.owner_space,c.use_transform_limit="Recorrido de clavícula","LOCAL",True
    for axis,bounds in zip("xyz",[(-16,20),(-8,8),(-18,18)]):
        setattr(c,"use_limit_"+axis,True)
        setattr(c,"min_"+axis,math.radians(bounds[0]))
        setattr(c,"max_"+axis,math.radians(bounds[1]))
    twist=p["Húmero • torsión"+side]
    twist.rotation_mode="XYZ"
    twist.bone.hide=True
    rig.data.collections["Mecanismo"].assign(twist.bone)
    drv=twist.driver_add("rotation_euler",1).driver
    v=drv.variables.new()
    v.name,v.type="roll","TRANSFORMS"
    target=v.targets[0]
    target.id,target.bone_target=rig,"Brazo"+side
    target.transform_type,target.transform_space,target.rotation_mode="ROT_Y","LOCAL_SPACE","SWING_TWIST_Y"
    drv.expression="-.65*roll"


def weights(name):
    i = skin.vertex_groups[name].index
    return np.array([next((g.weight for g in v.groups if g.group==i),0.) for v in skin.data.vertices])
def put(name,array,indices):
    group=skin.vertex_groups.get(name) or skin.vertex_groups.new(name=name)
    for i in indices:
        group.add([int(i)],float(array[i]),"REPLACE")


trunk,pelvis = weights("Cuerpo"),weights("Pelvis")
total=trunk+pelvis
ids=np.flatnonzero((z<2.75)&(total>0))
t=ease((z-1.97)/.75)
put("Pelvis",total*(1-t)**2,ids)
put("Abdomen • soporte",total*2*t*(1-t),ids)
put("Cuerpo",total*t*t,ids)
for side,s in ((".L",1),(".R",-1)):
    trunk=weights("Cuerpo")
    delta=trunk*.72*ease((s*x-.163)/.15)*ease((z-2.90)/.22)*(1-ease((z-3.22)/.12))
    ids=np.flatnonzero(delta>1.e-8)
    put("Cuerpo",trunk-delta,ids)
    put("Clavícula"+side,delta,ids)
    w=weights("Brazo"+side)
    a,b = (np.array(v) for v in (p["Brazo"+side].bone.head_local,p["Brazo"+side].bone.tail_local))
    d=b-a
    fraction=1-ease(((rest-a)@d/d.dot(d)-.03)/.86)
    delta=w*fraction
    ids=np.flatnonzero(delta>1.e-8)
    put("Brazo"+side,w-delta,ids)
    put("Húmero • torsión"+side,delta,ids)
    # Carry elbow flexion gradually across the forearm and distal upper arm.
    # Include the twist stations in each side of the blend, preserving their
    # internal split and the combined weight of the original limb chain.
    names=["Brazo"+side,"Húmero • torsión"+side,"Antebrazo"+side,"Torsión1"+side,"Torsión2"+side]
    ww=np.stack([weights(n) for n in names])
    arm_sum,fore_sum=ww[:2].sum(axis=0),ww[2:].sum(axis=0)
    total=arm_sum+fore_sum
    end=np.array(p["Antebrazo"+side].bone.tail_local)
    axis=(end-a)/np.linalg.norm(end-a)
    along=(rest-b)@axis
    ids=np.flatnonzero((abs(along)<.405)&(total>.00001))
    fore=ease(.5+along/.81)
    arm_rat=np.divide(ww[1],arm_sum,out=np.zeros_like(arm_sum),where=arm_sum>1.e-8)
    t1=np.divide(ww[3],fore_sum,out=np.zeros_like(fore_sum),where=fore_sum>1.e-8)
    t2=np.divide(ww[4],fore_sum,out=np.zeros_like(fore_sum),where=fore_sum>1.e-8)
    for n,ratio in zip(names,[(1-fore)*(1-arm_rat),(1-fore)*arm_rat,fore*(1-t1-t2),fore*t1,fore*t2]):
        put(n,total*np.maximum(0.,ratio),ids)

# A suspended wrist and softly grouped fingers. Directions are expressed in
# rig space, with positive Y behind the character. The thumb stays opposed.
scene.frame_set(1)
for curve in rig.animation_data.drivers:
    if any(word in curve.data_path for word in ("Torsión1","Torsión2","Base giro Palma")):
        curve.driver.expression=curve.driver.expression.replace("32*reposo","58*reposo")
    if "Base Brazo" in curve.data_path and curve.data_path.endswith(".location"):
        curve.driver.expression=curve.driver.expression.replace(".75049*rest",".79412*rest")
for side,s in ((".L",1),(".R",-1)):
    rest_driver("Base Brazo"+side,1,math.radians(s*45.5))
    p["Palma"+side]["Giro"]=0.
    for digit,bend,tip in [("Pulgar",2,4),("Índice",3,7),("Medio",4,10),("Menor",5,13)]:
        p[digit+".1"+side].rotation_euler.x=math.radians(bend)
        p[digit+".2"+side].rotation_euler.x=math.radians(tip)
    aim_helper("Base Palma"+side,"Palma"+side,(s*.015,-.055,-1))
    for digit,direction in [("Índice",(s*.055,-.12,-1)),("Medio",(-s*.025,-.06,-1)),
                            ("Menor",(-s*.085,.018,-1)),("Pulgar",(s*.13,-.77,-.75))]:
        aim_helper("Base giro "+digit+".1"+side,digit+".1"+side,direction)
    # Retain the old shoulder travel budget after the small change of rest angle.
    for curve in rig.animation_data.drivers:
        if p["Brazo"+side].path_from_id() in curve.data_path and "Recorrido articular" in curve.data_path:
            curve.driver.expression=curve.driver.expression.replace("+43*","+45.5*").replace("+-43*","+-45.5*")


# Bake small pose-dependent surface corrections. A local as-rigid-as-possible
# solve distributes joint strain over neighboring rings; outside the joint
# patch the solved displacement is identically zero.
def region(label,s):
    def ell(cx,cz,rx,rz):
        return ease(1-np.sqrt(((s*x-cx)/rx)**2+((z-cz)/rz)**2))
    if label=="hombro":
        return ell(.29,3.105,.40,.35)*ease((s*x-.156)/.065)
    if label=="codo":
        return ell(.88,2.79,.40,.29)
    if label=="muñeca":
        return ell(1.38,2.44,.30,.23)
    if label=="mano":
        return ell(1.54,2.38,.24,.20)
    return ease((z-1.80)/.29)*(1-ease((z-2.74)/.32))*ease((.45-abs(x))/.16)


class Patch:
    def __init__(self,mask):
        self.indices=np.flatnonzero(mask>.0001)
        self.rest=rest[self.indices]
        self.mask=mask[self.indices]
        lookup=np.full(len(rest),-1,int)
        lookup[self.indices]=np.arange(len(self.indices))
        original=lookup[edges]
        original=original[(original>=0).all(axis=1)]
        tree=KDTree(len(self.indices))
        for i,point in enumerate(self.rest):
            tree.insert(point,i)
        tree.balance()
        chords=[]
        for axis in np.eye(3):
            for offset in (.065,.13):
                for i,point in enumerate(self.rest+axis*offset):
                    _,j,d=tree.find(point)
                    if i!=j and d<offset*.55:
                        chords.append((i,j))
        pairs=np.unique(np.sort(np.vstack((original,chords)),axis=1),axis=0)
        self.i,self.j=pairs.T
        self.d=self.rest[self.i]-self.rest[self.j]
        self.w=1/np.maximum(.018,np.linalg.norm(self.d,axis=1))**1.38
        self.deg=self.scatter(self.w,self.w)
        self.pin=self.deg*(.033+1.6*(1-self.mask)**3)+.001
        self.diag=self.pin+self.deg

    def scatter(self,a,b):
        n=len(self.indices)
        return np.bincount(self.i,weights=a,minlength=n)+np.bincount(self.j,weights=b,minlength=n)

    def solve(self,posed):
        q0=posed[self.indices]
        q=q0.copy()
        for _ in range(3):
            cov=self.w[:,None,None]*(q[self.i]-q[self.j])[:,:,None]*self.d[:,None,:]
            flat=cov.reshape(-1,9)
            cov=np.stack([self.scatter(col,col) for col in flat.T],axis=1).reshape(-1,3,3)
            u,_,vh=np.linalg.svd(cov)
            u[:,:,-1]*=np.where(np.linalg.det(u@vh)<0,-1.,1.)[:,None]
            rot=u@vh
            goal=np.einsum("nij,nj->ni",(rot[self.i]+rot[self.j])*.5,self.d)
            b=self.w[:,None]*(goal-(q0[self.i]-q0[self.j]))
            rhs=np.stack([self.scatter(v,-v) for v in b.T],axis=1)
            # PCG with Dirichlet-like tethering to the original FK pose.
            delta=np.zeros_like(q)
            res=rhs.copy()
            direction=res/self.diag[:,None]
            rz=(res*direction).sum()
            for _ in range(25):
                edge=self.w[:,None]*(direction[self.i]-direction[self.j])
                ad=np.stack([self.scatter(v,-v) for v in edge.T],axis=1)+self.pin[:,None]*direction
                step=rz/max(1.e-25,(direction*ad).sum())
                delta+=step*direction
                res-=step*ad
                pre=res/self.diag[:,None]
                new_rz=(res*pre).sum()
                if new_rz<1.e-17:
                    break
                direction=pre+(new_rz/max(1.e-25,rz))*direction
                rz=new_rz
            q=q0+delta
        delta=(q-q0)*ease(self.mask/.25)[:,None]
        cap=np.minimum(1.,.060/np.maximum(1.e-8,np.linalg.norm(delta,axis=1)))
        return delta*cap[:,None]


edges=np.array([e.vertices[:] for e in skin.data.edges])
original_matrices={b.name:b.matrix_basis.copy() for b in p}
original_props={(b,key):p[b][key] for b,key in [
    ("Cuerpo","Brazos en descanso"),("Boca","MBP"),("Palma.L","Giro"),("Palma.R","Giro")]}
rig.animation_data.action=None
def neutral():
    for b in p:
        if not b.name.startswith(("Base ","Torsión","Húmero")):
            b.matrix_basis=Matrix.Identity(4)
    p["Cuerpo"]["Brazos en descanso"]=0.
    p["Boca"]["MBP"]=0.
    for side in (".L",".R"):
        p["Palma"+side]["Giro"]=0.
    update()


neutral()
filter_mod=skin.modifiers.get("Corrección de volumen en flexión")
filter_mod.show_viewport=False
probe=skin.shape_key_add(name="Sonda temporal",from_mix=False)
new_keys=[]
patches={}
def elbow_arc(side,patch,base):
    # Follow a rounded elbow with a tangent continuous into either shaft.
    # Using the same curve in the bind pose preserves its original silhouette.
    upper,lower=p["Brazo"+side],p["Antebrazo"+side]
    m=np.array(lower.bone.head_local)
    a=np.array(upper.bone.head_local)
    c=np.array(lower.bone.tail_local)
    unit=lambda d:d/np.linalg.norm(d)
    span=.34
    t_raw=.5+(patch.rest-m) @ unit(c-a)/(2*span)
    t=np.clip(t_raw,0,1)[:,None]
    def arc(a,m,c):
        start=m-span*unit(m-a)
        end=m+span*unit(c-m)
        return (1-t)**2*start+2*t*(1-t)*m+t*t*end
    bind=arc(a,m,c)
    bent=arc(np.array(upper.head),np.array(lower.head),np.array(lower.tail))
    qa=np.array(upper.matrix.to_quaternion() @ upper.bone.matrix_local.to_quaternion().inverted())
    qb=np.array(lower.matrix.to_quaternion() @ lower.bone.matrix_local.to_quaternion().inverted())
    if qa@qb<0: qb=-qb
    quat=(1-t)*qa+t*qb
    quat/=np.linalg.norm(quat,axis=1)[:,None]
    radial=patch.rest-bind
    u=2*np.cross(quat[:,1:],radial)
    rotated=radial+quat[:,:1]*u+np.cross(quat[:,1:],u)
    delta=bent+rotated-base[patch.indices]
    gate=ease((1-abs(2*t_raw-1))/.32)*ease(patch.mask/.40)
    delta*=gate[:,None]
    return delta*np.minimum(1,.065/np.maximum(1.e-8,np.linalg.norm(delta,axis=1)))[:,None]


def bake(label,side,bone,axis,degrees,manual=0):
    neutral()
    s=-1 if side==".R" else 1
    key_id=(label,side)
    if key_id not in patches:
        patches[key_id]=Patch(region(label,s))
    patch=patches[key_id]
    if manual:
        if manual==2:
            p["Cuerpo"]["Brazos en descanso"]=1.
            for name,matrix in original_matrices.items():
                if name.endswith(side) and name.split(".")[0] in ("Pulgar","Índice","Medio","Menor","Palma","Antebrazo"):
                    p[name].matrix_basis=matrix
        else:
            p[bone]["Giro"]=float(degrees)
    else:
        p[bone].rotation_euler[axis]=math.radians(degrees)
    update()
    base=xyz()
    offset=patch.solve(base)
    # Invert the local DQ transform of each vertex, not the nominal bone's
    # rotation. This preserves corrections across weighted skin transitions.
    mats=[]
    for v in np.eye(3)*.025:
        probe.data.foreach_set("co",(rest+v).astype(np.float32).ravel())
        probe.value=1.
        skin.data.update()
        bpy.context.view_layer.update()
        mats.append((xyz()[patch.indices]-base[patch.indices])/.025)
    probe.value=0.
    jac=np.stack(mats,axis=-1)
    delta=np.linalg.solve(jac,offset[:,:,None])[:,:,0]
    co=rest.copy()
    co[patch.indices]+=delta
    key=skin.shape_key_add(name=f"Apoyo {bone} {axis} {degrees:+d}",from_mix=False)
    key.data.foreach_set("co",co.astype(np.float32).ravel())
    new_keys.append((key,bone,axis,degrees,manual))
    if label=="codo":
        co=rest.copy()
        offset=elbow_arc(side,patch,base)
        co[patch.indices]+=np.linalg.solve(jac,offset[:,:,None])[:,:,0]
        arc_key=skin.shape_key_add(name=f"Arco de codo {side} {axis} {degrees:+d}",from_mix=False)
        arc_key.data.foreach_set("co",co.astype(np.float32).ravel())
        new_keys.append((arc_key,bone,axis,degrees,0))
    print(label,side,degrees,"max correction",round(np.linalg.norm(delta,axis=1).max(),4),flush=True)


for side,s in ((".L",1),(".R",-1)):
    for axis,degrees in [(2,74*s),(2,-45*s),(0,55),(0,-40)]:
        bake("hombro",side,"Brazo"+side,axis,degrees)
    for axis,degrees in [(0,75),(2,62*s),(2,-48*s)]:
        bake("codo",side,"Antebrazo"+side,axis,degrees)
    for angle in (48,-48):
        bake("muñeca",side,"Palma"+side,1,angle,1)
    bake("mano",side,"Palma"+side,0,1,2)
for axis,angle in [(0,22),(0,-20),(2,16),(2,-16),(1,22),(1,-22)]:
    bake("torso","","Cuerpo",axis,angle)
skin.shape_key_remove(probe)
filter_mod.show_viewport=True
neutral()
for key,bone,axis,deg,manual in new_keys:
    d=key.driver_add("value").driver
    if manual==2:
        variable(d,"rest",path("Cuerpo",'["Brazos en descanso"]'))
        d.expression=".72*smoothstep(.12,1.0,rest)"
        continue
    suffix='["Giro"]' if manual else f'.rotation_euler[{axis}]'
    variable(d,"angle",path(bone,suffix))
    variable(d,"rest",path("Cuerpo",'["Brazos en descanso"]'))
    units=1 if manual else math.pi/180
    shift=0
    if axis==2 and bone.startswith("Brazo"):
        shift=(-45.5 if bone.endswith(".L") else 45.5)*units
    if manual:
        shift=58.
    # Smooth activation starts just outside the neutral region; negative and
    # positive correctives are separate. Preserve small acting gestures.
    arc=key.name.startswith("Arco de codo")
    gain=.15 if arc else (.48 if bone.startswith("Antebrazo") else 1.)
    threshold=.18 if arc else .06
    activation=f"smoothstep({threshold},1,(angle+{shift:.6f}*rest)/({units*deg:.7f}))"
    if bone.startswith(("Brazo","Antebrazo")) or bone=="Cuerpo":
        # Bound the combined displacement for diagonal poses. A simple sum of
        # independent full-strength bend shapes can inflate the joint.
        axes=(0,1,2) if bone=="Cuerpo" else (0,2)
        terms=[]
        for test_axis in axes:
            varname="angle" if axis==test_axis else f"a{test_axis}"
            if axis!=test_axis:
                variable(d,varname,path(bone,f".rotation_euler[{test_axis}]"))
            if bone.startswith("Brazo"):
                span=55 if test_axis==0 else 65
                if test_axis==2:
                    varname=f"({varname}{'-' if bone.endswith('.L') else '+'}.794125*rest)"
            elif bone.startswith("Antebrazo"):
                span=75 if test_axis==0 else 62
            else:
                span=(22,22,16)[test_axis]
            terms.append(f"min(1,(({varname})/{math.radians(span):.5f})**2)")
        activation+="/max(1,sqrt("+"+".join(terms)+"))"
    d.expression=f"{gain}*"+activation

for b,matrix in original_matrices.items():
    p[b].matrix_basis=matrix
for (b,k),val in original_props.items():
    p[b][k]=val
rig.animation_data.action=action
for sc in (scene,sprite):
    sc.frame_set(1)
update()
scene.render.filepath="//Limonero_reposo.mp4"
sprite.render.filepath="//sprites/Reposo_"
# Rig is immediately accessible in Pose Mode; renders stay overlay-free.
rig.data.bones.active=rig.data.bones["Clavícula.L"]
for area in bpy.context.screen.areas:
    if area.type=="VIEW_3D":
        overlay=area.spaces.active.overlay
        overlay.show_overlays=True
        overlay.show_extras=overlay.show_relationship_lines=overlay.show_floor=False
        overlay.show_axis_x=overlay.show_axis_y=False
bpy.context.preferences.filepaths.save_version=0
bpy.ops.wm.save_as_mainfile(filepath=str(folder/"Limonero_v6.blend"))
