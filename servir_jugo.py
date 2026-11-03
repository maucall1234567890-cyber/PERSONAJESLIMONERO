"""Animate the existing Limonero FK rig; no prop, target or scene objects are created."""
from pathlib import Path
import math
import bpy
import numpy as np
from mathutils import Vector

out=Path(__file__).resolve().parent/"Limonero_servir"
out.mkdir(exist_ok=True)
scene=bpy.context.scene
scene.frame_set(1)
rig=next(o for o in scene.objects if o.type=="ARMATURE")
p=rig.pose.bones
inventory={(o.name,o.type) for o in bpy.data.objects}
original=rig.animation_data.action
original.use_fake_user=True
rig.animation_data.action=None
neutral={b.name:tuple(b.rotation_euler) for b in p}

def refresh():
    rig.update_tag()
    bpy.context.view_layer.update()

def rad(v):
    return np.radians(np.array(v,float))

def rot(a):
    x,y,z=a
    cx,cy,cz=np.cos(a)
    sx,sy,sz=np.sin(a)
    return np.array([[cz*cy,cz*sy*sx-sz*cx,cz*sy*cx+sz*sx],
                     [sz*cy,sz*sy*sx+cz*cx,sz*sy*cx-cz*sx],
                     [-sy,cy*sx,cy*cx]])

def unit(v):
    return v/np.linalg.norm(v)

def roty(a):
    return rot((0,a,0))

def set_angles(name,angles,frame):
    b=p[name]
    b.rotation_mode="XYZ"
    b.rotation_euler=rad(angles)
    b.keyframe_insert(data_path="rotation_euler",frame=frame,group=name)

def prop(name,field,value,frame):
    p[name][field]=float(value)
    p[name].keyframe_insert(data_path='["'+field+'"]',frame=frame,group=name)

def lerp(a,b,t):
    return np.array(a)*(1-t)+np.array(b)*t


def arm_pose(side,relax,target,thumb,aim,previous,orientation):
    s=1 if side==".L" else -1
    upper,lower,hand=[p[n+side] for n in ("Brazo","Antebrazo","Palma")]
    for b in (upper,lower,hand):
        b.rotation_euler=(0,0,0)
    hand["Giro"]=0.
    refresh()
    mu,mf,mp=[np.array(b.matrix) for b in (upper,lower,hand)]
    rf=np.linalg.inv(mu)@mf
    rp=np.linalg.inv(mf)@mp
    grip=np.array((0,.145,0))
    elbow_hint=np.array([s*(.61+.11*max(0,target[2]-2.55)),target[1]*.30,
                         np.clip(target[2]-.33,2.35,2.83)])
    def residual(q):
        u=mu[:3,:3]@rot(q[:3])
        pivot=mu[:3,3].copy()
        elev=np.clip((s*q[2]-.79412*relax)/1.29,0,1)**1.4
        pivot+=np.array([s*.033,0,.016])*elev
        elbow=pivot+u@rf[:3,3]
        f=u@rf[:3,:3]@rot(q[3:6])
        wrist=elbow+f@rp[:3,3]
        twist=s*(np.clip(rad(58*relax)+q[8],rad(-65),rad(65))-rad(58*relax))
        palm=f@rp[:3,:3]@roty(twist)@rot((q[6],0,q[7]))
        center=wrist+palm@grip
        return np.r_[14*(center-target),
                     orientation*1.65*(-s*palm[:,0]-thumb),
                     orientation*1.35*(palm[:,1]-aim),
                     .32*(elbow-elbow_hint), .10*(q-previous),
                     .038*q]
    variables=[(upper,0),(upper,1),(upper,2),(lower,0),(lower,1),(lower,2),(hand,0),(hand,2)]
    limits=np.array([[getattr(next(c for c in b.constraints if c.type=="LIMIT_ROTATION"),k+"_"+"xyz"[i]) for k in ("min","max")] for b,i in variables]+[[rad(-55),rad(55)]])
    # A forward elbow bend avoids the extended-arm singularity at pickup.
    seeds=[previous.copy(),rad((8,-s*12,-s*18,70,0,-s*12,5,0,0))]
    q=min((np.clip(v,limits[:,0],limits[:,1]) for v in seeds),key=lambda v:residual(v)@residual(v))
    h=.00025
    for _ in range(36):
        residual0=residual(q)
        jac=np.column_stack([(residual(q+j)-residual0)/h for j in np.eye(9)*h])
        hessian=jac.T@jac+.035*np.eye(9)
        grad=jac.T@residual0
        movable=~(((q<=limits[:,0]+1e-5)&(grad>0))|((q>=limits[:,1]-1e-5)&(grad<0)))
        step=np.zeros_like(q)
        step[movable]=-np.linalg.solve(hessian[np.ix_(movable,movable)],grad[movable])
        step*=min(1.,.36/max(1.e-8,abs(step).max()))
        for scale in [1,.5,.25,.08,.025]:
            trial=np.clip(q+scale*step,limits[:,0],limits[:,1])
            r=residual(trial)
            if r@r<residual0@residual0:
                q=trial
                break
        else:
            # Bounds can remove the descent component of the coupled GN step.
            # Use a projected first-order step to get off that face of the box.
            step=-grad*(.15/max(1e-8,abs(grad).max()))
            for scale in (1,.5,.2,.05,.01):
                trial=np.clip(q+scale*step,limits[:,0],limits[:,1])
                r=residual(trial)
                if r@r<residual0@residual0:
                    q=trial
                    break
            else:
                break
    if np.linalg.norm(residual(q)[:3])>1.5:
        print("Fit",side,np.round(target,2),np.round(residual(q)[:3]/14,2),flush=True)
    return q


# The spout follows one fixed point over the cup while pouring: rotate the
# imaginary handle around that point instead of merely flapping the wrist.
cup=np.array([.015,-.91,2.48])
handle=np.array([.84,-.92,2.79])
spout=handle+np.array([-.62,-.018,.24])
def tip_position(angle):
    return spout+roty(rad(-angle))@(handle-spout)

# frame, relaxed-arm control, jug grip, glass grip, tip degrees,
# orientation strength, relative finger closure per hand, body lean
keys=[
 (1,1,None,None,0,0.,0.,0.,0.),
 (8,.94,[.55,-.20,1.84],[-.53,-.18,1.78],0,.10,.05,.04,.18),
 (18,.48,[.63,-.64,2.19],[-.41,-.61,1.98],0,.60,.16,.22,.70),
 (25,.30,[.64,-.75,2.19],[-.35,-.74,2.02],0,.97,.90,.77,1.),
 (30,.30,[.64,-.75,2.19],[-.35,-.74,2.02],0,1.,1.,.86,1.),
 (40,.14,[.78,-.86,2.65],cup,0,1.,1.,.86,.70),
 (47,.14,handle,cup,0,1.,1.,.86,.64),
 (56,.12,tip_position(30),cup,30,1.,1.,.86,.79),
 (65,.12,tip_position(58),cup+[0,0,-.012],58,1.,1.,.88,.95),
 (76,.12,tip_position(60),cup+[0,0,-.024],60,1.,1.,.91,1.),
 (84,.12,tip_position(49),cup+[0,0,-.024],49,1.,1.,.91,.94),
 (92,.14,handle,cup+[0,0,-.024],0,1.,1.,.90,.74),
 (98,.14,handle+[.032,.025,-.075],cup+[0,0,-.024],0,1.,1.,.87,.68),
 (107,.30,[.64,-.75,2.19],[-.35,-.74,2.02],0,1.,1.,.86,1.),
 (112,.32,[.64,-.75,2.19],[-.35,-.74,2.02],0,.94,.20,.18,1.),
 (119,.72,[.57,-.43,1.98],[-.49,-.42,1.85],0,.30,.10,.10,.47),
 (129,1,None,None,0,0.,0.,0.,0.)]

rest_capture={}
for side in (".L",".R"):
    uu,ff,hh=[p[n+side] for n in ("Brazo","Antebrazo","Palma")]
    rest_capture[side]=np.r_[uu.rotation_euler[:],ff.rotation_euler[:],hh.rotation_euler.x,hh.rotation_euler.z,rad(hh["Giro"])]
prev={side:vec.copy() for side,vec in rest_capture.items()}
for frame,relax,jar,glass,tilt,ori,grasp_j,grasp_c,lean in keys:
    prop("Cuerpo","Brazos en descanso",relax,frame)
    set_angles("Cuerpo",(4.5*lean,2.2*lean,-1.8*lean),frame)
    set_angles("Cabeza",(7.3*lean,-1.6*lean,1.5*lean),frame)
    set_angles("Hoja",(-2.2*lean,0,2.8*lean),frame)
    for side,goal,grasp in [(".L",jar,grasp_j),(".R",glass,grasp_c)]:
        s=1 if side==".L" else -1
        set_angles("Clavícula"+side,(3.6*lean,0,s*(3.8 if side==".L" else 1.5)*lean),frame)
        if goal is None:
            q=rest_capture[side].copy()
        else:
            # Slightly diagonal palms cradle a cylinder. The thumb is above the
            # fingers at pickup. Only the pouring hand pronates during the tip.
            angle=tilt if side==".L" else 0.
            turn=roty(rad(-angle))
            thumb=turn@unit(np.array([s*.045,.06,1.]))
            aim=turn@unit(np.array([-s*.32,-1.,.035]))
            q=arm_pose(side,relax,np.asarray(goal),thumb,aim,prev[side],ori)
        prev[side]=q.copy()
        set_angles("Brazo"+side,np.degrees(q[:3]),frame)
        set_angles("Antebrazo"+side,np.degrees(q[3:6]),frame)
        set_angles("Palma"+side,np.degrees([q[6],0.,q[7]]),frame)
        prop("Palma"+side,"Giro",np.degrees(q[8]),frame)
        for digit,curl1,curl2 in [("Pulgar",34,24),("Índice",57,58),("Medio",65,61),("Menor",69,64)]:
            ratio=1 if side==".L" else .81
            for seg,curl in [(1,curl1),(2,curl2)]:
                n=f"{digit}.{seg}{side}"
                base=np.degrees(neutral[n])
                value=lerp(base,[curl*ratio,0,0],grasp)
                set_angles(n,value,frame)
    if frame in (25,47,65,76,92,107):
        refresh()
        for side,target in ((".L",jar),(".R",glass)):
            actual=p["Palma"+side].matrix@Vector((0,.145,0))
            print("Grip",frame,side,np.round(actual[:],3),round(np.linalg.norm(np.array(actual)-target),3),flush=True)

def prop_keys(bone,name,values):
    for f,v in values:
        prop(bone,name,v,f)

prop_keys("Cabeza","Mirada horizontal",[(1,-.08),(6,-.08),(12,.55),(28,.55),(39,-.54),(90,-.54),(102,.28),(114,.28),(124,-.08),(129,-.08)])
prop_keys("Cabeza","Mirada vertical",[(1,0),(12,-.80),(30,-.8),(45,-.50),(76,-.67),(108,-.82),(120,-.38),(129,0)])
prop_keys("Boca","MBP",[(1,1),(15,1),(39,.86),(49,1),(90,1),(105,.82),(119,1),(129,1)])
for side in (".L",".R"):
    prop_keys("Párpado"+side,"Cerrar",[(1,0),(34,0),(37,1),(38,1),(42,0),(99,0),(102,1),(103,1),(107,0),(129,0)])
action=rig.animation_data.action
action.name="Servir jugo • tomar, verter y soltar"
for layer in action.layers:
    for strip in layer.strips:
        for bag in strip.channelbags:
            for fc in bag.fcurves:
                for k in fc.keyframe_points:
                    k.interpolation="BEZIER"
                    k.handle_left_type=k.handle_right_type="AUTO_CLAMPED"
                fc.update()
for s in bpy.data.scenes:
    s.frame_start,s.frame_end,s.render.fps=1,128,24
    s.timeline_markers.clear()
    for f,label in [(1,"Reposo"),(25,"Tomar"),(47,"Alinear"),(65,"Verter"),(92,"Enderezar"),(107,"Apoyar"),(119,"Retirar")]:
        s.timeline_markers.new(label,frame=f)
    s.render.filepath="//Servir_jugo.mp4" if not s.render.film_transparent else "//sprites/Servir_"
    s.frame_set(1)
assert inventory=={(o.name,o.type) for o in bpy.data.objects}
bpy.context.preferences.filepaths.save_version=0
bpy.ops.wm.save_as_mainfile(filepath=str(out/"Limonero_servir.blend"))
