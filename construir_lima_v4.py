"""Blender 5.2 | Construye la escena y exporta sus imágenes en Limonero_v4."""
from pathlib import Path
import math
import bpy
import numpy as np
from mathutils import Vector

OUT = Path(__file__).resolve().parent / "Limonero_v4"
OUT.mkdir(exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
J = {}
skin = []
C = {}
for label in ("01 • Personaje", "02 • Controles", "03 • Estudio"):
    C[label[:2]] = bpy.data.collections.new(label)
    sc.collection.children.link(C[label[:2]])


def material(name, rgb, roughness=.45):
    m = bpy.data.materials.new(name)
    m.diffuse_color = (*rgb, 1)
    m.use_nodes = True
    bs = m.node_tree.nodes.get("Principled BSDF")
    bs.inputs["Base Color"].default_value = (*rgb, 1)
    bs.inputs["Roughness"].default_value = roughness
    return m


peel = material("Piel • lima amarilla", (.50,.73,.090),.54)
sun = material("Piel • amarillo cálido", (.90,.69,.104),.53)
for m in (peel,sun):
    m.node_tree.nodes.get("Principled BSDF").inputs["Specular IOR Level"].default_value = .26
ivory = material("Ojos • marfil", (1,.96,.81), .28)
ink = material("Pupila • café profundo", (.014,.009,.005), .58)
ink.node_tree.nodes.get("Principled BSDF").inputs["Specular IOR Level"].default_value = .13
glint = material("Reflejo • crema", (1,.99,.94))
green = material("Hoja • haz", (.075,.31,.034))
midrib = material("Hoja • venas", (.065,.235,.026))
oral = material("Boca • cavidad", (.115,.018,.025))
rose = material("Lengua • coral", (1,.106,.14))


def mesh(name, vs, fs, mat, joint=None):
    data = bpy.data.meshes.new(name)
    data.from_pydata(vs, [], fs)
    data.update()
    ob = bpy.data.objects.new(name, data)
    C["01"].objects.link(ob)
    data.materials.append(mat)
    data.polygons.foreach_set("use_smooth", np.ones(len(fs), dtype=bool))
    if joint:
        skin.append((ob,joint))
    return ob


def bone(name,a,b,parent=None):
    J[name] = (np.array(a),np.array(b),parent)
    return name


def smoothmin(a,b,k):
    return np.minimum(a,b)-np.maximum(k-abs(a-b),0)**2/(4*k)


def ell(p,c,r):
    return (np.sqrt(sum(((v-i)/j)**2 for v,i,j in zip(p,c,r)))-1)*min(r)


def capsule(p,a,b,r0,r1):
    d = np.subtract(b,a)
    u = np.clip(sum((v-i)*j for v,i,j in zip(p,a,d))/d.dot(d),0,1)
    return np.sqrt(sum((v-i-u*j)**2 for v,i,j in zip(p,a,d)))-(r0+(r1-r0)*u)


def surface(name, fn, bounds, step, mat):
    # A fixed tetrahedral split resolves ambiguous cells at the fingers and lips.
    axes = [np.arange(a,b+step,step,dtype=np.float32) for a,b in bounds]
    # Double the sampling resolution at the hands, where the thinnest
    # fingertip would otherwise span only three cells.
    for dim,axis in enumerate(axes):
        mid = (axis[1:]+axis[:-1])*.5
        detail = ((abs(mid)>1.36) if dim==0 else
                  (abs(mid)<.21) if dim==1 else ((mid>1.88)&(mid<2.64)))
        axes[dim] = np.sort(np.r_[axis,mid[detail]])
    f = fn(np.ix_(*axes))
    shape = tuple(n-1 for n in f.shape)
    corners = np.array(list(np.ndindex(2,2,2)))
    low = np.full(shape,np.inf,np.float32)
    high = -low
    for i,j,k in corners:
        block = f[i:i+shape[0],j:j+shape[1],k:k+shape[2]]
        np.minimum(low,block,out=low)
        np.maximum(high,block,out=high)
    cells = np.array(np.where((low<0)&(high>=0))).T
    grid_ids = np.stack([np.ravel_multi_index((cells+c).T,f.shape) for c in corners],axis=1)
    inside = f.ravel()[grid_ids]<0
    tris = []
    for tet in [(0,1,3,7),(0,3,2,7),(0,2,6,7),(0,6,4,7),(0,4,5,7),(0,5,1,7)]:
        case = inside[:,tet] @ np.array([1,2,4,8])
        for mask in range(1,15):
            grid = grid_ids[case==mask]
            if not len(grid):
                continue
            pos = [n for i,n in enumerate(tet) if mask&(1<<i)]
            neg = [n for n in tet if n not in pos]
            pairs = [(a,b) for a in pos for b in neg]
            if len(pairs)==4:
                pairs[2],pairs[3] = pairs[3],pairs[2]
            mids = np.array([(corners[a]+corners[b])*.5 for a,b in pairs])
            outward = corners[neg].mean(axis=0)-corners[pos].mean(axis=0)
            if np.cross(mids[1]-mids[0],mids[2]-mids[0]).dot(outward)<0:
                pairs.reverse()
            keys = np.stack([np.minimum(grid[:,a],grid[:,b])*f.size+np.maximum(grid[:,a],grid[:,b])
                             for a,b in pairs],axis=1)
            tris.append(keys[:,[0,1,2]])
            if len(pairs)==4:
                tris.append(keys[:,[0,2,3]])
    unique,indices = np.unique(np.concatenate(tris).ravel(),return_inverse=True)
    ia,ib = np.divmod(unique,f.size)
    a = np.stack(np.unravel_index(ia,f.shape),axis=1)
    b = np.stack(np.unravel_index(ib,f.shape),axis=1)
    t = (f.ravel()[ia]/(f.ravel()[ia]-f.ravel()[ib]))[:,None]
    vs = np.column_stack([ax[a[:,dim]]+(ax[b[:,dim]]-ax[a[:,dim]])*t[:,0]
                          for dim,ax in enumerate(axes)])
    ob = mesh(name,vs.tolist(),indices.reshape(-1,3).tolist(),mat)
    # Fair the finger webs before reduction; keep the approved body silhouette.
    web = ob.vertex_groups.new(name="Suavizado temporal")
    for vertex in ob.data.vertices:
        weight = min(1.,max(0.,(abs(vertex.co.x)-1.32)/.15))
        if weight:
            web.add([vertex.index],weight,"REPLACE")
    fair = ob.modifiers.new("Redondear comisuras de mano","SMOOTH")
    fair.factor,fair.iterations,fair.vertex_group = .72,7,web.name
    bpy.context.view_layer.objects.active = ob
    bpy.ops.object.modifier_apply(modifier=fair.name)
    ob.vertex_groups.remove(ob.vertex_groups["Suavizado temporal"])
    reduce = ob.modifiers.new("Simplificar superficie","DECIMATE")
    reduce.ratio = min(1,52000/len(vs))
    bpy.context.view_layer.objects.active = ob
    bpy.ops.object.modifier_apply(modifier=reduce.name)
    vertices = np.array([v.co[:] for v in ob.data.vertices])
    return ob,vertices


root = bone("Raíz",(0,0,.01),(0,0,.38))
pelvis = bone("Pelvis",(0,0,1.92),(0,0,2.22),root)
trunk = bone("Cuerpo",(0,0,2.16),(0,0,3.17),pelvis)
neck = bone("Cuello",(0,0,3.17),(0,0,3.41),trunk)
head = bone("Cabeza",(0,0,3.41),(0,0,5.38),neck)
segments = []
finger_paths = []
envelopes = [(trunk,.23),(neck,.12),(head,.75),(pelvis,.19)]
twists = []
for sign,side in [(1,".L"),(-1,".R")]:
    def xyz(x,y,z):
        return (sign*x,y,z)
    def chain(labels, pts, radii, parent):
        for i,label in enumerate(labels):
            joint = bone(label+side,pts[i],pts[i+1],parent)
            segments.append((pts[i],pts[i+1],radii[i],radii[i+1]))
            envelopes.append((joint,(radii[i]+radii[i+1])/2))
            parent = joint
        return parent
    palm = chain(["Brazo","Antebrazo","Palma"],
                 [xyz(.222,0,3.14),xyz(.88,-.012,2.79),xyz(1.40,-.035,2.44),xyz(1.61,-.050,2.30)],
                 [.114,.105,.084,.086],trunk)
    # The thumb leaves the radial side of the palm, above and in front of the fingers.
    # Longitudinal and radial coordinates across the palm. Three staggered
    # knuckles point along the hand; the opposable thumb sits on its own side.
    def hand(s,r,y):
        return (1.40+.84*s+.543*r,y,2.44-.543*s+.84*r)
    digits = [("Pulgar",hand(.064,.077,-.061),hand(.107,.172,-.098),hand(.19,.251,-.125)),
              ("Índice",hand(.227,.093,-.050),hand(.399,.155,-.054),hand(.551,.197,-.075)),
              ("Medio",hand(.249,-.008,-.052),hand(.428,-.015,-.059),hand(.59,-.031,-.081)),
              ("Menor",hand(.202,-.102,-.042),hand(.342,-.163,-.051),hand(.479,-.207,-.073))]
    for label,base,middle,tip in digits:
        radii = [.052,.044,.038] if label=="Pulgar" else ([.043,.035,.029] if label=="Menor" else [.047,.039,.032])
        chain([label+".1",label+".2"],[xyz(*base),xyz(*middle),xyz(*tip)],
              radii,palm)
        # Geometry follows one continuous tapered curve per finger; the
        # skeleton keeps its two separate phalanges for posing.
        del segments[-2:]
        if sign==1:
            start,mid,end = map(np.array,(base,middle,tip))
            control = 2*mid-.5*(start+end)
            ts = np.linspace(0,1,7)
            curve = [(1-t)**2*start+2*t*(1-t)*control+t*t*end for t in ts]
            finger_paths.append((curve,np.interp(ts,[0,.5,1],radii)))
    shin = chain(["Muslo","Pierna"],[xyz(.185,0,1.99),xyz(.225,-.017,1.15),xyz(.278,.015,.245)],
                 [.146,.132,.145],pelvis)
    foot = bone("Pie"+side,xyz(.278,.015,.245),xyz(.38,-.33,.125),shin)
    envelopes.append((foot,.15))
    a,b,_ = J["Antebrazo"+side]
    for index,fraction in ((1,.42),(2,.82)):
        start = a+(b-a)*fraction
        twist = bone("Torsión%d%s"%(index,side),start,start+.12*(b-a)/np.linalg.norm(b-a),"Antebrazo"+side)
        twists.append((twist,side,fraction))


def skull_radius(z):
    # Perfil torneado con interpolación cúbica de Hermite entre anillos.
    heights = np.array([3.27,3.36,3.47,3.66,3.94,4.22,4.47,4.74,4.96,5.15,5.28,5.34,5.42,5.51,5.55])
    widths = np.array([.12,.19,.43,.69,.90,1.00,1.015,.96,.85,.68,.49,.34,.225,.15,.018])
    slope = np.gradient(widths,heights)
    idx = np.clip(np.searchsorted(heights,z)-1,0,len(heights)-2)
    dz = heights[idx+1]-heights[idx]
    t = np.clip((z-heights[idx])/dz,0,1)
    return ((2*t**3-3*t**2+1)*widths[idx]+(t**3-2*t**2+t)*dz*slope[idx]
            +(-2*t**3+3*t**2)*widths[idx+1]+(t**3-t**2)*dz*slope[idx+1])


DEPTH = .88
def face_y(x,z):
    return -DEPTH*np.sqrt(np.maximum(.001, skull_radius(z)**2-x*x))


def smile_limits(x):
    u = (x+.008)/.365
    top = 3.86+.052*u*u
    return top,top-.244*np.maximum(0,1-u*u)**.77


def opening(p):
    x,y,z = p
    top,bot = smile_limits(x)
    # Pocket follows the curvature of the cheek; its rear wall stays inside the head.
    return np.maximum.reduce(np.broadcast_arrays(abs(x+.008)-.365, z-top,bot-z,
                                                  y-face_y(x,z)-.119))


def field(p):
    x,y,z = p
    # Mirroring only the distance evaluation gives both hands identical anatomy.
    folded = (abs(x),y,z)
    waist = np.clip((z-2.15)/.78,0,1)
    radius = .297-.089*waist*waist*(3-2*waist)
    dz = z-np.clip(z,2.19,3.12)
    torso = np.sqrt(x*x+(1.16*y)**2+(1.20*dz)**2)-radius
    for a,b,r0,r1 in segments[:len(segments)//2]:
        k = .074 if a[0]<.3 and a[2]>1.9 else (.031 if a[0]>=1.4 else .014)
        torso = smoothmin(torso,capsule(folded,a,b,r0,r1),k)
    # A flattened palm carries the knuckles instead of four intersecting bulbs.
    wx,wz = folded[0]-1.40,z-2.44
    s,r = .84*wx-.543*wz,.543*wx+.84*wz
    palm_field = ell((s,r,y),(.148,0,-.048),(.180,.138,.075))
    torso = smoothmin(torso,palm_field,.045)
    ix = np.flatnonzero(abs(x[:,0,0])>1.32)
    iy = np.flatnonzero(abs(y[0,:,0])<.25)
    iz = np.flatnonzero((z[0,0,:]>1.87)&(z[0,0,:]<2.70))
    if iz.size:
        subset = np.ix_(ix,iy,iz)
        local = (abs(x[ix]),y[:,iy],z[:,:,iz])
        union = torso[subset]
        for curve,radii in finger_paths:
            digit = np.minimum.reduce([capsule(local,a,b,r1,r2)
                                       for a,b,r1,r2 in zip(curve,curve[1:],radii,radii[1:])])
            union = smoothmin(union,digit,.036)
        torso[subset] = union
    toe_x = .326+.128*np.clip((-y+.04)/.53,0,1)
    foot = ell((folded[0]-toe_x,y,z),(0,-.16,.149),(.305,.44,.172))
    sole = np.maximum(foot,.022-z)
    torso = smoothmin(torso,sole,.11)
    torso = smoothmin(torso,capsule(p,(0,0,3.14),(0,0,3.46),.135,.145),.085)
    fruit = np.maximum.reduce(np.broadcast_arrays(np.sqrt(x*x+(y/DEPTH)**2)-skull_radius(z),
                                                   3.29-z,z-5.55))
    body = smoothmin(torso,fruit,.085)
    return body


# Build in thin horizontal slabs to bound memory, then weld the skin implicitly
# by polygonizing the single sampled scalar field.
def sliced_field(grid):
    x,y,z = grid
    return np.concatenate([field((x,y,z[:,:,i:i+8])) for i in range(0,z.shape[2],8)],axis=2)


body,verts = surface("Piel continua • cabeza a pies",sliced_field,
                     [(-2.09,2.09),(-1.01,.99),(-.055,5.62)],.019,sun)


def build_oral_loops(ob,points):
    """Replace a cheek disk with a welded lip, inner wall and recessed palate."""
    x,y,z = points.T
    radius = ((x+.008)/.57)**2+((z-3.775)/.247)**2-1
    selected = (radius<0)&(y<-.13)
    triangles = np.array([p.vertices[:] for p in ob.data.polygons])
    crossing = selected[triangles].any(axis=1)
    kept = triangles[~crossing].tolist()
    extra,edges,follow = [],{},{}
    # Clip triangles at the ellipse instead of following the staircase of grid edges.
    for tri in triangles[crossing&~selected[triangles].all(axis=1)]:
        polygon = []
        for i,j in zip(tri,np.roll(tri,-1)):
            if not selected[i]:
                polygon.append(i)
            if selected[i]!=selected[j]:
                edge = tuple(sorted((i,j)))
                if edge not in edges:
                    t = radius[i]/(radius[i]-radius[j])
                    edges[edge] = len(points)+len(extra)
                    extra.append(points[i]+t*(points[j]-points[i]))
                polygon.append(edges[edge])
        for i,j in zip(polygon,np.roll(polygon,-1)):
            if i>=len(points) and j>=len(points):
                follow[j] = i
        kept.extend([polygon[0],polygon[k],polygon[k+1]] for k in range(1,len(polygon)-1))
    points = np.vstack((points,np.array(extra)))
    boundary = [next(iter(follow))]
    while follow[boundary[-1]]!=boundary[0]:
        boundary.append(follow[boundary[-1]])
    assert len(boundary)==len(follow),"El corte debe ser un disco"
    outer = points[boundary]
    cx,cz = -.008,3.777
    dx,dz = outer[:,0]-cx,outer[:,2]-cz
    lo,hi = np.zeros(len(dx)),np.ones(len(dx))
    for _ in range(28):
        mid = (lo+hi)*.5
        px,pz = cx+mid*dx,cz+mid*dz
        top,bottom = smile_limits(px)
        inside = (pz<top)&(pz>bottom)&(abs(px-cx)<.365)
        lo,hi = np.where(inside,mid,lo),np.where(inside,hi,mid)
    xx,zz = cx+lo*dx,cz+lo*dz
    kept = np.array(kept)
    used = np.unique(kept)
    remap = np.empty(len(points),dtype=int)
    remap[used] = np.arange(len(used))
    vs = points[used].tolist()
    fs = remap[kept].tolist()
    depths = [0.]*len(vs)
    last = remap[boundary]
    def ring(px,pz,d):
        nonlocal last
        start = len(vs)
        vs.extend(np.column_stack((px,face_y(px,pz)+d,pz)).tolist())
        depths.extend([max(0.,d)]*len(px))
        current = np.arange(start,len(vs))
        for a,b,c,e in zip(last,np.roll(last,-1),np.roll(current,-1),current):
            fs.append([int(a),int(b),int(c),int(e)])
        last = current
    for blend in (.18,.4,.65,.82,.94,1.):
        px = outer[:,0]*(1-blend)+xx*blend
        pz = outer[:,2]*(1-blend)+zz*blend
        ring(px,pz,-.003*blend**4)
    for scale,depth in [(.992,.001),(.981,.009),(.965,.023),(.937,.051),
                        (.89,.083),(.78,.111),(.53,.129),(.25,.135)]:
        ring(cx+(xx-cx)*scale,cz+(zz-cz)*scale,depth)
    center = len(vs)
    vs.append((cx,float(face_y(cx,cz))+.138,cz))
    depths.append(.138)
    for a,b in zip(last,np.roll(last,-1)):
        fs.append([int(a),int(b),center])
    name = ob.name
    bpy.data.objects.remove(ob,do_unlink=True)
    ob = mesh(name,vs,fs,sun)
    return ob,np.array(vs),np.array(depths)


body,verts,depth = build_oral_loops(body,verts)
# Material color follows anatomy without a seam at the neck.
attribute = body.data.color_attributes.new(name="Pigmento",type="FLOAT_COLOR",domain="POINT")
x,y,z = verts.T
mix = np.clip((z-3.16)/.23,0,1)[:,None]
base = (1-mix)*np.array(sun.diffuse_color[:3])+mix*np.array(peel.diffuse_color[:3])
top,bot = smile_limits(x)
# Interpolate depth first, shade the lip per pixel. A binary vertex-color mask
# would expose the boundary triangles as green and red flecks when the mouth moves.
rest = body.data.attributes.new("Profundidad oral","FLOAT","POINT")
rest.data.foreach_set("value",depth.astype(np.float32))
body.data.color_attributes["Pigmento"].data.foreach_set("color",np.column_stack((base,np.ones(len(z)))).astype(np.float32).ravel())
node = sun.node_tree.nodes.new("ShaderNodeVertexColor")
node.layer_name = "Pigmento"
sun.node_tree.links.new(node.outputs["Color"],sun.node_tree.nodes.get("Principled BSDF").inputs["Base Color"])
tree = sun.node_tree
depth_input = tree.nodes.new("ShaderNodeAttribute")
depth_input.attribute_name = "Profundidad oral"
def ramp(name, stops):
    r = tree.nodes.new("ShaderNodeValToRGB")
    r.label = name
    r.color_ramp.interpolation = "EASE"
    for i,(position,color) in enumerate(stops):
        item = r.color_ramp.elements[i] if i<2 else r.color_ramp.elements.new(position)
        item.position,item.color = position,(*color,1)
    tree.links.new(depth_input.outputs["Fac"],r.inputs["Fac"])
    return r.outputs["Color"]
mask = ramp("Transición piel / labio",[(.003,(0,0,0)),(.013,(1,1,1))])
inside = ramp("Interior • coral a vino",[(.002,(.26,.062,.036)),(.04,(.15,.030,.034)),(.11,(.058,.010,.017))])
mix_shader = tree.nodes.new("ShaderNodeMixRGB")
for output,socket in [(mask,0),(node.outputs["Color"],1),(inside,2)]:
    tree.links.new(output,mix_shader.inputs[socket])
tree.links.new(mix_shader.outputs[0],tree.nodes.get("Principled BSDF").inputs["Base Color"])


def assign_weights(ob,vs):
    # Segment distance and a compact Gaussian falloff blend each joint to its neighbors.
    dist = []
    for label,r in envelopes:
        a,b,_ = J[label]
        d = b-a
        t = np.clip(np.sum((vs-a)*d,axis=1)/d.dot(d),0,1)
        ds = np.linalg.norm(vs-a-t[:,None]*d,axis=1)
        dist.append(ds-.43*r)
    dist = np.array(dist).T
    # Restrict across the midline below the crotch so bending one leg leaves the other planted.
    for i,(label,_) in enumerate(envelopes):
        if label.endswith((".L",".R")):
            opposite = vs[:,0]*(1 if label.endswith(".L") else -1)<0
            dist[(vs[:,2]<1.9)&opposite,i] += 1.
    sigma = np.where(abs(vs[:,0])>1.44,.033,np.where(vs[:,2]>3.30,.125,.131))
    ids = np.argsort(dist,axis=1)[:,:4]
    dd = np.take_along_axis(dist,ids,axis=1)
    w = np.exp(-((dd-dd[:,0:1])/sigma[:,None])**2*1.8)
    w[w<.005]=0
    w /= w.sum(axis=1)[:,None]
    # Across knees and elbows the weights follow the length of the limb,
    # keeping the whole cross-section together during a bend.
    weights = np.zeros_like(dist)
    np.put_along_axis(weights,ids,w,axis=1)
    labels = [name for name,r in envelopes]
    for side in (".L",".R"):
        for upper,lower,band in [("Brazo","Antebrazo",.235),("Muslo","Pierna",.26)]:
            first,second = upper+side,lower+side
            a,b,_ = J[first]
            _,c,_ = J[second]
            axis = (c-a)/np.linalg.norm(c-a)
            t = np.clip(.5+(vs-b) @ axis/(2*band),0,1)
            t = t*t*(3-2*t)
            i,j = labels.index(first),labels.index(second)
            total = weights[:,i]+weights[:,j]
            weights[:,i],weights[:,j] = total*(1-t),total*t
    # Keep the sternum attached to the chest. The shoulder transition runs
    # outwards across the socket, not down through the ribcage.
    x,_,z = vs.T
    def smooth(t):
        t = np.clip(t,0,1)
        return t*t*(3-2*t)
    for side,sign in ((".L",1),(".R",-1)):
        u = sign*x
        zone = (u>=0)&(u<.48)&(z>2.70)&(z<3.29)
        lateral = smooth((u-.24)/.13)
        height = smooth((z-2.84)/.24)
        shoulder = smooth((u-.163)/.208)*(height+(1-height)*lateral)
        weights[zone] = 0
        weights[zone,labels.index(trunk)] = (1-shoulder)[zone]
        weights[zone,labels.index("Brazo"+side)] = shoulder[zone]
        # Blend whole wrist cross sections instead of choosing a bone from
        # distance to the palm surface, which pinches the heel on flexion.
        wrist = (u-1.40)*.84-(z-2.44)*.543
        hand_zone = (u>1.08)&(z<2.72)&(z>1.78)
        fi,pi = labels.index("Antebrazo"+side),labels.index("Palma"+side)
        digit_ids = [i for i,n in enumerate(labels) if n.endswith(side) and
                     n.startswith(("Pulgar.","Índice.","Medio.","Menor."))]
        retained = smooth((wrist+.03)/.16)
        released = weights[:,digit_ids].sum(axis=1)*(1-retained)
        weights[np.ix_(hand_zone,digit_ids)] *= retained[hand_zone,None]
        hand_sum = weights[:,fi]+weights[:,pi]+released
        palm_share = smooth((wrist+.13)/.26)
        weights[hand_zone,fi] = (hand_sum*(1-palm_share))[hand_zone]
        weights[hand_zone,pi] = (hand_sum*palm_share)[hand_zone]
        for digit in ("Pulgar","Índice","Medio","Menor"):
            first,second = digit+".1"+side,digit+".2"+side
            a,b,_ = J[first]
            _,c,_ = J[second]
            axis = (c-a)/np.linalg.norm(c-a)
            share = smooth(.5+(vs-b) @ axis/.12)
            i,j = labels.index(first),labels.index(second)
            total = weights[:,i]+weights[:,j]
            weights[:,i],weights[:,j] = total*(1-share),total*share
        # Share only the forearm influence between a rigid base and two
        # twist stations. All three axes are collinear in the bind pose.
        a,b,_ = J["Antebrazo"+side]
        d = b-a
        t = np.clip((vs-a) @ d/d.dot(d),0,1)
        mid = smooth((t-.10)/.43)
        end = smooth((t-.53)/.42)
        fi = labels.index("Antebrazo"+side)
        total = weights[:,fi].copy()
        weights[:,fi] = total*(1-mid)
        weights = np.column_stack((weights,total*mid*(1-end),total*mid*end))
        labels.extend(["Torsión1"+side,"Torsión2"+side])
    ids = np.argsort(-weights,axis=1)[:,:4]
    w = np.take_along_axis(weights,ids,axis=1)
    w /= w.sum(axis=1)[:,None]
    groups = [ob.vertex_groups.new(name=n) for n in labels]
    for v,(indices,weights) in enumerate(zip(ids,w)):
        for i,value in zip(indices,weights):
            if value:
                groups[i].add([v],float(value),"REPLACE")


assign_weights(body,verts)
# Jaw shape keys move both the cavity and visible cheek around it.
body.shape_key_add(name="Base")
jaw = bone("Boca",(0,-.61,3.94),(0,-.62,3.69),head)
morphs = []
visemes = {"MBP":(1.,.022,0.), "A":(.95,1.28,.016),
           "E":(1.13,.60,.009), "O":(.57,1.15,.022), "U":(.43,.77,.018)}
def ease(t):
    t = np.clip(t,0.,1.)
    return t*t*(3-2*t)


def add_mouth_keys(ob,points,is_tongue=False):
    x,y,z = points.T
    top,bottom = smile_limits(x)
    height = top-bottom
    # The lip coordinates are authoritative. Adjacent cheek tissue absorbs a
    # diminishing fraction of travel; the eyes, chin silhouette and neck stay fixed.
    face = ease((-y-.14)/.26)*(1-ease((abs(x+.008)-.366)/.32))
    face *= ease((z-3.46)/.16)*(1-ease((z-3.925)/.20))
    face *= ease((skull_radius(z)-abs(x)-.045)/.19)
    lower = np.clip(top-z,0,height)*ease((z-3.39)/np.maximum(.05,bottom-3.39))
    for label,(width,opening_scale,lift) in visemes.items():
        co = points.copy()
        co[:,0] += (x+.008)*(width-1)*face
        co[:,2] += (lower*(1-opening_scale)+lift)*face
        if label in {"O","U"}:
            u = np.clip((x+.008)/.365,-1,1)
            # Soften the angular corner field away from the actual rim. Its cusp
            # is needed at the lips, but should never continue into the chin.
            distance = np.maximum.reduce([bottom-z,z-top,np.zeros_like(z)])
            arc = sum(w*np.sqrt(np.maximum(0,1-((x+.008+i*distance)/.365)**2))
                      for i,w in [(-.8,.20),(-.4,.20),(0,.20),(.4,.20),(.8,.20)])
            old_top,old_bottom = smile_limits(-.008+.365*u)
            mid = (old_top+old_bottom)*.5
            fraction = np.clip((z-mid)/np.maximum(.0001,(old_top-old_bottom)*.5),-1,1)
            radius = .151 if label=="O" else .114
            shift = 3.79-mid+(radius*arc-(old_top-old_bottom)*.5)*fraction
            co[:,2] = z+shift*face*np.exp(-(distance/.15)**2)
        co[:,1] += (face_y(co[:,0],co[:,2])-face_y(x,z))*(y<0)
        if is_tongue:
            # Pull the tongue behind the lip seal, preserving its own volume.
            co[:,1] += {"MBP":.085,"A":.014,"E":.030,"O":.027,"U":.043}[label]*face
        key = ob.shape_key_add(name=label)
        key.data.foreach_set("co",co.astype(np.float32).ravel())
        morphs.append((key,jaw,label))


add_mouth_keys(body,verts)


def disk(name, fn, mat, joint, rings=12, steps=64):
    vs = [fn(0.,0.)]
    fs = []
    for r in range(1,rings+1):
        for i in range(steps):
            angle = math.tau*i/steps
            vs.append(fn(r/rings*math.cos(angle),r/rings*math.sin(angle)))
        for i in range(steps):
            a = 1+(r-1)*steps+i
            b = 1+(r-1)*steps+(i+1)%steps
            fs.append((0,a,b) if r==1 else (a-steps,a,b,b-steps))
    return mesh(name,vs,fs,mat,joint)


def tongue_surface(u,v):
    x = .103+.183*u
    z = 3.710+.095*v+.061*u
    y = face_y(x,z)+.039-.046*(1-u*u-v*v)
    return x,y,z


tongue = disk("Lengua • interior de sonrisa",tongue_surface,rose,head)
tongue.shape_key_add(name="Base")
add_mouth_keys(tongue,np.array([v.co[:] for v in tongue.data.vertices]),True)
rose.node_tree.nodes.get("Principled BSDF").inputs["Roughness"].default_value = .44


for s,side in [(1,".L"),(-1,".R")]:
    cx,cz = s*.365,4.365
    rx,rz = .248,.425
    gaze = bone("Mirada"+side,(cx,face_y(cx,cz)+.09,cz),(cx,face_y(cx,cz)-.12,cz),head)
    lid = bone("Párpado"+side,(cx,face_y(cx,cz),cz+.40),(cx,face_y(cx,cz),cz+.20),head)
    def ocular(x,z):
        r2 = ((x-cx)/rx)**2+((z-cz)/rz)**2
        return face_y(x,z)-.009-.086*np.maximum(0.,1-r2)
    def patch(u,v,ox,oz,sx,sz,offset):
        x,z = cx+ox+sx*u,cz+oz+sz*v
        return x,ocular(x,z)-offset,z
    eyeball = disk("Óvalo marfil"+side,lambda u,v:patch(u,v,0,0,rx,rz,0),ivory,head)
    pupil = disk("Pupila"+side,lambda u,v:patch(u,v,-.015,-.006,.155,.291,.004),ink,gaze)
    lightspot = disk("Destello"+side,lambda u,v:patch(u,v,-.053,.112,.043,.079,.009),glint,gaze)
    def retract(x,y,z,q):
        # The globe settles back under the cheek as the lid travels over it.
        fold = q*q
        line = cz-.077+.040*((x-cx)/rx)**2
        nz = z+(line-z)*fold*.98
        offset = y-face_y(x,z)
        return (x,face_y(x,nz)+(1-fold)*offset+.032*fold,nz)
    for part in (eyeball,pupil,lightspot):
        part.shape_key_add(name="Ojo abierto")
        old = None
        points = [v.co[:] for v in part.data.vertices]
        for k in range(1,5):
            key = part.shape_key_add(name="Retraer %d%%"%(25*k))
            key.data.foreach_set("co",np.array([retract(*p,k/4) for p in points],np.float32).ravel())
            if old:
                key.relative_key = old
            old = key
            morphs.append((key,lid,"Cerrar",k))
    # A retracting lid mapped to the same eye surface, with no visible geometry at rest.
    def cap(t,a,b):
        angle = math.pi*t*b
        x = cx+rx*math.sin(angle)*a
        z = cz+rz*math.cos(angle)
        return retract(x,ocular(x,z)-.018,z,b)
    rows,cols = 24,32
    uv = [(i/rows,-1+2*j/cols) for i in range(rows+1) for j in range(cols+1)]
    fs = [(i*(cols+1)+j,i*(cols+1)+j+1,(i+1)*(cols+1)+j+1,(i+1)*(cols+1)+j)
          for i in range(rows) for j in range(cols)]
    lidob = mesh("Párpado retráctil"+side,[cap(t,a,0) for t,a in uv],fs,peel,head)
    lidob.shape_key_add(name="Recogido")
    # Several keys along the curved cornea prevent a halfway blink crossing the pupil.
    previous = None
    for k in range(1,5):
        key = lidob.shape_key_add(name="Cierre %d%%"%(25*k))
        key.data.foreach_set("co",np.array([cap(t,a,k/4) for t,a in uv],np.float32).ravel())
        if previous:
            key.relative_key = previous
        previous = key
        morphs.append((key,lid,"Cerrar",k))
    # The closed crease lies flush against the skin. At rest it is buried.
    crease_mat = material("Pliegue suave"+side,(.29,.41,.043))
    def fold_point(u,v,closed=False):
        x = cx+rx*.95*u
        z = cz-.077+.040*u*u+(.008*v if closed else 0)
        return (x,face_y(x,z)+(-.008 if closed else .035),z)
    crease = disk("Pliegue de cierre"+side,fold_point,crease_mat,head)
    crease.shape_key_add(name="Oculto")
    key = crease.shape_key_add(name="Ojo cerrado")
    # disk() samples a fixed sequence, reused to transfer the closed profile.
    coords = [fold_point(0,0,True)]
    for r in range(1,13):
        for i in range(64):
            theta = math.tau*i/64
            coords.append(fold_point(r/12*math.cos(theta),r/12*math.sin(theta),True))
    key.data.foreach_set("co",np.array(coords,np.float32).ravel())
    morphs.append((key,lid,"Cerrar",4))


def tube(name, points, radii, mat, joint):
    points = list(map(Vector,points))
    vs,fs = [],[]
    for i,p in enumerate(points):
        direction = (points[min(i+1,len(points)-1)]-points[max(0,i-1)]).normalized()
        a = direction.cross(Vector((0,1,0))).normalized()
        b = direction.cross(a)
        for j in range(10):
            v = p+float(radii[i])*(a*math.cos(j*math.tau/10)+b*math.sin(j*math.tau/10))
            vs.append(v[:])
            if i:
                n = 10*i
                fs.append((n+j-10,n+(j+1)%10-10,n+(j+1)%10,n+j))
    fs.extend([tuple(reversed(range(10))),tuple(range(len(vs)-10,len(vs)))])
    return mesh(name,vs,fs,mat,joint)


stem = bone("Pecíolo",(.105,.015,5.39),(.41,0,5.56),head)
blade = bone("Hoja",(.41,0,5.56),(1.24,-.03,5.22),stem)
ts = np.linspace(0,1,21)
tube("Tallo curvo",[(.105+.32*t,.015*(1-t),5.38+.19*np.sin(t*1.56)) for t in ts],
     [.030-.008*t for t in ts],midrib,stem)
def blade_pos(t,v):
    w = max(0.,math.sin(math.pi*t))**.80
    return (.408+.95*t,-.024-.075*w*(1-v*v)+.07*v*w,
            5.565-.345*t+.102*w+.239*w*v)
vs = [blade_pos(t,v) for t in np.linspace(0,1,41) for v in np.linspace(-1,1,17)]
fs = [(i*17+j,i*17+j+17,i*17+j+18,i*17+j+1) for i in range(40) for j in range(16)]
bladeob = mesh("Hoja • lámina plegada",vs,fs,green,blade)
thick = bladeob.modifiers.new("Espesor de la hoja","SOLIDIFY")
thick.thickness = .015
def vein_curve(name,uvs,rads):
    pts = [(x,y-.006,z) for x,y,z in (blade_pos(t,v) for t,v in uvs)]
    tube(name,pts,rads,midrib,blade)
vein_curve("Nervadura central",[(t,0) for t in ts],[.012*(1-t)+.002 for t in ts])
for i,t0 in enumerate([.20,.39,.57,.72]):
    for sign in [-1,1]:
        vein_curve("Nervadura %d"%i,[(t0+.15*t,sign*.79*t**.8) for t in ts],[.006*(1-t)+.001 for t in ts])


# Pose offsets sit above FK joints, leaving their own rotation channels editable.
offsets = []
for side,sign in ((".L",1),(".R",-1)):
    for part,drop,forward in [("Brazo",43,0),("Antebrazo",3,-10),("Palma",-4,3)]:
        target = part+side
        a,b,parent = J[target]
        helper = bone("Base "+target,a,a+np.array([0,.15,0]),parent)
        J[target] = (a,b,helper)
        offsets.append((helper,sign*drop,forward))
    for part in ["Palma"]+[name+".1" for name in ("Pulgar","Índice","Medio","Menor")]:
        target = part+side
        a,b,parent = J[target]
        direction = b-a
        axis = direction if part=="Palma" else np.cross(direction,[0,-1,0])
        helper = bone("Base giro "+target,a,a+.12*axis/np.linalg.norm(axis),parent)
        J[target] = (a,b,helper)
        offsets.append((helper,sign*32 if part=="Palma" else (7 if part=="Pulgar.1" else 15),0))
arm = bpy.data.armatures.new("FK • Limonero")
rig = bpy.data.objects.new("RIG • Limonero",arm)
C["02"].objects.link(rig)
bpy.context.view_layer.objects.active = rig
rig.select_set(True)
bpy.ops.object.mode_set(mode="EDIT")
for label,(a,b,parent) in J.items():
    bone_data = arm.edit_bones.new(label)
    bone_data.head,bone_data.tail = a,b
    bone_data.use_deform = not label.startswith("Base ")
    if label.startswith(("Brazo","Antebrazo","Palma","Pulgar.","Índice.","Medio.","Menor.")):
        bone_data.align_roll(Vector((0,-1,0)))
for label,(a,b,parent) in J.items():
    if parent:
        arm.edit_bones[label].parent = arm.edit_bones[parent]
bpy.ops.object.mode_set(mode="OBJECT")
rig.show_in_front = True
rig.scale = (.945,1,.971)
arm.display_type = "STICK"
def property_driver(driver,variable,label,prop):
    var = driver.variables.new()
    var.name = variable
    var.targets[0].id = rig
    var.targets[0].data_path = rig.pose.bones[label].path_from_id()+'["'+prop+'"]'
    return var


controller = rig.pose.bones[trunk]
controller["Brazos en descanso"] = 1.
controller.id_properties_ui("Brazos en descanso").update(min=0.,max=1.,description="0 = pose A; 1 = brazos sueltos junto al cuerpo")
for side in (".L",".R"):
    palm_ctrl = rig.pose.bones["Palma"+side]
    palm_ctrl["Giro"] = 0.
    palm_ctrl.id_properties_ui("Giro").update(min=-55.,max=55.,description="Pronación / supinación en grados, distribuida por el antebrazo")
def twist_driver(pb,side,fraction):
    sign = 1 if side==".L" else -1
    driver = pb.driver_add("rotation_euler",1).driver
    property_driver(driver,"reposo",trunk,"Brazos en descanso")
    property_driver(driver,"giro","Palma"+side,"Giro")
    driver.expression = f"{sign*fraction*math.pi/180:.10f}*min(65,max(-65,32*reposo+min(55,max(-55,giro))))"


for helper,drop,forward in offsets:
    pb = rig.pose.bones[helper]
    pb.rotation_mode = "XYZ"
    pb.bone.hide = True
    if helper.startswith("Base giro Palma"):
        twist_driver(pb,helper[-2:],1.)
        continue
    for axis,angle in [(1,drop),(0,forward)]:
        if angle:
            driver = pb.driver_add("rotation_euler",axis).driver
            property_driver(driver,"reposo",trunk,"Brazos en descanso")
            driver.expression = f"{math.radians(angle):.9f}*reposo"
for name,side,fraction in twists:
    pb = rig.pose.bones[name]
    pb.rotation_mode = "XYZ"
    pb.bone.hide = True
    twist_driver(pb,side,fraction)


def limit_rotation(name,x,y,z):
    pb = rig.pose.bones[name]
    pb.rotation_mode = "XYZ"
    pb.lock_location = pb.lock_scale = (True,True,True)
    con = pb.constraints.new("LIMIT_ROTATION")
    con.name = "Recorrido articular"
    con.owner_space = "LOCAL"
    con.use_transform_limit = True
    for axis,bounds in zip("xyz",(x,y,z)):
        setattr(con,"use_limit_"+axis,True)
        setattr(con,"min_"+axis,math.radians(bounds[0]))
        setattr(con,"max_"+axis,math.radians(bounds[1]))
    pb.lock_rotation = tuple(a==b for a,b in (x,y,z))
    return con


for name,axes in [(trunk,((-23,25),(-24,24),(-18,18))),
                  (neck,((-15,18),(-20,20),(-13,13))),
                  (head,((-20,22),(-38,38),(-20,20))),
                  (pelvis,((-15,15),(-20,20),(-12,12)))]:
    limit_rotation(name,*axes)
for side,sign in ((".L",1),(".R",-1)):
    shoulder = limit_rotation("Brazo"+side,(-48,65),(-30,30),(-80,80))
    # The lower stop includes the relaxation helper. Lowering the pose control
    # should not give the animator an extra 43 degrees through the chest.
    for attr,angle in [("min_z",-52 if sign==1 else -95),("max_z",95 if sign==1 else 52)]:
        driver = shoulder.driver_add(attr).driver
        property_driver(driver,"reposo",trunk,"Brazos en descanso")
        driver.expression = f"{math.pi/180:.10f}*({angle}+{sign*43}*reposo)"
    limit_rotation("Antebrazo"+side,(-12,95),(-12,12),(-72,72))
    limit_rotation("Palma"+side,(-43,54),(0,0),(-21,21))
    for digit in ("Pulgar","Índice","Medio","Menor"):
        for n in (1,2):
            splay = (20 if digit=="Pulgar" else 12) if n==1 else 3
            limit_rotation(digit+"."+str(n)+side,(-7,64 if digit=="Pulgar" else 76),(0,0),(-splay,splay))
    limit_rotation("Muslo"+side,(-65,45),(-25,25),(-35,35))
    limit_rotation("Pierna"+side,(-8,95),(-6,6),(-8,8))
    limit_rotation("Pie"+side,(-30,35),(-16,16),(-14,14))
groups = {name:arm.collections.new(name) for name in ("Tronco y piernas","Brazos y manos","Cara y hoja","Mecanismo")}
for pb in rig.pose.bones:
    if pb.name.startswith(("Base ","Torsión")):
        group = "Mecanismo"
    elif pb.name.startswith(("Brazo","Antebrazo","Palma","Pulgar.","Índice.","Medio.","Menor.")):
        group = "Brazos y manos"
        pb.bone.color.palette = "THEME04" if pb.name.endswith(".L") else "THEME03"
    elif pb.name.startswith(("Mirada","Párpado","Boca","Pecíolo","Hoja")):
        group = "Cara y hoja"
    else:
        group = "Tronco y piernas"
    groups[group].assign(pb.bone)
for ob,joint in [(body,None)]+skin:
    if joint:
        ob.vertex_groups.new(name=joint).add(list(range(len(ob.data.vertices))),1,"REPLACE")
    ob.parent = rig
    mod = ob.modifiers.new("Piel • pesos normalizados","ARMATURE")
    mod.object = rig
    mod.use_deform_preserve_volume = True
for key,label,prop,*stage in morphs:
    pb = rig.pose.bones[label]
    pb[prop] = 0.
    pb.id_properties_ui(prop).update(min=0,max=1)
    driver = key.driver_add("value").driver
    if label==jaw:
        for vowel in visemes:
            property_driver(driver,"v"+vowel,jaw,vowel)
        driver.expression = "v"+prop+"/max(1,"+"+".join("v"+v for v in visemes)+")"
    else:
        property_driver(driver,"nivel",label,prop)
        driver.expression = "min(1,max(0,4*nivel-%d))"%(stage[0]-1) if stage else "nivel"


# Authoring scene: still poses only.
sc.render.fps = 24
sc.frame_set(1)


def aim(ob,target):
    ob.rotation_euler = (Vector(target)-ob.location).to_track_quat("-Z","Y").to_euler()
def studio(name,data,loc):
    ob = bpy.data.objects.new(name,data)
    C["03"].objects.link(ob)
    ob.location = loc
    return ob
cam = studio("Cámara • tres cuartos",bpy.data.cameras.new("Ortográfica"),(6.2,-15,5.48))
aim(cam,(0,0,2.87))
cam.data.type,cam.data.ortho_scale = "ORTHO",6.74
sc.camera = cam
for name,loc,power,size,color in [
        ("Principal • suave",(-4,-4,8),555,2.1,(1,.98,.90)),
        ("Relleno",(3,-5,5),255,5,(1,1,1)),
        ("Luz de borde",(3,2,7),520,3,(.86,.94,1))]:
    light = bpy.data.lights.new(name,"AREA")
    ob = studio(name,light,loc)
    aim(ob,(0,0,3))
    light.energy,light.shape,light.size,light.color = power,"DISK",size,color
floor = mesh("Suelo • carbón",[(-100,-100,.016),(100,-100,.016),(100,100,.016),(-100,100,.016)],
             [(0,1,2,3)],material("Estudio • gris carbón",(.011,.012,.015),.89))
floor.data.materials[0].node_tree.nodes.get("Principled BSDF").inputs["Specular IOR Level"].default_value = .04
floor.data.materials[0].node_tree.nodes.get("Principled BSDF").inputs["Emission Color"].default_value = (.010,.011,.013,1)
floor.data.materials[0].node_tree.nodes.get("Principled BSDF").inputs["Emission Strength"].default_value = .64
C["01"].objects.unlink(floor)
C["03"].objects.link(floor)
world = bpy.data.worlds.new("Ambiente neutro")
sc.world = world
world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (.30,.32,.37,1)
world.node_tree.nodes["Background"].inputs[1].default_value = .45
sc.render.engine = "CYCLES"
sc.cycles.samples = 24
sc.cycles.use_denoising = True
sc.view_settings.view_transform = "Standard"
sc.render.resolution_x,sc.render.resolution_y = 504,686
sc.render.resolution_percentage = 100
sc.render.image_settings.color_mode = "RGBA"
for area in bpy.context.screen.areas:
    if area.type=="VIEW_3D":
        area.spaces.active.region_3d.view_perspective = "CAMERA"
        area.spaces.active.overlay.show_overlays = True
        area.spaces.active.overlay.show_extras = False
        area.spaces.active.overlay.show_relationship_lines = False
        area.spaces.active.shading.type = "MATERIAL"
        area.spaces.active.shading.color_type = "MATERIAL"
# Portable sprite scene uses the same character, lighting, animation and camera.
sprite = sc.copy()
sprite.use_fake_user = True
sprite.name = "02 • Juego 2D (alfa)"
sc.name = "01 • Referencia (estudio)"
def layer_collection(layer,collection):
    if layer.collection == collection:
        return layer
    for child in layer.children:
        found = layer_collection(child,collection)
        if found:
            return found
stage = bpy.data.collections.new("04 • Fondo de presentación")
sc.collection.children.link(stage)
C["03"].objects.unlink(floor)
stage.objects.link(floor)
sprite.render.film_transparent = True
sprite.render.resolution_x = sprite.render.resolution_y = 512
sprite.render.filepath = "//reposo/cuadro_"
sc.render.filepath = "//Presentacion.png"
bpy.context.view_layer.objects.active = rig
rig.select_set(True)
bpy.ops.object.mode_set(mode="POSE")
arm.bones.active = arm.bones["Palma.L"]
bpy.context.preferences.filepaths.save_version = 0
bpy.ops.wm.save_as_mainfile(filepath=str(OUT/"Limonero_v4.blend"))
bpy.ops.render.render(write_still=True,scene=sc.name)
controller["Brazos en descanso"] = 0.
rig.update_tag()
bpy.context.view_layer.update()
cam.location = (1.57,-8,2.78)
aim(cam,(1.57,-.03,2.23))
cam.data.ortho_scale = .99
sc.render.resolution_x,sc.render.resolution_y = 600,480
sc.render.filepath = "//Mano_detalle.png"
bpy.ops.render.render(write_still=True,scene=sc.name)
