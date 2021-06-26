from maya import cmds
from maya.api import OpenMaya as om
from .math_utils import vector_math as vm

def getVectorToSegment(start, end, vertex):
    vertex_p = om.MPoint(cmds.xform(vertex, q=1, ws=1, t=1))
    start_p = om.MPoint(cmds.xform(start, q=1, ws=1, t=1))
    end_p = om.MPoint(cmds.xform(end, q=1, ws=1, t=1))
    closestPnt =  vm.closestPointInLine(start_p, end_p, vertex_p)
    parentMatrix = om.MMatrix(cmds.xform(start, q=1, ws=1, m=1))
    return parentMatrix*(vertex_p-closestPnt)
    



joints = [u'joint1', u'joint2', u'joint3']
vertices = cmds.ls('pCylinder1.vtx[:]', fl=1)
bind_data = dict()
cmds.currentTime(0)
for vertex in vertices:
    closest_point, closest_seg = vm.getCloserSegment(vertex, joints)

    bind_data[vertex] = {'dist': initial_dist, 'segment':closest_seg, 'pos': cmds.xform(vertex, q=1, ws=1, t=1)}
deform_data = dict()
cmds.currentTime(20)
for vertex, data in bind_data.iteritems():
    closest_seg = data.get('segment')
    bind_pos = om.MPoint(data.get('pos'))
    vector = getVectorToSegment(closest_seg[0], closest_seg[1], vertex)
    delta = om.MPoint(cmds.xform(vertex, q=1, ws=1, t=1))-bind_pos
    deform_data[vertex] = {'dist': vector.length(), 'delta': list(delta), 'vector': list(vector)}
    
cmds.currentTime(0)
threshold = .05
wrong_vertices = list()    
for vertex in vertices:
     bind_dist = bind_data.get(vertex).get('dist')
     deform_dist = deform_data.get(vertex).get('dist')  
     if bind_dist - deform_dist > threshold:
         wrong_vertices.append(vertex)        
cmds.select(wrong_vertices)


from maya import cmds
from maya.api import OpenMaya as om
from RL_Maya.src.math_utils import vector_math as vm
from RL_Maya.src.maya_utils import attrUtils
import numpy as np
import math

def getVectorToSegment(start, end, vertex):
    vertex_p = om.MPoint(cmds.xform(vertex, q=1, ws=1, t=1))
    start_p = om.MPoint(cmds.xform(start, q=1, ws=1, t=1))
    end_p = om.MPoint(cmds.xform(end, q=1, ws=1, t=1))
    closestPnt =  vm.closestPointInLine(start_p, end_p, vertex_p)
    parentMatrix = om.MMatrix(cmds.xform(start, q=1, ws=1, m=1))
    return parentMatrix*(vertex_p-om.MPoint(closestPnt))

def getVerticesToEval(mesh, joints):
    skinClusters = [a for a in cmds.listHistory(mesh, pdo=1) if cmds.nodeType(a) == "skinCluster"]
    vertices = list()
    cmds.select(cl=1)
    for b in joints[:-1]:
        cmds.skinCluster(skinClusters[0], e=1, selectInfluenceVerts=b)
        vertices.extend(cmds.ls(sl=1, fl=1))
    cmds.select(cl=1)
    return vertices

mesh = "CHR_Body"
joints = ["B_Clavicle_L", "B_Arm_L", "B_Forearm_L"]
joints_pos = [cmds.xform(a, q=1, ws=1, t=1) for a in joints]
vertices = getVerticesToEval(mesh, joints)
bind_data = dict()
for vertex in vertices:
    pos = cmds.xform(vertex, q=1, ws=1, t=1)
    closest_point, closest_seg = vm.getCloserSegment(pos, joints_pos)
    initial_dist = vm.magnitude(pos-closest_point)
    vector = getVectorToSegment(joints[closest_seg[0]], joints[closest_seg[1]], vertex)
    bind_data[vertex] = {'dist': initial_dist,
                         'segment':closest_seg,
                         'pos': pos,
                         'vector': list(vector)}
animations = attrUtils.getAnimationValues(2)
deform_data = dict([(a,{'dist': 0.0 , 'vector': om.MVector()}) for a in bind_data.keys()])

with attrUtils.DisconnectCtx(joints, source=True, destination=False):
    for attr, values in animations:
        cmds.setAttr(f"{joints[1]}.{attr}", math.degrees(values[-1]))
        for vertex, data in bind_data.items():
            closest_seg = data.get('segment')
            bind_pos = om.MPoint(data.get('pos'))
            bind_dist = data.get('dist')
            bind_vector = om.MVector(data.get('vector'))
            vector = getVectorToSegment(joints[closest_seg[0]], joints[closest_seg[1]], vertex)
            deform_data[vertex]['dist'] += abs(bind_dist-vector.length())
            deform_data[vertex]['vector'] += om.MVector(vector)-bind_vector

threshold = 1
wrong_vertices = dict()    
for vertex, data in deform_data.items():
     if data.get('dist') > threshold:
         wrong_vertices[vertex] = data
cmds.select(list(wrong_vertices.keys()))



k = 6
centroids, closest = evalKMeans(k, raw_data)
groups = dict([(a, []) for a in range(k)])
for index, x in zip(wrong_vertices, closest):
    groups[x].append(f"{mesh}.vtx[{index}]")
cmds.select(groups[5])

'''
k means
import numpy as np

def initialize_centroids(points, k):
    """returns k centroids from the initial points"""
    centroids = points.copy()
    np.random.shuffle(centroids)
    return centroids[:k]
 
def closest_centroid(points, centroids):
    """returns an array containing the index to the nearest centroid for each point"""
    distances = np.sqrt(((points - centroids[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0) 

def move_centroids(points, closest, centroids):
    """returns the new centroids assigned from the points closest to them"""
    return np.array([points[closest==k].mean(axis=0) for k in range(centroids.shape[0])])

vectors = list()
for vtx in  wrong_vertices:
    vect = list(om.MVector(deform_data.get(vtx).get('vector')).normal())
    vect.extend(cmds.xform(vtx, q=1, ws=1, t=1))
    vectors.append(vect)
k = 3
groups = dict([(a, []) for a in range(k)])

points = np.array(vectors)
centroids = initialize_centroids(points, k)
prev_centroids = np.zeros_like(centroids)
for x in range(100):
    closest = closest_centroid(points, centroids)
    centroids = move_centroids(points, closest, centroids)
    if np.linalg.norm(centroids - prev_centroids) > 0:
       prev_centroids =  centroids.copy()
       print x
    else:
        print 'finished'
        break
closest = closest_centroid(points, centroids)
for i, x in enumerate(closest):
    groups[x].append(wrong_vertices[i])
cmds.select(groups[2])
'''
"""
PCA

import numpy as np

def standardize_data(X):
         
    '''
    This function standardize an array, its substracts mean value, 
    and then divide the standard deviation.
    
    param X: array 
    return: standardized array
    '''    
    rows, columns = X.shape
    
    standardizedArray = np.zeros(shape=(rows, columns))
    tempArray = np.zeros(rows)
    
    for column in range(columns):
        
        mean = np.mean(X[:,column])
        std = np.std(X[:,column])
        tempArray = np.empty(0)
        
        for element in X[:,column]:
            
            tempArray = np.append(tempArray, ((element - mean) / std))
 
        standardizedArray[:,column] = tempArray
    
    return standardizedArray
vectors = list()
for vtx in  wrong_vertices:
    vect = list(om.MVector(deform_data.get(vtx).get('vector')))
    #vect.extend(cmds.xform(vtx, q=1, ws=1, t=1))
    vectors.append(vect)
data = np.array(vectors)
X = standardize_data(data)
covariance_matrix = np.cov(X.T)   # The result is a Positive semidefinite matrix
eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
eigen_vec_svd, s, v = np.linalg.svd(X.T)
variance_explained = [(i/sum(eigen_values))*100 for i in eigen_values]
cumulative_variance_explained = np.cumsum(variance_explained)
projection_matrix = (eigen_vectors.T[:][:1]).T
X_pca = X.dot(projection_matrix)

"""
# Fuzzy c means

from fcmeans import fcm
from fcmeans import FuzzyCMeans
import numpy as np
from maya.api import OpenMaya as om2



points=np.array([cmds.xform(v, q=1, ws=1, t=1) for v in wrong_vertices])
vectors = list()
for i, vtx in  enumerate(wrong_vertices):
    vect = list(om.MVector(deform_data.get(vtx).get('vector')).normal())
    vect.extend(points[i])
    vectors.append(vect)

raw_data = np.array(vectors)

cluster_n = 3
expo = 2
min_err = 0.001 
max_iter = 500 
verbose = 1
m,c, m1 = FuzzyCMeans.fcm(raw_data,cluster_n,expo,min_err,max_iter,verbose)
#groups = dict([(a, []) for a in range(k)])

#for i, x in enumerate(m1):
#    groups[x].append(wrong_vertices[i])
#cmds.select(groups[2])

# API get the mesh
selList = om2.MSelectionList()
selList.add('pCylinder1')
dagPath = selList.getDagPath(0)
selMesh = om2.MFnMesh(dagPath)

# Get vert list
vertList = list(set(selMesh.getVertices()[1]))
lenVertList = len(vertList)

# Initial color info
tempColor = om2.MColor([0.0,0.0,0.0])
vertexColorList = om2.MColorArray(lenVertList, tempColor)
for i, x in enumerate(m.T):
    index = int(wrong_vertices[i].split('[')[-1][:-1])
    vertexColorList[index].r = x[0]
    vertexColorList[index].g = x[1]
    vertexColorList[index].b = x[2]

# Sets vert colors
selMesh.setVertexColors(vertexColorList, vertList)


# Create joints on most influenced vertex position

for j in range(cluster_n):
    cmds.select( d=True )
    jnt = cmds.joint()
    cmds.xform(jnt, ws=1, t=points[np.argmax(m.T[:,j])])

#####----------------------------
## goblin 
#####----------------------------
from maya import cmds
from maya.api import OpenMaya as om
from RL_Maya.src.math_utils import vector_math as vm
from RL_Maya.src.maya_utils import attrUtils
from RL_Maya.src.math_utils import k_means
from RL_Maya.src.math_utils import fuzzyCMeans

import numpy as np
import math

def getVectorToSegment(start, end, vertex):
    vertex_p = om.MPoint(cmds.xform(vertex, q=1, ws=1, t=1))
    start_p = om.MPoint(cmds.xform(start, q=1, ws=1, t=1))
    end_p = om.MPoint(cmds.xform(end, q=1, ws=1, t=1))
    closestPnt =  vm.closestPointInLine(start_p, end_p, vertex_p)
    parentMatrix = om.MMatrix(cmds.xform(start, q=1, ws=1, m=1))
    return parentMatrix*(vertex_p-om.MPoint(closestPnt))

def getVerticesToEval(mesh, joints):
    skinClusters = [a for a in cmds.listHistory(mesh, pdo=1) if cmds.nodeType(a) == "skinCluster"]
    vertices = list()
    cmds.select(cl=1)
    for b in joints[:-1]:
        cmds.skinCluster(skinClusters[0], e=1, selectInfluenceVerts=b)
        vertices.extend(cmds.ls(sl=1, fl=1))
    cmds.select(cl=1)
    return vertices

def createJoint(pos, joints):
    segments_pos = [np.array(cmds.xform(a, q=1, ws=1, t=1)) for a in joints]
    closes_pnt, closerSeg = vm.getCloserSegment(pos, segments_pos)
    zAxis = vm.normalize(pos - closes_pnt)
    yAxis = vm.normalize(segments_pos[closerSeg[1]] - segments_pos[closerSeg[0]])
    xAxis = vm.normalize(np.cross(yAxis, zAxis)) 
    zAxis = vm.normalize(np.cross(xAxis, yAxis))
    jnt = cmds.joint()
    grp = cmds.group(jnt)
    matrix = list(xAxis)
    matrix.append(0)
    matrix.extend(yAxis)
    matrix.append(0)
    matrix.extend(zAxis)
    matrix.append(0)
    matrix.extend(pos)
    matrix.append(1)
    cmds.xform(grp, os=1, m=matrix)
    seg = np.array(segments_pos[2])-np.array(segments_pos[0])
    vect = np.array(pos)-np.array(segments_pos[0])
    w = (np.dot(vect, seg) / np.linalg.norm(seg)**2)
    cmds.parentConstraint(joints[0], joints[1], grp, mo=1)
    cmds.parentConstraint(joints[0], grp, e=True, w=1-w)
    cmds.parentConstraint(joints[1], grp, e=True, w=w)


mesh = "CHR_Body"
joints = ["B_Clavicle_L", "B_Arm_L", "B_Forearm_L"]
joints_pos = [cmds.xform(a, q=1, ws=1, t=1) for a in joints]
vertices = getVerticesToEval(mesh, joints)
bind_data = dict()
for vertex in vertices:
    pos = cmds.xform(vertex, q=1, ws=1, t=1)
    closest_point, closest_seg = vm.getCloserSegment(pos, joints_pos)
    initial_dist = vm.magnitude(pos-closest_point)
    vector = getVectorToSegment(joints[closest_seg[0]], joints[closest_seg[1]], vertex)
    bind_data[vertex] = {'dist': initial_dist,
                         'segment':closest_seg,
                         'pos': pos,
                         'vector': list(vector)}
animations = attrUtils.getAnimationValues(frame_num=2)
deform_data = dict([(a,{'dist': 0.0 , 'vector': om.MVector()}) for a in bind_data.keys()])

data = bind_data[vertex]
with attrUtils.DisconnectCtx(joints, source=True, destination=False):
    for attr, values in animations:
        cmds.setAttr(f"{joints[1]}.{attr}", math.degrees(values[-1]))
        for vertex, data in bind_data.items():
            closest_seg = data.get('segment')
            bind_pos = om.MPoint(data.get('pos'))
            bind_dist = data.get('dist')
            bind_vector = om.MVector(data.get('vector'))
            vector = getVectorToSegment(joints[closest_seg[0]], joints[closest_seg[1]], vertex)
            deform_data[vertex]['dist'] += abs(bind_dist-vector.length())
            deform_data[vertex]['vector'] += om.MVector(vector)-bind_vector

threshold = 1
wrong_vertices = dict()    
for vertex, data in deform_data.items():
    index = int("".join([a for a in vertex.split(".")[-1] if a.isdigit()]))
    if data.get('dist') > threshold:
        wrong_vertices[vertex] = data
cmds.select(cl=1)
for vtx in wrong_vertices.keys():
    cmds.select(vtx, add=1)

points=np.array([cmds.xform(v, q=1, ws=1, t=1) for v in wrong_vertices])
vectors = list()
for i, vtx in  enumerate(wrong_vertices):
    vect = list(om.MVector(deform_data.get(vtx).get('vector')).normal())
    vect.extend(points[i])
    vectors.append(vect)

raw_data = np.array(vectors)
expo = 2
min_err = 0.001 
max_iter = 500 
verbose = 0
cluster_n = 4
m, centroids, m1 = fuzzyCMeans.fcm(raw_data, cluster_n, expo, min_err, max_iter, verbose)
duplicated = [cmds.duplicate(a, po=1, rc=1, n=f"rl_driver{i}")[0] for i,a in enumerate(joints)]
cmds.parent(duplicated[2], duplicated[1])
cmds.parent(duplicated[1], duplicated[0])
cmds.parent(duplicated[0], w=1)
for j in range(cluster_n):
    cmds.select(d=True )
    createJoint(points[np.argmax(m.T[:,j])], duplicated)
    # cmds.xform(jnt, ws=1, t=points[np.argmax(m.T[:,j])])




from maya.api import OpenMaya as om
from RL_Maya.src.maya_utils import mNode
import numpy as np
from RL_Maya.src.math_utils import vector_math as vm
from RL_Maya.src.math_utils import fuzzyCMeans

def getTrianglePoints(mesh, triangles):
    if isinstance(mesh, mNode.MNode):
        node = mesh
    else:
        node = mNode.MNode(mesh)
    mfn = node.getShape().getBestFn()
    positions = np.array(mfn.getPoints())
    vtx_pos = positions[triangles.flatten()]
    # reshape in 3X4 matrix (triangleXpoint)
    return vtx_pos.reshape(-1,3,4)

def getVerticesToEval(mesh, joints):
    # todo get data from node intead usign command
    skinClusters = [a for a in cmds.listHistory(mesh, pdo=1) if cmds.nodeType(a) == "skinCluster"]
    vertices = list()
    cmds.select(cl=1)
    for b in joints[:-1]:
        cmds.skinCluster(skinClusters[0], e=1, selectInfluenceVerts=b)
        vertices.extend([int(a.split("[")[-1][:-1]) for a in cmds.ls(sl=1, fl=1)])
    cmds.select(cl=1)
    return list(set(vertices))

def getVertextriangles(all_triangles, vertices):
    vertex_is_in_triangle = np.all(np.isin(all_triangles, vertices), axis=1)
    return all_triangles[np.where(vertex_is_in_triangle)]
cmds.currentTime(0)
joints = ["joint1", "joint2", "joint3"]    
mesh = mNode.MNode("pCylinder1")
mfn = mesh.getShape().getBestFn()
positions = np.array(mfn.getPoints(space=om.MSpace.kWorld))
vectors = np.delete(positions, -1, 1) # remove last element of each point
normals = np.array(mfn.getNormals())
triangle_counts, triangle_vertices =  mfn.getTriangles()
all_triangles = np.array(triangle_vertices).reshape(-1,3)
vertices = getVerticesToEval(mesh, joints)
#vertices = [int(a.split("[")[-1][:-1]) for a in cmds.ls(sl=1, fl=1)]
triangles = getVertextriangles(all_triangles, vertices)
#triangle_points = getTrianglePoints(mesh, triangles])

joints_pos = [cmds.xform(a, q=1, ws=1, t=1) for a in joints]
vertex_data = np.zeros((len(vertices), 2), np.int16)
for vertex in vertices:
    closest_point, closest_seg = vm.getCloserSegment(vectors[vertex], joints_pos)
    vertex_data[vertex] = closest_seg

triangle_bind = dict()
for i, tri in enumerate(triangles):
    start_seg_idnx = np.bincount(vertex_data[tri][:,0]).argmax()
    end_seg_idnx = np.bincount(vertex_data[tri][:,1]).argmax()
    tri_segment = [start_seg_idnx, end_seg_idnx]
    start = cmds.xform(joints[tri_segment[0]], q=1, ws=1, t=1)
    end = cmds.xform(joints[tri_segment[1]], q=1, ws=1, t=1)
    vol = vm.getTriangleVolume(start, end, vectors[tri])
    # plane = cmds.polyPlane(sx=1, sy=1)[0]
    # [cmds.xform(f"{plane}.vtx[{i}]", ws=1, t=a) for i, a in enumerate(proj)]
    # cmds.xform(f"{plane}.vtx[3]", ws=1, t=proj[-1])
    triangle_bind[i] = {"vertices": list(tri), "segment": tri_segment, "volume": vol}
cmds.currentTime(20)    
triangle_deform = dict()
deformed_positions = np.array(mfn.getPoints(space=om.MSpace.kWorld))
deformed_vectors = np.delete(deformed_positions, -1, 1)
for i, data in triangle_bind.items():
    start = cmds.xform(joints[data.get("segment")[0]], q=1, ws=1, t=1)
    end = cmds.xform(joints[data.get("segment")[1]], q=1, ws=1, t=1)
    vol = vm.getTriangleVolume(start, end, deformed_vectors[data.get("vertices")])
    #plane = cmds.polyPlane(sx=1, sy=1, n="p_"+"_".join([str(a) for a in data.get("vertices")]))[0]
    #[cmds.xform(f"{plane}.vtx[{i}]", ws=1, t=a) for i, a in enumerate(proj)]
    #cmds.xform(f"{plane}.vtx[3]", ws=1, t=proj[-1])
    triangle_deform[i] = {"volume": vol}

cmds.currentTime(0)    
volume_vtx = np.zeros((len(vertices)), np.float16)
for i, data in triangle_bind.items():
    delta = abs(triangle_deform[i].get("volume")-triangle_bind[i].get("volume"))
    volume_vtx[triangle_bind[i].get("vertices")]+=delta

raw_data = list()
wrong_points = list()
wrong_vtx = list()
for x, vol in enumerate(volume_vtx):
    if vol>.1:
        pos = vectors[x]
        normal = normals[x]*vol
        data = list(pos)
        data.extend(normal)
        raw_data.append(data)
        wrong_points.append(pos)
        wrong_vtx.append(x)
        

raw_data = np.array(raw_data)
expo = 2
min_err = 0.001 
max_iter = 500 
verbose = 0
cluster_n = 4
m, centroids, m1 = fuzzyCMeans.fcm(raw_data, cluster_n, expo, min_err, max_iter, verbose)
duplicated = [cmds.duplicate(a, po=1, rc=1, n=f"rl_driver{i}")[0] for i,a in enumerate(joints)]
cmds.parent(duplicated[2], duplicated[1])
cmds.parent(duplicated[1], duplicated[0])
[cmds.parentConstraint(a, b) for a,b in zip(joints, duplicated)]
#cmds.parent(duplicated[0], w=1)
cluster_joints = list()
for j in range(cluster_n):
    cmds.select(d=True )
    #jnt = cmds.joint()
    cluster_joints.append(createJoint(wrong_points[np.argmax(m.T[:,j])], duplicated))
    #cmds.xform(jnt, ws=1, t=wrong_points[np.argmax(m.T[:,j])])
skin_cluster = "skinCluster1"
cmds.skinCluster(skin_cluster, e=1, wt=0, ai=cluster_joints)

mesh_vtx = "pCylinder1.vtx[{}]"
for vtx, weights in zip(wrong_vtx, m.T):
    max_indices = np.argpartition(weights, -2)[-2:]
    for i in max_indices:
        cmds.skinPercent(skin_cluster, mesh_vtx.format(vtx), transformValue=[(cluster_joints[i], weights[i])])



