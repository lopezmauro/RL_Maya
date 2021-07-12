import math
import numpy as np
from maya import cmds
from maya.api import OpenMaya as om
from rl_vp.maya_utils import mUtils, skinCluster, meshes, attrUtils, transforms, rewards
from rl_vp.math_utils import vector_math as vm
from rl_vp.math_utils import fuzzyCMeans

DELTA_VOLUME_ATTR = "deltaVolume"
CLUSTER_VERTICES_ATTR = "clustersVertices"
CLUSTER_JOINTS_ATTR = "clustersJoints"
CLUSTER_WEIGHTS_ATTR = "clustersWeights"
ALL_METADATA_ATTR = [DELTA_VOLUME_ATTR, CLUSTER_VERTICES_ATTR,
                     CLUSTER_JOINTS_ATTR, CLUSTER_WEIGHTS_ATTR]


def getVolumeChange(mesh, joints):
    mesh = mUtils.MNode(mesh)
    mfn = mesh.getShape().getBestFn()
    positions = np.array(mfn.getPoints(space=om.MSpace.kWorld))[:, :3]
    triangle_counts, triangle_vertices = mfn.getTriangles()
    all_triangles = np.array(triangle_vertices).reshape(-1, 3)
    vertices = skinCluster.getInfluencesVertices(mesh, joints)
    triangles = meshes.getVertextriangles(all_triangles, vertices)
    bind_data = rewards.getTriangleBindData(joints, triangles, positions)
    bind_volume = rewards.getTrianglesVolume(positions, bind_data)
    deformation_volume = list()
    animations = attrUtils.getAnimationValues(2)
    default_mxt = cmds.xform(joints[1], q=1, ws=1, m=1)
    decendant = set(joints)
    for jnt in joints:
        decendant.update(cmds.listRelatives(jnt, ad=1, type="joint"))
    with attrUtils.DisconnectCtx(decendant, source=True, destination=False):
        for attr, values in animations:
            cmds.setAttr(f"{joints[1]}.{attr}", math.degrees(values[-1]))
            deformed_positions = np.array(mfn.getPoints(space=om.MSpace.kWorld))
            deformation_volume.append(rewards.getTrianglesVolume(deformed_positions, bind_data))
        cmds.xform(joints[1], ws=1, m=default_mxt)
    delta_volume_vtx = np.zeros((len(positions)), np.float16)
    for i, data in bind_data.items():
        for deformed in deformation_volume:
            delta = abs(bind_volume[i] - deformed[i])
            delta_volume_vtx[data.get("vertices")] += delta
    return delta_volume_vtx


def getVolumeLossVertices(delta_volume_vtx, tol=0.03):
    wrong_vtx = list()
    for x, vol in enumerate(delta_volume_vtx):
        if vol > tol:
            wrong_vtx.append(x)
    return wrong_vtx


def createClustersJoints(mesh, joints, wrong_vtx, delta_volume_vtx, cluster_n=4):
    mesh_node = mUtils.MNode(mesh)
    mfn = mesh_node.getShape().getBestFn()
    positions = np.array(mfn.getPoints(space=om.MSpace.kWorld))[:, :3]
    normals = np.array(mfn.getNormals())
    # default fuzzyCMeans arguments
    expo = 2
    min_err = 0.001
    max_iter = 500
    verbose = 0
    raw_data = list()
    for x in wrong_vtx:
        pos = positions[x]
        vol_normal = normals[x]
        data = list(pos)
        data.extend(vol_normal)
        data.append(delta_volume_vtx[x])
        raw_data.append(data)
    raw_data = np.array(raw_data)
    m, centroids, m1 = fuzzyCMeans.fcm(raw_data, cluster_n, expo, min_err, max_iter, verbose)
    cluster_joints = list()
    for j in range(cluster_n):
        indx = np.argmax(m.T[:, j])
        vtx = wrong_vtx[indx]
        cluster_joints.append(transforms.createAimedJoint(positions[vtx], joints))
    return cluster_joints, m.T


def applyClustersDefaultSkin(mesh, clusters_joints, clusters_vertices, clusters_weights, prune=0.1):
    mesh_node = mUtils.MNode(mesh)
    mfn = mesh_node.getShape().getBestFn()
    positions = np.array(mfn.getPoints(space=om.MSpace.kWorld))[:, :3]
    skin_cluster = skinCluster.getDeformersFromMesh(mesh_node)[0]
    faceVertices = meshes.getFacesVertices(mesh_node)
    vertexFaces = meshes.getVerticesFaces(faceVertices)
    vertices_distances = list()
    joints_positions = [np.array(cmds.xform(a, q=1, ws=1, t=1)) for a in clusters_joints]
    for i, joint_pos in enumerate(joints_positions):
        start_vtx = vm.getCloserIndex(joint_pos, positions[clusters_vertices])
        dist_dict = meshes.getGeodesicDistances(clusters_vertices[start_vtx], clusters_vertices, positions, vertexFaces, faceVertices)
        vertices_distances.append([dist_dict[a] for a in clusters_vertices])

    vtx_dist = np.array(vertices_distances).T
    max_dist = np.amax(vtx_dist, axis=1)
    influences = skinCluster.getSkinInfluences(skin_cluster)
    missing_inf = [a for a in clusters_joints if a not in influences.values()]
    if missing_inf:
        cmds.skinCluster(skin_cluster, e=1, wt=0, ai=missing_inf)
    new_weighst = dict()
    for i, vtx in enumerate(clusters_vertices):
        normal_dist = 1 - (vtx_dist[i] / max_dist[i])
        weights = clusters_weights[i]
        dist_weights = weights * normal_dist
        if prune:
            mask = np.greater_equal(dist_weights, prune)
            dist_weights *= mask
        vtx_weighst = dict()
        for i, w in enumerate(dist_weights):
            vtx_weighst[clusters_joints[i]] = w
        new_weighst[vtx] = vtx_weighst

    weightMap = skinCluster.getSkinWeightsMapping(skin_cluster, normalize=False, tol=0.0001)
    for k, v in new_weighst.items():
        weightMap[k].update(v)
    normal_weights = skinCluster.normalizeWeights(weightMap)
    relaxedMap = skinCluster.relaxWeightMapping(normal_weights, faceVertices, vertexFaces, iterations=1)
    skinCluster.setSkinWeights(skin_cluster, relaxedMap)


def setMetadataAttribute(node, attr_name, data, storable=False):
    if not cmds.objExists(f"{node}.{attr_name}"):
        cmds.addAttr(node, longName=attr_name, dt="string", storable=storable)
    cmds.setAttr(f"{node}.{attr_name}", str(data), type="string")


def clearMetadata(node):
    for attr_name in ALL_METADATA_ATTR:
        if cmds.objExists(f"{node}.{attr_name}"):
            cmds.setAttr(f"{node}.{attr_name}", "", type="string")


def initData(mesh, joints):
    clearMetadata(mesh)
    delta_volume_vtx = getVolumeChange(mesh, joints)
    data_list = delta_volume_vtx.tolist()
    setMetadataAttribute(mesh, DELTA_VOLUME_ATTR, str(data_list))
    return delta_volume_vtx


def selectAffectedVertices(mesh, tolerance=0.03, select=True):
    if not cmds.objExists(f"{mesh}.{DELTA_VOLUME_ATTR}"):
        raise BaseException(f"unable to find {DELTA_VOLUME_ATTR} in {mesh} was not inited")
    delta_str = cmds.getAttr(f"{mesh}.{DELTA_VOLUME_ATTR}")
    if not delta_str:
        raise BaseException(f"unable to find {DELTA_VOLUME_ATTR} data in {mesh} was not inited")
    delta_volume_vtx = np.array(eval(delta_str))
    wrong_vtx = getVolumeLossVertices(delta_volume_vtx, tolerance)
    setMetadataAttribute(mesh, CLUSTER_VERTICES_ATTR, wrong_vtx, storable=True)
    if select:
        cmds.select(cl=1)
        for i in wrong_vtx:
            cmds.select(f"{mesh}.vtx[{i}]", add=1)
    return wrong_vtx


def initClustersJoints(mesh, joints, cluster_n=4, tolerance=0.03):
    if not cmds.objExists(f"{mesh}.{DELTA_VOLUME_ATTR}"):
        raise BaseException(f"unable to find {DELTA_VOLUME_ATTR} in {mesh} was not inited")
    delta_str = cmds.getAttr(f"{mesh}.{DELTA_VOLUME_ATTR}")
    if not delta_str:
        raise BaseException(f"unable to find {DELTA_VOLUME_ATTR} data in {mesh} was not inited")
    wrong_vtx_str = ""
    if not cmds.objExists(f"{mesh}.{CLUSTER_VERTICES_ATTR}"):
        wrong_vtx_str = str(selectAffectedVertices(mesh, tolerance=tolerance, select=False))
    else:
        wrong_vtx_str = cmds.getAttr(f"{mesh}.{CLUSTER_VERTICES_ATTR}")
    if not wrong_vtx_str:
        raise BaseException(f"unable to find {CLUSTER_VERTICES_ATTR} data in {mesh} was not inited")
    delta_volume_vtx = np.array(eval(delta_str))
    clusters_vertices = np.array(eval(wrong_vtx_str))

    cluster_joints_str = ""
    if cmds.objExists(f"{mesh}.{CLUSTER_JOINTS_ATTR}"):
        cluster_joints_str = cmds.getAttr(f"{mesh}.{CLUSTER_JOINTS_ATTR}")
    if cluster_joints_str:
        old_joints = [a for a in eval(cluster_joints_str) if cmds.objExists(a)]
        to_delete = set()
        for jnt in old_joints:
            prnt = cmds.listRelatives(jnt, p=1)
            if prnt and cmds.objExists(f"{prnt[0]}.{transforms.JOINT_METADATA}"):
                to_delete.add(prnt[0])
            else:
                to_delete.add(jnt)
        cmds.delete(to_delete)

    cluster_joints, cluster_weights = createClustersJoints(mesh,
                                                           joints,
                                                           clusters_vertices,
                                                           delta_volume_vtx,
                                                           cluster_n)
    setMetadataAttribute(mesh, CLUSTER_JOINTS_ATTR, cluster_joints, storable=True)
    data_list = cluster_weights.tolist()
    setMetadataAttribute(mesh, CLUSTER_WEIGHTS_ATTR, str(data_list), storable=True)
    return cluster_joints, cluster_weights


def setDefaultSkin(mesh):
    if not cmds.objExists(f"{mesh}.{CLUSTER_JOINTS_ATTR}"):
        raise BaseException(f"mesh {mesh} has not the cluster joints information, please recreate cluster joints")
    if not cmds.objExists(f"{mesh}.{CLUSTER_VERTICES_ATTR}"):
        raise BaseException(f"mesh {mesh} has not the cluster vertices information, please recreate cluster joints")
    if not cmds.objExists(f"{mesh}.{CLUSTER_WEIGHTS_ATTR}"):
        raise BaseException(f"mesh {mesh} has not the cluster weights information, please recreate cluster joints")
    wrong_vtx_str = cmds.getAttr(f"{mesh}.{CLUSTER_VERTICES_ATTR}")
    cluster_joints_str = cmds.getAttr(f"{mesh}.{CLUSTER_JOINTS_ATTR}")
    clusters_weights_str = cmds.getAttr(f"{mesh}.{CLUSTER_WEIGHTS_ATTR}")
    clusters_joints = [a for a in eval(cluster_joints_str) if cmds.objExists(a)]
    clusters_vertices = np.array(eval(wrong_vtx_str))
    clusters_weights = np.array(eval(clusters_weights_str))
    applyClustersDefaultSkin(mesh, clusters_joints, clusters_vertices, clusters_weights)
