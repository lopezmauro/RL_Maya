import math
import numpy as np
from maya import cmds
from maya.api import OpenMaya as om
from . import mUtils, skinCluster, meshes, attrUtils, transforms
from ..math_utils import vector_math as vm
from ..math_utils import fuzzyCMeans


def getTriangleBindData(joints, triangles, positions):
    joints_pos = [cmds.xform(a, q=1, ws=1, t=1) for a in joints]
    vertices = set(triangles.flatten())
    vertex_data = np.zeros((max(vertices)+1, 2), np.int16)
    for vertex in vertices:
        closest_point, closest_seg = vm.getCloserSegment(positions[vertex][:3], joints_pos)
        vertex_data[vertex] = closest_seg
    triangle_bind = dict()
    for i, tri in enumerate(triangles):
        start_seg_idnx = np.bincount(vertex_data[tri][:, 0]).argmax()
        end_seg_idnx = np.bincount(vertex_data[tri][:, 1]).argmax()
        tri_segment = [joints[start_seg_idnx], joints[end_seg_idnx]]
        triangle_bind[i] = {"vertices": list(tri), "segment": tri_segment}
    return triangle_bind


def getTrianglesVolume(positions, triangle_bind):
    np_positions = np.array(positions)[:, :3]
    result = dict()
    for i, data in triangle_bind.items():
        start = cmds.xform(data.get("segment")[0], q=1, ws=1, t=1)
        end = cmds.xform(data.get("segment")[1], q=1, ws=1, t=1)
        vol = vm.getTriangleVolume(start, end, np_positions[data.get("vertices")])
        result[i] = vol
    return result