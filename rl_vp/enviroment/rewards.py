import numpy as np
from maya import cmds
from maya.api import OpenMaya as om
from rl_vp.math_utils import vector_math as vm


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


def getAgentSegmentSide(agent_pos, drivers_pos):
    ba = drivers_pos[1]-drivers_pos[0]
    ca = drivers_pos[2]-drivers_pos[0]
    # get middle plane vector, project segment into total limb extension
    # and move that pojection into the correct position
    m = drivers_pos[0]+(ca.normal()*((ba * ca) / ca.length()))
    # check if the limb is straight, the prev projection is lenght 0
    if m.length() == 0:
        return .1
    # get the prjection betwen the midle plane and the agent position
    pb = agent_pos-drivers_pos[1]
    proj = (m.normal()*((pb * m) / m.length()))
    # check if the projection is on the same direction than the closer segment
    direction = ((proj-agent_pos).normal()*ba.normal())
    return direction


def getAgentCollisionValue(agent_pos, drivers_pos):
    np_drivers_pos = np.array(drivers_pos)[:, :3]
    np_agent_pos = om.MPoint(agent_pos)
    ba = np_drivers_pos[1] - np_drivers_pos[0]
    ca = np_drivers_pos[2] - np_drivers_pos[0]
    # get middle plane vector, project segment into total limb extension
    # and move that pojection into the correct position
    middle = np_drivers_pos[0] + vm.projectVector(ca, ba)
    if vm.magnitude(middle) == 0:  # drivers straight
        distToStart = vm.magnitude(np_drivers_pos[0] - list(np_agent_pos)[:3])
        distToEnd = vm.magnitude(np_drivers_pos[-1] - list(np_agent_pos)[:3])
        return distToEnd-distToStart
    # matrix that define the intersection plane
    x_axis = vm.normalize(np_drivers_pos[1]-middle)
    y_axis = vm.normalize(np_drivers_pos[-1]-middle)
    z_axis = np.cross(x_axis, y_axis)
    y_axis = np.cross(z_axis, x_axis)
    matrx = list()
    for ax in [x_axis, y_axis, z_axis]:
        matrx.extend(ax)
        matrx.append(0)
    matrx.extend(middle)
    matrx.append(1)
    # distance from the agent to the plane
    mmtx = om.MMatrix(matrx)
    realtive_pos = np_agent_pos*mmtx.inverse()
    return realtive_pos[1]
