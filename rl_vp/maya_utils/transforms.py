from maya import cmds
import numpy as np
from rl_vp.math_utils import vector_math as vm
JOINT_METADATA = "isJoint"


def createAimedJoint(pos, joints):
    cmds.select(d=True)
    segments_pos = [np.array(cmds.xform(a, q=1, ws=1, t=1)) for a in joints]
    closes_pnt, closerSeg = vm.getCloserSegment(pos, segments_pos)
    zAxis = vm.normalize(pos - closes_pnt)
    yAxis = vm.normalize(segments_pos[closerSeg[1]] - segments_pos[closerSeg[0]])
    xAxis = vm.normalize(np.cross(yAxis, zAxis))
    yAxis = vm.normalize(np.cross(zAxis, xAxis))
    jnt = cmds.joint()
    grp = cmds.group(jnt)
    cmds.addAttr(grp, longName=JOINT_METADATA, at="bool", keyable=False)
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
    return jnt