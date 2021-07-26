from maya.api import OpenMaya as om
from rl_vp.math_utils import vector_math as vm


def getObservation(drivers_mtx, agent_mtx, restVector):
    """standalone get observation since is ti will be usead for the enviroment
    and the node

    Args:
        drivers_mtx ([type]): [description]
        agent_mtx ([type]): [description]
        restVector ([type]): [description]

    Returns:
        [type]: [description]
    """
    observation = list()
    for drv_mtx in drivers_mtx:
        # get relative position
        localMat = agent_mtx*drv_mtx.inverse()
        localTrf = om.MTransformationMatrix(localMat)
        rbd_lTr = localTrf.translation(om.MSpace.kObject)
        observation.extend([rbd_lTr.x, rbd_lTr.y, rbd_lTr.z])
    localTrf = om.MTransformationMatrix(drivers_mtx[1])
    rbd_lOri = localTrf.rotation(asQuaternion=True)
    observation.extend(rbd_lOri)
    # state = list()
    # observation.extend([curr_vector.x, curr_vector.y, curr_vector.z])
    observation.extend([restVector.x*restVector.x,
                        restVector.y*restVector.y,
                        restVector.z*restVector.z])
    observation.extend([restVector.x, restVector.y, restVector.z])
    # return np.array(state)
    featuresNorm, _, _ = vm.featNorm(observation)
    return featuresNorm

def getRestVector(drivers_mtx, agent_mtx):
    drivers_pos = [list(a)[12:15] for a in drivers_mtx]
    agent_pos = list(agent_mtx)[12:15]
    closestPnt, _ = vm.getCloserSegment(agent_pos, drivers_pos)
    return om.MVector(om.MPoint(agent_pos)-om.MPoint(closestPnt))