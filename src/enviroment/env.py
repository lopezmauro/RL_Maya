import math
import numpy as np
from maya import cmds
from maya.api import OpenMaya as om
from ..maya_utils import mUtils, rewards
from ..math_utils import vector_math as vm
ACTIONS_MULTIPLIERS = [("tx", 1), ("ty", 1), ("tz", 1)]
# ACTIONS_MULTIPLIERS = [("ty", 1), ("tz", 1)]
ACTIONS_PENALTY = {"tx": 1, "ty": 1, "tz": 0}


class Enviroment():

    def __init__(self, agent, drivers, maxFrame=20, hasAnimation=False):
        self.agent = mUtils.MNode(agent)
        self.drivers = [mUtils.MNode(a) for a in drivers]
        self.restVector = om.MVector()
        self.action_space = len(ACTIONS_MULTIPLIERS)
        self.currentFrame = 0
        self.maxFrame = maxFrame
        self.agent_pos = None
        self.agent_mtx = None
        self.drivers_mtx = list()
        self.drivers_pos = list()
        self.hasAnimation = hasAnimation
        self.animations = [("attr", [0])]
        # self.mesh = mUtils.MNode(mesh)
        # self.mfn = self.mesh.getShape().getBestFn()
        # triangle_counts, triangle_vertices = self.mfn.getTriangles()
        # self.all_triangles = np.array(triangle_vertices).reshape(-1, 3)
        if not hasAnimation:
            self.animations = self.createAnimations()
            self.currAttr = self.animations[0][0]
            self.currAnim = self.animations[0][1]
        self.reInit(agent)

    def reInit(self, agent):
        self.agent = mUtils.MNode(agent)
        self.currentFrame = 0
        cmds.currentTime(self.currentFrame)
        self.setAction([0]*self.action_space)
        self.updateStatesCache(init=True)
        state = self.getState()
        self.observation_space = state.size
        curr_coll = rewards.getAgentCollisionValue(self.agent_pos, self.drivers_pos)
        self.startSide = math.copysign(1, curr_coll)
        # positions = np.array(self.mfn.getPoints(space=om.MSpace.kWorld))[:, :3]
        # self.vertices = skinCluster.getInfluencesVertices(self.mesh, [str(self.agent)], 0.05)
        # self.triangles = meshes.getVertextriangles(self.all_triangles, self.vertices)
        # self.bind_data = rewards.getTriangleBindData(self.drivers, self.triangles, positions)
        # self.bind_volume = rewards.getTrianglesVolume(positions, self.bind_data)
        return state

    def step(self, action, addFrame=True):
        if self.hasAnimation:
            cmds.currentTime(self.currentFrame)
        elif hasattr(self.drivers[1], self.currAttr):
            plug = getattr(self.drivers[1], self.currAttr)
            plug.setFloat(self.currAnim[self.currentFrame])
        else:
            raise ValueError("Unable to create animation")
        self.setAction(action)
        observation = self.getState()
        reward = self.getReward()
        done = False
        info = ""
        if addFrame:
            self.currentFrame += 1
        if self.currentFrame >= self.maxFrame:
            done = True
        return observation, reward, done, info

    def reset(self):
        self.currentFrame = 0
        if self.hasAnimation:
            cmds.currentTime(self.currentFrame)
        else:
            self.resetJointDriver()
        self.setAction([0]*self.action_space)
        return self.getState()

    def createAnimations(self):
        animations = [("ry", np.linspace(0, math.radians(120), self.maxFrame)),
                      ("ry", np.linspace(0, math.radians(-120), self.maxFrame)),
                      ("rz", np.linspace(0, math.radians(120), self.maxFrame)),
                      ("rz", np.linspace(0, math.radians(-120), self.maxFrame)),
                      ("rx", np.linspace(0, math.radians(180), self.maxFrame)),
                      ("rx", np.linspace(0, math.radians(-180), self.maxFrame))]
        return animations

    def resetJointDriver(self):
        for attr, values in self.animations:
            if not hasattr(self.drivers[1], attr):
                continue
            plug = getattr(self.drivers[1], attr)
            plug.setFloat(values[0])

    def setAction(self, action):
        for act, attr in zip(action, ACTIONS_MULTIPLIERS):
            plug = getattr(self.agent, attr[0])
            plug.set(float(attr[1]*act))

    def updateStatesCache(self, init=False):
        self.agent_pos = self.agent.getPosition()
        self.agent_mtx = self.agent.getMatrix()
        self.drivers_mtx = [a.getMatrix() for a in self.drivers]
        self.drivers_pos = [a.getPosition() for a in self.drivers]
        if init:
            self.closest_point, self.closest_seg = self.getCloserSegment()
            self.restVector = self.getAgentToSegmentVector()
        self.curr_vector = self.getAgentToSegmentVector()

    def getAgentToSegmentVector(self):
        parentMatrix = self.drivers_mtx[self.closest_seg[0]]
        closestPnt = vm.closestPointInLine(self.drivers_pos[self.closest_seg[0]],
                                           self.drivers_pos[self.closest_seg[1]],
                                           self.agent_pos)
        return parentMatrix*(om.MPoint(self.agent_pos)-om.MPoint(closestPnt))

    def getRBDState(self):
        observation = list()
        for drv_mtx in self.drivers_mtx:
            # get relative position
            localMat = self.agent_mtx*drv_mtx.inverse()
            localTrf = om.MTransformationMatrix(localMat)
            rbd_lTr = localTrf.translation(om.MSpace.kObject)
            observation.extend([rbd_lTr.x, rbd_lTr.y, rbd_lTr.z])
        """
        for driver in self.drivers:
            matrx = driver.attr('worldMatrix').get()
            localTrf = pmc.datatypes.TransformationMatrix(matrx)
            rbd_lOri = localTrf.rotation(asQuaternion=True)
            observation.extend(rbd_lOri)
        localMat = self.agent_mtx * self.drivers_mtx[1].inverse()
        localTrf = om.MTransformationMatrix(localMat)
        rbd_lTr = localTrf.translation(om.MSpace.kObject)
        observation.extend([rbd_lTr.x, rbd_lTr.y, rbd_lTr.z])
        """
        localTrf = om.MTransformationMatrix(self.drivers_mtx[1])
        rbd_lOri = localTrf.rotation(asQuaternion=True)
        observation.extend(rbd_lOri)
        return observation

    def getState(self):
        self.updateStatesCache()
        state = self.getRBDState()
        # state = list()
        # state.extend([self.curr_vector.x, self.curr_vector.y, self.curr_vector.z])
        state.extend([self.restVector.x*self.restVector.x,
                      self.restVector.y*self.restVector.y,
                      self.restVector.z*self.restVector.z])
        state.extend([self.restVector.x, self.restVector.y, self.restVector.z])
        # return np.array(state)
        featuresNorm, mean, std = vm.featNorm(state)
        return featuresNorm

    def getPoseRwd(self):
        rewards = list()
        # distance from oprimal volume preserv
        delta_dist = self.restVector.length()-self.curr_vector.length()
        # check that is in the same direction
        dot_p = 1-(self.curr_vector.normal()*self.restVector.normal())
        rewards = delta_dist+dot_p
        return np.exp(-3 * (rewards ** 2))
        # return np.exp(-3 * rewards)

    def getPoseRwdOld(self):
        positions = np.array(self.mfn.getPoints(space=om.MSpace.kWorld))[:, :3]
        pose_volume = rewards.getTrianglesVolume(positions, self.bind_data)
        vol_change = 1
        for i, bind_vol in self.bind_volume.items():
            delta = abs(bind_vol - pose_volume[i])
            vol_change -= delta
        return vol_change

    def getCollisionReward(self):
        rew = .1
        scale = 2
        curr_coll = rewards.getAgentCollisionValue(self.agent_pos, self.drivers_pos)
        curr_side = math.copysign(1, curr_coll)
        if curr_side != self.startSide:
            amount_collision = np.exp(abs(curr_coll))
            if amount_collision > scale:
                rew = scale - amount_collision
        return rew

    def getGasPenalty(self):
        penalty = 0
        values = list()
        multipliers = list()
        for attr, multipl in ACTIONS_PENALTY.items():
            plug = getattr(self.agent, attr)
            values.append(abs(plug.get()))
            multipliers.append(multipl)
        penalty = sum(vm.normalize(values)*multipliers)
        if penalty > 2:
            return -2
        elif penalty > .01:
            return penalty*-1
        return .0

    def getReward(self):
        rew = self.getPoseRwd() + self.getCollisionReward() + self.getGasPenalty()
        return rew

    def getCloserSegment(self):
        return vm.getCloserSegment(self.agent_pos, self.drivers_pos)

    def render(self):
        pass

    