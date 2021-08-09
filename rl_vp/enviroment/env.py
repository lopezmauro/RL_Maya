import math
import random
import logging
import numpy as np
from maya import cmds
from maya.api import OpenMaya as om
from rl_vp.maya_utils import mUtils
from rl_vp.math_utils import vector_math as vm
from rl_vp.enviroment import observation as obs_utils
from rl_vp.enviroment import rewards as rew_utils
from rl_vp.enviroment import constants

logger = logging.getLogger(__name__)
DRIVER_ATTR = {"rx":(-120, 120), "ry":(-40, 120), "rz":(-120, 120)}

class Enviroment():

    def __init__(self, agent, drivers, maxFrame=20, hasAnimation=False, driver_attrs=DRIVER_ATTR):
        self.agent = mUtils.MNode(agent)
        self.drivers = [mUtils.MNode(a) for a in drivers]
        self.maxFrame = maxFrame
        self.action_space = len(constants.ACTIONS_MULTIPLIERS)
        self.drivers_attrs = driver_attrs
        self.hasAnimation = hasAnimation
        # self.mesh = mUtils.MNode(mesh)
        # self.mfn = self.mesh.getShape().getBestFn()
        # triangle_counts, triangle_vertices = self.mfn.getTriangles()
        # self.all_triangles = np.array(triangle_vertices).reshape(-1, 3)
        self.reInit(agent)

    def reInit(self, agent):
        self.agent = mUtils.MNode(agent)
        self.currentFrame = 0
        self.animations = self.createAnimations()
        self.agent_pos = None
        self.agent_mtx = None
        self.rest_distance = 1.0
        self.restVector = om.MVector()
        self.drivers_mtx = list()
        self.drivers_pos = list()
        self.closest_seg = list()
        self.reset(getState=False)
        self.updateStatesCache()
        _, self.closest_seg = self.getCloserSegment()
        self.restVector = self.getAgentToSegmentVector()
        self.rest_distance = vm.magnitude(self.restVector)
        curr_coll = rew_utils.getAgentCollisionValue(self.agent_pos, self.drivers_pos)
        self.startSide = math.copysign(1, curr_coll)
        state = self.getState()
        self.observation_space = state.size
        # positions = np.array(self.mfn.getPoints(space=om.MSpace.kWorld))[:, :3]
        # self.vertices = skinCluster.getInfluencesVertices(self.mesh, [str(self.agent)], 0.05)
        # self.triangles = meshes.getVertextriangles(self.all_triangles, self.vertices)
        # self.bind_data = rew_utils.getTriangleBindData(self.drivers, self.triangles, positions)
        # self.bind_volume = rew_utils.getTrianglesVolume(positions, self.bind_data)
        return

    def step(self, action, addFrame=True):
        if self.hasAnimation:
            cmds.currentTime(self.currentFrame)
        for attr, value in self.animations[self.currentFrame]:
            if hasattr(self.drivers[1], attr):
                plug = getattr(self.drivers[1], attr)
                plug.setFloat(value)
            else:
                raise ValueError("Unable to create animation")
        self.setAction(action)
        observation = self.getState()
        reward = self.getReward()
        logger.debug(f"Action {action} Reward {reward}")
        done = False
        info = ""
        if addFrame:
            self.currentFrame += 1
        if self.currentFrame >= self.maxFrame:
            done = True
        return observation, reward, done, info

    def reset(self, getState=True):
        self.currentFrame = 0
        if self.hasAnimation:
            cmds.currentTime(self.currentFrame)
        else:
            self.resetJointDriver()
        self.setAction([0]*self.action_space)
        if not getState:
            return
        return self.getState()

    def createAnimations(self):
        animations = dict()
        all_animations = dict()

        for attr, limits in self.drivers_attrs.items():
            max_value =  math.radians(random.randrange(limits[0], limits[1]))
            all_animations[attr] = np.linspace(0, max_value, self.maxFrame)
        for x in range(self.maxFrame):
            animations[x] = list()
            for attr in self.drivers_attrs.keys():
                animations[x].append((attr, all_animations[attr][x]))
        """
        for frame in range(self.maxFrame):
            frame_values = list()
            for attr, limits in self.drivers_attrs.items():
                value =  math.radians(random.randrange(limits[0], limits[1]))
                frame_values.append((attr, value))
            animations[frame] = frame_values
        """
        return animations

    def resetJointDriver(self):
        for attr in self.drivers_attrs.keys():
            if not hasattr(self.drivers[1], attr):
                continue
            plug = getattr(self.drivers[1], attr)
            plug.setFloat(0)

    def setAction(self, action):
        for act, attr in zip(action, constants.ACTIONS_MULTIPLIERS):
            plug = getattr(self.agent, attr[0])
            plug.set(float(attr[1]*act*self.rest_distance))
            # plug.set(float(attr[1]*act))

    def updateStatesCache(self):
        self.agent_pos = self.agent.getPosition()
        self.agent_mtx = self.agent.getMatrix()
        self.drivers_mtx = [a.getMatrix() for a in self.drivers]
        self.drivers_pos = [a.getPosition() for a in self.drivers]
        if self.closest_seg:
            self.curr_vector = self.getAgentToSegmentVector()

    def getAgentToSegmentVector(self):
        closestPnt = vm.closestPointInLine(self.drivers_pos[self.closest_seg[0]],
                                           self.drivers_pos[self.closest_seg[1]],
                                           self.agent_pos)
        return om.MPoint(self.agent_pos)-om.MPoint(closestPnt)
        
    def getState(self):
        self.updateStatesCache()
        return obs_utils.getObservation(self.drivers_mtx, self.agent_mtx, self.restVector)
        

    def getPoseRwd(self):
        rewards = list()
        # distance from oprimal volume preserv
        delta_dist = self.restVector.length()-self.curr_vector.length()
        # check that is in the same direction
        dot_p = 1-(self.curr_vector.normal()*self.restVector.normal())
        rewards = delta_dist+dot_p
        return np.exp(-3 * (rewards ** 2))
        # return np.exp(-3 * rewards)

    def getCollisionReward(self):
        rew = .1
        curr_coll = rew_utils.getAgentCollisionValue(self.agent_pos, self.drivers_pos)
        curr_side = math.copysign(1, curr_coll)
        if curr_side != self.startSide:
            rew = 1-np.exp(abs(curr_coll)/(self.rest_distance/4.0))
        return rew

    def getGasPenalty(self):
        penalty = 0
        values = list()
        for attr, multipl in constants.ACTIONS_PENALTY.items():
            plug = getattr(self.agent, attr)
            curr_val=abs(plug.get())
            if curr_val > 0.0:
                values.append((curr_val/self.rest_distance)*multipl)
        penalty = sum(values)
        if penalty > 2:
            return -2
        elif penalty > .01:
            return penalty*-1
        return .0

    def getReward(self):
        pose_rew = self.getPoseRwd()
        coll_rew = self.getCollisionReward()
        gas_rew = self.getGasPenalty()
        logger.debug(f"Pose Rew {pose_rew} Collision Rew {coll_rew} Gas Penalty {gas_rew}")
        rew = pose_rew + coll_rew + gas_rew
        return rew

    def getCloserSegment(self):
        return vm.getCloserSegment(self.agent_pos, self.drivers_pos)

    def render(self):
        pass

    