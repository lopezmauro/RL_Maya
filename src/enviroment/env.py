import math
import numpy as np
from maya import cmds
from maya.api import OpenMaya as om
from ..maya_utils import mUtils, rewards, skinCluster, meshes
from ..math_utils import vector_math as vm


class Enviroment():

    def __init__(self, agent, drivers, mesh, maxFrame=20, hasAnimation=False):
        self.agent = mUtils.MNode(agent)
        self.drivers = [mUtils.MNode(a) for a in drivers]
        self.restVector = om.MVector()
        self.action_space = 4
        self.currentFrame = 0
        self.maxFrame = maxFrame
        self.agent_pos = None
        self.agent_mtx = None
        self.drivers_mtx = list()
        self.drivers_pos = list()
        self.hasAnimation = hasAnimation
        self.animations = [("attr", [0])]
        self.mesh = mUtils.MNode(mesh)
        self.mfn = self.mesh.getShape().getBestFn()
        triangle_counts, triangle_vertices = self.mfn.getTriangles()
        self.all_triangles = np.array(triangle_vertices).reshape(-1, 3)
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
        positions = np.array(self.mfn.getPoints(space=om.MSpace.kWorld))[:, :3]
        self.vertices = skinCluster.getInfluencesVertices(self.mesh, [str(self.agent)], 0.05)
        self.triangles = meshes.getVertextriangles(self.all_triangles, self.vertices)
        self.bind_data = rewards.getTriangleBindData(self.drivers, self.triangles, positions)
        self.bind_volume = rewards.getTrianglesVolume(positions, self.bind_data)
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
        self.agent.tx.setFloat(float(action[0]))
        self.agent.ty.setFloat(float(action[1]))
        self.agent.tz.setFloat(float(action[2]))
        self.agent.rx.setFloat(90.0 * float(action[3]))

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
        #state.extend([self.curr_vector.x, self.curr_vector.y, self.curr_vector.z])
        state.extend([self.restVector.x*self.restVector.x,
                      self.restVector.y*self.restVector.y,
                      self.restVector.z*self.restVector.z])
        state.extend([self.restVector.x, self.restVector.y, self.restVector.z])
        # return np.array(state)
        featuresNorm, mean, std = vm.featNorm(state)
        return featuresNorm

    def getPoseRwdOld(self):
        rewards = list()
        # distance from oprimal volume preserv
        delta_dist = self.restVector.length()-self.curr_vector.length()
        # check that is in the same direction
        dot_p = 1-(self.curr_vector.normal()*self.restVector.normal())
        rewards = delta_dist+dot_p
        return np.exp(-3 * (rewards ** 2))
        # return np.exp(-3 * rewards)

    def getPoseRwd(self):
        positions = np.array(self.mfn.getPoints(space=om.MSpace.kWorld))[:, :3]
        pose_volume = rewards.getTrianglesVolume(positions, self.bind_data)
        vol_change = 1
        for i, bind_vol in self.bind_volume.items():
            delta = abs(bind_vol - pose_volume[i])
            vol_change -= delta
        return vol_change

    def getCollisionReward(self):
        rew = .1
        ba = self.drivers_pos[1]-self.drivers_pos[0]
        ca = self.drivers_pos[2]-self.drivers_pos[0]
        # get middle plane vector, project segment into total limb extension
        # and move that pojection into the correct position
        m = self.drivers_pos[0]+(ca.normal()*((ba * ca) / ca.length()))
        # check if the limb is straight, the prev projection is lenght 0
        if m.length() == 0:
            return rew
        # get the prjection betwen the midle plane and the agent position
        pb = self.agent_pos-self.drivers_pos[1]
        proj = (m.normal()*((pb * m) / m.length()))
        # check if the projection is on the same direction than the closer segment
        multiply = 1
        if self.closest_seg[1] == 2:
            # if the start position is in the upper side reverse the direction
            multiply = -1
        direction = ((proj-self.agent_pos).normal()*ba.normal())*multiply
        if direction < 0:
            # rew = -1*(1-np.exp(-20 * ((proj-p).length())))
            amount_collision = (proj-self.agent_pos).length()
            if amount_collision > 2:
                rew = -4
            else:
                rew = -2**amount_collision
        return rew

    def getGasPenalty(self):
        translY = abs(self.agent.ty.asFloat())
        if translY > 1:
            return -2
        elif translY > .01:
            return 1-(2**translY)
            # return 1-np.exp(translX**2)
        return .0

    def getReward(self):
        rew = self.getPoseRwd() + self.getCollisionReward()  # + self.getGasPenalty()
        return rew

    def getCloserSegment(self):
        return vm.getCloserSegment(self.agent_pos, self.drivers_pos)

    def render(self):
        pass

    