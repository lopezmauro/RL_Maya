import sys
from rl_vp.enviroment import observation, constants
from tensorflow.keras.models import load_model
from maya.api import OpenMaya as om
import os


def maya_useNewAPI():
    """
    The presence of this function tells Maya that the plugin produces, and
    expects to be passed, objects created using the Maya Python API 2.0.
    """
    pass


nodeName = 'rlvp'
nodeID = om.MTypeId(0x60011)


class RLVPNode(om.MPxNode):
    '''A template Maya Python DG Node.'''

    def __init__(self):
        om.MPxNode.__init__(self)
        self.filePath = ''
        self. model = None

    def getOrLoad(self, filePath):
        '''A interface for loading and caching Keras models'''
        if filePath == self.filePath:
            return False
        if not os.path.exists(filePath):
            return f"Unable to find file {filePath}"
        self.filePath = filePath
        try:
            self.model = load_model(filePath)
        except Exception as e:
            return str(e)
        return False

    def compute(self, plug, datablock):
        if plug not in RLVPNode.outputs:
            return None
        filePath_h = datablock.inputValue(RLVPNode.filePath)
        error = self.getOrLoad(filePath_h.asString())
        if error:
            om.MGlobal.displayError(error)
            return None
        if self.model is None:
            return None
        
        start_mtx = datablock.inputValue(RLVPNode.start).asMatrix()
        mid_mtx = datablock.inputValue(RLVPNode.mid).asMatrix()
        end_mtx = datablock.inputValue(RLVPNode.end).asMatrix()
        parent_mtx = datablock.inputValue(RLVPNode.parent_matrix).asMatrix()
        initialized = datablock.inputValue(RLVPNode.initialized).asShort()
        rest_vect = None
        if not initialized:
            rest_vect = observation.getRestVector([start_mtx, mid_mtx, end_mtx], parent_mtx)
            rest_h = datablock.outputValue(RLVPNode.rest_vector)
            rest_h.set3Double(rest_vect.x, rest_vect.y, rest_vect.z)
            init_h = datablock.outputValue(RLVPNode.initialized)
            init_h.setShort(1)
        else:
            rest_vect = datablock.inputValue(RLVPNode.rest_vector).asVector()
        print(rest_vect)
        obs = observation.getObservation([start_mtx, mid_mtx, end_mtx], parent_mtx, rest_vect)
        action = self.model.predict(obs.reshape(1, obs.size))
        for act, attr, mult in zip(action[0], RLVPNode.outputs, RLVPNode.outputs_mult):
            attr_h = datablock.outputValue(attr)
            print(float(act*mult*rest_vect.length()))
            attr_h.setDouble(float(act*mult*rest_vect.length()))
            datablock.setClean(attr)
        # The plug was successfully computed
        return self


def creator():
    return RLVPNode()


def initialize():
    # (1) Setup attributes
    nAttr = om.MFnNumericAttribute()
    tAttr = om.MFnTypedAttribute()
    mAttr = om.MFnMatrixAttribute()
    kDouble = om.MFnNumericData.kDouble  # Maya's float type
    kString = om.MFnData.kString
    kBool = om.MFnNumericData.kBoolean

    RLVPNode.filePath = tAttr.create('filePath', 'fp', kString)
    tAttr.usedAsFilename = True

    RLVPNode.start = mAttr.create( "start", "start")
    mAttr.hidden = False
    mAttr.keyable = False
    RLVPNode.mid = mAttr.create( "mid", "mid")
    mAttr.hidden = False
    mAttr.keyable = False
    RLVPNode.end = mAttr.create( "end", "end")
    mAttr.hidden = False
    mAttr.keyable = False

    RLVPNode.parent_matrix = mAttr.create( "parentMatrix", "parentMatrix")
    mAttr.hidden = False
    mAttr.keyable = False

    restX = nAttr.create( "restX", "restX", kDouble )
    restY = nAttr.create( "restY", "restY", kDouble )
    restZ = nAttr.create( "restZ", "restZ", kDouble )
    RLVPNode.rest_vector = nAttr.create( "restVector", "restVector", restX, restY, restZ )
    nAttr.hidden = False
    nAttr.keyable = False

    RLVPNode.initialized = nAttr.create("initialized", "initialized", kBool, 0)
    tAttr.hidden = False
    tAttr.keyable = False


    RLVPNode.outputs = list()
    RLVPNode.outputs_mult = list()
    for attr, mult in constants.ACTIONS_MULTIPLIERS:
        # (2) Setup the output attributes
        RLVPNode.outputs.append(nAttr.create(attr, attr, kDouble))
        RLVPNode.outputs_mult.append(mult)
        nAttr.writable = False
        nAttr.storable = False
        nAttr.readable = True

    # (3) Add the attributes to the node
    RLVPNode.addAttribute(RLVPNode.filePath)
    RLVPNode.addAttribute(RLVPNode.start)
    RLVPNode.addAttribute(RLVPNode.mid)
    RLVPNode.addAttribute(RLVPNode.end)
    RLVPNode.addAttribute(RLVPNode.parent_matrix)
    RLVPNode.addAttribute(RLVPNode.rest_vector)
    RLVPNode.addAttribute(RLVPNode.initialized)
    for attr in RLVPNode.outputs:
        RLVPNode.addAttribute(attr)
        RLVPNode.attributeAffects(RLVPNode.start, attr)
        RLVPNode.attributeAffects(RLVPNode.mid, attr)
        RLVPNode.attributeAffects(RLVPNode.end, attr)
        RLVPNode.attributeAffects(RLVPNode.filePath, attr)


def initializePlugin(obj):
    plugin = om.MFnPlugin(obj, "Autodesk", "1.0", "Any")

    try:
        plugin.registerNode(nodeName, nodeID, creator, initialize)
    except Exception:
        sys.stderr.write("Failed to register node\n")
        raise


def uninitializePlugin(obj):
    plugin = om.MFnPlugin(obj)

    try:
        plugin.deregisterNode(nodeID)
    except Exception:
        sys.stderr.write("Failed to deregister node\n")
        pass
