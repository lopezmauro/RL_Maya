import logging
import copy
from . import meshes
from .mUtils import mNode
from maya import cmds
_logger = logging.getLogger(__name__)


def getDeformersFromMesh(sourceMesh, nodeType="skinCluster"):
    history = cmds.listHistory(str(sourceMesh), pruneDagObjects=True) or list()
    deformers = [a for a in history if 'geometryFilter' in cmds.nodeType(a, i=1)][::-1]
    if not deformers:
        _logger.debug("unable to find any deformers in {}".format(str(sourceMesh)))
        return []
    if not nodeType:
        return [mNode.MNode(a) for a in deformers]
    nodes = list(mNode.MNode(a) for a in deformers if cmds.nodeType(a) == nodeType)
    if not nodes:
        _logger.debug("unable to find a {} in {}".format(nodeType, str(sourceMesh)))
    return nodes


def getSkinInfluences(skinCluster, asMNode=False):
    """get the skin influences 
    Args:
        skinCluster (str,MNode): source skin
    Returns:
        dict: influenceIndex: influence partial name
    """    
    skinClusterNode = mNode.MNode(skinCluster)
    matrixPlug = skinClusterNode.matrix
    influences = dict()
    for indx in matrixPlug.getExistingArrayAttributeIndices():
        currPreBindlug = matrixPlug[indx]
        jntPlug = currPreBindlug.source()
        if jntPlug.isNull:
            continue
        influence = mNode.MNode(jntPlug.node())
        if asMNode:
            influences[indx] = influence
        else:
            influences[indx] = str(influence)
    return influences


def getSkinWeightsMapping(skinCluster, normalize=True, tol=0.0001):
    """get each vertex influence and weight

    Args:
        skinCluster (str- omx.XNode): source skincluster
        searchForJoints (bool, optional): if the influence is not a joint
        keep searching upsream connection until get a joint. Usesfull with localized skincluster. Defaults to True.

    Returns:
        list(dict): for each vertex, a dict {influenceName: weight}
    """    
    skin = mNode.MNode(skinCluster)
    influences = getSkinInfluences(skin)
    weightMap = dict()
    for i in range(skin.weightList.numElements()):  # for each vertex
        plugWeights = skin.weightList[i].weights  # access the actual weight attr
        vertDict = {}
        for j in plugWeights.getExistingArrayAttributeIndices():  # for each joint
            weightFloat = plugWeights[j].get()
            if weightFloat < tol:
                continue
            inf = influences.get(j)
            if not inf:
                continue
            vertDict[str(inf)] = weightFloat
        allWeigts = sum(vertDict.values())
        if allWeigts != 1 and normalize:
            # normalize data
            for k, v in vertDict.items():
                vertDict[k] = float(v)/allWeigts
        weightMap[i] = vertDict
    return weightMap


def normalizeWeights(weightMaping):
    result = dict()
    for indx, values in weightMaping.items():
        weight_sum = sum(values.values())
        if weight_sum == 1:
            result[indx] = values
            continue
        # normalize data
        norm_values = dict()
        for k, v in values.items():
            norm_values[k] = float(v)/weight_sum
        result[indx] = norm_values
    return result


def setSkinWeights(skinCluster, weightMaping):
    tagetSkin = mNode.MNode(skinCluster)
    targInfluences = getSkinInfluences(tagetSkin)
    for i, vtxW in weightMaping.items():  # for each vertex
        plugWeights = tagetSkin.weightList[i].weights
        # First reset values to zero:
        nb_weights = plugWeights.numElements()
        for j in range(nb_weights):  # for each joint
            plugWeights[j].set(0)
        # set weights
        influences = list()
        weights = list()
        for infIdx, inflName in targInfluences.items():
            inflWeigt = 0
            if inflName in vtxW:
                inflWeigt = vtxW[inflName]
            influences.append(infIdx)
            weights.append(inflWeigt)
        for jntIdx, value in zip(influences, weights):
            plugWeights[jntIdx].set(value)


def relaxWeightMapping(weightMap, faceVertices, vertexFaces,
                       vertexList=None, relaxStreght=1.0,
                       iterations=1, influencesToSkip=[]):
    relaxedWeights = copy.deepcopy(weightMap) 
    if not vertexList:
        vertexList = weightMap.keys()
    connectedVertices = meshes.getConnectedVertices(vertexList, vertexFaces, faceVertices)
    for x in range(iterations):
        for currIdx, connected in connectedVertices.items():
            newWeigts = dict()
            # add missing influences
            missingInfl = set()
            for con in connected:
                for inf in relaxedWeights[con].keys():
                    if inf in relaxedWeights[currIdx].keys():
                        continue
                    missingInfl.add(inf)
            for inf in missingInfl:
                relaxedWeights[currIdx][inf] = 0.0
            for inf, weight in relaxedWeights[currIdx].items():
                # get all neighbor vertices that share the same influence
                if inf in influencesToSkip:
                    continue
                newW = 0.0
                for con in connected:
                    connW = relaxedWeights[con]
                    if inf not in connW:
                        continue
                    newW += connW[inf]
                # average those weights
                newW /= len(connected)
                newWeigts[inf] = weight+(newW-weight)*relaxStreght
            allWeigts = sum(newWeigts.values())
            if allWeigts != 1:
                # normalize data
                for k, v in newWeigts.items():
                    newWeigts[k] = float(v) / allWeigts
            relaxedWeights[currIdx] = newWeigts
    return relaxedWeights


def getInfluencesVertices(mesh, influences, tolerance=.01):
    skinClusters = getDeformersFromMesh(mesh)
    weightMap = getSkinWeightsMapping(skinClusters[0], normalize=False, tol=tolerance)
    vertices = list()
    for k, v in weightMap.items():
        for a in v.keys():
            if a in influences:
                vertices.append(k)
                break
    return list(set(vertices))
