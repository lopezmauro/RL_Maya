import numpy as np
import math
from maya import cmds

def getAnimationValues(frame_num):
    animations = [("ry", np.linspace(0, math.radians(120), frame_num)),
                  ("ry", np.linspace(0, math.radians(-120), frame_num)),
                  ("rz", np.linspace(0, math.radians(120), frame_num)),
                  ("rz", np.linspace(0, math.radians(-120), frame_num)),
                  ("rx", np.linspace(0, math.radians(180), frame_num)),
                  ("rx", np.linspace(0, math.radians(-180), frame_num))]
    return animations

def getConnectionPairs(node, source=True, destination=True):
    connectionPairs = list()
    if source:
        conns = cmds.listConnections(node, plugs=True, connections=True, destination=False)
        if conns:
            connectionPairs.extend(zip(conns[1::2], conns[::2]))
    
    if destination:
        conns = cmds.listConnections(node, plugs=True, connections=True, source=False)
        if conns:
            connectionPairs.extend(zip(conns[::2], conns[1::2]))
    return connectionPairs


class DisconnectCtx():
    def __init__(self, nodes, source=True, destination=True):
        self.connectionPairs = list()
        for node in nodes:
            self.connectionPairs.extend(getConnectionPairs(node, source, destination))
          
    def __enter__(self):
        for srcAttr, destAttr in self.connectionPairs:
            cmds.disconnectAttr(srcAttr, destAttr)
      
    def __exit__(self, exc_type, exc_value, exc_traceback):
        for srcAttr, destAttr in self.connectionPairs:
            cmds.connectAttr(srcAttr, destAttr)
