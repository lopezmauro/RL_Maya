import numpy as np
from PIL import Image
import cv2
import os

#dir_path = os.path.dirname(os.path.realpath(__file__))
#jointsPos3d = np.load(os.path.join(dir_path, 'joints.npy')) 
#agentsPos3d = np.load(os.path.join(dir_path, 'agent.npy') )

# the dict!
class Renderer():
    def __init__(self, size=30, spf=100, bboxMin=[-6.0,-6.0], bboxMax=[6.0,6.0]):
        self.SIZE = size
        self.SPF = spf
        self.bboxMin = np.array(bboxMin)
        self.bboxMax = np.array(bboxMax)
        self.colors = {"joint": (255, 175, 0),
                       "agent": (0, 255, 0)}

        self.pix_size = ((self.bboxMax-self.bboxMin)/self.SIZE)

    def getPixel(self, pos):
        return tuple(((pos-self.bboxMin)/self.pix_size).astype(np.int))

    def drawFrame(self, jointsPos3d, agentPos3d):
        jointsPos2d = np.delete(jointsPos3d, obj=0, axis=1)
        agentsPos2d = np.delete(agentPos3d, obj=0, axis=1)
        env = np.zeros((self.SIZE, self.SIZE, 3), dtype=np.uint8)  # starts an rbg of our size
        for pos in jointsPos2d:
            pixel = self.getPixel(pos)
            env[pixel[0]][pixel[1]] = self.colors['joint']

        for pos in agentsPos2d:
            pixel = self.getPixel(pos)
            env[pixel[0]][pixel[1]] = self.colors['agent']

        img = Image.fromarray(env, 'RGB')  # reading to rgb. Apparently. Even tho color definitions are bgr. ???
        img = img.resize((300, 300))  # resizing so we can see our agent in all its glory.
        cv2.imshow("image", np.array(img))  # show it!
        key = cv2.waitKey(self.SPF)#pauses for 3 seconds before fetching next image
        if key == 27:#if ESC is pressed, exit loop
                cv2.destroyAllWindows()


