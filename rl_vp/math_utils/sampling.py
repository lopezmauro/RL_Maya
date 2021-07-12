import numpy as np
import math
import random
import itertools
# https://www.cs.ubc.ca/~rbridson/docs/bridson-siggraph07-poissondisk.pdf

def getDistanceSquare(a, b):
    return np.sum(abs(a - b)**2)


def isValid(candidate, radius, minBBox, maxBBox, grid, samples):
    # check if the candidate is outside of border
    
    if (candidate < minBBox).any() or (candidate > maxBBox).any():
        return False

    cellSize = radius/math.sqrt(2)
    cell = ((candidate-minBBox)/cellSize).astype(int)
    minCell = cell - 2
    maxCell = cell + 2
    searchEnd = np.array(grid.shape)
    searchStart = np.zeros_like(searchEnd)
    minIndices = np.where(minCell > 0)
    searchStart[minIndices] = minCell[minIndices]
    maxIndices = np.where(np.less_equal(maxCell, searchEnd))
    searchEnd[maxIndices] = maxCell[maxIndices]

    indices = [range(a, b) for a, b in zip(searchStart, searchEnd)]
    for index in itertools.product(*indices):
        pointIndex = grid[index]-1

        if pointIndex != -1:
            distanceSquare = getDistanceSquare(samples[int(pointIndex)], candidate)

            if distanceSquare < radius**2:
                return False

    return True


def poissionDiscSampling(radius=30, bundingBox=((0, 0, 0), (1000, 1000, 1000)), sampleRetrial=1, maxTrials=100000):
    
    samples = []
    activeList = []
    cellSize = radius/math.sqrt(2)
    bboxMin = np.array(bundingBox[0])
    bboxMax = np.array(bundingBox[1])
    sampleRegionSize = bboxMax-bboxMin
    grid = np.zeros(np.ceil(sampleRegionSize/cellSize).astype(int))
    trial = 0
    center = (sampleRegionSize*.5)+bboxMin
    activeList = [center]

    while len(activeList) > 0:
        activeIndex = random.randrange(len(activeList))
        spawnCentre = activeList[activeIndex]
        candidateAccepted = False

        for _ in range(sampleRetrial):
            angle = random.random() * math.pi * 2
            direction = np.array([math.sin(angle), math.cos(angle), math.tan(angle)])
            candidate = spawnCentre + direction * random.uniform(radius, 2*radius)
            trial += 1

            if isValid(candidate, radius, bboxMin, bboxMax, grid, samples):
                samples.append(candidate)
                coord = (candidate-bboxMin)/cellSize
                grid[int(coord[0])][int(coord[1])][int(coord[2])] = len(samples)
                candidateAccepted = True
                activeList.append(candidate)
                break

        if candidateAccepted is False:
            activeList.pop(activeIndex)
        if trial > maxTrials:
            print("Limit trials reached!!")
            break
    return samples
