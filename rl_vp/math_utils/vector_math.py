import numpy as np


def magnitude(x):
    return np.linalg.norm(x)


def normalize(x):
    return np.array(x) / magnitude(x)


def pointToLineDistance(start, end, point):
    """ segment line AB, point P, where each one is an array([x, y]) """
    A = np.array(start)
    B = np.array(end)
    P = np.array(point)
    if all(A == P) or all(B == P):
        return 0
    if all(A == B):
        return magnitude(P-A)
    if np.arccos(np.dot((P - A) / magnitude(P - A), (B - A) / magnitude(B - A))) > np.pi / 2:
        return magnitude(P - A)
    if np.arccos(np.dot((P - B) / magnitude(P - B), (A - B) / magnitude(A - B))) > np.pi / 2:
        return magnitude(P - B)
    return magnitude(np.cross(A-B, A-P))/magnitude(B-A)


def closestPointInLine(start, end, point):
    a = np.array(start)
    b = np.array(end)
    p = np.array(point)
    n = normalize(b - a)
    ap = p - a
    t = ap @ n
    if t >= magnitude(b - a):
        return b
    x = a + t * n  # x is a point on line
    return x


def getCloserSegment(pos, segments_pos):
    closest_dist = None
    closest_seg = None
    closest_point = None
    for i, seg_pos in enumerate(segments_pos[:-1]):
        curr_dist = pointToLineDistance(seg_pos, segments_pos[i+1], pos)
        if not closest_dist or curr_dist < closest_dist:
            closest_dist = curr_dist
            closest_seg = (i, i+1)
            closest_point = closestPointInLine(seg_pos, segments_pos[i+1], pos)
    return closest_point, closest_seg

def featNorm(features):
    """Normalize features by mean and standard deviation.
    in order to be able to dernomalize them afterwards
    Args:
        features (np.array): un normlized np.array

    Returns:
        tuple: (normalizedFeatures, mean, standardDeviation)
    """

    mean = np.mean(features, axis=0)
    std = np.std(features - mean, axis=0)
    featuresNorm = (features - mean) / (std + np.finfo(np.double).eps)
    return (featuresNorm, mean, std)


def featDenorm(featuresNorm, mean, std):
    """Denormalize features by mean and standard deviation

    Args:
        features_norm (np.array): normlized np.array
        mean (float):  average of the array elements
        std (np.array): standard deviation, a measure of the spread of the array elements

    Returns:
        np.array: un normlized np.array
    """
    features = (featuresNorm * std) + mean
    return features


def easeInOutCubic(currentTime, start, end, totalTime):
    """
    Args:
        currentTime (float): is the current time (or position) of the tween.
        start (float): is the beginning value of the property.
        end (float): is the change between the beginning and 
            destination value of the property.
        totalTime (float): is the total time of the tween.
    Returns:
        float: normalized interpoltion value
    """
    currentTime /= totalTime/2
    if currentTime < 1:
        return end/2*currentTime*currentTime*currentTime + start
    currentTime -= 2
    return end/2*(currentTime*currentTime*currentTime + 2) + start


def getTriangleVolume(start, end, triangle_position):
    """project the triangle over the plane formed between the segment and the triangle normal
    and return the volume of that prism
    Args:
        start (np.array): line segment start
        end (np.array): line segment end
        triangle_position (list): list of np.array defining a triangle
    Returns:
        float: projected triangle volume
    """
    points = [np.array(a) for a in triangle_position]
    x_axis = np.array(start) - np.array(end)
    x_axis = x_axis / np.linalg.norm(x_axis)
    cross_vect = np.cross((triangle_position[2]- triangle_position[1]), (triangle_position[0]- triangle_position[1]))
    cross_lenght = np.linalg.norm(cross_vect)
    y_axis = cross_vect / cross_lenght
    # if the normal is parallel to the segment, create a pyramid intead of a prism
    if round(np.dot(y_axis, x_axis), 3) in [1, -1]:
        # projection = points.copy()
        start_heights = sum([np.linalg.norm(a-start) for a in points])
        end_heights = sum([np.linalg.norm(a-end) for a in points])
        heights_mean = min([start_heights, end_heights])/len(points)
    else:
        z_axis = np.cross(y_axis, x_axis)
        y_axis = np.cross(z_axis, x_axis)
        norm_x = x_axis / np.linalg.norm(x_axis)
        norm_y = y_axis / np.linalg.norm(y_axis)
        norm_z = z_axis / np.linalg.norm(z_axis)
        root_mtx = np.zeros((4, 4))
        root_mtx[0][0:3] = norm_x
        root_mtx[1][0:3] = norm_y
        root_mtx[2][0:3] = norm_z
        root_mtx[3][0:3] = np.array(start)
        root_mtx[3][-1] = 1
        root_mtx_inv = np.linalg.inv(root_mtx)
        points = [np.append(a, 1) for a in triangle_position]
        bind_pos = [np.dot(a, root_mtx_inv) for a in points]
        heights_mean = sum([a[1] for a in bind_pos])/len(points)
    return (cross_lenght/2.0) * heights_mean


def getCloserIndex(point, points):
    """the closes index of a point in a point list
    Args:
        points (list): list of 3 values list Ex: [[1,2,3],[5,6,7],...]
        point (list): 3 values list Ex:[1,2,3]
        numOfOutput (int): how many closest indixes return
    Returns:
        list: indices of the closest elements of the points list
    """
    pointsAr = np.asarray(points)
    distances = np.sum((pointsAr - np.array(point))**2, axis=1)
    return np.argmin(distances)


def projectVector(source, projection):
    np_source = np.array(source)
    np_projection = np.array(projection)
    return normalize(np_source) * ((np_projection @ np_source) / magnitude(np_source))
