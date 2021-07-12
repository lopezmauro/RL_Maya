import numpy as np

def initializeCentroids(points, k):
    """returns k centroids from the initial points"""
    centroids = points.copy()
    np.random.shuffle(centroids)
    return centroids[:k]
 
def closestCentroid(points, centroids):
    """returns an array containing the index to the nearest centroid for each point"""
    distances = np.sqrt(((points - centroids[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0) 

def moveCentroids(points, closest, centroids):
    """returns the new centroids assigned from the points closest to them"""
    return np.array([points[closest == k].mean(axis=0) for k in range(centroids.shape[0])])

def evalKMeans(k, raw_data):
    points = np.array(raw_data)
    centroids = initializeCentroids(points, k)
    prev_centroids = np.zeros_like(centroids)
    for x in range(100):
        closest = closestCentroid(points, centroids)
        centroids = moveCentroids(points, closest, centroids)
        if np.linalg.norm(centroids - prev_centroids) > 0:
           prev_centroids = centroids.copy()
        else:
            break
    closest = closestCentroid(points, centroids)
    return centroids, closest