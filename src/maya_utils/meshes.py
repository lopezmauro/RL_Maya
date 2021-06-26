from .mUtils import mNode
import numpy as np


def getFacesVertices(mesh):
    """Returns the index of the vertex that conform each face
    Args:
        meshFn (UsdGeom.Mesh): USD mesh prim
    Returns:
        list of list: vertex indices for each face ex: [[1,2,3,4],[2,3,5,6],...]
    """
    meshNode = mNode.MNode(mesh).getShape()
    fn = meshNode.getBestFn()
    faceVertexCount, faceVerticesIndex = fn.getVertices()
    faceVertices = list()
    i = 0
    for vtxCount in faceVertexCount:
        faceVertices.append(list(faceVerticesIndex[i:i+vtxCount]))
        i += vtxCount
    return faceVertices


def getVerticesFaces(faceVertices):
    """given a list with all the vertices indices for each face
    returns a list for each vertex and wich faces are conneted to it
    Args:
        faceVertices (list): [description]
    Returns:
        list: list of faces connected to the vertices
    """
    result = list()
    # cast them into a dict in order to not worry about the initial list size
    vertexFaces = dict()
    for f, fVtx in enumerate(faceVertices):
        for vtx in fVtx:
            vertexFaces.setdefault(vtx, set()).add(f)
    # convert dict to list for faster post iteration
    indices = list(vertexFaces.keys())
    indices.sort()
    for idx in indices:
        result.append(vertexFaces[idx])
    return result


def nearVertices(vertex, vertexFaces, faceVertices):
    """return all surrounding vertices indices
    Args:
        vertex (int): vertex index
        vertexFaces (list): list with all the faces id connected to the each vertiex
        faceVertices (list): list of each vertex id that conform the face
    Returns:
        set: all vertices idices of connected to the overtex
    """
    result = set()
    faces = vertexFaces[vertex]
    for f in faces:
        result.update(faceVertices[f])
    return result


def getNearVerticesDistances(vertex, points, vertexFaces, faceVertices):
    conn_vtx = nearVertices(vertex, vertexFaces, faceVertices)
    distances = dict()
    for vtx in conn_vtx:
        distances[vtx] = np.linalg.norm(points[vertex]-points[vtx])
    return distances


def getGeodesicDistances(start, verticesList, points, vertexFaces, faceVertices):
    distances = dict()
    vertices_list = list(verticesList)
    if start not in vertices_list:
        vertices_list.append(start)
    for vtx in vertices_list:
        dist = getNearVerticesDistances(vtx, points, vertexFaces, faceVertices)
        distances[vtx] = dict([(k, v) for k, v in dist.items() if k in vertices_list])
    geodesic_dist = {start: 0}
    geodesic_dist.update(distances[start])
    for vtx in vertices_list:
        curr_dist = geodesic_dist.copy()
        for indx, dist in geodesic_dist.items():
            near_dist = dict([(k, v+dist) for k, v in distances[indx].items() if k not in curr_dist])
            curr_dist.update(near_dist)
        geodesic_dist.update(curr_dist)
    return geodesic_dist


def getConnectedVertices(vertices, vertexFaces, faceVertices):
    result = dict()
    for vertex in vertices:
        result[vertex] = nearVertices(vertex, vertexFaces, faceVertices)
    return result


def getVertextriangles(all_triangles, vertices):
    vertex_is_in_triangle = np.all(np.isin(all_triangles, vertices), axis=1)
    return all_triangles[np.where(vertex_is_in_triangle)]



