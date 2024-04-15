import torch
import numpy as np

edgeLength = 2.0

class PlainMesh:

    vertices = None
    faces = None
    edges = None
    # valid only in ClothMesh
    springs = None
    verticesMass = None
    fixedVertices = None
    springs = None
    springType = None

    def __init__(self, filePath) -> None:
        with open(filePath, 'r') as file:
            verticesList = []
            facesList = []
            edgesList = []
            while True:
                currentLine = file.readline()
                if not currentLine:
                    break
                currentData = currentLine[2: -1].split(' ')
                if currentLine[0] == 'v':
                    verticesList.append([float(x) for x in currentData])
                else:
                    facesList.append(sorted([int(x) - 1 for x in currentData]))
            self.vertices = np.array(verticesList)
            self.faces = np.array(facesList)
            for face in facesList:
                edgesList.append([face[0], face[1]])
                edgesList.append([face[0], face[2]])
                edgesList.append([face[1], face[2]])
            self.edges = np.unique(np.array(edgesList), axis = 0)

    def getVerticesCount(self):
        return self.vertices.shape[0]

    def getFacesCount(self):
        return self.faces.shape[0]
    
    def getEdgesCount(self):
        return self.edges.shape[0]
    
    def debug(self):
        for i in range(self.getVerticesCount()):
            print("v", self.vertices[i])
        for i in range(self.getFacesCount()):
            print("f", self.faces[i])


class ClothMesh(PlainMesh):

    def __init__(self, filePath, rho, fixed) -> None:
        PlainMesh.__init__(self, filePath)
        n = self.getVerticesCount()
        m = self.getFacesCount()
        self.verticesMass = np.zeros(n)
        self.verticesType = np.zeros(n)
        self.fixedVertices = []
        print("test")
        for i in range(n):
            incidentArea = 0
            for j in range(m):
                if i in self.faces[j]:
                    print(i)
                    incidentArea += self.getFaceArea(j) / 3
            self.verticesMass[i] = incidentArea * rho
        print("test1")
        if fixed:
            for i in range(n):
                if (self.vertices[i] == [0, 0, 0]).all() or (self.vertices[i] == [0, edgeLength, 0]).all():
                    self.fixedVertices.append(i)
        self.buildSprings()
    
    def getFaceArea(self, FaceID):
        verticesList = [self.vertices[x] for x in self.faces[FaceID]]
        a = torch.from_numpy(verticesList[1] - verticesList[0])
        b = torch.from_numpy(verticesList[2] - verticesList[0])
        area = 0.5 * torch.norm(torch.cross(a, b, -1))
        return area.item()
    
    def buildSprings(self):
        self.springs = self.edges
        self.springType = [0] * self.getEdgesCount()
        for edge in self.edges:
            incidentVertices = []
            for face in self.faces:
                if edge[0] in face and edge[1] in face:
                    for vertex in face:
                        if vertex not in edge:
                            incidentVertices.append(vertex)
            if len(incidentVertices) == 2:
                incidentVertices.sort()
                self.springs = np.row_stack((self.springs, np.array(incidentVertices)))
                self.springType.append(1)
        # print(self.springs)  

    def debug(self):
        for i in range(self.getVerticesCount()):
            print("mass", self.vertices[i], self.verticesMass[i])

