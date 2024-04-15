import torch
from mesh import *

device = torch.device("cuda")

dHat = 2 * 10 ** -2 # threshold distance

def pointEdgeDistance(edge, point):
    edge = edge.to(device)
    edgeVector = edge[1] - edge[0]
    edgeLength = torch.norm(edgeVector)
    unitVector = edgeVector / edgeLength
    pointToStart = point - edge[0]
    projLength = torch.dot(pointToStart, unitVector)
    if projLength <= 0:
        distance = torch.norm(pointToStart)
    elif projLength >= edgeLength:
        distance = torch.norm(point - edge[1])
    else:
        projPoint = edge[0] + unitVector * projLength
        distance = torch.norm(point - projPoint)
    return distance

class CoefMatrix:
    
    coef2D = None
    coef1D = None

    def __init__(self, size) -> None:
        # print(11111)
        self.coef2D = torch.zeros(size * (size + 1) // 2, 2).to(device)
        self.coef1D = torch.zeros(size * size, 2).to(device)
        cnt1 = 0
        cnt2 = 0
        for i in range(size):
            for j in range(size - i):
                self.coef2D[cnt1] = torch.tensor([i / (size - 1), j / (size - 1)]).to(device)
                cnt1 += 1
            for j in range(size):
                self.coef1D[cnt2] = torch.tensor([i / (size - 1), -j / (size - 1)]).to(device)
                cnt2 += 1
        torch.save(self.coef2D, "./coef2D.pt")
        torch.save(self.coef1D, "./coef1D.pt")

coef = CoefMatrix(20)

def pointTriDistance(triangle, point):
    vec1 = triangle[1] - triangle[0]
    vec2 = triangle[2] - triangle[0]
    samples = torch.matmul(coef.coef2D, torch.stack((vec1, vec2))) + triangle[0]   
    samples = torch.norm(samples - point, dim = 1)
    distance = torch.min(samples)
    return torch.where(distance < 10 ** -8, 1, distance)

def edgeDistance(edge1, edge2):
    vec1 = edge1[1] - edge1[0]
    vec2 = edge2[1] - edge2[0]
    samples = torch.matmul(coef.coef1D, torch.stack((vec1, vec2))) + edge1[0] - edge2[0] 
    samples = torch.norm(samples, dim = 1)
    distance = torch.min(samples)
    return torch.where(distance < 10 ** -8, 1, distance)

def evalSingleEnergy(d):
    return torch.where(d <= dHat, -torch.square(d - dHat) * torch.log(d / dHat), torch.zeros(1).to(device))

def computeContactEnergy(clothPosition, rigidPosition, clothDynamic, rigidDynamic):
    clothMesh = clothDynamic.mesh
    rigidMesh = rigidDynamic.mesh 
    gradResult = torch.zeros(3 * clothDynamic.vNum).to(device)
    rPos = rigidPosition.reshape((rigidDynamic.vNum, 3))
    return gradResult
    def allRigidClothTriangle(position):
        result = torch.zeros(1).to(device)
        cPos = position.reshape((clothDynamic.vNum, 3))
        for face in clothMesh.faces:
            func = lambda p: evalSingleEnergy(pointTriDistance(torch.stack([cPos[face[0]], cPos[face[1]], cPos[face[2]]]), p))
            result += torch.sum(torch.vmap(func)(rPos))
        print(result.item())
        return result
    
    def allClothTriangle(position):
        result = torch.zeros(1).to(device)
        cPos = position.reshape((clothDynamic.vNum, 3))
        for face in clothMesh.faces:
            func = lambda p: evalSingleEnergy(pointTriDistance(torch.stack([cPos[face[0]], cPos[face[1]], cPos[face[2]]]), p))
            result += torch.sum(torch.vmap(func)(cPos))
        return result

    def toAllRigidTriangle(position):
        result = torch.zeros(1).to(device)
        cPos = position.reshape((clothDynamic.vNum, 3))
        for face in rigidMesh.faces:
            func = lambda p: evalSingleEnergy(pointTriDistance(torch.stack([rPos[face[0]], rPos[face[1]], rPos[face[2]]]), p))
            result += torch.sum(torch.vmap(func)(cPos))
        print(result.item())
        return result

    def allEdge(position):
        result = torch.zeros(1).to(device)
        cPos = position.reshape((clothDynamic.vNum, 3))
        cEdge = torch.zeros(clothDynamic.eNum, 2, 3).to(device)
        for i in range(clothDynamic.eNum):
            edge = clothMesh.edges[i]
            cEdge[i] = torch.stack((cPos[edge[0]], cPos[edge[1]])).to(device)
        # for edge1 in clothMesh.edges:
        #     edge1Tensor = torch.stack((cPos[edge1[0]], cPos[edge1[1]]))
        #     func = lambda e: evalSingleEnergy(edgeDistance(edge1Tensor, e))
        #     result += torch.sum(torch.vmap(func)(cEdge))
        for edge1 in rigidMesh.edges:
            edge1Tensor = torch.stack((rPos[edge1[0]], rPos[edge1[1]]))
            func = lambda e: evalSingleEnergy(edgeDistance(edge1Tensor, e))
            result += torch.sum(torch.vmap(func)(cEdge))
        print(result.item())
        return result

    # cloth - rigid point - face
    gradResult += torch.autograd.functional.jacobian(toAllRigidTriangle, clothPosition).view(-1).to(device)
    # rigid - cloth point - face
    gradResult += torch.autograd.functional.jacobian(allRigidClothTriangle, clothPosition).view(-1).to(device)
    # cloth - cloth point - face
    # gradResult += torch.autograd.functional.jacobian(allClothTriangle, clothPosition).view(-1).to(device)
    # edge - edge
    gradResult += torch.autograd.functional.jacobian(allEdge, clothPosition).view(-1).to(device)
    return gradResult

