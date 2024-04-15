from mesh import *
from distance import *
import torch
import time

gravity = -1
stiffness = [8, 2]
delta = 10 ** -2
damping = 1 - 2.5 * 10 ** -2

def eulerToRotationMatrix(angles):
    theta_x, theta_y, theta_z = angles[0], angles[1], angles[2]
    
    Rx = torch.tensor([[1, 0, 0],
                       [0, torch.cos(theta_x), -torch.sin(theta_x)],
                       [0, torch.sin(theta_x), torch.cos(theta_x)]])
    Ry = torch.tensor([[torch.cos(theta_y), 0, torch.sin(theta_y)],
                       [0, 1, 0],
                       [-torch.sin(theta_y), 0, torch.cos(theta_y)]])
    Rz = torch.tensor([[torch.cos(theta_z), -torch.sin(theta_z), 0],
                       [torch.sin(theta_z), torch.cos(theta_z), 0],
                       [0, 0, 1]])
    rotation_matrix = torch.matmul(Rz, torch.matmul(Ry, Rx))
    return rotation_matrix

class Dynamics:

    position = None
    velocity = None
    Fext = None
    mesh = None
    vNum = None
    fNum = None
    eNum = None

    def __init__(self, mesh, translation, rotation) -> None:
        self.position = torch.tensor([]).to(device)
        self.mesh = mesh
        self.vNum = mesh.getVerticesCount()
        self.fNum = mesh.getFacesCount()
        self.eNum = mesh.getEdgesCount()
        rotationMatrix = eulerToRotationMatrix(rotation)
        # initialize position
        for pos in mesh.vertices:
            f32Pos = torch.tensor(pos).to(torch.float32)
            configuredPos = torch.matmul(rotationMatrix, f32Pos) + translation 
            self.position = torch.cat((self.position, configuredPos.to(device)), dim = 0)
        # initialize velocity
        self.velocity = torch.zeros((3 * self.vNum, )).to(device)
        # initialize external forces
        self.Fext = torch.zeros((3 * self.vNum, ))
    
    def exportOBJ(self, filePath, check = False):
        with open(filePath, 'w') as file:
            for i in range(self.vNum):
                file.write("v {:.6f} {:.6f} {:.6f}\n".format(self.position[3 * i], self.position[3 * i + 1], self.position[3 * i + 2]))
            for face in self.mesh.faces:
                file.write("f {} {} {}\n".format(face[0] + 1, face[1] + 1, face[2] + 1))
        if check:
            torch.save(self.velocity, "./checkpoint.pt")


class RigidDynamics(Dynamics):

    uniformVelocity = None

    def __init__(self, rigidMesh, translation, rotation, velocity) -> None:
        Dynamics.__init__(self, rigidMesh, translation, rotation)
        # initialize velocity
        self.velocity = torch.cat([velocity] * self.vNum, dim = 0)
        self.uniformVelocity = velocity.to(device)

    def timeStep(self):
        temp = torch.reshape(self.position, (self.vNum, 3))
        temp = temp + self.uniformVelocity * delta
        self.position = torch.reshape(temp, (3 * self.vNum,))


class ClothDynamics(Dynamics):

    sNum = None
    mass = None
    massInv = None
    incident = None
    K = None

    def __init__(self, clothMesh, translation, rotation) -> None:
        Dynamics.__init__(self, clothMesh, translation, rotation)
        # initialize mass
        mass = torch.eye(3 * self.vNum).to(device)
        for i in range(self.vNum):
            mass[3 * i, 3 * i] *= clothMesh.verticesMass[i]
            mass[3 * i + 1, 3 * i + 1] *= clothMesh.verticesMass[i]
            mass[3 * i + 2, 3 * i + 2] *= clothMesh.verticesMass[i]
        self.massInv = mass.inverse()
        # initialize gravity
        self.Fext = torch.cat([torch.tensor([0, 0, gravity])] * self.vNum, dim = 0).to(device)
        for fixed in clothMesh.fixedVertices:
            self.Fext[fixed * 3 + 2] = 0
        # initialize spring
        self.sNum = clothMesh.springs.shape[0]
        self.originalLength = torch.zeros((self.sNum, )).to(device)
        self.K = torch.eye(self.sNum).to(device)
        self.incident = torch.zeros(3 * self.sNum, 3 * self.vNum).to(device)
        for i in range(self.sNum):
            spring = clothMesh.springs[i]
            self.originalLength[i] = torch.norm(torch.tensor(clothMesh.vertices[spring[1]] - clothMesh.vertices[spring[0]]))
            self.K[i, i] *= stiffness[self.mesh.springType[i]]
            self.incident[3 * i: 3 * i + 3, 3 * spring[0]: 3 * spring[0] + 3] = torch.eye(3)
            self.incident[3 * i: 3 * i + 3, 3 * spring[1]: 3 * spring[1] + 3] = torch.eye(3) * -1

    def computeElasticEnergy(self):
        def evalEnergy(pos):
            relPos = torch.mv(self.incident, pos)
            relPos = relPos.reshape((self.sNum, 3))
            squareLength = torch.square(torch.norm(relPos, dim = 1) - self.originalLength)
            squareLength = 0.5 * torch.mv(self.K, squareLength)
            print(torch.sum(squareLength).item())
            return torch.sum(squareLength)
        # evalEnergy(self.position)
        gradResult = torch.autograd.functional.jacobian(evalEnergy, self.position).view(-1)
        # hessianResult = torch.autograd.functional.hessian(evalEnergy, self.position)
        return gradResult#, hessianResult

    def timeStep(self, rigidDynamics: RigidDynamics): 
        dInv = delta * delta * self.massInv
        dInv = dInv.to(device)
        xHat = delta * delta * self.Fext + damping * delta * self.velocity + self.position
        iterCnt = 0
        # BFGS
        def truncate(G, threshold):
            if torch.norm(G) < threshold:
                return torch.zeros_like(G)
            else:
                return G 
        originalPosition = self.position
        self.velocity = (xHat - self.position) / delta
        self.position = xHat
        for vertex in self.mesh.fixedVertices:
            print(vertex)
            self.position[3 * vertex: 3 * vertex + 3] = originalPosition[3 * vertex: 3 * vertex + 3]
            self.velocity[3 * vertex: 3 * vertex + 3] = torch.zeros(3)
            xHat[3 * vertex: 3 * vertex + 3] = originalPosition[3 * vertex: 3 * vertex + 3]
        GNow = torch.eye(3 * self.vNum).to(device)
        G = computeContactEnergy(self.position, rigidDynamics.position, self, rigidDynamics)
        KG = truncate(self.computeElasticEnergy(), 3 * 10 ** -3)
        # print(torch.norm(G), torch.norm(KG))
        gNxt = self.position + damping * torch.mv(dInv, KG + G) - xHat
        dg = gNxt
        eta = 0.01
        while iterCnt <= 200:
            iterCnt += 1
            dx = -eta * torch.mv(GNow, gNxt)
            self.position += dx
            self.velocity += dx / delta
            for vertex in self.mesh.fixedVertices:
                self.position[3 * vertex: 3 * vertex + 3] = originalPosition[3 * vertex: 3 * vertex + 3]
                self.velocity[3 * vertex: 3 * vertex + 3] = torch.zeros(3)
            print(iterCnt, torch.norm(dx))
            if torch.norm(dx) < 3 * 10 ** -4:
                break
            gNow = gNxt
            G = computeContactEnergy(self.position, rigidDynamics.position, self, rigidDynamics)
            KG = truncate(self.computeElasticEnergy(), 3 * 10 ** -3)
            # print(torch.norm(G), torch.norm(KG))
            gNxt = self.position + damping * torch.mv(dInv, KG + G) - xHat
            dg = gNxt - gNow
            T = torch.eye(3 * self.vNum).to(device) - torch.matmul(dx.unsqueeze(1), dg.unsqueeze(0)) / torch.dot(dx, dg)
            GNow = torch.matmul(T, torch.matmul(GNow, torch.transpose(T, 0, 1))) + torch.matmul(dx.unsqueeze(1), dx.unsqueeze(0)) / torch.dot(dx, dg)
            # if iterCnt % 5 == 0:
            #     eta += 0.02
