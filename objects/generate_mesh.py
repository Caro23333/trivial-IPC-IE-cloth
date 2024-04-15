from re import M
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
from scipy.spatial import Delaunay

# 生成网格点
np.random.seed(int(time.time()))
meshSize = int(sys.argv[2])
edgeLength = 2.0

def getInnerPoints():
    if sys.argv[1] == 'u':
        samples = np.linspace(1 / (meshSize + 1), meshSize / (meshSize + 1), meshSize)
        res = np.zeros((meshSize ** 2, 2))
        for i in range(meshSize):
            for j in range(meshSize):
                res[i * meshSize + j] = [samples[i], samples[j]]
        res += (np.random.rand(meshSize ** 2, 2) - np.full((meshSize ** 2, 2), 0.5)) / (1.5 * meshSize)
        return res
    else:
        return np.random.rand(meshSize ** 2, 2)

def getBoundaryPoints():
    if sys.argv[1] == 'u':
        boundaryPoints = np.row_stack((np.linspace([0, 0], [0, meshSize / (meshSize + 1)], meshSize + 1), 
                                       np.linspace([0, 1], [meshSize / (meshSize + 1), 1], meshSize + 1),
                                       np.linspace([1, 1], [1, 1 / (meshSize + 1)], meshSize + 1),
                                       np.linspace([1, 0], [1 / (meshSize + 1), 0], meshSize + 1)))
        return boundaryPoints
    else:
        boundaryPoints = np.column_stack((np.transpose(np.full((meshSize), 0)), np.random.rand(meshSize, 1))) # left boundary
        boundaryPoints = np.row_stack((boundaryPoints, 
                                       np.column_stack((np.transpose(np.full((meshSize), 1)), np.random.rand(meshSize, 1))))) # right boundary
        boundaryPoints = np.row_stack((boundaryPoints, 
                                       np.column_stack((np.random.rand(meshSize, 1), np.transpose(np.full((meshSize), 0)))))) # bottom boundary
        boundaryPoints = np.row_stack((boundaryPoints, 
                                       np.column_stack((np.random.rand(meshSize, 1), np.transpose(np.full((meshSize), 1)))))) # top boundary
        return boundaryPoints

def getCornerPoints():
    return [[0, 0], [0, 1], [1, 0], [1, 1]]

points = np.row_stack((getInnerPoints(), getBoundaryPoints()))
if sys.argv[1] != 'u':
    points = np.row_stack((points, getCornerPoints()))
points = edgeLength * points

# 进行三角剖分
tri = Delaunay(points)


# 可视化
plt.triplot(points[:,0], points[:,1], tri.simplices)
# plt.plot(points[:,0], points[:,1], 'o')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Delaunay Triangulation')
plt.show()

# 保存到文件
meshID = int(sys.argv[3])
fileName = "cloth" + str(meshID) + ".obj"
with open(fileName, 'w') as file:
    for i in range(points.shape[0]):
        file.write("v {:.6f} {:.6f} {:.6f}\n".format(points[i, 0], 0, points[i, 1]))
    for t in tri.simplices:
        file.write("f {} {} {}\n".format(t[0] + 1, t[1] + 1, t[2] + 1))
