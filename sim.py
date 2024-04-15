import torch
import os
import numpy as np
import time
from dynamics import *

meshDir = "objects"
resultDir = "result"
maxSteps = 600


print("1")
clothMesh = ClothMesh(os.path.join(meshDir, "cloth10.obj"), 1, fixed = True)
clothDynamic = ClothDynamics(clothMesh, torch.tensor([-1, -1, 2.4]), torch.tensor([0, 0, 0]))
rigidMesh = PlainMesh(os.path.join(meshDir, "rigid5.obj"))
rigidDynamic = RigidDynamics(rigidMesh, torch.tensor([0, 0, -10]), torch.tensor([0, 0, 0]), torch.tensor([0, 0, 0]))
startTime = time.time()
for i in range(maxSteps):
    print(i)
    rigidDynamic.timeStep()
    clothDynamic.timeStep(rigidDynamic)
    rigidDynamic.exportOBJ(os.path.join(resultDir, "rigid_frame_" + str(i + 1) + ".obj"))
    clothDynamic.exportOBJ(os.path.join(resultDir, "cloth_frame_" + str(i + 1) + ".obj"), (i + 1) % 50 == 0)
    print("frame {} finished, {:.2f} seconds cost".format(i + 1, time.time() - startTime))
    # input()