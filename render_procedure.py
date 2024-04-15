import os

for i in range(0, 201):
    os.system("blender -b default_scene.blend --python render.py " + str(i))