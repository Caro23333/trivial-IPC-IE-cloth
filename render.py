import bpy
import os
import sys

frameDir = "my_data_10"
outputPath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "image")

clothMaterial = bpy.data.materials["ClothMaterial"]
rigidMaterial = bpy.data.materials["RigidMaterial"]

frameNum = int(sys.argv[5])

for file in os.listdir(frameDir):
    if file.startswith("shell" + str(frameNum) + ".obj"):
    # if file.startswith("rigid_frame_" + str(frameNum) + ".obj") or file.startswith("cloth_frame_" + str(frameNum) + ".obj"):
        # determine path
        objPath = os.path.join(frameDir, file)
        targetOBJName = file.split('.')[0]
        # import .obj file
        # bpy.ops.wm.obj_import(filepath = objPath, forward_axis = "Y", up_axis = "Z")
        bpy.ops.wm.obj_import(filepath = objPath)
        newOBJ = bpy.context.selected_objects[0]
        newOBJ.name = targetOBJName + "_new"
        bpy.context.scene.collection.objects.link(newOBJ)
        # check if the target object exists
        if targetOBJName in bpy.data.objects:
            targetOBJ = bpy.data.objects[targetOBJName]
            bpy.data.objects.remove(targetOBJ, do_unlink = True)
        newOBJ.name = targetOBJName
        newOBJ.data.materials.clear()
        # if file.startswith("rigid_frame_"):
        #     newOBJ.data.materials.append(rigidMaterial)
        # else:
        #     newOBJ.data.materials.append(clothMaterial)
        #     # surface subdivision
        # bpy.ops.object.modifier_add(type = "SUBSURF")
        # modifier = newOBJ.modifiers[-1]
        # modifier_levels = 1
        #     modifier.render_levels = 2
        #     # solidify
        #     bpy.ops.object.modifier_add(type = "SOLIDIFY")
        #     modifier = newOBJ.modifiers[-1]
        #     modifier.thickness = 0.01
        newOBJ.data.materials.append(rigidMaterial)

bpy.context.scene.render.engine = "BLENDER_EEVEE"
# bpy.context.scene.cycles.device = "GPU"
# bpy.context.scene.cycles.feature_set = "SUPPORTED"
# bpy.context.preferences.addons['cycles'].preferences.compute_device_type = "CUDA"
# render
bpy.context.scene.render.filepath = os.path.join(outputPath, "frame_" + str(frameNum) + ".png")
bpy.ops.render.render(write_still = True)