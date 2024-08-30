import bpy
import bmesh
# ExportHelper is a helper class, defines filename and invoke() function which calls the file selector.
from bpy_extras.io_utils import ExportHelper

from mathutils import Vector, Quaternion
from math import radians
import numpy as np
import os
import pickle
from bpy.props import (BoolProperty, EnumProperty,
                       FloatProperty, PointerProperty)
from bpy.types import (PropertyGroup)

SMPL_JOINT_NAMES = {
    0:  'Pelvis',
    1:  'L_Hip',        4:  'L_Knee',            7:  'L_Ankle',           10: 'L_Foot',
    2:  'R_Hip',        5:  'R_Knee',            8:  'R_Ankle',           11: 'R_Foot',
    3:  'Spine1',       6:  'Spine2',            9:  'Spine3',            12: 'Neck',            15: 'Head',
    13: 'L_Collar',     16: 'L_Shoulder',       18: 'L_Elbow',            20: 'L_Wrist',         22: 'L_Hand',
    14: 'R_Collar',     17: 'R_Shoulder',       19: 'R_Elbow',            21: 'R_Wrist',         23: 'R_Hand',
}
smpl_joints = 21  # len(SMPL_JOINT_NAMES)
proj_dir = 'D:/BlenderProj/SAGE'  # todo Change this dir
# End SMPL globals


def rodrigues_from_pose(armature, bone_name):
    # Ensure that rotation mode is AXIS_ANGLE so the we get a correct readout of current pose
    armature.pose.bones[bone_name].rotation_mode = 'AXIS_ANGLE'
    axis_angle = armature.pose.bones[bone_name].rotation_axis_angle

    angle = axis_angle[0]

    rodrigues = Vector((axis_angle[1], axis_angle[2], axis_angle[3]))
    rodrigues.normalize()
    rodrigues = rodrigues * angle
    return rodrigues


def set_pose_from_rodrigues(armature, bone_name, rodrigues, rodrigues_reference=None):
    rod = Vector((rodrigues[0], rodrigues[1], rodrigues[2]))
    angle_rad = rod.length
    axis = rod.normalized()

    if armature.pose.bones[bone_name].rotation_mode != 'QUATERNION':
        armature.pose.bones[bone_name].rotation_mode = 'QUATERNION'

    quat = Quaternion(axis, angle_rad)

    if rodrigues_reference is None:
        armature.pose.bones[bone_name].rotation_quaternion = quat
    else:
        # SMPL-X is adding the reference rodrigues rotation to the relaxed hand rodrigues rotation, so we have to do the same here.
        # This means that pose values for relaxed hand model cannot be interpreted as rotations in the local joint coordinate system of the relaxed hand.
        # https://github.com/vchoutas/smplx/blob/f4206853a4746139f61bdcf58571f2cea0cbebad/smplx/body_models.py#L1190
        #   full_pose += self.pose_mean
        rod_reference = Vector(
            (rodrigues_reference[0], rodrigues_reference[1], rodrigues_reference[2]))
        rod_result = rod + rod_reference
        angle_rad_result = rod_result.length
        axis_result = rod_result.normalized()
        quat_result = Quaternion(axis_result, angle_rad_result)
        armature.pose.bones[bone_name].rotation_quaternion = quat_result

        """
        rod_reference = Vector((rodrigues_reference[0], rodrigues_reference[1], rodrigues_reference[2]))
        angle_rad_reference = rod_reference.length
        axis_reference = rod_reference.normalized()
        quat_reference = Quaternion(axis_reference, angle_rad_reference)

        # Rotate first into reference pose and then add the target pose
        armature.pose.bones[bone_name].rotation_quaternion = quat_reference @ quat
        """
    return


def updata_location():
    obj = bpy.context.object
    bpy.ops.object.mode_set(mode='OBJECT')

    j_regressor_male = None

    if j_regressor_male is None:
        regressor_path = os.path.join(
            proj_dir, "data", "smpl_joint_regressor_male.npz")
        with np.load(regressor_path) as data:
            j_regressor_male = data['joint_regressor']

    j_regressor = j_regressor_male
    armature = obj.parent
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='EDIT')

    bone_rotations = {}
    for pose_bone in armature.pose.bones:
        pose_bone.rotation_mode = 'AXIS_ANGLE'
        axis_angle = pose_bone.rotation_axis_angle
        bone_rotations[pose_bone.name] = (
            axis_angle[0], axis_angle[1], axis_angle[2], axis_angle[3])

    # Set model in default pose
    for bone in armature.pose.bones:
        reset_poseshape()
        bone.rotation_mode = 'AXIS_ANGLE'
        bone.rotation_axis_angle = (0, 0, 1, 0)

    # Reset corrective poseshapes if used
    context = bpy.context

    # Get vertices with applied skin modifier
    depsgraph = context.evaluated_depsgraph_get()
    object_eval = obj.evaluated_get(depsgraph)
    mesh_from_eval = object_eval.to_mesh()

    # Get Blender vertices as numpy matrix
    vertices_np = np.zeros((len(mesh_from_eval.vertices)*3), dtype=np.float)
    mesh_from_eval.vertices.foreach_get("co", vertices_np)
    vertices_matrix = np.reshape(
        vertices_np, (len(mesh_from_eval.vertices), 3))
    object_eval.to_mesh_clear()  # Remove temporary mesh

    # Note: Current joint regressor uses 6890 vertices as input which is slow numpy operation
    joint_locations = j_regressor @ vertices_matrix

    # Set new bone joint locations
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='EDIT')

    for index in range(smpl_joints):
        bone = armature.data.edit_bones[SMPL_JOINT_NAMES[index]]
        bone.head = (0.0, 0.0, 0.0)
        bone.tail = (0.0, 0.0, 0.1)

        bone_start = Vector(joint_locations[index])
        bone.translate(bone_start)

    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.context.view_layer.objects.active = obj

    # Restore pose
    for pose_bone in armature.pose.bones:
        pose_bone.rotation_mode = 'AXIS_ANGLE'
        pose_bone.rotation_axis_angle = bone_rotations[pose_bone.name]

    return {'FINISHED'}


def reset_poseshape():
    obj = bpy.context.object

    if obj.type == 'ARMATURE':
        obj = bpy.context.object.children[0]

    for key_block in obj.data.shape_keys.key_blocks:
        if key_block.name.startswith("Pose"):
            key_block.value = 0.0
    return {'FINISHED'}


def rodrigues_to_mat(rotvec):
    theta = np.linalg.norm(rotvec)
    r = (rotvec/theta).reshape(3, 1) if theta > 0. else rotvec
    cost = np.cos(theta)
    mat = np.asarray([[0, -r[2], r[1]],
                      [r[2], 0, -r[0]],
                      [-r[1], r[0], 0]], dtype=object)
    return (cost*np.eye(3) + (1-cost)*r.dot(r.T) + np.sin(theta)*mat)


def rodrigues_to_posecorrective_weight(pose):
    joints_posecorrective = len(SMPL_JOINT_NAMES)
    rod_rots = np.asarray(pose).reshape(joints_posecorrective, 3)
    mat_rots = [rodrigues_to_mat(rod_rot) for rod_rot in rod_rots]
    bshapes = np.concatenate([(mat_rot - np.eye(3)).ravel()
                             for mat_rot in mat_rots[1:]])
    return (bshapes)


def set_pose_shape():
    obj = bpy.context.object
    # Get armature pose in rodrigues representation
    if obj.type == 'ARMATURE':
        armature = obj
        obj = bpy.context.object.children[0]
    else:
        armature = obj.parent
    pose = [0.0] * (len(SMPL_JOINT_NAMES) * 3)

    for index in range(len(SMPL_JOINT_NAMES)):
        joint_name = SMPL_JOINT_NAMES[index]
        joint_pose = rodrigues_from_pose(armature, joint_name)
        pose[index*3 + 0] = joint_pose[0]
        pose[index*3 + 1] = joint_pose[1]
        pose[index*3 + 2] = joint_pose[2]

    poseweights = rodrigues_to_posecorrective_weight(pose)
    # Set weights for pose corrective shape keys
    for index, weight in enumerate(poseweights):
        obj.data.shape_keys.key_blocks["Pose%03d" % index].value = weight


def add_gender():
    gender = 'male'
    print("Adding gender: " + gender)
    objects_path = os.path.join(
        proj_dir, "data", "smpl-model-20200803.blend", "Object")
    object_name = "SMPL-mesh-" + gender
    bpy.ops.wm.append(filename=object_name, directory=str(objects_path))
    # Select imported mesh
    bpy.context.view_layer.objects.active = bpy.data.objects[object_name]
    bpy.data.objects[object_name].select_set(True)
    
    return {'FINISHED'}


def add_texture(obj):
    texture = "smplx_texture_m_alb.png"  # 可选
    # obj = bpy.context.object
    mat = obj.data.materials[0]
    links = mat.node_tree.links
    nodes = mat.node_tree.nodes

    # Find texture node
    node_texture = None
    for node in nodes:
        if node.type == 'TEX_IMAGE':
            node_texture = node
            break

    # Find shader node
    node_shader = None
    for node in nodes:
        if node.type.startswith('BSDF'):
            node_shader = node
            break

    if node_texture is None:
        node_texture = nodes.new(type="ShaderNodeTexImage")

    if texture not in bpy.data.images:
        texture_path = os.path.join(proj_dir, "data", texture)
        image = bpy.data.images.load(texture_path)
    else:
        image = bpy.data.images[texture]

    node_texture.image = image
    # Link texture node to shader node if not already linked
    if len(node_texture.outputs[0].links) == 0:
        links.new(node_texture.outputs[0], node_shader.inputs[0])

    # Switch viewport shading to Material Preview to show texture
    if bpy.context.space_data:
        if bpy.context.space_data.type == 'VIEW_3D':
            bpy.context.space_data.shading.type = 'MATERIAL'

    return obj


def load_pose(file_path, fId=0):
    obj = bpy.context.object

    if obj.type == 'MESH':
        armature = obj.parent
    else:
        armature = obj
        obj = armature.children[0]
        # mesh needs to be active object for recalculating joint locations
        bpy.context.view_layer.objects.active = obj

    file_path = file_path
    print("Loading: " + file_path)

    data = np.load(file_path)
    len_all = data["poses"].shape[0]

    if fId > len_all or fId < 0:
        return
    translation = np.array(data["trans"][fId]).reshape(3)
    global_orient = np.array(data["poses"][fId, :3]).reshape(3)
    body_pose = np.array(data["poses"][fId, 3:66]).reshape(smpl_joints, 3)

    if global_orient is not None:
        set_pose_from_rodrigues(armature, "Pelvis", global_orient)
    for index in range(smpl_joints):
        pose_rodrigues = body_pose[index]
        # body pose starts with left_hip
        bone_name = SMPL_JOINT_NAMES[index + 1]
        set_pose_from_rodrigues(armature, bone_name, pose_rodrigues)

    if translation is not None:
        # Set translation
        armature.location = (translation[0], -translation[2], translation[1])

    set_pose_shape()


def snap_ground():
    bpy.ops.object.mode_set(mode='OBJECT')

    obj = bpy.context.object
    if obj.type == 'ARMATURE':
        armature = obj
        obj = bpy.context.object.children[0]
    else:
        armature = obj.parent

    # Get vertices with applied skin modifier in object coordinates
    depsgraph = bpy.context.evaluated_depsgraph_get()
    object_eval = obj.evaluated_get(depsgraph)
    mesh_from_eval = object_eval.to_mesh()

    # Get vertices in world coordinates
    matrix_world = obj.matrix_world
    vertices_world = [matrix_world @
                      vertex.co for vertex in mesh_from_eval.vertices]
    z_min = (min(vertices_world, key=lambda item: item.z)).z
    object_eval.to_mesh_clear()  # Remove temporary mesh

    # Adjust height of armature so that lowest vertex is on ground plane.
    # Do not apply new armature location transform so that we are later able to show loaded poses at their desired height.
    armature.location.z = armature.location.z - z_min
    return {'FINISHED'}


def load_animation(file_path, obj):
    # fId = 100
    # obj = bpy.context.object

    if obj.type == 'MESH':
        armature = obj.parent
    else:
        armature = obj
        obj = armature.children[0]
        # mesh needs to be active object for recalculating joint locations
        bpy.context.view_layer.objects.active = obj

    print("Loading: " + file_path)

    # data = np.load(file_path)
    file = open(file_path, 'rb')
    data = pickle.load(file)
    len_all = data["poses"].shape[0]

    start_frame = 0
    end_frame = len_all - 1
    bpy.context.scene.render.fps = 60
    bpy.context.scene.frame_start = start_frame
    bpy.context.scene.frame_end = end_frame 
    bpy.context.scene.frame_current = start_frame 

    for fId in range(0, len_all):
        # if fId % 10 == 0:
        #     print(f"{fId}/{len_all}")
        print(fId)
        translation = np.array(data["trans"][fId]).reshape(3)
        global_orient = np.array(data["poses"][fId, :3]).reshape(3)
        body_pose = np.array(data["poses"][fId, 3:66]).reshape(smpl_joints, 3)

        if global_orient is not None:
            set_pose_from_rodrigues(armature, "Pelvis", global_orient)
            armature.pose.bones["Pelvis"].keyframe_insert('rotation_quaternion', frame=fId)
        for index in range(smpl_joints):
            pose_rodrigues = body_pose[index]
            # body pose starts with left_hip
            bone_name = SMPL_JOINT_NAMES[index + 1]
            set_pose_from_rodrigues(armature, bone_name, pose_rodrigues)
            armature.pose.bones[bone_name].keyframe_insert('rotation_quaternion', frame=fId)

        if translation is not None:
            # Set translation
            # armature.location = (translation[0], -translation[2], translation[1])
            armature.location = (translation[0], translation[1], translation[2])
            armature.keyframe_insert('location', frame=fId)
        armature.rotation_euler = (-np.pi / 2, 0, 0)
        

def test_add(file_path, fId=0):
    gender = 'male'
    print("Adding gender: " + gender)
    objects_path = os.path.join(
        proj_dir, "data", "smpl-model-20200803.blend", "Object")
    object_name = "SMPL-mesh-" + gender
    bpy.ops.wm.append(filename=object_name, directory=str(objects_path))
    # Select imported mesh
    bpy.context.view_layer.objects.active = bpy.data.objects[object_name]
    bpy.data.objects[object_name].select_set(True)
    
    new_name = f"male{fId}"
    bpy.data.objects[object_name].name = new_name
    
    texture = "m_01_alb.002.png"  
    # texture = "f_01_alb.002.png"
    obj = bpy.context.object
    # obj.name = f'SMPL_male{fId}'
    mat = obj.data.materials[0]
    links = mat.node_tree.links
    nodes = mat.node_tree.nodes

    # Find texture node
    node_texture = None
    for node in nodes:
        if node.type == 'TEX_IMAGE':
            node_texture = node
            break

    # Find shader node
    node_shader = None
    for node in nodes:
        if node.type.startswith('BSDF'):
            node_shader = node
            break

    if node_texture is None:
        node_texture = nodes.new(type="ShaderNodeTexImage")

    if texture not in bpy.data.images:
        texture_path = os.path.join(proj_dir, "data", texture)
        image = bpy.data.images.load(texture_path)
    else:
        image = bpy.data.images[texture]

    node_texture.image = image
    # Link texture node to shader node if not already linked
    if len(node_texture.outputs[0].links) == 0:
        links.new(node_texture.outputs[0], node_shader.inputs[0])

    # Switch viewport shading to Material Preview to show texture
    if bpy.context.space_data:
        if bpy.context.space_data.type == 'VIEW_3D':
            bpy.context.space_data.shading.type = 'MATERIAL'
            
        obj = bpy.context.object

    if obj.type == 'MESH':
        armature = obj.parent
    else:
        armature = obj
        obj = armature.children[0]
        # mesh needs to be active object for recalculating joint locations
        bpy.context.view_layer.objects.active = obj

    # file_path = "C:/Users/admin/Desktop/07_01_poses.npz"
    print("Loading: " + file_path)

    data = np.load(file_path)
    len_all = data["poses"].shape[0]

    if fId > len_all or fId < 0:
        return
    
    # -----------------------------------------------------------------------------------------------
    up_idx = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    low_idx = [1, 2, 4, 5, 7, 8, 10, 11]
    gt_pose = data["poses"][:, :66].reshape(-1, 22, 3)
    gt_pose[:, up_idx] *= 0
    gt_pose = gt_pose.reshape(-1, 66)
    # -----------------------------------------------------------------------------------------------

    translation = np.array(data["trans"][fId]).reshape(3)
    global_orient = np.array(gt_pose[fId, :3]).reshape(3)
    body_pose = np.array(gt_pose[fId, 3:66]).reshape(smpl_joints, 3)

    if global_orient is not None:
        set_pose_from_rodrigues(armature, "Pelvis", global_orient)
    for index in range(smpl_joints):
        pose_rodrigues = body_pose[index]
        # body pose starts with left_hip
        bone_name = SMPL_JOINT_NAMES[index + 1]
        set_pose_from_rodrigues(armature, bone_name, pose_rodrigues)

    if translation is not None:
        # Set translation
        armature.location = (translation[0] * 1.5, -translation[2] * 1.5, translation[1])
    armature.scale = (0.5, 0.5, 0.5)
    armature.rotation_euler = (-1.5708, -0.174533, 0)

    set_pose_shape()
    return armature

if __name__ == "__main__":
    filepath ='C:/Users/hanfeng/Desktop/vis/dataset-76610.pkl'
    add_gender()
    obj = bpy.context.object
    obj = add_texture(obj)
    load_animation(filepath, obj)