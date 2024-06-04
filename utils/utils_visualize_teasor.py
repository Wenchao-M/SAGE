# MIT License
# Copyright (c) 2022 ETH Sensing, Interaction & Perception Lab
#
# This code is based on https://github.com/eth-siplab/AvatarPoser
# Copyright (c) Meta Platforms, Inc. All Rights Reserved
import os

import pyrender
from scipy.spatial.transform import Rotation

os.environ["PYOPENGL_PLATFORM"] = "egl"
import torch
import cv2
import numpy as np
import trimesh
# from body_visualizer.mesh.mesh_viewer import MeshViewer
from utils.mesh_viewer import MeshViewer
from body_visualizer.tools.vis_tools import colors

from human_body_prior.tools.omni_tools import copy2cpu as c2c
from tqdm import tqdm
from utils import utils_transform

line_between_point_upper = [[0, 3], [3, 6], [6, 9], [9, 12], [9, 13], [9, 14], [12, 15], [13, 16], [14, 17], [16, 18],
                            [17, 19], [18, 20], [19, 21]]
line_between_point_lower = [[0, 1], [0, 2], [1, 4], [2, 5], [4, 7], [5, 8], [7, 10], [8, 11]]


class CheckerBoard:
    def __init__(self, white=(247, 246, 244), black=(146, 163, 171)):
        self.white = np.array([0.969, 0.968, 0.960, 1.])  # np.array(white) / 255.0
        self.black = np.array([0.572, 0.639, 0.655, 1.])  # p.array(black) / 255.0
        self.verts, self.faces, self.texts = None, None, None
        self.offset = None

    @staticmethod
    def gen_checker_xy(black, white, square_size=0.5, xlength=50.0, ylength=50.0):
        """
        generate a checker board in parallel to x-y plane
        starting from (0, 0) to (xlength, ylength), in meters
        return: trimesh.Trimesh
        """
        xsquares = int(xlength / square_size)
        ysquares = int(ylength / square_size)
        verts, faces, texts = [], [], []
        fcount = 0
        height = -0.12
        for i in range(xsquares):
            for j in range(ysquares):
                p1 = np.array([i * square_size, j * square_size, height])
                p2 = np.array([(i + 1) * square_size, j * square_size, height])
                p3 = np.array([(i + 1) * square_size, (j + 1) * square_size, height])

                verts.extend([p1, p2, p3])
                faces.append([fcount * 3, fcount * 3 + 1, fcount * 3 + 2])
                fcount += 1

                p1 = np.array([i * square_size, j * square_size, height])
                p2 = np.array([(i + 1) * square_size, (j + 1) * square_size, height])
                p3 = np.array([i * square_size, (j + 1) * square_size, height])

                verts.extend([p1, p2, p3])
                faces.append([fcount * 3, fcount * 3 + 1, fcount * 3 + 2])
                fcount += 1

                if (i + j) % 2 == 0:
                    texts.append(black)
                    texts.append(black)
                else:
                    texts.append(white)
                    texts.append(white)

        # now compose as mesh
        mesh = trimesh.Trimesh(
            vertices=np.array(verts) + np.array([-5, -5, 0]), faces=np.array(faces), process=False,
            face_colors=np.array(texts))
        return mesh


def get_body_vert(part):
    # get the body part
    import json
    f = open('smpl_vert.json', 'r')
    content = f.read()
    smpl_vert_all = json.loads(content)
    res = []
    # 'hips'目前不知道该放到哪里
    if part == 'lower':
        body_part = ['rightUpLeg', 'leftLeg', 'leftToeBase', 'leftFoot', 'rightFoot', 'rightLeg', 'rightToeBase',
                     'leftUpLeg', 'hips']
    elif part == 'upper':
        body_part = ['rightHand', 'leftArm', 'spine1', 'spine2',
                     'leftShoulder', 'rightShoulder', 'head', 'rightArm', 'leftHandIndex1',
                     'rightHandIndex1', 'leftForeArm', 'rightForeArm', 'neck', 'spine',
                     'leftHand']
    else:
        return
    for key in body_part:
        res += smpl_vert_all[key]
    # print(a.keys())
    return res


def get_vector_mesh(sparse_mat):
    mesh_head = trimesh.creation.axis(origin_size=0.02, axis_length=0.15)
    mesh_lh = trimesh.creation.axis(origin_size=0.02, axis_length=0.15)
    mesh_rh = trimesh.creation.axis(origin_size=0.02, axis_length=0.15)
    mesh_head.apply_transform(sparse_mat[0])
    mesh_lh.apply_transform(sparse_mat[1])
    mesh_rh.apply_transform(sparse_mat[2])

    return [mesh_head, mesh_lh, mesh_rh]


"""
# --------------------------------
# Visualize avatar using body pose information and body model
# --------------------------------
"""


def save_animation(body_pose, savepath, bm, color_map=None, fps=60, resolution=(800, 800)):
    imw, imh = resolution
    mv = MeshViewer(width=imw, height=imh, use_offscreen=True)
    faces = c2c(bm.f)
    img_array = []
    body_vert_list = []
    generator = CheckerBoard()
    checker_mesh = generator.gen_checker_xy(generator.black, generator.white)
    checker_mesh.apply_transform(
        trimesh.transformations.rotation_matrix(-90, (0, 0, 10))
    )
    checker_mesh.apply_transform(
        trimesh.transformations.rotation_matrix(30, (10, 0, 0))
    )
    checker_mesh.apply_transform(trimesh.transformations.scale_matrix(0.5))

    part_vert_idx = get_body_vert('lower')
    begin_frame = 310
    end_frame = 334
    all_item = tqdm(range(begin_frame, end_frame))
    color_cur = [0.57, 0.55, 0.64]
    rotate_angle = 4.65
    scale_fac = 0.8  # 0.55
    # color_cur = [0.9, 0.9, 0.9]
    add_axis = False
    vert_color = np.tile(color_cur + [0.50], (6890, 1))
    # vert_color = np.tile([0.9, 0.9, 0.9, 0.6], (6890, 1))
    # vert_color[part_vert_idx] = np.array([[0.9, 0.9, 0.9, 0.9]]).repeat(len(part_vert_idx), 0)
    for fId in all_item:
        orien_mesh_list = []
        body_mesh = trimesh.Trimesh(
            vertices=c2c(body_pose.v[fId]),
            faces=faces,
            # vertex_colors=color_map[fId],
            vertex_colors=vert_color,  # np.tile(color_cur + [1.], (6890, 1)),
            # face_colors=np.tile(color_cur + [0.10], (6890, 1))
            # vertex_colors=np.tile([.74, .06, .88, 0.8], (6890, 1))
        )
        num_faces = faces.shape[0]
        face_colors = np.tile([0.9, 0.9, 0.9, 0.3], (num_faces, 1))
        body_mesh.visual.face_colors = face_colors
        # samples = []
        # for dim in range(3):
        #     dim_samples = torch.linspace(body_pose.Jtr[fId, 0].cpu()[dim], body_pose.Jtr[fId, 1].cpu()[dim], 100)
        #     samples.append(dim_samples)
        # samples = torch.stack(samples, dim=1)
        #
        # point = trimesh.Trimesh(vertices = samples)

        body_mesh.apply_transform(
            trimesh.transformations.rotation_matrix(rotate_angle, (0, 0, 10))
        )
        body_mesh.apply_transform(
            trimesh.transformations.rotation_matrix(30, (10, 0, 0))
        )
        body_mesh.apply_transform(trimesh.transformations.scale_matrix(scale_fac))

        joint_mesh_list = []
        for i in range(22):
            if i == 0:
                joint_mesh = trimesh.creation.uv_sphere(0.02)
                color = [1.0, 0.0, 0.0]  # set color,  [R, G, B]
            else:
                joint_mesh = trimesh.creation.uv_sphere(0.01)
                color = [0.1, 0.1, 0.1]
            joint_mesh.visual.vertex_colors = np.tile(color, (len(joint_mesh.vertices), 1))

            trasform_joint = torch.eye(4)
            transition = body_pose.Jtr[fId, i]
            trasform_joint[:3, 3] = transition
            joint_mesh.apply_transform(trasform_joint)
            joint_mesh.apply_transform(
                trimesh.transformations.rotation_matrix(rotate_angle, (0, 0, 10))
            )
            joint_mesh.apply_transform(
                trimesh.transformations.rotation_matrix(30, (10, 0, 0))
            )
            joint_mesh.apply_transform(trimesh.transformations.scale_matrix(scale_fac))
            joint_mesh_list.append(joint_mesh)

        line_list = []
        for line in line_between_point_upper:
            point1 = c2c(body_pose.Jtr[fId, line[0]])
            point2 = c2c(body_pose.Jtr[fId, line[1]])
            cylinder = trimesh.creation.cylinder(radius=0.003, segment=[point1, point2])
            cylinder.visual.vertex_colors = np.tile([0.0, 0.0, 1.0], (len(cylinder.vertices), 1))

            cylinder.apply_transform(
                trimesh.transformations.rotation_matrix(rotate_angle, (0, 0, 10))
            )
            cylinder.apply_transform(
                trimesh.transformations.rotation_matrix(30, (10, 0, 0))
            )
            cylinder.apply_transform(trimesh.transformations.scale_matrix(scale_fac))
            line_list.append(cylinder)

        for line in line_between_point_lower:
            point1 = c2c(body_pose.Jtr[fId, line[0]])
            point2 = c2c(body_pose.Jtr[fId, line[1]])
            cylinder = trimesh.creation.cylinder(radius=0.003, segment=[point1, point2])
            cylinder.visual.vertex_colors = np.tile([0.0, 1.0, 0.0], (len(cylinder.vertices), 1))

            cylinder.apply_transform(
                trimesh.transformations.rotation_matrix(rotate_angle, (0, 0, 10))
            )
            cylinder.apply_transform(
                trimesh.transformations.rotation_matrix(30, (10, 0, 0))
            )
            cylinder.apply_transform(trimesh.transformations.scale_matrix(scale_fac))
            line_list.append(cylinder)

        # if add_axis:
        #     sparse_angle = body_pose.full_pose[fId, :66].reshape(22, 3)[[15, 20, 21]]  # (3, 3)
        #     sparse_matrix = utils_transform.aa2matrot(sparse_angle)  # (3, 3, 3)
        #     sparse_pos = body_pose.Jtr[fId, :22][[15, 20, 21]]  # (3, 3)
        #     mat_all = torch.cat((sparse_matrix, sparse_pos[..., None]), dim=-1)  # (3, 3, 4)
        #     last_row = torch.tensor([0., 0., 0., 1.]).reshape(1, 1, 4).repeat(3, 1, 1).to(mat_all.device)
        #     mat_all = torch.cat((mat_all, last_row), dim=1)
        #     orien_mesh_list_tmp = get_vector_mesh(mat_all.cpu())
        #
        #     for orien_mesh in orien_mesh_list_tmp:
        #         orien_mesh.apply_transform(
        #             trimesh.transformations.rotation_matrix(rotate_angle, (0, 0, 10))
        #         )
        #         orien_mesh.apply_transform(
        #             trimesh.transformations.rotation_matrix(30, (10, 0, 0))
        #         )
        #         orien_mesh.apply_transform(trimesh.transformations.scale_matrix(scale_fac))
        #     orien_mesh_list += orien_mesh_list_tmp
        #
        # if begin_frame + 120 < fId < end_frame:
        #     prev_first_vert = body_vert_list[fId - begin_frame - 40]
        #     prev_second_vert = body_vert_list[fId - begin_frame - 80]
        #     prev_third_vert = body_vert_list[fId - begin_frame - 120]
        #     mesh_vert_list = [prev_first_vert, prev_second_vert, prev_third_vert]
        #     transparency = [.8, .8, .8]
        #     body_mesh_tmp_list = []
        #     for i in range(3):
        #         cur_mesh = trimesh.Trimesh(
        #             vertices=mesh_vert_list[i],
        #             faces=faces,
        #             # vertex_colors=color_map[fId],
        #             vertex_colors=vert_color  # np.tile(color_cur + [transparency[i]], (6890, 1)),
        #             # vertex_colors=np.tile([.74, .06, .88, 0.8], (6890, 1))
        #         )
        #         cur_mesh.apply_transform(
        #             trimesh.transformations.rotation_matrix(rotate_angle, (0, 0, 10))
        #         )
        #         cur_mesh.apply_transform(
        #             trimesh.transformations.rotation_matrix(30, (10, 0, 0))
        #         )
        #         cur_mesh.apply_transform(trimesh.transformations.scale_matrix(scale_fac))
        #         body_mesh_tmp_list.append(cur_mesh)
        #         # if add_axis:
        #         #     sparse_angle = body_pose.full_pose[fId - 40 * (i + 1), :66].reshape(22, 3)[[15, 20, 21]]  # (3, 3)
        #         #     sparse_matrix = utils_transform.aa2matrot(sparse_angle)  # (3, 3, 3)
        #         #     sparse_pos = body_pose.Jtr[fId - 40 * (i + 1), :22][[15, 20, 21]]  # (3, 3)
        #         #     mat_all = torch.cat((sparse_matrix, sparse_pos[..., None]), dim=-1)  # (3, 3, 4)
        #         #     last_row = torch.tensor([0., 0., 0., 1.]).reshape(1, 1, 4).repeat(3, 1, 1).to(mat_all.device)
        #         #     mat_all = torch.cat((mat_all, last_row), dim=1)
        #         #     orien_mesh_list_tmp = get_vector_mesh(mat_all.cpu())
        #         #
        #         #     for orien_mesh in orien_mesh_list_tmp:
        #         #         orien_mesh.apply_transform(
        #         #             trimesh.transformations.rotation_matrix(rotate_angle, (0, 0, 10))
        #         #         )
        #         #         orien_mesh.apply_transform(
        #         #             trimesh.transformations.rotation_matrix(30, (10, 0, 0))
        #         #         )
        #         #         orien_mesh.apply_transform(trimesh.transformations.scale_matrix(scale_fac))
        #         #     orien_mesh_list += orien_mesh_list_tmp
        #     mv.set_static_meshes([body_mesh] + body_mesh_tmp_list + orien_mesh_list)  # orien_mesh_list
        # else:
        #     # mv.set_static_meshes([checker_mesh, body_mesh])
        #     mv.set_static_meshes([body_mesh] + orien_mesh_list)  # orien_mesh_list
        mv.set_static_meshes([body_mesh] + joint_mesh_list + line_list)
        body_image = mv.render(render_wireframe=False, RGBA=True)

        body_image = body_image.astype(np.uint8)
        body_image = cv2.cvtColor(body_image, cv2.COLOR_BGR2RGB)

        body_vert_list.append(c2c(body_pose.v[fId]))
        img_array.append(body_image)

    out = cv2.VideoWriter(savepath, cv2.VideoWriter_fourcc(*"mp4v"), fps, resolution)

    for i in range(len(img_array)):
        out.write(img_array[i])
    tqdm.write(f"Videos {savepath} have been saved.")

    file_name = savepath.split('/')[-1].split('.')[0]
    img_save_dir = f'outputs/Video_smpl/{file_name}'
    if not os.path.exists(img_save_dir):
        os.makedirs(img_save_dir, exist_ok=True)
    for i in range(len(img_array)):
        img_save_path = os.path.join(img_save_dir, f'{i}.jpg')
        cv2.imwrite(img_save_path, img_array[i])
        print(f"save to {img_save_path}")
    all_item.close()
    out.release()
