# MIT License
# Copyright (c) 2022 ETH Sensing, Interaction & Perception Lab
#
# This code is based on https://github.com/eth-siplab/AvatarPoser
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import os

os.environ["PYOPENGL_PLATFORM"] = "egl"
import cv2
import numpy as np
import trimesh
from body_visualizer.mesh.mesh_viewer import MeshViewer
from body_visualizer.tools.vis_tools import colors

from human_body_prior.tools.omni_tools import copy2cpu as c2c
from tqdm import tqdm


class CheckerBoard:
    def __init__(self, white=(247, 246, 244), black=(146, 163, 171)):
        self.white = np.array(white) / 255.0
        self.black = np.array(black) / 255.0
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
        for i in range(xsquares):
            for j in range(ysquares):
                p1 = np.array([i * square_size, j * square_size, 0])
                p2 = np.array([(i + 1) * square_size, j * square_size, 0])
                p3 = np.array([(i + 1) * square_size, (j + 1) * square_size, 0])

                verts.extend([p1, p2, p3])
                faces.append([fcount * 3, fcount * 3 + 1, fcount * 3 + 2])
                fcount += 1

                p1 = np.array([i * square_size, j * square_size, 0])
                p2 = np.array([(i + 1) * square_size, (j + 1) * square_size, 0])
                p3 = np.array([i * square_size, (j + 1) * square_size, 0])

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

    # 上身的setting
    # begin_frame = 240
    # end_frame = 360
    # begin_frame = 120
    # end_frame = 560
    begin_frame = 0
    end_frame = 420
    all_item = tqdm(range(begin_frame, end_frame))

    for fId in all_item:
        color_cur = [1., 1., 1.]
        body_mesh = trimesh.Trimesh(
            vertices=c2c(body_pose.v[fId]),
            faces=faces,
            # vertex_colors=color_map[fId],
            vertex_colors=np.tile(color_cur + [.8], (6890, 1)),
            # vertex_colors=np.tile([.74, .06, .88, 0.8], (6890, 1))
        )
        body_mesh.apply_transform(
            trimesh.transformations.rotation_matrix(-90, (0, 0, 10))
        )
        body_mesh.apply_transform(
            trimesh.transformations.rotation_matrix(30, (10, 0, 0))
        )
        body_mesh.apply_transform(trimesh.transformations.scale_matrix(0.5))
        # if begin_frame + 31 < fId < end_frame:
        # prev_first_vert = body_vert_list[fId - begin_frame - 10]
        # prev_second_vert = body_vert_list[fId - begin_frame - 18]
        # prev_third_vert = body_vert_list[fId - begin_frame - 30]
        # if begin_frame + 240 < fId < end_frame:
        #     prev_first_vert = body_vert_list[fId - begin_frame - 80]
        #     prev_second_vert = body_vert_list[fId - begin_frame - 160]
        #     prev_third_vert = body_vert_list[fId - begin_frame - 240]
        if begin_frame + 120 < fId < end_frame:
            prev_first_vert = body_vert_list[fId - begin_frame - 40]
            prev_second_vert = body_vert_list[fId - begin_frame - 80]
            prev_third_vert = body_vert_list[fId - begin_frame - 120]
            mesh_vert_list = [prev_first_vert, prev_second_vert, prev_third_vert]
            transparency = [.8, .8, .8]
            body_mesh_tmp_list = []
            for i in range(3):
                cur_mesh = trimesh.Trimesh(
                    vertices=mesh_vert_list[i],
                    faces=faces,
                    # vertex_colors=color_map[fId],
                    vertex_colors=np.tile(color_cur + [transparency[i]], (6890, 1)),
                    # vertex_colors=np.tile([.74, .06, .88, 0.8], (6890, 1))
                )
                cur_mesh.apply_transform(
                    trimesh.transformations.rotation_matrix(-90, (0, 0, 10))
                )
                cur_mesh.apply_transform(
                    trimesh.transformations.rotation_matrix(30, (10, 0, 0))
                )
                cur_mesh.apply_transform(trimesh.transformations.scale_matrix(0.5))
                body_mesh_tmp_list.append(cur_mesh)

            mv.set_static_meshes([body_mesh] + body_mesh_tmp_list)
        else:
            # mv.set_static_meshes([checker_mesh, body_mesh])
            mv.set_static_meshes([body_mesh])
        # mv.set_static_meshes([checker_mesh, body_mesh])
        body_image = mv.render(render_wireframe=False)
        body_image = body_image.astype(np.uint8)
        body_image = cv2.cvtColor(body_image, cv2.COLOR_BGR2RGB)

        body_vert_list.append(c2c(body_pose.v[fId]))
        img_array.append(body_image)

    out = cv2.VideoWriter(savepath, cv2.VideoWriter_fourcc(*"mp4v"), fps, resolution)

    for i in range(len(img_array)):
        out.write(img_array[i])
    tqdm.write(f"Videos {savepath} have been saved.")

    img_save_dir = 'outputs/lower_vqvae_maskroot_code256/Video/img_bio21_full'
    if not os.path.exists(img_save_dir):
        os.makedirs(img_save_dir, exist_ok=True)
    for i in range(len(img_array) - 120, len(img_array)):
        img_save_path = os.path.join(img_save_dir, f'{i}.jpg')
        cv2.imwrite(img_save_path, img_array[i])
        print(f"save to {img_save_path}")
    all_item.close()
    out.release()
