# MIT License
# Copyright (c) 2022 Guy Tevet
#
# This code is based on https://github.com/GuyTevet/motion-diffusion-model
# Copyright (c) Meta Platforms, Inc. All Rights Reserved
# import argparse
# import json
# import os
# from argparse import ArgumentParser
# from yacs.config import CfgNode as CN
#
#
# def get_args():
#     parser = ArgumentParser(description='Train Motion Capture Network')
#     parser.add_argument('--cfg',
#                         help='experiment configure file name',
#                         # required=True,
#                         default="config/lower.yaml",
#                         type=str)
#     args = parser.parse_args()
#     cfg = CN(new_allowed=True)
#     cfg.merge_from_file(args.cfg)
#     return cfg
