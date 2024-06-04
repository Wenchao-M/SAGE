# MIT License
# Copyright (c) 2022 Guy Tevet
#
# This code is based on https://github.com/GuyTevet/motion-diffusion-model
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

# from diffusion import gaussian_diffusion as gd
# from diffusion.respace import space_timesteps, SpacedDiffusion

from diffusion_simplify import gaussian_diffusion as gd_simplify
from diffusion_simplify.respace import space_timesteps as space_timesteps_simplify
from diffusion_simplify.respace import SpacedDiffusion as SpaceDiffusionSimplify


def create_gaussian_diffusion_simplified(args, is_train=True):
    steps = args.DIFFUSION_STEPS  # 1000
    scale_beta = 1.0
    timestep_respacing = None if is_train else args.TIMESTEP_RESPACING
    rescale_timesteps = False

    betas = gd_simplify.get_named_beta_schedule(args.NOISE_SCHEDULE, steps, scale_beta)

    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpaceDiffusionSimplify(
        dataset=args.DATASET,  # "amass"
        use_timesteps=space_timesteps_simplify(steps, timestep_respacing),  # set(0, 200, 400, 600, 800)
        betas=betas,  # (1000,)
        model_mean_type=gd_simplify.ModelMeanType.START_X,
        model_var_type=gd_simplify.ModelVarType.FIXED_SMALL,
        loss_type=gd_simplify.LossType.MSE,
        rescale_timesteps=rescale_timesteps,  # False
    )
