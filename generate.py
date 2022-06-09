# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import os
import re
import json
import random
from typing import List, Optional

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
from torch.nn import functional as F
from torch_utils import misc

import legacy
from volumetric_rendering import cal_m2c
from training.training_loop import save_image_grid
from torchvision.utils import save_image
#----------------------------------------------------------------------------

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

#----------------------------------------------------------------------------

class CommaSeparatedList(click.ParamType):
    name = 'list'

    def convert(self, value, param, ctx):
        _ = param, ctx
        if value is None or value.lower() == 'none' or value == '':
            return []
        return value.split(',')

#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=num_range, help='List of random seeds')
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--ema', help='Use EMA for generator', type=bool, default=True, metavar='BOOL', show_default=True)
@click.option('--num_steps', help='Number of samples for a ray', type=int, default=96, metavar='int', show_default=True)
@click.option('--mode', help='Comma-separated list or "none"', type=CommaSeparatedList(), default='random,rotation,translation', show_default=True)
@click.option('--num_random', help='Number of image generation for random', type=int, default=30, metavar='int', show_default=True)
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')

def generate_images(
    ctx: click.Context,
    network_pkl: str,
    seeds: Optional[List[int]],
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
    class_idx: Optional[int],
    ema: bool,
    num_steps: int,
    mode: CommaSeparatedList,
    num_random: int
):

    print('Loading networks from "%s"...' % network_pkl)
    print(f'"Use G-EMA : {ema}"')
    device = torch.device('cuda')

    if False:
        G_kwargs = {'class_name': 'training.networks.Generator', 'z_dim': 512, 'w_dim': 512, 'triplane_channels': 96,
         'triplane_res': 128, 'feat_channels': 33, 'feat_res': 32,
         'mapping_kwargs': {'num_layers': 8, 'cam_condition': True},
         'synthesis_kwargs': {'channel_base': 32768, 'channel_max': 512, 'num_fp16_res': 4, 'conv_clamp': 256,
                              'cam_data_sample': True, 'importance_sampling': True, 'point_scaling': True, 'num_steps': 48}}

        common_kwargs = {'c_dim': 0, 'img_resolution': 128, 'img_channels': 3}

        G = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).eval().requires_grad_(False).to(device)

        with dnnlib.util.open_url(network_pkl) as f:
            resume_data = legacy.load_network_pkl(f)
        if ema:
            for name, module in [('G_ema', G)]:
                misc.copy_params_and_buffers(resume_data[name], module, require_all=False)
        else:
            for name, module in [('G', G)]:
                misc.copy_params_and_buffers(resume_data[name], module, require_all=False)
    else:
        with dnnlib.util.open_url(network_pkl) as f:
            if ema:
                G = legacy.load_network_pkl(f)['G_ema'].to(device)
            else:
                G = legacy.load_network_pkl(f)['G'].to(device)

    # Change the num_stpes
    # G.synthesis.num_steps = num_steps

    if seeds is None:
        ctx.fail('--seeds option is required when not using --projected-w')

    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            ctx.fail('Must specify class label with --class when using a conditional network')
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print ('warn: --class=lbl ignored when running on an unconditional network')

    c2i_default =torch.Tensor([[[9.0579, 0.0000, 0.0000, 0.0000],
         [0.0000, 9.0579, 0.0000, 0.0000],
         [0.0000, 0.0000, 1.0000, 0.0000]]])

# ---------------------------rotation---------------------------------
    if 'rotation' in mode:
        os.makedirs(os.path.join(outdir, 'rotation'), exist_ok=True)
        # theta = [0.4, 0.2, 0, -0.2, -0.4]
        # phi = [0, 0, 0, 0, 0]
        #
        # theta_0 = [0] * 5
        # phi_0 = [0] * 5
        #
        # gw = 5
        # gh = 1

        theta = [0.4, 0.4, 0.4, 0, 0, 0, -0.4, -0.4, -0.4,]
        phi = [0.3, 0, -0.3, 0.3, 0, -0.3, 0.3, 0, -0.3,]

        theta_0 = [0] * 9
        phi_0 = [0] * 9

        gw = 3
        gh = 3

        batch_size = len(theta)


        # for seed_idx, seed in enumerate(seeds):
        #     print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        #     z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        #     c2i = c2i_default.repeat(batch_size, 1, 1).to(device)
        #     m2c = cal_m2c(theta, phi).to(device)
        #     img = G(z.repeat(gw*gh,1), label.repeat(gw*gh,1), m2c, c2i, truncation_psi=truncation_psi, noise_mode=noise_mode)
        #     # save_image_grid(img[:,:3].cpu(), f'{outdir}/rotation/seed{seed:04d}.png', drange=[-1,1], grid_size=(gw, gh))
        #     for i in range(len(img)):
        #         save_image(img[i, :3].cpu().clamp(-1,1) / 2 + 0.5, f'{outdir}/rotation/seed{seed:04d}_{i}.png')
        #         save_image(img[i, 3:6].cpu().clamp(-1, 1) / 2 + 0.5, f'{outdir}/rotation/seed{seed:04d}_{i}_low.png')
        #         super_low = F.interpolate(F.interpolate(img[i:i+1, :3].cpu().clamp(-1,1) / 2 + 0.5, (32,32)), (128,128),mode='bilinear')
        #         save_image(super_low[0], f'{outdir}/rotation/seed{seed:04d}_{i}_super_low_again.png')

        for seed_idx, seed in enumerate(seeds):
            print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
            z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
            c2i = c2i_default.repeat(batch_size, 1, 1).to(device)
            m2c = cal_m2c(theta, phi).to(device)
            m2c_2 = cal_m2c(theta_0, phi_0).to(device)
            img = G(z.repeat(gw*gh,1), label.repeat(gw*gh,1), m2c, c2i, m2c_2=m2c_2, c2i_2=c2i, swap_prob=0, truncation_psi=truncation_psi, noise_mode=noise_mode)
            save_image_grid(img[:,:3].cpu(), f'{outdir}/rotation/gconfix_seed{seed:04d}.png', drange=[-1,1], grid_size=(gw, gh))
            for i in range(len(img)):
                save_image(img[i, :3].cpu().clamp(-1,1) / 2 + 0.5, f'{outdir}/rotation/gconfix_seed{seed:04d}_{i}.png')
                # save_image(img[i, 3:6].cpu().clamp(-1,1) / 2 + 0.5, f'{outdir}/rotation/gconfix_seed{seed:04d}_{i}_low.png')
                # super_low = F.interpolate(F.interpolate(img[i:i + 1, :3].cpu().clamp(-1, 1) / 2 + 0.5, (32, 32)),
                #                           (128, 128), mode='bilinear', align_corners=True)
                # save_image(super_low[0], f'{outdir}/rotation/gconfix_seed{seed:04d}_{i}_super_low_again.png')


# ---------------------------translation---------------------------------
    if 'translation' in mode:
        os.makedirs(os.path.join(outdir, 'translation'), exist_ok=True)
        theta = [0] * 9
        phi = [0] * 9

        theta_0 = [0] * 9
        phi_0 = [0] * 9


        translation = [[-0.1, 0, 0], [0, 0, 0], [0.1, 0, 0], [-0.1, 0.1, 0], [0, 0.1, 0], [0.1, 0.1, 0], [-0.1, -0.1, 0], [0, -0.1, 0], [0.1, -0.1, 0]]

        gw = 3
        gh = 3
        batch_size = len(theta)

        for seed_idx, seed in enumerate(seeds):
            print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
            z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
            c2i = c2i_default.repeat(batch_size, 1, 1).to(device)
            m2c = cal_m2c(theta, phi, translation).to(device)
            m2c_2 = cal_m2c(theta_0, phi_0).to(device)
            img = G(z.repeat(gw*gh,1), label.repeat(gw*gh,1), m2c, c2i, m2c_2=m2c_2, c2i_2=c2i, swap_prob=0, truncation_psi=truncation_psi, noise_mode=noise_mode)
            for i in range(len(img)):
                save_image(img[i, :3].cpu().clamp(-1,1) / 2 + 0.5, f'{outdir}/translation/gconfix_seed{seed:04d}_{i}.png')
                # save_image(img[i, 3:6].cpu().clamp(-1, 1) / 2 + 0.5, f'{outdir}/translation/gconfix_seed{seed:04d}_{i}_low.png')

# ----------------------random sample--------------------------------------------
    if 'random' in mode:
        os.makedirs(os.path.join(outdir, 'random'), exist_ok=True)
        with open('./ffhq_camera_params_addflip.json', 'r') as f:
            cam_param_dict = json.load(f)
        image_name = list(cam_param_dict.keys())
        batch_size = 8
        i = 0
        while True:
            selected_cam_name = random.choices(image_name, k=batch_size)
            m2w = torch.Tensor(list(map(lambda x: cam_param_dict[x]['m2w'], selected_cam_name))).to(device)
            w2c = torch.Tensor(list(map(lambda x: cam_param_dict[x]['w2c'], selected_cam_name))).to(device)
            c2i = torch.Tensor(list(map(lambda x: cam_param_dict[x]['c2i'], selected_cam_name))).to(device)
            m2c = torch.bmm(w2c, m2w)

            z = torch.from_numpy(np.random.randn(batch_size, G.z_dim)).to(device)
            img = G(z, label.repeat(batch_size,1), m2c, c2i, swap_prob=1, truncation_psi=truncation_psi, noise_mode=noise_mode)
            for j in range(len(img)):
                save_image(img[j, :3].cpu().clamp(-1,1) / 2 + 0.5, f'{outdir}/random/{i*batch_size + j}.png')
                if i*batch_size + j == num_random:
                    break
            if i * batch_size + j == num_random:
                break
            i += 1
#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
