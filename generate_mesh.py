import os
import re
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
import mrcfile

#----------------------------------------------------------------------------

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]


def create_samples(N=256, voxel_origin=[0, 0, 0], cube_length=2.0):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array(voxel_origin) - cube_length / 2
    voxel_size = cube_length / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 3)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    return samples.unsqueeze(0), voxel_origin, voxel_size

def feature_sample(triplane, query_points, triplane_channels = 96):
    xy_plane, xz_plane, yz_plane = triplane.split(int(triplane_channels / 3), dim=1)

    query_points_xy = torch.cat((query_points[:, 0:1], query_points[:, 1:2]), dim=1)
    query_points_xz = torch.cat((query_points[:, 0:1], query_points[:, 2:3]), dim=1)
    query_points_yz = torch.cat((query_points[:, 1:2], query_points[:, 2:3]), dim=1)


    # xy_feature = F.grid_sample(xy_plane, query_points_xy.permute(0, 2, 3, 1).clamp(-1,1))
    # xz_feature = F.grid_sample(xz_plane, query_points_xz.permute(0, 2, 3, 1).clamp(-1,1))
    # yz_feature = F.grid_sample(yz_plane, query_points_yz.permute(0, 2, 3, 1).clamp(-1,1))

    xy_feature = F.grid_sample(xy_plane, query_points_xy.permute(0, 2, 3, 1).clamp(-1,1))
    xz_feature = F.grid_sample(xz_plane, query_points_xz.permute(0, 2, 3, 1).clamp(-1,1))
    yz_feature = F.grid_sample(yz_plane, query_points_yz.permute(0, 2, 3, 1).clamp(-1,1))

    return xy_feature + xz_feature + yz_feature
#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=num_range, help='List of random seeds')
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--projected-w', help='Projection result file', type=str, metavar='FILE')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
def generate_meshs(
    ctx: click.Context,
    network_pkl: str,
    seeds: Optional[List[int]],
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
    class_idx: Optional[int],
    projected_w: Optional[str]
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    # Generate curated MetFaces images without truncation (Fig.10 left)
    python generate.py --outdir=out --trunc=1 --seeds=85,265,297,849 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

    \b
    # Generate uncurated MetFaces images with truncation (Fig.12 upper left)
    python generate.py --outdir=out --trunc=0.7 --seeds=600-605 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

    \b
    # Generate class conditional CIFAR-10 images (Fig.17 left, Car)
    python generate.py --outdir=out --seeds=0-35 --class=1 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/cifar10.pkl

    \b
    # Render an image from projected W
    python generate.py --outdir=out --projected_w=projected_w.npz \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl
    """

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')

    if True:
        G_kwargs = {'class_name': 'training.networks.Generator', 'z_dim': 512, 'w_dim': 512, 'triplane_channels': 96,
         'triplane_res': 128, 'feat_channels': 33, 'feat_res': 32,
         'mapping_kwargs': {'num_layers': 8, 'cam_condition': True},
         'synthesis_kwargs': {'channel_base': 32768, 'channel_max': 512, 'num_fp16_res': 4, 'conv_clamp': 256,
                              'cam_data_sample': True, 'importance_sampling': True, 'point_scaling': True, 'num_steps': 24}}

        common_kwargs = {'c_dim': 0, 'img_resolution': 128, 'img_channels': 3}

        G = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).eval().requires_grad_(False).to(device)

        with dnnlib.util.open_url(network_pkl) as f:
            resume_data = legacy.load_network_pkl(f)
        for name, module in [('G_ema', G)]:
            misc.copy_params_and_buffers(resume_data[name], module, require_all=False)
    else:
        with dnnlib.util.open_url(network_pkl) as f:
            G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    os.makedirs(outdir, exist_ok=True)

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

    # Generate images.
    theta = [0]
    phi = [0]

    theta_0 = [0]
    phi_0 = [0]

    truncation_psi = 0.5
    batch_size = len(theta)
    c2i_default =torch.Tensor([[[9.0579, 0.0000, 0.0000, 0.0000],
         [0.0000, 9.0579, 0.0000, 0.0000],
         [0.0000, 0.0000, 1.0000, 0.0000]]])

    for seed_idx, seed in enumerate(seeds):
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        c2i = c2i_default.repeat(batch_size, 1, 1).to(device)
        m2c = cal_m2c(theta, phi).to(device)

        voxel_resolution = 128
        voxel_origin = [0,0,0]
        cube_length = 2.0

        samples, voxel_origin, voxel_size = create_samples(voxel_resolution, voxel_origin, cube_length)
        samples = samples.to(z.device)
        sigmas = torch.zeros((batch_size, samples.shape[1], 1), device=z.device)

        ws = G.mapping(z, label, m2c, c2i, truncation_psi=truncation_psi, truncation_cutoff=None, swap_prob=1)
        triplane = G.synthesis(ws, m2c, c2i, return_triplane=True)
        chunk_size = 100000
        for b in range(batch_size):
            head = 0
            while head < samples.shape[1]:
                tail = head + chunk_size
                feature = feature_sample(triplane, samples[b:b + 1, head:tail].permute(0, 2, 1).unsqueeze(-1))
                output = G.synthesis.tri_plane_decoder(feature.unsqueeze(-3)).permute(0, 2, 1, 3).squeeze(2)
                sigmas[b:b+1, head:tail] = output[..., -1:]
                head += chunk_size
        sigmas = sigmas.reshape((voxel_resolution, voxel_resolution, voxel_resolution)).cpu().numpy()
        with mrcfile.new_mmap(os.path.join(outdir, f'{seed}.mrc'), overwrite=True, shape=sigmas.shape, mrc_mode=2) as mrc:
            mrc.data[:] = sigmas

    theta = [0.4, 0.2, 0, -0.2, -0.4]
    phi = [0, 0, 0, 0, 0]

    theta_0 = [0, 0, 0, 0, 0]
    phi_0 = [0, 0, 0, 0, 0]

    gw = 5
    gh = 1
    batch_size = len(theta)

    c2i = c2i_default.repeat(batch_size, 1, 1).to(device)
    m2c = cal_m2c(theta, phi).to(device)
    m2c_2 = cal_m2c(theta_0, phi_0).to(device)
    img = G(z.repeat(gw * gh, 1), label.repeat(gw * gh, 1), m2c, c2i, m2c_2=m2c_2, c2i_2=c2i, swap_prob=0,
            truncation_psi=truncation_psi, noise_mode=noise_mode)
    save_image_grid(img[:, :3].cpu(), f'{outdir}/gconfix_seed{seed:04d}.png', drange=[-1, 1], grid_size=(gw, gh))

    for seed_idx, seed in enumerate(seeds):
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        c2i = c2i_default.repeat(batch_size, 1, 1).to(device)
        m2c = cal_m2c(theta, phi).to(device)
        m2c_2 = cal_m2c(theta_0, phi_0).to(device)
        img = G(z.repeat(gw*gh,1), label.repeat(gw*gh,1), m2c, c2i, m2c_2=m2c_2, c2i_2=c2i, swap_prob=0, truncation_psi=truncation_psi, noise_mode=noise_mode)
        save_image_grid(img[:,:3].cpu(), f'{outdir}/gconfix_seed{seed:04d}.png', drange=[-1,1], grid_size=(gw, gh))



#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_meshs() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
