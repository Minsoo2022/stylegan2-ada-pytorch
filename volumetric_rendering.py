import torch
from torch.nn import functional as F


# https://github.com/marcoamonteiro/pi-GAN/blob/master/generators/volumetric_rendering.py#L18
def fancy_integration(rgb_sigma, z_vals, device, noise_std=0.5, last_back=False, white_back=False, clamp_mode='relu', fill_mode=None):
    """Performs NeRF volumetric rendering."""

    rgbs = rgb_sigma[..., :-1]
    sigmas = rgb_sigma[..., -1:]

    deltas = z_vals[:, :, 1:] - z_vals[:, :, :-1]
    delta_inf = 1e10 * torch.ones_like(deltas[:, :, :1])
    deltas = torch.cat([deltas, delta_inf], -2)

    noise = torch.randn(sigmas.shape, device=device) * noise_std

    if clamp_mode == 'softplus':
        alphas = 1 - torch.exp(-deltas * (F.softplus(sigmas + noise)))
    elif clamp_mode == 'relu':
        alphas = 1 - torch.exp(-deltas * (F.relu(sigmas + noise)))
    else:
        raise "Need to choose clamp mode"

    alphas_shifted = torch.cat([torch.ones_like(alphas[:, :, :1]), 1 - alphas + 1e-10], -2)
    weights = alphas * torch.cumprod(alphas_shifted, -2)[:, :, :-1]
    weights_sum = weights.sum(2)

    if last_back:
        weights[:, :, -1] += (1 - weights_sum)

    rgb_final = torch.sum(weights * rgbs, -2)
    depth_final = torch.sum(weights * z_vals, -2)

    if white_back:
        rgb_final = rgb_final + 1 - weights_sum

    if fill_mode == 'debug':
        rgb_final[weights_sum.squeeze(-1) < 0.9] = torch.tensor([1., 0, 0], device=rgb_final.device)
    elif fill_mode == 'weight':
        rgb_final = weights_sum.expand_as(rgb_final)

    return rgb_final, depth_final, weights


# https://github.com/marcoamonteiro/pi-GAN/blob/master/generators/volumetric_rendering.py#L58
def get_initial_rays_image(n, num_steps, device, resolution, ray_start, ray_end, perturb_points=False):
    """Returns sample points, z_vals, and ray directions in image space."""
    W, H = resolution
    # Create full screen NDC (-1 to +1) coords [x, y, 0, 1].
    # Y is flipped to follow image memory layouts.
    x, y = torch.meshgrid(torch.linspace(-1, 1, W, device=device),
                          torch.linspace(1, -1, H, device=device))
    x = x.T.flatten()
    y = y.T.flatten()
    # DO NOT negate z; it is handled by the `w2c` matrix in our case.
    z = torch.ones_like(x, device=device)

    rays_d_image = torch.stack([x, y, z], -1)
    # No normalisation
    # rays_d_image = rays_d_image / torch.linalg.norm(rays_d_image, dim=-1, keepdim=True)

    z_vals = torch.linspace(ray_start, ray_end, num_steps, device=device).view(1, num_steps, 1).expand(W * H, -1, -1)

    # TODO: different perturbation for the samples in the batch
    if perturb_points:
        distance_between_points = z_vals[:, 1:2, :] - z_vals[:, 0:1, :]
        offset = (torch.rand(z_vals.shape, device=device) - 0.5) * distance_between_points
        z_vals = z_vals + offset

    points = rays_d_image.unsqueeze(1) * z_vals

    points = points.expand(n, -1, -1, -1)
    z_vals = z_vals.expand(n, -1, -1, -1)
    rays_d_image = rays_d_image.expand(n, -1, -1)

    return points, z_vals, rays_d_image


def transform_points(i2m, directions_image, points_image):
    origin = i2m[:, :3, 3].view(-1, 1, 1, 3)
    direction = torch.matmul(
        i2m[:, :3, :3].view(-1, 1, 1, 3, 3),
        directions_image.unsqueeze(-2).unsqueeze(-1)).squeeze(-1)
    points = torch.matmul(
        i2m[:, :3, :3].view(-1, 1, 1, 3, 3),
        points_image.unsqueeze(-1)).squeeze(-1) + origin
    return origin, direction, points


def get_i2m(intrinsics, extrinsics):
    intrinsics_4x4 = torch.zeros_like(extrinsics)
    intrinsics_4x4[:, :3, :3] = intrinsics[:, :3, :3]
    intrinsics_4x4[:, 3, 3] = 1

    # return torch.linalg.inv(torch.bmm(intrinsics_4x4, extrinsics))
    return torch.inverse(torch.bmm(intrinsics_4x4, extrinsics))



# https://github.com/marcoamonteiro/pi-GAN/blob/master/generators/volumetric_rendering.py#L205
def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
    """
    Sample @N_importance samples from @bins with distribution defined by @weights.
    Inputs:
        bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
        weights: (N_rays, N_samples_)
        N_importance: the number of samples to draw from the distribution
        det: deterministic or not
        eps: a small number to prevent division by zero
    Outputs:
        samples: the sampled samples
    Source: https://github.com/kwea123/nerf_pl/blob/master/models/rendering.py
    """
    N_rays, N_samples_ = weights.shape
    weights = weights + eps  # prevent division by zero (don't do inplace op!)
    pdf = weights / torch.sum(weights, -1, keepdim=True)  # (N_rays, N_samples_)
    cdf = torch.cumsum(pdf, -1)  # (N_rays, N_samples), cumulative distribution function
    cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], -1)  # (N_rays, N_samples_+1)
                                                              # padded to 0~1 inclusive

    if det:
        u = torch.linspace(0, 1, N_importance, device=bins.device)
        u = u.expand(N_rays, N_importance)
    else:
        u = torch.rand(N_rays, N_importance, device=bins.device)
    u = u.contiguous()

    inds = torch.searchsorted(cdf, u)
    below = torch.clamp_min(inds - 1, 0)
    above = torch.clamp_max(inds, N_samples_)

    inds_sampled = torch.stack([below, above], -1).view(N_rays, 2 * N_importance)
    cdf_g = torch.gather(cdf, 1, inds_sampled)
    cdf_g = cdf_g.view(N_rays, N_importance, 2)
    bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, N_importance, 2)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom[denom < eps] = 1  # denom equals 0 means a bin has weight 0, in which case it will not be sampled
                            # anyway, therefore any value for it is fine (set to 1 here)

    samples = bins_g[..., 0] + (u - cdf_g[..., 0]) / denom * (bins_g[..., 1]-bins_g[..., 0])
    return samples
