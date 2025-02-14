import math
import numpy as np
import torch


def get_named_beta_schedule(schedule_name, num_diffussion_steps):

    if schedule_name == "linear":
        # I'll need to check the paper for the reasoning behind the start and end values
        scale = 1000 / num_diffussion_steps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linespace(
            beta_start, beta_end, num_diffussion_steps, dtype=np.float64
        )

    elif schedule_name == "cosine":
        return beta_for_alpha_bar(
            num_diffussion_steps,
            lambda t: math.cos(((t + 0.008) / 1.008) * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"Unknown beta schedule: {schedule_name}")


def beta_for_alpha_bar(num_diffussion_steps, alpha_bar, max_beta=0.999):
    betas = []
    for i in range(num_diffussion_steps):
        t1 = alpha_bar(i / num_diffussion_steps)
        t2 = alpha_bar((i + 1) / num_diffussion_steps)

        beta = min((1 - t2 / t1), max_beta)
        betas.append(beta)
    return np.array(betas)


def _extract_into_tensor(arr, timestep, broadcast_shape):

    res = torch.from_numpy(arr).to(device=timestep.device)[timestep].float()

    while len(res.shape) < len(broadcast_shape):
        res = res[:, ..., None]

    return res.expand(broadcast_shape)


class GaussianDiffusion:

    def __init__(self, betas):
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas

        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas

        self.alphas_cum_prod = np.cumprod(alphas, axis=0)

        self.sqrt_alphas_cum_prod = np.sqrt(self.alphas_cum_prod)

        self.sqrt_one_minus_alphas_cum_prod = np.sqrt(1.0 - self.alphas_cum_prod)

    def q_sample(self, x_start, t, noise=None):

        if not noise:
            noise = torch.randn_like(x_start)

        return (
            _extract_into_tensor(self.sqrt_alphas_cum_prod, t, x_start.shape) * x_start
        ) + (
            _extract_into_tensor(self.sqrt_one_minus_alphas_cum_prod, t, x_start.shape)
            * noise
        )
