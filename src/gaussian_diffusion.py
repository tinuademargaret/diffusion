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
        res = res[..., None]

    return res.expand(broadcast_shape)


class GaussianDiffusion:

    def __init__(self, betas):
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas

        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas

        self.alphas_cum_prod = np.cumprod(alphas, axis=0)
        # right shift
        self.alphas_cum_prod_prev = np.append(1.0, self.alphas_cum_prod[:-1])
        # left shift
        self.alphas_cum_prod_next = np.append(self.alphas_cum_prod[1:], 0.0)

        # terms or the prior q{xt|xt-1}
        self.sqrt_alphas_cum_prod = np.sqrt(self.alphas_cum_prod)
        self.sqrt_one_minus_alphas_cum_prod = np.sqrt(1.0 - self.alphas_cum_prod)

        # terms for the posterior q{xt-1 | xt, x0}
        self.posterior_variance = self.betas * (
            (1.0 - self.alphas_cum_prod_prev) / (1.0 - self.alphas_cum_prod)
        )
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        # coefficient of x0 in the mean
        self.posterior_mean_coeff1 = self.betas * (
            np.sqrt(self.alphas_cum_prod_prev) / (1.0 - self.alphas_cum_prod)
        )
        # coefficient of xt in the mean
        self.posterior_mean_coeff2 = (
            np.sqrt(alphas) * (1.0 - self.alphas_cum_prod_prev)
        ) / (1.0 - self.alphas_cum_prod)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def q_sample(self, x_start, t, noise=None):

        if not noise:
            noise = torch.randn_like(x_start)

        return (
            _extract_into_tensor(self.sqrt_alphas_cum_prod, t, x_start.shape) * x_start
        ) + (
            _extract_into_tensor(self.sqrt_one_minus_alphas_cum_prod, t, x_start.shape)
            * noise
        )

    def q_posterior_mean_varience(self, x_start, x_t, t):

        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coeff1, t, x_t.shape) * x_start
        ) + (_extract_into_tensor(self.posterior_mean_coeff2, t, x_t.shape) * x_t)

        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)

        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )

        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, model, x, t, model_kwargs=None):

        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]

        model_output = model(x, self._scale_timesteps(t), **model_kwargs)
        # split model output to noise and variance
        model_output, model_var_values = torch.split(model_output, C, dim=1)

        min_log = _extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)
        max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)

        frac = (model_var_values + 1) / 2
        model_log_variance = frac * max_log + (1 - frac) * min_log
        model_variance = torch.exp(model_log_variance)
