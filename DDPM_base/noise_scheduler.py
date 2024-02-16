import torch

class NoiseScheduler:
    def __init__(self, num_timesteps, beta_start, beta_end):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1 - self.betas
        self.alpha_cum_prod = torch.cumprod(self.alphas, 0)
        self.sqrt_alpha_cum_prod = torch.sqrt(self.alpha_cum_prod)
        self.sqrt_one_minus_alpha_cum_prod = torch.sqrt(1 - self.alpha_cum_prod)

    def add_noise(self, img, noise, timestep):
        img_shape = img.shape
        batch_size = img_shape[0]

        sqrt_alpha_cum_prod = self.sqrt_alpha_cum_prod[timestep].reshape(batch_size)
        sqrt_one_minus_alpha_cum_prod = self.sqrt_one_minus_alpha_cum_prod[timestep].reshape(batch_size)

        for _ in range(len(img_shape)-1):
            sqrt_alpha_cum_prod = sqrt_alpha_cum_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_cum_prod = sqrt_one_minus_alpha_cum_prod.unsqueeze(-1)

        return sqrt_alpha_cum_prod * img + sqrt_one_minus_alpha_cum_prod * noise
    
    def sample_previous_timestep(self, xt, noise_pred, timestep):
        x0 = (xt - self.sqrt_one_minus_alpha_cum_prod[timestep] * noise_pred) / self.sqrt_alpha_cum_prod[timestep]
        x0 = torch.clamp(x0, -1, 1)

        mean = xt - (self.betas[timestep] * noise_pred) / self.sqrt_one_minus_alpha_cum_prod[timestep]
        mean = mean / torch.sqrt(self.alphas[timestep])

        if timestep == 0:
            return mean, x0
        else:
            variance = (1-self.alpha_cum_prod[timestep-1]) / (1 - self.alpha_cum_prod[timestep])
            variance = variance * (self.betas[timestep])
            sigma = torch.sqrt(variance)
            z = torch.randn_like(xt)
            return mean + sigma * z, x0
