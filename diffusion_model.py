import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ContextEncoder(nn.Module):
    def __init__(self, seq_len, input_dim=2, hidden_dim=64, context_dim=128):
        """
        Encodes historical returns and volatility into a context vector.
        input_dim: 2 (returns, volatility)
        """
        super().__init__()
        # Using a simple LSTM to capture sequential non-linear dynamics
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, 
                            num_layers=2, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_dim, context_dim)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        _, (hidden, _) = self.lstm(x)
        # Take the hidden state of the last layer
        last_hidden = hidden[-1]
        context = self.fc(last_hidden)
        return F.silu(context)

class ConditionalDenoiser(nn.Module):
    def __init__(self, target_dim=1, context_dim=128, time_dim=64, hidden_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, hidden_dim)  # Changed output dim to hidden_dim
        )
        
        # Main MLP processing the noisy input and context
        self.input_mlp = nn.Sequential(
            nn.Linear(target_dim, hidden_dim),
            nn.SiLU()
        )
        
        self.context_mlp = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.SiLU()
        )
        
        # Residual blocks for non-linear dynamics
        self.res_block1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        self.res_block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        self.output_mlp = nn.Linear(hidden_dim, target_dim)

    def forward(self, x, time, context):
        # x: (batch_size, target_dim) - noisy next-step return
        # time: (batch_size,)
        # context: (batch_size, context_dim)
        
        t_emb = self.time_mlp(time)
        x_emb = self.input_mlp(x)
        c_emb = self.context_mlp(context)
        
        # Combine all features
        h = x_emb + t_emb + c_emb
        
        # Residual blocks
        h = h + self.res_block1(h)
        h = h + self.res_block2(h)
        
        return self.output_mlp(h)

class ConditionalDiffusionModel(nn.Module):
    def __init__(self, seq_len, target_dim=1, timesteps=100):
        super().__init__()
        self.timesteps = timesteps
        
        # Modules
        self.encoder = ContextEncoder(seq_len=seq_len)
        self.denoiser = ConditionalDenoiser(target_dim=target_dim)
        
        # Define noise schedule (Linear schedule)
        beta_start = 1e-4
        beta_end = 0.02
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        
    def forward(self, x_0, history_data):
        """
        x_0: (batch_size, target_dim) - Actual next step return
        history_data: (batch_size, seq_len, 2) - Historical (return, vol)
        Returns: loss
        """
        device = x_0.device
        batch_size = x_0.shape[0]
        self.betas = self.betas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        
        # 1. Generate context
        context = self.encoder(history_data)
        
        # 2. Sample random timesteps
        t = torch.randint(0, self.timesteps, (batch_size,), device=device).long()
        
        # 3. Sample noise
        noise = torch.randn_like(x_0)
        
        # 4. Add noise to x_0 based on t
        alpha_cumprod_t = self.alphas_cumprod[t].view(-1, 1)
        x_t = torch.sqrt(alpha_cumprod_t) * x_0 + torch.sqrt(1 - alpha_cumprod_t) * noise
        
        # 5. Predict noise using the denoiser
        predicted_noise = self.denoiser(x_t, t, context)
        
        # 6. Calculate loss (MSE between actual noise and predicted noise)
        loss = F.mse_loss(noise, predicted_noise)
        return loss

    @torch.no_grad()
    def sample(self, history_data, num_samples=1):
        """
        Generate synthetic next-step return distributions given historical data.
        history_data: (batch_size, seq_len, 2)
        """
        device = history_data.device
        batch_size = history_data.shape[0]
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        
        context = self.encoder(history_data)
        
        # Start from pure Gaussian noise
        x_t = torch.randn((batch_size, 1), device=device)
        
        # Iteratively denoise
        for i in reversed(range(0, self.timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            
            predicted_noise = self.denoiser(x_t, t, context)
            
            alpha_t = self.alphas[t].view(-1, 1)
            alpha_cumprod_t = self.alphas_cumprod[t].view(-1, 1)
            beta_t = self.betas[t].view(-1, 1)
            
            # Predict x_{t-1}
            mean = (1 / torch.sqrt(alpha_t)) * (x_t - ((1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t)) * predicted_noise)
            
            if i > 0:
                noise = torch.randn_like(x_t)
                # Compute variance
                var = beta_t  # or \tilde{\beta}_t
                x_t = mean + torch.sqrt(var) * noise
            else:
                x_t = mean
                
        return x_t

# Example usage/test
if __name__ == "__main__":
    batch_size = 16
    seq_len = 30 # 30 previous time steps
    
    # Dummy historical data: (batch, seq_len, 2 features [returns, vol])
    history = torch.randn(batch_size, seq_len, 2)
    # Dummy target: (batch, 1 feature [next return])
    next_return = torch.randn(batch_size, 1)
    
    model = ConditionalDiffusionModel(seq_len=seq_len)
    
    # Training step
    loss = model(next_return, history)
    print(f"Training Loss: {loss.item():.4f}")
    
    # Inference / Sampling step
    generated_returns = model.sample(history)
    print(f"Generated samples shape: {generated_returns.shape}")
