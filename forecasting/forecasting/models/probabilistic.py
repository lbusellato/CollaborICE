import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,  # Set batch_first=True for better performance
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.d_model = d_model

    def forward(self, src):
        # src shape: [B, T, input_dim]
        src = self.input_projection(src) * math.sqrt(self.d_model)  # [B, T, d_model]
        src = self.pos_encoder(src)  # [B, T, d_model]
        # No need to permute since we're using batch_first=True
        memory = self.transformer_encoder(src)  # [B, T, d_model]
        return memory


class CrossAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True  # Set batch_first=True for better performance
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory):
        # tgt shape: [B, T, d_model]
        # memory shape: [B, T, d_model]

        # No need to permute since we're using batch_first=True
        attn_output, _ = self.multihead_attn(tgt, memory, memory)
        output = self.norm(tgt + self.dropout(attn_output))

        return output


class LatentSpace(nn.Module):
    def __init__(self, d_model, latent_dim):
        super(LatentSpace, self).__init__()
        # Mean and log variance for VAE-style sampling
        self.fc_mu = nn.Linear(d_model, latent_dim)
        self.fc_logvar = nn.Linear(d_model, latent_dim)

    def forward(self, x):
        # x shape: [B, T, d_model]
        # Pool temporal dimension to get shape [B, d_model]
        x_pooled = x.mean(dim=1)

        mu = self.fc_mu(x_pooled)  # [B, latent_dim]
        logvar = self.fc_logvar(x_pooled)  # [B, latent_dim]

        return mu, logvar

    def sample(self, mu, logvar, num_samples=1):
        # mu, logvar shape: [B, latent_dim]
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(num_samples, mu.size(0), mu.size(1), device=mu.device)  # [N, B, latent_dim]
        z = mu.unsqueeze(0) + eps * std.unsqueeze(0)  # [N, B, latent_dim]
        return z


class ConditionedDecoder(nn.Module):
    def __init__(self, latent_dim, condition_dim, d_model, output_dim, sequence_length):
        super(ConditionedDecoder, self).__init__()
        self.latent_proj = nn.Linear(latent_dim + condition_dim, d_model)
        self.gru = nn.GRU(d_model, d_model, num_layers=2, batch_first=True)
        self.output_proj = nn.Linear(d_model, output_dim)
        self.sequence_length = sequence_length

    def forward(self, z, condition):
        # z shape: [N, B, latent_dim]
        # condition shape: [N, B, condition_dim]

        # Concatenate latent and condition
        z_cond = torch.cat([z, condition], dim=-1)  # [N, B, latent_dim + condition_dim]

        # Project to hidden state
        N, B, _ = z_cond.shape
        hidden = self.latent_proj(z_cond)  # [N, B, d_model]
        hidden = hidden.view(1, N * B, -1).repeat(2, 1, 1)  # [2, N*B, d_model] for 2-layer GRU

        # Create initial input
        dummy_input = torch.zeros(N * B, self.sequence_length, hidden.size(-1), device=z.device)

        # Generate sequence
        output, _ = self.gru(dummy_input, hidden)  # [N*B, seq_len, d_model]

        # Reshape and project to output dimension
        output = output.reshape(N, B, self.sequence_length, -1)  # [N, B, seq_len, d_model]
        output = self.output_proj(output)  # [N, B, seq_len, output_dim]

        return output


class QuaternionDecoder(ConditionedDecoder):
    def __init__(self, latent_dim, condition_dim, d_model, sequence_length):
        super(QuaternionDecoder, self).__init__(latent_dim, condition_dim, d_model, 4, sequence_length)

    def forward(self, z, condition):
        output = super().forward(z, condition)
        # Normalize quaternions to unit length
        output = F.normalize(output, p=2, dim=-1)
        return output


class ProbV2(nn.Module):
    def __init__(
        self,
        traj_input_dim=3,
        rot_input_dim=4,
        d_model=256,
        nhead=8,
        num_layers=4,
        traj_latent_dim=64,
        rot_latent_dim=32,
        num_samples=10,
        T_in=30,
        T_out=30,
        dropout=0.1,
    ):
        super(ProbV2, self).__init__()

        self.num_samples = num_samples
        self.traj_latent_dim = traj_latent_dim
        self.rot_latent_dim = rot_latent_dim

        # Trajectory encoder
        self.traj_encoder = TransformerEncoder(traj_input_dim, d_model, nhead, num_layers, dropout)

        # Rotation encoder
        self.rot_encoder = TransformerEncoder(rot_input_dim, d_model, nhead, num_layers, dropout)

        # Cross-attention mechanisms
        self.traj_to_rot_attention = CrossAttention(d_model, nhead, dropout)
        self.rot_to_traj_attention = CrossAttention(d_model, nhead, dropout)

        # Latent spaces
        self.traj_latent = LatentSpace(d_model, traj_latent_dim)
        self.rot_latent = LatentSpace(d_model, rot_latent_dim)

        # Decoders
        self.traj_decoder = ConditionedDecoder(traj_latent_dim, rot_latent_dim, d_model, traj_input_dim, T_out)
        self.rot_decoder = QuaternionDecoder(rot_latent_dim, traj_latent_dim, d_model, T_out)

        self.T_in = T_in
        self.T_out = T_out

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using best practices for this kind of architecture"""
        # Xavier uniform for general weights
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # Orthogonal initialization for recurrent weights (better for GRUs)
        for name, param in self.named_parameters():
            if "gru" in name and "weight" in name:
                nn.init.orthogonal_(param)

        # Special initialization for the latent space projections
        for m in [self.traj_latent.fc_mu, self.traj_latent.fc_logvar, self.rot_latent.fc_mu, self.rot_latent.fc_logvar]:
            nn.init.orthogonal_(m.weight)
            nn.init.zeros_(m.bias)

        # Special initialization for linear projections
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, trajectory_input, rotation_input):
        """
        Forward pass through the model.

        Args:
            trajectory_input: Tensor of shape [B, T, 3] containing trajectory inputs
            rotation_input: Tensor of shape [B, T, 4] containing rotation inputs (quaternions)

        Returns:
            Dictionary containing:
            - trajectory_samples: Tensor of shape [N, B, T, 3]
            - rotation_samples: Tensor of shape [N, B, T, 4]
            - traj_mu, traj_logvar: trajectory latent parameters
            - rot_mu, rot_logvar: rotation latent parameters
        """
        # Encode inputs
        traj_features = self.traj_encoder(trajectory_input)  # [B, T, d_model]
        rot_features = self.rot_encoder(rotation_input)  # [B, T, d_model]

        # Apply cross-attention to condition each on the other
        traj_conditioned = self.rot_to_traj_attention(traj_features, rot_features)
        rot_conditioned = self.traj_to_rot_attention(rot_features, traj_features)

        # Convert to latent spaces
        traj_mu, traj_logvar = self.traj_latent(traj_conditioned)
        rot_mu, rot_logvar = self.rot_latent(rot_conditioned)

        # Sample from latent spaces
        traj_z = self.traj_latent.sample(traj_mu, traj_logvar, self.num_samples)  # [N, B, traj_latent_dim]
        rot_z = self.rot_latent.sample(rot_mu, rot_logvar, self.num_samples)  # [N, B, rot_latent_dim]

        # Decode to get trajectory and rotation samples, conditioning each on the other
        trajectory_samples = self.traj_decoder(traj_z, rot_z)  # [N, B, T, 3]
        rotation_samples = self.rot_decoder(rot_z, traj_z)  # [N, B, T, 4]

        return {
            "trajectory_samples": trajectory_samples,  # [N, B, T, 3]
            "rotation_samples": rotation_samples,  # [N, B, T, 4]
            "traj_mu": traj_mu,
            "traj_logvar": traj_logvar,
            "rot_mu": rot_mu,
            "rot_logvar": rot_logvar,
        }


class ProbV2Loss(nn.Module):
    def __init__(
        self,
        kl_weight=0.05,  # KL regularization (reduced from 0.1)
        traj_recon_weight=2.0,  # Trajectory reconstruction (increased)
        rot_recon_weight=2.0,  # Rotation reconstruction (increased)
        consistency_weight=0.3,  # Cross-modal consistency (reduced slightly)
        ade_weight=2.0,  # Average Displacement Error
        fde_weight=3.0,  # Final Displacement Error
        diversity_weight=0.1,
    ):  # Sample diversity
        super(ProbV2Loss, self).__init__()
        self.kl_weight = kl_weight
        self.traj_recon_weight = traj_recon_weight
        self.rot_recon_weight = rot_recon_weight
        self.consistency_weight = consistency_weight
        self.ade_weight = ade_weight
        self.fde_weight = fde_weight
        self.diversity_weight = diversity_weight

    def quaternion_distance(self, q1, q2):
        """
        Calculate the distance between two quaternions.
        The quaternion distance is calculated as 1 - |dot(q1, q2)|
        """
        # Ensure q1 and q2 are normalized
        q1 = F.normalize(q1, p=2, dim=-1)
        q2 = F.normalize(q2, p=2, dim=-1)

        # Calculate dot product and take absolute value
        # (Since q and -q represent the same rotation)
        dot_product = torch.sum(q1 * q2, dim=-1)
        dot_product_abs = torch.abs(dot_product)

        # Clamp to avoid numerical issues
        dot_product_abs = torch.clamp(dot_product_abs, 0, 1)

        # Calculate distance
        distance = 1 - dot_product_abs

        return distance

    def kl_divergence_loss(self, mu, logvar):
        """
        Calculate KL divergence loss between the latent distribution and a standard Gaussian.
        """
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return kl_loss

    def sample_consistency_loss(self, traj_samples, rot_samples):
        """
        Calculate consistency loss to ensure that trajectory and rotation samples are coherent.
        Uses the correlation between pairwise distances in both spaces.
        """
        N, B, T, _ = traj_samples.shape

        # Flatten samples for pairwise distance computation
        traj_flat = traj_samples.reshape(N * B, -1)  # [N*B, T*3]

        # Calculate pairwise distances within the same batch for trajectories
        traj_pdist = torch.cdist(traj_flat, traj_flat)  # [N*B, N*B]

        # Reshape rotation samples for vectorized quaternion distance computation
        rot_reshaped = rot_samples.reshape(N * B, T, 4)  # [N*B, T, 4]

        # Normalize all quaternions
        rot_normalized = F.normalize(rot_reshaped, p=2, dim=-1)  # [N*B, T, 4]

        # Prepare for broadcasting:
        # q1 shape: [N*B, 1, T, 4] - will broadcast to [N*B, N*B, T, 4]
        # q2 shape: [1, N*B, T, 4] - will broadcast to [N*B, N*B, T, 4]
        q1 = rot_normalized.unsqueeze(1)  # [N*B, 1, T, 4]
        q2 = rot_normalized.unsqueeze(0)  # [1, N*B, T, 4]

        # Compute dot product between all pairs of quaternions
        # This gives a tensor of shape [N*B, N*B, T]
        dot_product = torch.sum(q1 * q2, dim=-1)  # [N*B, N*B, T]

        # Take absolute value (since q and -q represent the same rotation)
        dot_product_abs = torch.abs(dot_product)

        # Clamp to avoid numerical issues
        dot_product_abs = torch.clamp(dot_product_abs, 0, 1)

        # Calculate quaternion distance for all pairs
        quat_dist = 1 - dot_product_abs  # [N*B, N*B, T]

        # Average over time steps
        rot_pdist = quat_dist.mean(dim=-1)  # [N*B, N*B]

        # Create mask to only compare samples from the same input batch
        batch_idx = torch.arange(B, device=traj_samples.device).repeat_interleave(N)
        mask = (batch_idx.unsqueeze(1) == batch_idx.unsqueeze(0)).float()

        # Remove self-comparisons
        self_mask = torch.eye(N * B, device=traj_samples.device)
        mask = mask * (1 - self_mask)

        # Normalize distance matrices to [0, 1] range for meaningful comparison
        traj_pdist_norm = traj_pdist / (traj_pdist.max() + 1e-8)
        rot_pdist_norm = rot_pdist / (rot_pdist.max() + 1e-8)

        # Calculate consistency loss as the absolute difference between normalized distances
        diff = (traj_pdist_norm - rot_pdist_norm).abs()
        consistency = (diff * mask).sum() / (mask.sum() + 1e-8)

        return consistency

    def diversity_loss(self, trajectory_samples):
        """
        Encourage diversity among trajectory samples by maximizing pairwise distances
        """
        N, B, T, D = trajectory_samples.shape

        # Reshape for pairwise distance computation
        traj_flat = trajectory_samples.view(N, B, -1)  # [N, B, T*D]

        # Calculate batch-wise diversity (only compare samples within same batch)
        diversity = 0.0

        for b in range(B):
            # Extract samples for this batch element
            batch_samples = traj_flat[:, b, :]  # [N, T*D]

            # Calculate pairwise distances between samples
            pdist = torch.cdist(batch_samples, batch_samples)  # [N, N]

            # Remove self-comparisons by zeroing diagonal
            mask = 1.0 - torch.eye(N, device=pdist.device)
            masked_pdist = pdist * mask

            # Calculate diversity as mean of pairwise distances
            if mask.sum() > 0:
                batch_diversity = masked_pdist.sum() / mask.sum()
                diversity += batch_diversity

        # Average diversity across all batches
        diversity = diversity / B

        # Return negative diversity to be minimized
        return -diversity

    def forward(self, model_output, y, y_rotations):
        """
        Calculate loss for the probabilistic trajectory and rotation forecasting model.

        Args:
            model_output: Dictionary containing model outputs
            y: Tensor of shape [B, T, 3] containing ground truth trajectories
            y_rotations: Tensor of shape [B, T, 4] containing ground truth rotations

        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary containing individual loss components
        """
        trajectory_samples = model_output["trajectory_samples"]  # [N, B, T, 3]
        rotation_samples = model_output["rotation_samples"]  # [N, B, T, 4]
        traj_mu = model_output["traj_mu"]  # [B, traj_latent_dim]
        traj_logvar = model_output["traj_logvar"]  # [B, traj_latent_dim]
        rot_mu = model_output["rot_mu"]  # [B, rot_latent_dim]
        rot_logvar = model_output["rot_logvar"]  # [B, rot_latent_dim]

        # Calculate KL divergence losses for both latent spaces
        traj_kl_loss = self.kl_divergence_loss(traj_mu, traj_logvar)
        rot_kl_loss = self.kl_divergence_loss(rot_mu, rot_logvar)
        kl_loss = traj_kl_loss + rot_kl_loss

        # Calculate reconstruction losses using all samples
        N, B, T, _ = trajectory_samples.shape

        # Expand ground truth to match samples shape
        expanded_y = y.unsqueeze(0).expand(N, -1, -1, -1)  # [N, B, T, 3]
        expanded_y_rotations = y_rotations.unsqueeze(0).expand(N, -1, -1, -1)  # [N, B, T, 4]

        # MSE for trajectory reconstruction
        traj_recon_loss = F.mse_loss(trajectory_samples, expanded_y, reduction="none").sum(dim=-1).mean()

        # Quaternion distance for rotation reconstruction
        rot_recon_loss = self.quaternion_distance(rotation_samples, expanded_y_rotations).mean()

        # Calculate consistency loss
        consistency_loss = self.sample_consistency_loss(trajectory_samples, rotation_samples)

        # Calculate ADE (average over all timesteps)
        ade_loss = torch.mean(torch.norm(trajectory_samples - expanded_y, dim=-1))

        # Calculate FDE (only final timestep)
        fde_loss = torch.mean(torch.norm(trajectory_samples[:, :, -1, :] - y[:, -1, :].unsqueeze(0), dim=-1))

        # Calculate diversity loss to encourage diverse samples
        div_loss = self.diversity_loss(trajectory_samples)

        # Combine all losses with weights
        total_loss = (
            self.kl_weight * kl_loss
            + self.traj_recon_weight * traj_recon_loss
            + self.rot_recon_weight * rot_recon_loss
            + self.consistency_weight * consistency_loss
            + self.ade_weight * ade_loss
            + self.fde_weight * fde_loss
            + self.diversity_weight * div_loss
        )

        # Create loss dictionary for monitoring
        loss_dict = {
            "total_loss": total_loss,
            "traj_kl_loss": traj_kl_loss,
            "rot_kl_loss": rot_kl_loss,
            "traj_recon_loss": traj_recon_loss,
            "rot_recon_loss": rot_recon_loss,
            "consistency_loss": consistency_loss,
            "ade_loss": ade_loss,
            "fde_loss": fde_loss,
            "diversity_loss": div_loss,
        }

        return total_loss, loss_dict


def __test__():
    import time

    model = ProbV2(T_out=30)
    loss_fn = ProbV2Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    optimizer.zero_grad()
    x_train = torch.randn(32, 30, 3)
    x_rot_train = torch.randn(32, 30, 4)
    y_train = torch.randn(32, 30, 3)

    start_time = time.time()
    out = model(x_train, x_rot_train)
    loss, _ = loss_fn(out, y_train, x_rot_train)
    loss.backward()
    optimizer.step()
    end_time = time.time()

    print(f"Loss: {loss.item():.4f}, Time taken: {end_time - start_time:.4f} seconds")


def __fps_test__():
    model = ProbV2(T_in=30, T_out=30)
    timings = []

    model = model.to("cuda")
    model.eval()
    xin = torch.randn(1, 30, 3).to("cuda")
    xrot = torch.randn(1, 30, 4).to("cuda")
    model(xin, xrot)  # Warmup

    with torch.no_grad():
        for _ in range(100):
            xin = torch.randn(1, 30, 3).to("cuda")
            xrot = torch.randn(1, 30, 4).to("cuda")
            start_time = time.time()
            out = model(xin, xrot)
            end_time = time.time()
            timings.append(end_time - start_time)

    print(f"Mean inference time: {sum(timings) / len(timings) * 1000:.4f} ms (FPS: {1 / (sum(timings) / len(timings)):.2f})")


if __name__ == "__main__":
    __test__()
    __fps_test__()
