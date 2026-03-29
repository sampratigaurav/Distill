"""
PyTorch models for the Universal Data Sanitization API.

Two anomaly-detection architectures that accept a dynamic `input_dim`
so they can work with ANY tabular or embedding dataset.
"""

import torch
import torch.nn as nn


class DynamicAutoencoder(nn.Module):
    """
    Symmetric autoencoder whose layer sizes are derived from `input_dim`.

    Architecture (encoder → bottleneck → decoder):
        input_dim → h1 → h2 → bottleneck → h2 → h1 → input_dim

    Anomalies are detected via high reconstruction error.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim

        # Derive hidden sizes relative to input dimensionality.
        # Ensures the network scales gracefully from 4-feature CSVs
        # to 2048-dim image embeddings.
        h1 = max(input_dim // 2, 4)
        h2 = max(input_dim // 4, 2)
        bottleneck = max(input_dim // 8, 1)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(inplace=True),
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(inplace=True),
            nn.Linear(h2, bottleneck),
        )

        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(inplace=True),
            nn.Linear(h2, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(inplace=True),
            nn.Linear(h1, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the reconstruction of `x`."""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample MSE reconstruction error (no reduction)."""
        recon = self.forward(x)
        return torch.mean((x - recon) ** 2, dim=1)


class DynamicDeepSVDD(nn.Module):
    """
    Deep Support Vector Data Description (Deep SVDD).

    Maps inputs into a compact hypersphere in latent space.  Points
    that land far from the learned center `c` are flagged as anomalies.

    Architecture:
        input_dim → h1 → h2 → latent_dim

    Reference: Ruff et al., "Deep One-Class Classification" (ICML 2018)
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim

        h1 = max(input_dim // 2, 4)
        h2 = max(input_dim // 4, 2)
        latent_dim = max(input_dim // 8, 1)

        self.network = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(inplace=True),
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(inplace=True),
            nn.Linear(h2, latent_dim),
        )

        # Center of the hypersphere — set after a forward pass on
        # initial training data via `initialize_center()`.
        self.register_buffer("center", torch.zeros(latent_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project `x` into the latent space."""
        return self.network(x)

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Squared L2 distance from each sample's embedding to `center`."""
        latent = self.forward(x)
        return torch.sum((latent - self.center) ** 2, dim=1)

    @torch.no_grad()
    def initialize_center(self, dataloader: torch.utils.data.DataLoader) -> None:
        """
        Compute the initial hypersphere center `c` as the mean of all
        network outputs on the provided data.  Should be called once
        before training begins.
        """
        embeddings = []
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            embeddings.append(self.forward(batch))

        all_embeddings = torch.cat(embeddings, dim=0)
        self.center = torch.mean(all_embeddings, dim=0)
