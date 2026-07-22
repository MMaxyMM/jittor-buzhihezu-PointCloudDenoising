"""KNN-local Point Transformer encoder for diffusion ablations."""

import jittor as jt
from jittor import nn

from .feature import get_knn_idx


def _gather_neighbors(values, indices):
    return jt.stack(
        [values[batch_idx][indices[batch_idx]]
         for batch_idx in range(values.shape[0])],
        dim=0,
    )


class LocalPointTransformerBlock(nn.Module):
    def __init__(self, channels: int, condition_dim: int):
        super().__init__()
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)
        self.position = nn.Sequential(
            nn.Linear(3, channels),
            nn.ReLU(),
            nn.Linear(channels, channels),
        )
        self.attention = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Linear(channels, channels),
        )
        self.output = nn.Linear(channels, channels)
        self.film = nn.Linear(condition_dim, channels * 2)
        self.activation = nn.ReLU()

    def execute(self, features, geometry, neighbor_indices, condition):
        batch_size, num_points, channels = features.shape
        neighbors = _gather_neighbors(features, neighbor_indices)
        neighbor_geometry = _gather_neighbors(geometry, neighbor_indices)
        relative_position = (
            geometry.unsqueeze(2) - neighbor_geometry
        )
        position_embedding = self.position(
            relative_position.reshape(-1, 3)
        ).reshape(
            batch_size, num_points, neighbor_indices.shape[-1], channels
        )

        query = self.query(features.reshape(-1, channels)).reshape(
            batch_size, num_points, 1, channels
        )
        key = self.key(neighbors.reshape(-1, channels)).reshape(
            neighbors.shape
        )
        value = self.value(neighbors.reshape(-1, channels)).reshape(
            neighbors.shape
        )
        logits = self.attention(
            (query - key + position_embedding).reshape(-1, channels)
        ).reshape(neighbors.shape)
        weights = nn.softmax(logits, dim=2)
        aggregated = (
            weights * (value + position_embedding)
        ).sum(dim=2)
        output = self.output(
            aggregated.reshape(-1, channels)
        ).reshape(batch_size, num_points, channels)

        film = self.film(condition)
        gamma = film[:, :channels].reshape(batch_size, 1, channels)
        beta = film[:, channels:].reshape(batch_size, 1, channels)
        return self.activation(
            features + output * (1.0 + gamma) + beta
        )


class LocalPointTransformerEncoder(nn.Module):
    """Permutation-equivariant local attention with relative XYZ encoding."""

    def __init__(
        self,
        k: int,
        embedding_dim: int,
        condition_dim: int,
        input_dim: int = 6,
        num_blocks: int = 3,
    ):
        super().__init__()
        self.k = k
        self.embedding_dim = embedding_dim
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.ReLU(),
        )
        self.blocks = nn.ModuleList(
            [
                LocalPointTransformerBlock(
                    embedding_dim, condition_dim
                )
                for _ in range(num_blocks)
            ]
        )

    def execute(self, features, geometry, condition):
        batch_size, num_points, _ = features.shape
        projected = self.input_projection(
            features.reshape(-1, features.shape[-1])
        ).reshape(batch_size, num_points, self.embedding_dim)
        neighbor_indices = get_knn_idx(
            geometry, geometry, self.k + 1
        )[:, :, 1:]
        for block in self.blocks:
            projected = block(
                projected, geometry, neighbor_indices, condition
            )
        return projected
