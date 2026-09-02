import torch
import torch.nn as nn


class CrossAttention(nn.Module):
    """
    Cross-Attention bidirectionnel entre les mémoires
    spatiale et sémantique.

    Entrées :
        H_spatial :
            (batch, sequence_length, hidden_size)

        H_semantic :
            (batch, sequence_length, hidden_size)

    Sorties :
        spatial_from_semantic :
            (batch, hidden_size)

        semantic_from_spatial :
            (batch, hidden_size)

        attention_s2m :
            (batch, 1, sequence_length - 1)

        attention_m2s :
            (batch, 1, sequence_length - 1)
    """

    def __init__(
        self,
        hidden_size=128,
        num_heads=4
    ):
        super().__init__()

        # ==========================================
        # Spatial → Semantic
        # ==========================================

        self.spatial_to_semantic = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True
        )

        # ==========================================
        # Semantic → Spatial
        # ==========================================

        self.semantic_to_spatial = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True
        )

    def forward(
        self,
        H_spatial,
        H_semantic
    ):

        # ==========================================
        # 1. Mémoire spatiale
        # ==========================================

        # Dernier état = requête
        spatial_query = H_spatial[:, -1:, :]

        # 9 états précédents = mémoire
        spatial_memory = H_spatial[:, :-1, :]


        # ==========================================
        # 2. Mémoire sémantique
        # ==========================================

        # Dernier état = requête
        semantic_query = H_semantic[:, -1:, :]

        # 9 états précédents = mémoire
        semantic_memory = H_semantic[:, :-1, :]


        # ==========================================
        # 3. Spatial → Semantic
        # ==========================================

        spatial_from_semantic, attention_s2m = (
            self.spatial_to_semantic(
                query=spatial_query,
                key=semantic_memory,
                value=semantic_memory
            )
        )


        # ==========================================
        # 4. Semantic → Spatial
        # ==========================================

        semantic_from_spatial, attention_m2s = (
            self.semantic_to_spatial(
                query=semantic_query,
                key=spatial_memory,
                value=spatial_memory
            )
        )


        # ==========================================
        # 5. Supprimer la dimension temporelle
        # ==========================================

        spatial_from_semantic = (
            spatial_from_semantic.squeeze(1)
        )

        semantic_from_spatial = (
            semantic_from_spatial.squeeze(1)
        )


        return (
            spatial_from_semantic,
            semantic_from_spatial,
            attention_s2m,
            attention_m2s
        )