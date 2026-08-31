import torch
import torch.nn as nn

from models.geographical_attention import GeographicalAttention


class SpatialEncoder(nn.Module):
    """
    Encodeur spatial du modèle S2-LSTM.

    Pipeline :

        7 caractéristiques
              ↓
             MLP
              ↓
            LSTM
              ↓
    Attention géographique
              ↓
        h_spatial

    Entrées :
        x :
            (batch, sequence_length, 7)

        locations :
            (batch, sequence_length, 2)

    Sorties :
        h_spatial :
            (batch, hidden_size)

        attention_weights :
            (batch, sequence_length - 1)
    """

    def __init__(
        self,
        input_size=7,
        embedding_size=64,
        hidden_size=128
    ):

        super().__init__()

        # ==========================================
        # 1. Projection spatiale
        # ==========================================

        self.mlp = nn.Sequential(

            nn.Linear(
                input_size,
                embedding_size
            ),

            nn.ReLU(),

            nn.Linear(
                embedding_size,
                embedding_size
            )
        )

        # ==========================================
        # 2. LSTM spatial
        # ==========================================

        self.lstm = nn.LSTM(

            input_size=embedding_size,

            hidden_size=hidden_size,

            batch_first=True
        )

        # ==========================================
        # 3. Attention géographique
        # ==========================================

        self.geographical_attention = (
            GeographicalAttention(
                hidden_size=hidden_size
            )
        )

    def forward(self, x, locations):

        # ==========================================
        # 1. MLP
        # ==========================================

        spatial_embedding = self.mlp(x)

        # (batch, sequence_length, 64)


        # ==========================================
        # 2. LSTM
        # ==========================================

        hidden_states, _ = self.lstm(
            spatial_embedding
        )

        # (batch, sequence_length, 128)


        # ==========================================
        # 3. Attention géographique
        # ==========================================

        h_spatial, attention_weights = (
            self.geographical_attention(
                hidden_states,
                locations
            )
        )

        # h_spatial :
        # (batch, 128)

        return h_spatial, attention_weights