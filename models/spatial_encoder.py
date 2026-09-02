import torch
import torch.nn as nn

from models.geographical_attention import GeographicalAttention


class SpatialEncoder(nn.Module):
    """
    Encodeur spatial du modèle S2-LSTM.

    Entrée :
        x :
            (batch, sequence_length, 7)

        locations :
            (batch, sequence_length, 2)

    Sorties normales :
        h_spatial :
            (batch, 128)

        attention_weights :
            (batch, sequence_length - 1)

    Avec return_sequence=True :
        h_spatial :
            (batch, 128)

        H_spatial :
            (batch, sequence_length, 128)

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

        self.geographical_attention = GeographicalAttention(
            hidden_size=hidden_size
        )

    # ==========================================
    # FORWARD
    # ==========================================

    def forward(
        self,
        x,
        locations,
        return_sequence=False
    ):

        # ==========================================
        # 1. MLP
        # ==========================================

        spatial_embedding = self.mlp(x)

        # spatial_embedding :
        # (batch, sequence_length, 64)


        # ==========================================
        # 2. LSTM
        # ==========================================

        hidden_states, _ = self.lstm(
            spatial_embedding
        )

        # hidden_states :
        # (batch, sequence_length, 128)
        #
        # Cette variable représente H_spatial.


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
        #
        # attention_weights :
        # (batch, sequence_length - 1)


        # ==========================================
        # 4. Retour des résultats
        # ==========================================

        if return_sequence:

            return (
                h_spatial,
                hidden_states,
                attention_weights
            )

        return (
            h_spatial,
            attention_weights
        )