import torch
import torch.nn as nn


class SemanticEncoder(nn.Module):
    """
    Encodeur sémantique basé sur un MLP suivi d'un LSTM.

    Entrée :
        x :
            (batch, sequence_length, 5)

    Sortie par défaut :
        h_semantic :
            (batch, 128)

    Si return_sequence=True :
        h_semantic :
            (batch, 128)

        outputs :
            (batch, sequence_length, 128)
    """

    def __init__(
        self,
        input_size=5,
        embedding_size=64,
        hidden_size=128
    ):
        super().__init__()

        # Projection des features sémantiques
        self.embedding = nn.Sequential(
            nn.Linear(input_size, embedding_size),
            nn.ReLU()
        )

        # Encodage temporel
        self.lstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            batch_first=True
        )

    def forward(self, x, return_sequence=False):
        # Projection des données
        x = self.embedding(x)

        # LSTM
        outputs, (h_n, c_n) = self.lstm(x)

        # Dernier état caché
        h_semantic = h_n[-1]

        # Si on veut également récupérer toute la séquence
        if return_sequence:
            return h_semantic, outputs

        return h_semantic