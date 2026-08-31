import torch
import torch.nn as nn


class GeographicalAttention(nn.Module):
    """
    Attention géographique basée sur les distances GPS.

    Entrées :
        hidden_states :
            (batch, sequence_length, hidden_size)

        locations :
            (batch, sequence_length, 2)

    Sorties :
        context :
            (batch, hidden_size)

        attention_weights :
            (batch, sequence_length - 1)
    """

    def __init__(self, hidden_size):
        super().__init__()

        self.hidden_size = hidden_size

    def haversine_distance(self, locations1, locations2):
        """
        Calcule la distance Haversine entre deux positions GPS.

        locations1 :
            (batch, 2)

        locations2 :
            (batch, sequence_length, 2)

        Retour :
            distances :
                (batch, sequence_length)
                en kilomètres
        """

        R = 6371.0

        # Position 1
        lat1 = torch.deg2rad(locations1[:, 0])
        lon1 = torch.deg2rad(locations1[:, 1])

        # Positions historiques
        lat2 = torch.deg2rad(locations2[:, :, 0])
        lon2 = torch.deg2rad(locations2[:, :, 1])

        # Différences
        dlat = lat2 - lat1.unsqueeze(1)
        dlon = lon2 - lon1.unsqueeze(1)

        # Formule de Haversine
        a = (
            torch.sin(dlat / 2) ** 2
            +
            torch.cos(lat1).unsqueeze(1)
            * torch.cos(lat2)
            * torch.sin(dlon / 2) ** 2
        )

        c = 2 * torch.atan2(
            torch.sqrt(a),
            torch.sqrt(1 - a + 1e-8)
        )

        return R * c

    def forward(self, hidden_states, locations):

        # ==========================================
        # 1. Position courante
        # ==========================================

        current_location = locations[:, -1, :]

        # (batch, 2)


        # ==========================================
        # 2. Historique uniquement
        # ==========================================

        historical_locations = locations[:, :-1, :]

        historical_states = hidden_states[:, :-1, :]

        # historical_locations :
        # (batch, sequence_length - 1, 2)

        # historical_states :
        # (batch, sequence_length - 1, hidden_size)


        # ==========================================
        # 3. Distance Haversine
        # ==========================================

        distances = self.haversine_distance(
            current_location,
            historical_locations
        )

        # (batch, sequence_length - 1)


        # ==========================================
        # 4. Largeur de bande sigma
        # ==========================================

        sigma = distances.median(
            dim=1,
            keepdim=True
        ).values

        sigma = sigma.clamp(
            min=1e-3
        )


        # ==========================================
        # 5. Score géographique
        # ==========================================

        scores = torch.exp(
            -(distances ** 2)
            /
            (2 * sigma ** 2)
        )


        # ==========================================
        # 6. Normalisation
        # ==========================================

        attention_weights = scores / (
            scores.sum(
                dim=1,
                keepdim=True
            )
            + 1e-8
        )

        # (batch, sequence_length - 1)


        # ==========================================
        # 7. Représentation spatiale finale
        # ==========================================

        context = torch.sum(
            attention_weights.unsqueeze(-1)
            * historical_states,
            dim=1
        )

        # (batch, hidden_size)

        return context, attention_weights