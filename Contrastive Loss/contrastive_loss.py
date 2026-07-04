import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """Classic contrastive loss (Hadsell, Chopra & LeCun, 2006).

    Pulls same-class embedding pairs together, pushes different-class pairs
    apart up to `margin` (beyond which they incur zero penalty).

    label: 1.0 if the pair is from the same class, 0.0 otherwise.
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, emb1: torch.Tensor, emb2: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        dist = F.pairwise_distance(emb1, emb2)

        loss_similar = label * dist.pow(2)
        loss_dissimilar = (1 - label) * F.relu(self.margin - dist).pow(2)

        return 0.5 * (loss_similar + loss_dissimilar).mean()