"""Functional API for losses.
"""

import torch
import torch.nn.functional as F


# Borrowed from https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/losses/_functional.py
def dice_score(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    smooth: float = 0.0,
    eps: float = 1e-7,
    dims=None,
) -> torch.Tensor:
    assert y_hat.size() == y.size()
    if dims is not None:
        intersection = torch.sum(y_hat * y, dim=dims)
        cardinality = torch.sum(y_hat + y, dim=dims)
    else:
        intersection = torch.sum(y_hat * y)
        cardinality = torch.sum(y_hat + y)
    dice_score = (2.0 * intersection + smooth) / (cardinality + smooth).clamp_min(eps)
    return dice_score


def info_nce(query: torch.Tensor, positive: torch.Tensor, temperature: float):
    """
    Computes the InfoNCE loss using negatives from other entries in the batch.

    Parameters
    ----------
    query : torch.Tensor
        The query vectors of shape (batch_size, embedding_dim).
    positive : torch.Tensor
        The positive vectors of shape (batch_size, embedding_dim).
    temperature : float
        A denominator applied after computing cosine similarity.

    Returns
    -------
    torch.Tensor
        The computed InfoNCE loss.
    """

    batch_size = query.shape[0]
    device = query.device

    # Cosine similarity
    query = F.normalize(query, dim=-1)
    positive = F.normalize(positive, dim=-1)
    similarity = torch.matmul(query, positive.T) / temperature

    # Create labels
    labels = torch.arange(batch_size, device=device)

    # Compute loss
    return F.cross_entropy(similarity, labels)
