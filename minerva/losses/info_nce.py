
import torch
from torch.nn.modules.loss import _Loss
from minerva.losses._functional import info_nce


class InfoNCELoss(_Loss):
    """
    A constrastive loss class for the 'Contrastive Predictive Coding'
    self-supervised learning technique

    References
    ----------
    Aaron van den Oord, Yazhe Li, Oriol Vinyals.
    "Representation Learning with Contrastive Predictive Coding", 2019
    """

    def __init__(self, temperature: float = 0.07):
        """
        Initialize the InfoNCELoss

        Parameters
        ----------
        temperature : float
            A denominator applied after computing cosine similarity.
            Defaults to 0.07
        """
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, query: torch.Tensor, positive: torch.Tensor):
        """
        Computes the InfoNCE loss using negatives from other entries in the batch.
        
        Parameters
        ----------
        query : torch.Tensor
            The query vectors of shape (batch_size, embedding_dim).
        positive : torch.Tensor
            The positive vectors of shape (batch_size, embedding_dim).
        
        Returns
        -------
        torch.Tensor
            The computed InfoNCE loss.
        """
        return info_nce(query, positive, self.temperature)