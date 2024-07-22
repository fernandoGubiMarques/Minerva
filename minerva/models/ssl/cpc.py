import lightning as L
import torch
from torch import nn, optim
from minerva.losses.infonce import InfoNCELoss
from typing import Tuple


class ContrastivePredictiveCoding(L.LightningModule):
    """
    Contrastive Predictive Coding (CPC) model for self-supervised learning.

    References
    ----------
    Aaron van den Oord, Yazhe Li, Oriol Vinyals.
    "Representation Learning with Contrastive Predictive Coding", 2019
    """

    def __init__(
        self,
        backbone: nn.Module,
        autoregressive: nn.Module,
        predictors: nn.ModuleList,
        learning_rate: float,
        min_context_size: int = 0,
    ):
        """
        Initialize the ConstrastivePredictiveCoding module. 

        Parameters
        ----------
        backbone : nn.Module
            The backbone network for feature extraction.
        autoregressive : nn.Module
            The autoregressive model for context representation.
        predictors : nn.ModuleList
            A list of predictor networks for future prediction.
        learning_rate : float
            The learning rate for optimization.
        min_context_size : int
            The minimum context size for the autoregressive model, by default 0.
        """
        super().__init__()

        self.backbone = backbone
        self.autoregressive = autoregressive
        self.predictors = predictors
        self.learning_rate = learning_rate
        self.min_context_size = min_context_size
        self.loss = InfoNCELoss()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            The input data.

        Returns
        -------
        torch.Tensor
            The predicted output and projected input.
        """
        z = self.backbone(x)
        sequence_length = z.shape[1]
        context_end = torch.randint(
            self.min_context_size, sequence_length - len(self.predictors), (1,)
        ).item()

        c = self.autoregressive(z[:, :context_end])
        if isinstance(c, tuple):
            c = c[0]
        c = c[:, -1]

        y_proj = z[:, context_end : context_end + len(self.predictors)]
        y_pred = torch.stack([pred(c) for pred in self.predictors], dim=1)

        return y_pred, y_proj

    def _single_step(
        self, batch: torch.Tensor, batch_idx: int, step_name: str
    ) -> torch.Tensor:
        """
        Perform a single training/validation/test step.

        Parameters
        ----------
        batch : torch.Tensor
            The input batch of data.
        batch_idx : int
            The index of the batch.
        step_name : str
            The name of the step (train, val, test).

        Returns
        -------
        torch.Tensor
            The loss value for the batch.
        """
        y_pred, y_proj = self(batch)
        loss = self.loss(y_pred, y_proj)
        self.log(f"{step_name}_loss", loss)
        return loss

    def training_step(self, batch: torch.Tensor, batch_index: int) -> torch.Tensor:
        return self._single_step(batch, batch_index, "train")

    def validation_step(self, batch: torch.Tensor, batch_index: int) -> torch.Tensor:
        return self._single_step(batch, batch_index, "val")

    def test_step(self, batch: torch.Tensor, batch_index: int) -> torch.Tensor:
        return self._single_step(batch, batch_index, "test")

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)
