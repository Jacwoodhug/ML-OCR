"""Full CRNN model: Backbone -> Sequence Model -> CTC Head."""

import torch
import torch.nn as nn

from src.model.backbone import MobileNetV3Backbone
from src.model.sequence import SequenceModel


class CRNN(nn.Module):
    """Convolutional Recurrent Neural Network for text recognition.

    Architecture:
        MobileNetV3-Small backbone (stride 4) -> BiLSTM -> Linear -> LogSoftmax
        Trained with CTC loss.
    """

    def __init__(
        self,
        num_classes: int,
        backbone_pretrained: bool = True,
        lstm_hidden_size: int = 256,
        lstm_num_layers: int = 2,
        lstm_dropout: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes

        # CNN backbone: (B, 3, H, W) -> (B, C, 1, W/4)
        self.backbone = MobileNetV3Backbone(pretrained=backbone_pretrained)
        backbone_out = self.backbone.out_channels

        # Sequence model: (B, T, C) -> (B, T, hidden*2)
        self.sequence = SequenceModel(
            input_size=backbone_out,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            dropout=lstm_dropout,
        )

        # CTC head: (B, T, hidden*2) -> (B, T, num_classes)
        self.head = nn.Linear(lstm_hidden_size * 2, num_classes)

        # Stride of the backbone (used to compute CTC input lengths)
        self.stride = 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, 3, H, W) input image tensor

        Returns:
            (T, B, num_classes) log-probabilities for CTC loss
            where T = W // stride
        """
        # Backbone: (B, 3, H, W) -> (B, C, 1, W')
        features = self.backbone(x)

        # Reshape: (B, C, 1, W') -> (B, W', C)
        b, c, h, w = features.shape
        features = features.squeeze(2)     # (B, C, W')
        features = features.permute(0, 2, 1)  # (B, W', C)

        # Sequence model: (B, T, C) -> (B, T, hidden*2)
        seq_out = self.sequence(features)

        # CTC head: (B, T, hidden*2) -> (B, T, num_classes)
        logits = self.head(seq_out)

        # CTC loss expects (T, B, C)
        log_probs = logits.permute(1, 0, 2).log_softmax(2)

        return log_probs

    def compute_input_lengths(self, image_widths: torch.Tensor) -> torch.Tensor:
        """Compute the number of CTC timesteps for each image in a batch.

        Args:
            image_widths: (B,) tensor of original (padded) image widths in pixels

        Returns:
            (B,) tensor of sequence lengths after backbone stride
        """
        return image_widths // self.stride
