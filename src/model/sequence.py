"""Bidirectional LSTM sequence model for CRNN."""

import torch
import torch.nn as nn


class BidirectionalLSTM(nn.Module):
    """Single bidirectional LSTM layer with a linear projection."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, input_size)
        Returns:
            (B, T, output_size)
        """
        output, _ = self.lstm(x)  # (B, T, hidden*2)
        output = self.linear(output)  # (B, T, output_size)
        return output


class SequenceModel(nn.Module):
    """Stacked BiLSTM sequence model.

    Takes feature sequences from the backbone and models inter-character
    dependencies for CTC decoding.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.output_size = hidden_size * 2  # bidirectional

        layers = []
        for i in range(num_layers):
            in_sz = input_size if i == 0 else hidden_size * 2
            out_sz = hidden_size * 2  # intermediate projection
            layers.append(BidirectionalLSTM(in_sz, hidden_size, out_sz))
            if dropout > 0 and i < num_layers - 1:
                layers.append(nn.Dropout(dropout))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, input_size) feature sequence

        Returns:
            (B, T, hidden_size * 2) contextualized sequence
        """
        return self.layers(x)
