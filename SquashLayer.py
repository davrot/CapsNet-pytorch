import torch


class SquashLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        lengths2 = (input**2).sum(dim=2)
        return input * (lengths2 / (1 + lengths2) / lengths2.sqrt()).unsqueeze(-1)
