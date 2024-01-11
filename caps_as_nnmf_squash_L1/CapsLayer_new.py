import torch
import torch.nn.functional as F

from nnmf import NNMFLayer, COMPARISSON_TOLERANCE
from nnmf.parameters import NonNegativeParameter


class CapsLayer(NNMFLayer):
    input_dim: int
    input_caps: int
    output_dim: int
    output_caps: int
    weights: torch.nn.Parameter
    threshold: float
    number_of_iterations: int

    def __init__(
        self,
        input_caps: int,
        input_dim: int,
        output_caps: int,
        output_dim: int,
        threshold: float = 0.00001,
        number_of_iterations: int = 5,
    ) -> None:
        super().__init__(
            n_iterations=number_of_iterations,
            backward_method="fixed point",
            h_update_rate=1,
            sparsity_rate=1,
            keep_h=False,
            activate_secure_tensors=False,
        )
        self.input_dim: int = input_dim
        self.input_caps: int = input_caps
        self.output_dim: int = output_dim
        self.output_caps: int = output_caps
        self.threshold: float = threshold
        self.number_of_iterations: int = number_of_iterations

        self.weight: NonNegativeParameter = NonNegativeParameter(
            torch.Tensor(input_caps, output_caps, input_dim, output_dim)
        )

        # self.normalize_dim = -1
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.uniform_(self.weight, a=0, b=1)
        self.normalize_weights()

    @torch.no_grad()
    def normalize_weights(self) -> None:
        assert self.threshold >= 0

        weight_data = F.normalize(self.weight.data, p=1, dim=1) # May contain negative values if Madam not used

        weight_data= torch.clamp(
            weight_data,
            min=float(self.threshold),
            max=None,
            out=self.weight.data,
        )

        self.weight.data = F.normalize(self.weight.data, p=1, dim=1)

    def _reset_h(self, x):
        self.h = torch.full(
            (x.shape[0], self.input_caps, self.output_caps, self.output_dim),
            1.0 / float(self.output_dim),
            dtype=x.dtype,
            device=x.device,
        )

    def _reconstruct(self, h):
        return torch.einsum(
            "biof,iodf->bid", h, self.weight
        )  # b: batch, i: input_caps, o: output_caps, d: input_dim, f: output_dim

    def _forward(self, nnmf_update):
        return torch.einsum(
            "bid,iodf->biof", nnmf_update, self.weight
        )  # b: batch, i: input_caps, o: output_caps, d: input_dim, f: output_dim

    def _process_h(self, h):
        h = F.normalize(h, p=1, dim=-1)
        # apply sparsity
        # h = self.sparsity(F.relu(h))
        return h

    def _check_forward(self, input):
        assert self.weight.sum(1, keepdim=True).allclose(
            torch.ones_like(self.weight), atol=COMPARISSON_TOLERANCE
        ), self.weight.sum(1)
        assert (self.weight >= 0).all(), self.weight.min()
        assert (input >= 0).all(), input.min()

    def routing(self, x):
        reconstruct = self._reconstruct(self.h)
        alpha = torch.einsum("bid,bid->bi", reconstruct, x)
        # alpha = F.normalize(alpha, p=1, dim=-1)
        return alpha

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        super().forward(input)
        alpha = self.routing(input)
        output = torch.einsum("biof,bi->bof", self.h, alpha)
        output = F.normalize(output, p=1, dim=-1)
        return output

    def norm_weights(self) -> None:
        self.normalize_weights()