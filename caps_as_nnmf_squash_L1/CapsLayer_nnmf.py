import torch


class CapsLayer(torch.nn.Module):
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
        super().__init__()
        self.input_dim: int = input_dim
        self.input_caps: int = input_caps
        self.output_dim: int = output_dim
        self.output_caps: int = output_caps
        self.threshold: float = threshold
        self.number_of_iterations: int = number_of_iterations

        self.weights: torch.nn.Parameter = torch.nn.Parameter(
            torch.Tensor(input_caps, output_caps, input_dim, output_dim)
        )

        torch.nn.init.uniform_(self.weights, a=0.0, b=1.0)

        self.norm_weights()

        self.functional_nnmf_sbs_bp = FunctionalNNMFSbSBP.apply

    @torch.no_grad()
    def norm_weights(self) -> None:
        assert self.threshold >= 0

        self.weights.data = self.weights.data / (
            self.weights.data.sum(dim=1, keepdim=True) + 1e-20
        )

        torch.clamp(
            self.weights.data,
            min=float(self.threshold),
            max=None,
            out=self.weights.data,
        )

        self.weights.data = self.weights.data / (
            self.weights.data.sum(dim=1, keepdim=True) + 1e-20
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input / (input.sum(dim=-1, keepdim=True) + 1e-20)

        h = self.functional_nnmf_sbs_bp(input, self.weights, self.number_of_iterations)

        # Routing
        reconstruct = (h.unsqueeze(-2) * self.weights).sum(
            -1
        )  # alpha: B, in_caps, out_caps, in_dim, --out_dim--
        alpha = (reconstruct * input.unsqueeze(-2)).sum(
            -1
        )  # alpha: B, in_caps, out_caps, --in_dim--
        alpha = alpha / (alpha.sum(dim=-1, keepdim=True) + 1e-20)

        output = (h * alpha.unsqueeze(-1)).sum(
            -3
        )  # output: B, --in_caps--, out_caps, out_dim

        return output


class FunctionalNNMFSbSBP(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore
        ctx, input: torch.Tensor, weights: torch.Tensor, number_of_iterations: int
    ) -> torch.Tensor:
        output = torch.full(
            (input.shape[0], input.shape[1], weights.shape[-3], weights.shape[-1]),
            1.0 / float(weights.shape[-1]),
            dtype=input.dtype,
            device=input.device,
        )

        for _ in range(0, number_of_iterations):
            h_w = output.unsqueeze(-2) * weights.unsqueeze(0)
            h_w = h_w / (h_w.sum(dim=-1, keepdim=True) + 1e-20)
            h_w = (h_w * input.unsqueeze(-2).unsqueeze(-1)).sum(
                dim=-2
            )  # input: B, in_caps, in_dim => B, in_caps, 1, in_dim, 1
            output = h_w / (
                h_w.sum(dim=-1, keepdim=True) + 1e-20
            )  # h: B, in_caps, out_caps, <out_dim>

        ctx.save_for_backward(
            input,
            weights,
            output,
        )

        return output

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> tuple[None | torch.Tensor, None | torch.Tensor, None]:
        # ##############################################
        # Get the variables back
        # ##############################################
        (
            input,
            weights,
            output,
        ) = ctx.saved_tensors

        # ##############################################
        # Default output
        # ##############################################
        grad_input: None | torch.Tensor = None
        grad_weights: None | torch.Tensor = None
        grad_number_of_iterations: None = None

        # #################################################
        # Calculate backprop error (grad_input)
        # #################################################
        backprop_r: torch.Tensor = weights.unsqueeze(0) * output.unsqueeze(-2)

        backprop_bigr: torch.Tensor = backprop_r.sum(dim=-1)

        backprop_z: torch.Tensor = backprop_r * (
            1.0 / (backprop_bigr + 1e-20)
        ).unsqueeze(-1)

        grad_input = (backprop_z * grad_output.unsqueeze(-2)).sum(-1).sum(-2)

        del backprop_z

        # #################################################
        # Backprop
        # #################################################
        backprop_f: torch.Tensor = output.unsqueeze(-2) * (
            input.unsqueeze(-2) / (backprop_bigr**2 + 1e-20)
        ).unsqueeze(-1)

        result_omega: torch.Tensor = backprop_bigr.unsqueeze(
            -1
        ) * grad_output.unsqueeze(-2)
        result_omega -= (backprop_r * grad_output.unsqueeze(-2)).sum(-1).unsqueeze(-1)
        result_omega *= backprop_f

        del backprop_f
        grad_weights = result_omega.sum(0)

        del result_omega
        del backprop_bigr
        del backprop_r

        return (
            grad_input,
            grad_weights,
            grad_number_of_iterations,
        )
