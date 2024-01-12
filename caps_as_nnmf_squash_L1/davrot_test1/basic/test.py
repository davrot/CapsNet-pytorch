import torch

from MarginLoss import MarginLoss


def test(
    model: torch.nn.Sequential,
    test_loader: torch.utils.data.DataLoader,
    torch_device: torch.device,
    with_reconstruction: bool,
    mask_layer_position: int,
    probability_layer_position: int,
    loss_fn: MarginLoss,
    reconstruction_alpha: float,
    mask_reconstruction: bool,
) -> tuple[float, float]:
    with torch.no_grad():
        model.eval()
        test_loss: float = 0
        correct: int = 0
        for data, target in test_loader:
            data = data.to(torch_device)
            target = target.to(torch_device)

            if with_reconstruction:
                model[mask_layer_position].enable = mask_reconstruction
                model[mask_layer_position].target = target.clone()

            output = model(data)
            probability = model[probability_layer_position].probability
            assert probability is not None
            test_loss += loss_fn(probability, target, size_average=False).item()

            if with_reconstruction:
                test_loss += (
                    reconstruction_alpha
                    * torch.nn.functional.mse_loss(
                        output,
                        data.flatten(start_dim=1, end_dim=-1),
                        reduction="sum",
                    ).item()
                )

            correct += int(
                (probability.argmax(dim=1).cpu() == target.cpu()).sum().item()
            )

        test_loss /= len(test_loader.dataset)
        print(
            f"\nTest set: Average loss: {test_loss:.4f}, "
            f"Accuracy: {correct}/{len(test_loader.dataset)} "
            f"({100.0 * correct / len(test_loader.dataset):.0f}%)\n"
        )

        test_error = 100.0 - float(100.0 * correct / len(test_loader.dataset))
        return test_loss, test_error
