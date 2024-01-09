import torch

from MarginLoss import MarginLoss


def train(
    epoch: int,
    model: torch.nn.Sequential,
    train_loader: torch.utils.data.DataLoader,
    torch_device: torch.device,
    with_reconstruction: bool,
    mask_layer_position: int,
    probability_layer_position: int,
    optimizer: torch.optim.Adam,
    loss_fn: MarginLoss,
    reconstruction_alpha: float,
    log_interval: int,
) -> None:
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(torch_device)
        target = target.to(torch_device)

        optimizer.zero_grad()

        if with_reconstruction:
            model[mask_layer_position].enable = True
            model[mask_layer_position].target = target.clone()

        output = model(data)
        probability = model[probability_layer_position].probability
        assert probability is not None
        loss = loss_fn(probability, target)

        if with_reconstruction:
            loss = loss + reconstruction_alpha * torch.nn.functional.mse_loss(
                output, data.flatten(start_dim=1, end_dim=-1)
            )

        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print(
                f"Train Epoch: {epoch} "
                f"[{batch_idx * len(data)}/{len(train_loader.dataset)} "
                f"({100.0 * batch_idx * len(data)/ len(train_loader.dataset):.0f}%)]"
                f"\tLoss: {loss.item():.6f}"
            )
