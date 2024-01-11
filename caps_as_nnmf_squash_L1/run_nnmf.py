import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from torch.utils.tensorboard import SummaryWriter

import torch

from data_loader import data_loader
from network_nnmf import network
from MarginLoss import MarginLoss

from train_nnmf import train
from test_nnmf import test

import os

tb = SummaryWriter()

torch.set_default_dtype(torch.float32)
use_gpu: bool = torch.cuda.is_available()
torch_device: torch.device = (
    torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
)

# Parameter
batch_size: int = 128
test_batch_size: int = 100
epochs: int = 250
lr: float = 0.001
seed: int = 1
log_interval: int = 10
routing_iterations: int = 10

with_reconstruction: bool = False
mask_reconstruction: bool = True
reconstruction_alpha: float = 0.0005

m_pos: float = 0.9
m_neg: float = 0.1
lambda_: float = 0.5
patience: int = 15
min_lr: float = 1e-6
model_path: str = "model"


nnmf_number_of_iterations: int = 5
nnmf_threshold: float = 0.00001
nnmf_weight_noise_range: list[float] = [0.0, 1.0]

# -------
# Network structure:

number_of_classes: int = 10
conv1_in_channels: int = 1
conv1_out_channels: int = 256
conv1_kernel_size: int = 9
conv1_stride: int = 1
conv2_kernel_size: int = 9
conv2_stride: int = 2
primary_caps_output_dim: int = 8
primary_caps_output_caps: int = 32
number_of_primary_caps_yx = 36  # 6 * 6
caps_layer_output_dim: int = 16
fc1_out_features: int = 512
fc2_out_features: int = 1024
fc3_out_features: int = 784
# -------

torch.manual_seed(seed)
if use_gpu:
    torch.cuda.manual_seed(seed)

train_loader, test_loader = data_loader(
    batch_size=batch_size, test_batch_size=test_batch_size
)

model, probability_layer_position, mask_layer_position, nnmf_layer_position = network(
    number_of_classes=number_of_classes,
    conv1_in_channels=conv1_in_channels,
    conv1_out_channels=conv1_out_channels,
    conv1_kernel_size=conv1_kernel_size,
    conv1_stride=conv1_stride,
    conv2_kernel_size=conv2_kernel_size,
    conv2_stride=conv2_stride,
    primary_caps_output_dim=primary_caps_output_dim,
    primary_caps_output_caps=primary_caps_output_caps,
    number_of_primary_caps_yx=number_of_primary_caps_yx,
    caps_layer_output_dim=caps_layer_output_dim,
    fc1_out_features=fc1_out_features,
    fc2_out_features=fc2_out_features,
    fc3_out_features=fc3_out_features,
    routing_iterations=routing_iterations,
    with_reconstruction=with_reconstruction,
    nnmf_number_of_iterations=nnmf_number_of_iterations,
    nnmf_threshold=nnmf_threshold,
    nnmf_weight_noise_range=nnmf_weight_noise_range,
)

model.to(torch_device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, verbose=True, patience=patience, min_lr=min_lr
)
loss_fn = MarginLoss(m_pos=m_pos, m_neg=m_neg, lambda_=lambda_)

for epoch in range(1, epochs + 1):
    train(
        epoch=epoch,
        model=model,
        train_loader=train_loader,
        torch_device=torch_device,
        with_reconstruction=with_reconstruction,
        mask_layer_position=mask_layer_position,
        probability_layer_position=probability_layer_position,
        optimizer=optimizer,
        loss_fn=loss_fn,
        reconstruction_alpha=reconstruction_alpha,
        log_interval=log_interval,
        nnmf_layer_position=nnmf_layer_position,
    )

    test_loss, test_error = test(
        model=model,
        test_loader=test_loader,
        torch_device=torch_device,
        with_reconstruction=with_reconstruction,
        mask_layer_position=mask_layer_position,
        probability_layer_position=probability_layer_position,
        loss_fn=loss_fn,
        reconstruction_alpha=reconstruction_alpha,
        mask_reconstruction=mask_reconstruction,
    )

    tb.add_scalar("loss", test_loss, epoch)
    tb.add_scalar("error", test_error, epoch)
    tb.flush()

    scheduler.step(test_loss)

    os.makedirs(model_path, exist_ok=True)

    torch.save(
        model.state_dict(),
        os.path.join(
            model_path,
            f"{epoch:03d}_model_dict_{routing_iterations}_routing_reconstruction{with_reconstruction}.pth",
        ),
    )

tb.close()
