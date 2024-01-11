import os
import argparse

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from torch.utils.tensorboard import SummaryWriter

import torch

from data_loader import data_loader
from network_nnmf import network
from MarginLoss import MarginLoss

from train_nnmf import train
from test_nnmf import test

from nnmf.optimizer import Madam
from nnmf.parameters import NonNegativeParameter

tb = SummaryWriter()
experiment_name = tb.get_logdir().split("/")[-1]

torch.set_default_dtype(torch.float32)
use_gpu: bool = torch.cuda.is_available()
torch_device: torch.device = (
    torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
)

# -------
parser = argparse.ArgumentParser(description="CapsNet NNMF")

parser.add_argument("--batch-size", type=int, default=128, help="batch size (default: 128)")
parser.add_argument("--test-batch-size", type=int, default=100, help="test batch size (default: 100)")
parser.add_argument("--epochs", type=int, default=100, help="number of epochs to train (default: 100)")
parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
parser.add_argument("--log-interval", type=int, default=10, help="how many batches to wait before logging training status")
# -------
# Network structure:
parser.add_argument("--number-of-classes", type=int, default=10, help="number of classes (default: 10)")
parser.add_argument("--conv1-in-channels", type=int, default=1, help="conv1 in channels (default: 1)")
parser.add_argument("--conv1-out-channels", type=int, default=256, help="conv1 out channels (default: 256)")
parser.add_argument("--conv1-kernel-size", type=int, default=9, help="conv1 kernel size (default: 9)")
parser.add_argument("--conv1-stride", type=int, default=1, help="conv1 stride (default: 1)")
parser.add_argument("--conv2-kernel-size", type=int, default=9, help="conv2 kernel size (default: 9)")
parser.add_argument("--conv2-stride", type=int, default=2, help="conv2 stride (default: 2)")
parser.add_argument("--primary-caps-output-dim", type=int, default=8, help="primary caps output dim (default: 8)")
parser.add_argument("--primary-caps-output-caps", type=int, default=32, help="primary caps output caps (default: 32)")
parser.add_argument("--number-of-primary-caps-yx", type=int, default=36, help="number of primary caps yx (default: 36)")
parser.add_argument("--caps-layer-output-dim", type=int, default=16, help="caps layer output dim (default: 16)")
parser.add_argument("--fc1-out-features", type=int, default=512, help="fc1 out features (default: 512)")
parser.add_argument("--fc2-out-features", type=int, default=1024, help="fc2 out features (default: 1024)")
parser.add_argument("--fc3-out-features", type=int, default=784, help="fc3 out features (default: 784)")
# reconstruction
parser.add_argument("--with-reconstruction", action="store_true", default=False, help="use reconstruction loss (default: False)")
parser.add_argument("--mask-reconstruction", action="store_true", default=True, help="mask reconstruction loss (default: True)")
parser.add_argument("--reconstruction-alpha", type=float, default=0.0005, help="reconstruction loss weight (default: 0.0005)")
# loss function
parser.add_argument("--m-pos", type=float, default=0.9, help="m+ parameter for margin loss (default: 0.9)")
parser.add_argument("--m-neg", type=float, default=0.1, help="m- parameter for margin loss (default: 0.1)")
parser.add_argument("--loss-lambda", type=float, default=0.5, help="lambda parameter for margin loss (default: 0.5)")
# optimizer
parser.add_argument("--optimizer", type=str, default="adam", help="optimizer (default: adam)", choices=["adam", "madam"])
parser.add_argument("--lr", type=float, default=0.001, help="learning rate (default: 0.001)")
parser.add_argument("--patience", type=int, default=15, help="patience for ReduceLROnPlateau (default: 15)")
parser.add_argument("--min-lr", type=float, default=1e-6, help="min lr for ReduceLROnPlateau (default: 1e-6)")
# nnmf
parser.add_argument("--nnmf-number-of-iterations", type=int, default=5, help="number of iterations for NNMF (default: 5)")
parser.add_argument("--nnmf-threshold", type=float, default=0.00001, help="threshold for NNMF (default: 0.00001)")
# model path
parser.add_argument("--model-path", type=str, default="models", help="path to save models (default: models)")
# -------
args = parser.parse_args()

torch.manual_seed(args.seed)
if use_gpu:
    torch.cuda.manual_seed(args.seed)

train_loader, test_loader = data_loader(
    batch_size=args.batch_size, test_batch_size=args.test_batch_size
)
model, probability_layer_position, mask_layer_position, nnmf_layer_position = network(
    number_of_classes=args.number_of_classes,
    conv1_in_channels=args.conv1_in_channels,
    conv1_out_channels=args.conv1_out_channels,
    conv1_kernel_size=args.conv1_kernel_size,
    conv1_stride=args.conv1_stride,
    conv2_kernel_size=args.conv2_kernel_size,
    conv2_stride=args.conv2_stride,
    primary_caps_output_dim=args.primary_caps_output_dim,
    primary_caps_output_caps=args.primary_caps_output_caps,
    number_of_primary_caps_yx=args.number_of_primary_caps_yx,
    caps_layer_output_dim=args.caps_layer_output_dim,
    fc1_out_features=args.fc1_out_features,
    fc2_out_features=args.fc2_out_features,
    fc3_out_features=args.fc3_out_features,
    with_reconstruction=args.with_reconstruction,
    nnmf_number_of_iterations=args.nnmf_number_of_iterations,
    nnmf_threshold=args.nnmf_threshold,
)

model.to(torch_device)

if args.optimizer == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print("Using Adam optimizer.")
elif args.optimizer == "madam":
    nnmf_params = [p for n, p in model.named_parameters() if n.endswith("weights") or isinstance(p, NonNegativeParameter)]
    other_params = [p for n, p in model.named_parameters() if not n.endswith("weights") and not isinstance(p, NonNegativeParameter)]
    optimizer = Madam(params=[
        {
            "params": nnmf_params,
            "lr": args.lr,
            "NNMF": True,
            "foreach": False,
        },
        {
            "params": other_params,
            "lr": args.lr,
        },
    ])
    print(f"Using Madam optimizer on {len(nnmf_params)} nnmf parameters and {len(other_params)} other parameters.")


scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, verbose=True, patience=args.patience, min_lr=args.min_lr
)
loss_fn = MarginLoss(m_pos=args.m_pos, m_neg=args.m_neg, lambda_=args.loss_lambda)

# save parameters in args to tensorboard
tb.add_hparams(vars(args), {})

print("Experiment name:", experiment_name)
for epoch in range(1, args.epochs + 1):
    train_loss = train(
        epoch=epoch,
        model=model,
        train_loader=train_loader,
        torch_device=torch_device,
        with_reconstruction=args.with_reconstruction,
        mask_layer_position=mask_layer_position,
        probability_layer_position=probability_layer_position,
        optimizer=optimizer,
        loss_fn=loss_fn,
        reconstruction_alpha=args.reconstruction_alpha,
        log_interval=args.log_interval,
        nnmf_layer_position=nnmf_layer_position,
    )

    test_loss, test_error = test(
        model=model,
        test_loader=test_loader,
        torch_device=torch_device,
        with_reconstruction=args.with_reconstruction,
        mask_layer_position=mask_layer_position,
        probability_layer_position=probability_layer_position,
        loss_fn=loss_fn,
        reconstruction_alpha=args.reconstruction_alpha,
        mask_reconstruction=args.mask_reconstruction,
    )

    tb.add_scalar("train_loss", train_loss, epoch)
    tb.add_scalar("loss", test_loss, epoch)
    tb.add_scalar("error", test_error, epoch)
    tb.flush()

    scheduler.step(test_loss)

    model_dir = os.path.join(args.model_path, experiment_name)
    os.makedirs(model_dir, exist_ok=True)

    torch.save(
        model.state_dict(),
        os.path.join(
            model_dir,
            f"{epoch:03d}_nnmf_model_dict_{args.nnmf_number_of_iterations}_iter_{'reconstruction' if args.with_reconstruction else ''}.pth",
        ),
    )

tb.close()
