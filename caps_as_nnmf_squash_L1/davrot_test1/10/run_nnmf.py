import os


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

from make_parser import make_parser

tb = SummaryWriter()
experiment_name = tb.get_logdir().split("/")[-1]

torch.set_default_dtype(torch.float32)
use_gpu: bool = torch.cuda.is_available()
torch_device: torch.device = (
    torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
)

parser = make_parser()
args = parser.parse_args()

# -------


torch.manual_seed(args.seed)
if use_gpu:
    torch.cuda.manual_seed(args.seed)

train_loader, test_loader = data_loader(
    dataset=args.dataset,
    batch_size=args.batch_size,
    test_batch_size=args.test_batch_size,
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
    nnmf_params = [
        p
        for n, p in model.named_parameters()
        if n.endswith("weights") or isinstance(p, NonNegativeParameter)
    ]
    other_params = [
        p
        for n, p in model.named_parameters()
        if not n.endswith("weights") and not isinstance(p, NonNegativeParameter)
    ]
    optimizer = Madam(
        params=[
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
        ]
    )
    print(
        f"Using Madam optimizer on {len(nnmf_params)} nnmf parameters and {len(other_params)} other parameters."
    )
else:
    raise ValueError(f"Unknown optimizer: {args.optimizer}")

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
            f"{args.dataset}_{epoch:03d}_nnmf_model_dict_{args.nnmf_number_of_iterations}_iter{'_reconstruction' if args.with_reconstruction else ''}.pth",
        ),
    )

tb.close()
