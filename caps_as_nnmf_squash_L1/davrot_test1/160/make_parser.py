import argparse


def make_parser():
    parser = argparse.ArgumentParser(description="CapsNet NNMF")

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="number of epochs to train (default: 100)",
    )
    parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="how many batches to wait before logging training status",
    )
    # -------
    # Dataset:
    parser.add_argument(
        "--dataset",
        type=str,
        default="fashion-mnist",
        help="dataset (default: mnist)",
        choices=["mnist", "fashion-mnist", "cifar10"],
    )
    parser.add_argument(
        "--batch-size", type=int, default=128, help="batch size (default: 128)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=100,
        help="test batch size (default: 100)",
    )
    # -------
    # Network structure:
    parser.add_argument(
        "--number-of-classes",
        type=int,
        default=10,
        help="number of classes (default: 10)",
    )
    parser.add_argument(
        "--conv1-in-channels",
        type=int,
        default=1,
        help="conv1 in channels (default: 1)",
    )
    parser.add_argument(
        "--conv1-out-channels",
        type=int,
        default=256,
        help="conv1 out channels (default: 256)",
    )
    parser.add_argument(
        "--conv1-kernel-size",
        type=int,
        default=9,
        help="conv1 kernel size (default: 9)",
    )
    parser.add_argument(
        "--conv1-stride", type=int, default=1, help="conv1 stride (default: 1)"
    )
    parser.add_argument(
        "--conv2-kernel-size",
        type=int,
        default=9,
        help="conv2 kernel size (default: 9)",
    )
    parser.add_argument(
        "--conv2-stride", type=int, default=2, help="conv2 stride (default: 2)"
    )
    parser.add_argument(
        "--primary-caps-output-dim",
        type=int,
        default=8,
        help="primary caps output dim (default: 8)",
    )
    parser.add_argument(
        "--primary-caps-output-caps",
        type=int,
        default=32,
        help="primary caps output caps (default: 32)",
    )
    parser.add_argument(
        "--number-of-primary-caps-yx",
        type=int,
        default=36,
        help="number of primary caps yx (default: 36)",
    )
    parser.add_argument(
        "--caps-layer-output-dim",
        type=int,
        default=16,
        help="caps layer output dim (default: 16)",
    )
    parser.add_argument(
        "--fc1-out-features",
        type=int,
        default=512,
        help="fc1 out features (default: 512)",
    )
    parser.add_argument(
        "--fc2-out-features",
        type=int,
        default=1024,
        help="fc2 out features (default: 1024)",
    )
    parser.add_argument(
        "--fc3-out-features",
        type=int,
        default=784,
        help="fc3 out features (default: 784)",
    )
    # reconstruction
    parser.add_argument(
        "--with-reconstruction",
        action="store_true",
        default=False,
        help="use reconstruction loss (default: False)",
    )
    parser.add_argument(
        "--mask-reconstruction",
        action="store_true",
        default=True,
        help="mask reconstruction loss (default: True)",
    )
    parser.add_argument(
        "--reconstruction-alpha",
        type=float,
        default=0.0005,
        help="reconstruction loss weight (default: 0.0005)",
    )
    # loss function
    parser.add_argument(
        "--m-pos",
        type=float,
        default=0.9,
        help="m+ parameter for margin loss (default: 0.9)",
    )
    parser.add_argument(
        "--m-neg",
        type=float,
        default=0.1,
        help="m- parameter for margin loss (default: 0.1)",
    )
    parser.add_argument(
        "--loss-lambda",
        type=float,
        default=0.5,
        help="lambda parameter for margin loss (default: 0.5)",
    )
    # optimizer
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        help="optimizer (default: adam)",
        choices=["adam", "madam"],
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="patience for ReduceLROnPlateau (default: 15)",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=1e-6,
        help="min lr for ReduceLROnPlateau (default: 1e-6)",
    )
    # nnmf
    parser.add_argument(
        "--nnmf-number-of-iterations",
        type=int,
        default=5,
        help="number of iterations for NNMF (default: 5)",
    )
    parser.add_argument(
        "--nnmf-threshold",
        type=float,
        default=0.00001,
        help="threshold for NNMF (default: 0.00001)",
    )
    # model path
    parser.add_argument(
        "--model-path",
        type=str,
        default="models",
        help="path to save models (default: models)",
    )

    return parser


# -------
