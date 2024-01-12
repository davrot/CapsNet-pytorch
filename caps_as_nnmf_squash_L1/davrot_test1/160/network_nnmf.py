import torch
from ProbabilityLayer import ProbabilityLayer
from MaskLayer import MaskLayer

# from SquashLayer_mod import SquashLayer
from PrimaryCapsReshapeLayer import PrimaryCapsReshapeLayer
from CapsLayer_nnmf_160 import CapsLayer

# from CapsLayer_new import CapsLayer


def network(
    number_of_classes: int = 10,
    conv1_in_channels: int = 1,
    conv1_out_channels: int = 256,
    conv1_kernel_size: int = 9,
    conv1_stride: int = 1,
    conv2_kernel_size: int = 9,
    conv2_stride: int = 2,
    primary_caps_output_dim: int = 8,
    primary_caps_output_caps: int = 32,
    number_of_primary_caps_yx=36,  # 6 * 6
    caps_layer_output_dim: int = 16,
    fc1_out_features: int = 512,
    fc2_out_features: int = 1024,
    fc3_out_features: int = 784,
    with_reconstruction: bool = True,
    nnmf_number_of_iterations: int = 5,
    nnmf_threshold: float = 0.00001,
) -> tuple[torch.nn.Sequential, int, int, int]:
    fc1_in_features: int = caps_layer_output_dim * number_of_classes
    number_of_primary_caps: int = primary_caps_output_caps * number_of_primary_caps_yx
    conv2_out_channels: int = primary_caps_output_dim * primary_caps_output_caps

    model = torch.nn.Sequential()
    model.append(
        torch.nn.Conv2d(
            in_channels=conv1_in_channels,
            out_channels=conv1_out_channels,
            kernel_size=conv1_kernel_size,
            stride=conv1_stride,
        )
    )
    model.append(torch.nn.ReLU())
    model.append(
        torch.nn.Conv2d(
            in_channels=conv1_out_channels,
            out_channels=conv2_out_channels,
            kernel_size=conv2_kernel_size,
            stride=conv2_stride,
        )
    )
    model.append(
        PrimaryCapsReshapeLayer(
            output_caps=primary_caps_output_caps, output_dim=primary_caps_output_dim
        )
    )
    model.append(torch.nn.ReLU())
    nnmf_layer_position: int = len(model)
    model.append(
        CapsLayer(
            input_caps=number_of_primary_caps,
            input_dim=primary_caps_output_dim,
            output_caps=number_of_classes,
            output_dim=caps_layer_output_dim,
            number_of_iterations=nnmf_number_of_iterations,
            threshold=nnmf_threshold,
        )
    )
    # model.append(torch.nn.ReLU())
    # model.append(SquashLayer())
    # model.append(torch.nn.ReLU())
    probability_layer_position: int = len(model)
    model.append(ProbabilityLayer())

    if with_reconstruction:
        mask_layer_position: int = len(model)
        model.append(MaskLayer())
        model.append(torch.nn.Flatten(start_dim=1, end_dim=-1))
        model.append(
            torch.nn.Linear(in_features=fc1_in_features, out_features=fc1_out_features)
        )
        model.append(torch.nn.ReLU())
        model.append(
            torch.nn.Linear(in_features=fc1_out_features, out_features=fc2_out_features)
        )
        model.append(torch.nn.ReLU())
        model.append(
            torch.nn.Linear(in_features=fc2_out_features, out_features=fc3_out_features)
        )
        model.append(torch.nn.Sigmoid())
    else:
        mask_layer_position = -1

    return model, probability_layer_position, mask_layer_position, nnmf_layer_position
