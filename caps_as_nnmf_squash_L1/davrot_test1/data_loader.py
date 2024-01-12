import torch
import torchvision as tv


def data_loader(
    dataset: str, batch_size: int, test_batch_size: int
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    if dataset == "mnist":
        train_loader = torch.utils.data.DataLoader(
            tv.datasets.MNIST(
                "~/data",
                train=True,
                download=True,
                transform=tv.transforms.Compose(
                    [
                        tv.transforms.Pad(2),
                        tv.transforms.RandomCrop(28),
                        tv.transforms.ToTensor(),
                    ]
                ),
            ),
            batch_size=batch_size,
            shuffle=True,
        )

        test_loader = torch.utils.data.DataLoader(
            tv.datasets.MNIST(
                "~/data",
                train=False,
                transform=tv.transforms.Compose([tv.transforms.ToTensor()]),
            ),
            batch_size=test_batch_size,
            shuffle=False,
        )
    elif dataset == "fashion-mnist":
        train_loader = torch.utils.data.DataLoader(
            tv.datasets.FashionMNIST(
                "~/data",
                train=True,
                download=True,
                transform=tv.transforms.Compose(
                    [
                        tv.transforms.Pad(2),
                        tv.transforms.RandomCrop(28),
                        tv.transforms.ToTensor(),
                    ]
                ),
            ),
            batch_size=batch_size,
            shuffle=True,
        )

        test_loader = torch.utils.data.DataLoader(
            tv.datasets.FashionMNIST(
                "~/data",
                train=False,
                transform=tv.transforms.Compose([tv.transforms.ToTensor()]),
            ),
            batch_size=test_batch_size,
            shuffle=False,
        )
    elif dataset == "cifar10":
        train_loader = torch.utils.data.DataLoader(
            tv.datasets.CIFAR10(
                "~/data",
                train=True,
                download=True,
                transform=tv.transforms.Compose(
                    [
                        transforms.RandomCrop(28),
                        transforms.RandomHorizontalFlip(0.5),
                        transforms.ColorJitter(
                            brightness=0.5, contrast=0.1, saturation=0.1, hue=0.15
                        ),
                        transforms.ToTensor(),
                    ]
                ),
            ),
            batch_size=batch_size,
            shuffle=True,
        )

        test_loader = torch.utils.data.DataLoader(
            tv.datasets.CIFAR10(
                "~/data",
                train=False,
                transform=tv.transforms.Compose(
                    [
                        transforms.CenterCrop(28),
                        transforms.ToTensor(),
                    ]
                ),
            ),
            batch_size=test_batch_size,
            shuffle=False,
        )

    return train_loader, test_loader
