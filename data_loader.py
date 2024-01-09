import torch
import torchvision as tv


def data_loader(
    batch_size: int, test_batch_size: int
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    train_loader = torch.utils.data.DataLoader(
        tv.datasets.MNIST(
            "../data",
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
            "../data",
            train=False,
            transform=tv.transforms.Compose([tv.transforms.ToTensor()]),
        ),
        batch_size=test_batch_size,
        shuffle=False,
    )

    return train_loader, test_loader
