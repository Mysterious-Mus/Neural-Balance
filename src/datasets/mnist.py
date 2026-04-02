from __future__ import annotations

from typing import Tuple

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision import transforms as T


def get_mnist_loaders(
    batch_size: int,
    seed: int,
    *,
    train_fraction: float = 1.0,
    test: bool = False,
    normalize: bool = True,
    data_root: str = "./data",
) -> DataLoader:
    """Build a MNIST DataLoader (train or test)."""
    transform_list = [T.ToTensor()]
    if normalize:
        # Normalize around mean 0.5, std 0.5 after scaling to [0, 1].
        transform_list.append(T.Normalize((0.5,), (0.5,)))
    transform = T.Compose(transform_list)

    mnist_train = datasets.MNIST(
        root=data_root, train=True, download=True, transform=transform
    )
    mnist_test = datasets.MNIST(
        root=data_root, train=False, download=True, transform=transform
    )

    if train_fraction < 1.0:
        print("using a fraction of the training dataset")
        labels = [label for _, label in mnist_train]
        _, subset_indices = train_test_split(
            range(len(mnist_train)),
            test_size=train_fraction,
            random_state=seed,
            stratify=labels,
        )
        mnist_train = Subset(mnist_train, subset_indices)

    if not test:
        return DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    return DataLoader(mnist_test, batch_size=batch_size, shuffle=False)


def get_train_and_test_loaders(
    batch_size: int,
    seed: int,
    *,
    train_fraction: float = 1.0,
    normalize: bool = True,
    data_root: str = "./data",
) -> Tuple[DataLoader, DataLoader]:
    train = get_mnist_loaders(
        batch_size,
        seed,
        train_fraction=train_fraction,
        test=False,
        normalize=normalize,
        data_root=data_root,
    )
    test = get_mnist_loaders(
        batch_size,
        seed,
        train_fraction=1.0,
        test=True,
        normalize=normalize,
        data_root=data_root,
    )
    return train, test

